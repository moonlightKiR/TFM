"""
model.py
--------
Transformer Decoder-Only (Causal LM) implementado 100% desde cero con PyTorch.
Arquitectura modernizada al estilo LLaMA/Mistral con:

  ✦ RMSNorm          — más estable y rápido que LayerNorm
  ✦ SwiGLU           — activación gated (mejor que GELU puro)
  ✦ RoPE             — Rotary Position Embeddings (mejor generalización)
  ✦ GQA              — Grouped Query Attention (memoria eficiente)
  ✦ Flash Attention  — via F.scaled_dot_product_attention (MPS + CUDA)
  ✦ Weight tying     — embedding ↔ LM head (ahorro de parámetros)
  ✦ 6 tamaños        — micro / tiny / small / medium / large / xl

Uso:
    from model import SmallLM, SLMConfig
    cfg = SLMConfig.small(vocab_size=8000)
    model = SmallLM(cfg)
    loss, logits = model(input_ids, labels=labels)   # train
    tokens = model.generate(input_ids)               # inference
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuración
# ============================================================================

@dataclass
class SLMConfig:
    """Hiperparámetros del modelo."""
    vocab_size:   int   = 8_000
    n_positions:  int   = 512
    n_layer:      int   = 6
    n_head:       int   = 6
    n_kv_head:    int   = 6
    n_embd:       int   = 384
    ffn_mult:     float = 8/3
    dropout:      float = 0.0
    bias:         bool  = False
    rope_theta:   float = 10_000.0
    architecture: str   = "transformer"   # "transformer" | "hybrid"
    rnn_ratio:    float = 0.5             # fracción de capas GRU en modo hybrid

    # ------------------------------------------------------------------
    # Tamaños predefinidos
    # ------------------------------------------------------------------
    @classmethod
    def micro(cls, vocab_size: int) -> "SLMConfig":
        """~2M parámetros — prueba en segundos, cualquier CPU."""
        return cls(vocab_size=vocab_size,
                   n_layer=2, n_head=4, n_kv_head=4, n_embd=128,
                   n_positions=256)

    @classmethod
    def tiny(cls, vocab_size: int) -> "SLMConfig":
        """~7M parámetros — CPU en minutos, ideal para 1-5 PDFs."""
        return cls(vocab_size=vocab_size,
                   n_layer=4, n_head=4, n_kv_head=4, n_embd=256,
                   n_positions=512)

    @classmethod
    def small(cls, vocab_size: int) -> "SLMConfig":
        """~22M parámetros — Mac M1/M2/M3 MPS, 5-20 PDFs."""
        return cls(vocab_size=vocab_size,
                   n_layer=6, n_head=6, n_kv_head=6, n_embd=384,
                   n_positions=512)

    @classmethod
    def medium(cls, vocab_size: int) -> "SLMConfig":
        """~85M parámetros — GPU >=8 GB, 20-100 PDFs."""
        return cls(vocab_size=vocab_size,
                   n_layer=12, n_head=8, n_kv_head=8, n_embd=512,
                   n_positions=1024)

    @classmethod
    def large(cls, vocab_size: int) -> "SLMConfig":
        """~250M parámetros — GPU >=16 GB, corpus grande."""
        return cls(vocab_size=vocab_size,
                   n_layer=16, n_head=12, n_kv_head=4, n_embd=768,
                   n_positions=2048)

    @classmethod
    def xl(cls, vocab_size: int) -> "SLMConfig":
        """~500M parámetros — GPU >=24 GB, corpus muy grande."""
        return cls(vocab_size=vocab_size,
                   n_layer=24, n_head=16, n_kv_head=4, n_embd=1024,
                   n_positions=2048)

    def save(self, path: Path | str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "SLMConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


# ============================================================================
# Bloques de arquitectura LLaMA-style
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Más simple y rápido que LayerNorm: solo normaliza por RMS y escala con γ.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Rota Q y K según su posición sin parámetros extra.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096,
                 theta: float = 10_000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        ))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Aplica RoPE a x: (B, n_head, T, head_dim)."""
        B, H, T, D = x.shape
        freqs = self.freqs_cis[offset: offset + T]
        xc = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
        out = torch.view_as_real(xc * freqs.unsqueeze(0).unsqueeze(0))
        return out.reshape(B, H, T, D).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) con RoPE y Flash Attention.
    - n_kv_head == n_head  → MHA estándar
    - n_kv_head == 1       → MQA
    - 1 < n_kv_head < n_head → GQA (LLaMA 2/3, Mistral)
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0

        self.n_head    = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep     = cfg.n_head // cfg.n_kv_head
        self.head_dim  = cfg.n_embd // cfg.n_head
        self.dropout_p = cfg.dropout

        self.q_proj  = nn.Linear(cfg.n_embd, cfg.n_head    * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj  = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        self.proj_drop = nn.Dropout(cfg.dropout)

        # RoPE — no necesita máscara manual: is_causal=True en sdpa
        self.rope = RotaryEmbedding(self.head_dim, cfg.n_positions, cfg.rope_theta)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        return (x.unsqueeze(3)
                  .expand(B, H, T, self.n_rep, D)
                  .reshape(B, H * self.n_rep, T, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Flash Attention (optimizado para CUDA y MPS via PyTorch sdpa)
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.o_proj(out))


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network (LLaMA, PaLM, Gemma).
    SwiGLU(x) = SiLU(x @ W_gate) * (x @ W_up) → @ W_down
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        hidden = int(cfg.n_embd * cfg.ffn_mult)
        hidden = (hidden + 255) // 256 * 256  # múltiplo de 256

        self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up_proj   = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop      = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.drop(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    """Pre-RMSNorm + GQA + SwiGLU — estilo LLaMA."""

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn      = GroupedQueryAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.n_embd)
        self.ffn       = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ============================================================================
# Bloque RNN (GRU) con pre-normalización y conexión residual
# ============================================================================

class GRUBlock(nn.Module):
    """
    Bloque GRU con Pre-RMSNorm y conexión residual.
    Captura dependencias secuenciales locales de forma eficiente:
      O(T) en memoria vs O(T²) del Transformer.

    Diseño inspirado en Griffin/Hawk (Google DeepMind, 2024).

    NOTA: nn.GRU no soporta bfloat16 en MPS (error de verficiación de tipos
    en MetalPerformanceShadersGraph). Por ello, el GRU interno se mantiene
    siempre en float32, con conversión automática en forward().
    El resto del bloque (norm, out_proj) sí usa el dtype del modelo.
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.n_embd)
        self.gru  = nn.GRU(
            input_size=cfg.n_embd,
            hidden_size=cfg.n_embd,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0,
        )
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop     = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------
    # Mantener GRU en float32 aunque el resto del modelo use bfloat16/half
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs) -> "GRUBlock":
        super().to(*args, **kwargs)
        self.gru.float()   # restaurar GRU a float32 tras cualquier conversión
        return self

    def bfloat16(self) -> "GRUBlock":
        super().bfloat16()
        self.gru.float()
        return self

    def half(self) -> "GRUBlock":
        super().half()
        self.gru.float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_embd) → (B, T, n_embd) con residual."""
        residual = x
        x_norm   = self.norm(x)
        # GRU siempre en float32 (compatibilidad MPS)
        gru_out, _ = self.gru(x_norm.float())
        gru_out    = gru_out.to(x.dtype)          # volver al dtype del modelo
        out = self.drop(self.out_proj(gru_out))
        return residual + out


# ============================================================================
# SmallLM — solo Transformer
# ============================================================================

class SmallLM(nn.Module):
    """
    Small Language Model — Transformer Decoder-Only desde cero.
    Arquitectura LLaMA-style con RMSNorm + SwiGLU + RoPE + Flash Attention.
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f  = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=scale)

        n_params = sum(p.numel() for p in self.parameters())
        n_emb    = self.tok_emb.weight.numel()
        print(
            f"🤖 SmallLM [{self._size_label()}] inicializado:\n"
            f"   Layers: {cfg.n_layer} | Heads Q: {cfg.n_head} | "
            f"Heads KV: {cfg.n_kv_head} | Hidden: {cfg.n_embd}\n"
            f"   Norm: RMSNorm | Attn: GQA+RoPE | FFN: SwiGLU\n"
            f"   Vocab: {cfg.vocab_size:,} | Contexto: {cfg.n_positions}\n"
            f"   Parámetros totales: {n_params/1e6:.2f}M "
            f"(únicos: {(n_params - n_emb)/1e6:.2f}M con weight tying)"
        )

    def _size_label(self) -> str:
        thresholds = [(3e6,"micro"),(10e6,"tiny"),(30e6,"small"),
                      (150e6,"medium"),(400e6,"large")]
        n = sum(p.numel() for p in self.parameters())
        for lim, label in thresholds:
            if n < lim:
                return label
        return "xl"

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, RMSNorm):
            nn.init.ones_(m.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor] | torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.cfg.n_positions, \
            f"Secuencia {T} > n_positions {self.cfg.n_positions}"

        x = self.drop(self.tok_emb(input_ids))
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return loss, logits

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Genera tokens con nucleus (top-p) y/o top-k sampling."""
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            ctx    = generated[:, -self.cfg.n_positions:]
            logits = self(ctx)[:, -1, :]

            if repetition_penalty != 1.0:
                for tid in generated[0].unique():
                    if logits[0, tid] > 0:
                        logits[0, tid] /= repetition_penalty
                    else:
                        logits[0, tid] *= repetition_penalty

            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits[logits < kth] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = probs.sort(descending=True)
            cumulative  = sorted_probs.cumsum(-1)
            to_remove   = cumulative - sorted_probs > top_p
            sorted_probs[to_remove] = 0.0
            sorted_probs /= sorted_probs.sum(-1, keepdim=True)

            next_token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]
            generated  = torch.cat([generated, next_token.view(1, 1)], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated

    def save(self, save_dir: Path | str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.save(save_dir / "config.json")
        torch.save(self.state_dict(), save_dir / "model.pt")
        print(f"✅ Modelo guardado en: {save_dir}")

    @classmethod
    def load(cls, load_dir: Path | str, map_location: str = "cpu") -> "SmallLM":
        load_dir = Path(load_dir)
        cfg   = SLMConfig.load(load_dir / "config.json")
        model = cls(cfg)
        state = torch.load(load_dir / "model.pt",
                           map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model


# ============================================================================
# HybridSmallLM — alterna bloques Transformer y GRU
# ============================================================================

class HybridSmallLM(SmallLM):
    """
    Arquitectura híbrida Transformer + GRU, inspirada en Griffin/Hawk
    (Google DeepMind, 2024).

    Alterna bloques de atención (Transformer) y recurrencia (GRU):

        Token emb → [Transformer] → [GRU] → [Transformer] → [GRU] → ... → LM Head

    Ventajas sobre el Transformer puro:
      - GRU captura dependencias secuenciales locales con O(T) memoria
      - Transformer captura relaciones globales con O(T²) pero "ve" todo
      - La combinación aprovecha lo mejor de ambos mundos
      - Converge más rápido con datasets pequeños/medianos

    El parámetro `rnn_ratio` (en SLMConfig) controla qué fracción de las
    capas son GRU. Por ejemplo, rnn_ratio=0.5 con n_layer=6 da:
        [Attn, GRU, Attn, GRU, Attn, GRU]
    """

    def __init__(self, cfg: SLMConfig):
        # Inicializa la clase base (SmallLM) pero reemplazamos self.blocks
        nn.Module.__init__(self)   # saltar super().__init__() de SmallLM
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)

        # Construir lista de bloques alternados según rnn_ratio
        # rnn_ratio=0.5 → cada 2 capas, 1 es GRU
        blocks = []
        n_gru = max(1, round(cfg.n_layer * cfg.rnn_ratio))
        # Posiciones donde poner GRU: distribuidas uniformemente
        gru_positions = set(
            round(i * (cfg.n_layer - 1) / max(n_gru - 1, 1))
            for i in range(n_gru)
        ) if n_gru > 1 else {cfg.n_layer // 2}

        self.block_types: list[str] = []
        for i in range(cfg.n_layer):
            if i in gru_positions:
                blocks.append(GRUBlock(cfg))
                self.block_types.append("GRU")
            else:
                blocks.append(TransformerBlock(cfg))
                self.block_types.append("Attn")

        self.blocks  = nn.ModuleList(blocks)
        self.norm_f  = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight", "out_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=scale)

        n_params = sum(p.numel() for p in self.parameters())
        n_emb    = self.tok_emb.weight.numel()
        layout   = " → ".join(self.block_types)
        print(
            f"🤖 HybridSmallLM [{self._size_label()}] inicializado:\n"
            f"   Layout: {layout}\n"
            f"   Heads Q: {cfg.n_head} | Heads KV: {cfg.n_kv_head} | "
            f"Hidden: {cfg.n_embd}\n"
            f"   Vocab: {cfg.vocab_size:,} | Contexto: {cfg.n_positions}\n"
            f"   Parámetros totales: {n_params/1e6:.2f}M "
            f"(únicos: {(n_params - n_emb)/1e6:.2f}M con weight tying)"
        )

    @classmethod
    def load(cls, load_dir: Path | str, map_location: str = "cpu") -> "HybridSmallLM":
        load_dir = Path(load_dir)
        cfg   = SLMConfig.load(load_dir / "config.json")
        model = cls(cfg)
        state = torch.load(load_dir / "model.pt",
                           map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model


# ============================================================================
# Factory: construir el modelo correcto según cfg.architecture
# ============================================================================

def build_model(cfg: SLMConfig) -> SmallLM | HybridSmallLM:
    """Construye SmallLM o HybridSmallLM según cfg.architecture."""
    if cfg.architecture == "hybrid":
        return HybridSmallLM(cfg)
    return SmallLM(cfg)


# ============================================================================
# Smoke-test
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print(" Smoke-test: Transformer vs Híbrido")
    print("=" * 65)

    for arch in ["transformer", "hybrid"]:
        cfg = SLMConfig.small(vocab_size=8_000)
        cfg.n_positions  = 32
        cfg.architecture = arch
        model = build_model(cfg)

        B, T = 2, 16
        x      = torch.randint(0, cfg.vocab_size, (B, T))
        labels = x.clone()
        loss, logits = model(x, labels=labels)
        print(f"   [{arch:11s}] loss={loss.item():.4f} | logits={list(logits.shape)}")

        prompt = x[:1, :4]
        gen    = model.generate(prompt, max_new_tokens=8, temperature=0.8)
        print(f"   [{arch:11s}] generate OK -> {gen.shape[1]} tokens total\n")
