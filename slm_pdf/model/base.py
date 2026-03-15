"""
base.py — SmallLM: Transformer Decoder-Only desde cero.

SmallLM es el modelo base (solo bloques Transformer). HybridSmallLM hereda
de este y alterna bloques Transformer con GRU.

Este módulo contiene:
  - SmallLM: modelo completo con forward, generate, save, load
  - _init_weights: inicialización de pesos estilo LLaMA
  - Utilidades de tamaño y etiquetado
"""

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import SLMConfig
from model.shared import RMSNorm
from model.transformer.block import TransformerBlock


class SmallLM(nn.Module):
    """
    Small Language Model — Transformer Decoder-Only desde cero.

    Arquitectura moderna estilo LLaMA con:
      - RMSNorm (pre-normalización)
      - SwiGLU Feed-Forward Networks
      - Rotary Position Embeddings (RoPE)
      - Grouped Query Attention (GQA)
      - Flash Attention via F.scaled_dot_product_attention
      - Weight tying (embedding ↔ LM head)

    Parameters
    ----------
    cfg : SLMConfig
        Configuración del modelo (tamaño, cabezas, dimensiones, etc.)
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.cfg = cfg

        # Embedding de tokens
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)

        # Stack de bloques Transformer
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])

        # Normalización final + cabeza de lenguaje
        self.norm_f  = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying

        # Inicialización de pesos
        self.apply(self._init_weights)
        # Escalar las proyecciones de salida de atención y FFN
        scale = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=scale)

        # Log del modelo
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
        """Etiqueta de tamaño basada en el número total de parámetros."""
        thresholds = [(3e6, "micro"), (10e6, "tiny"), (30e6, "small"),
                      (150e6, "medium"), (400e6, "large")]
        n = sum(p.numel() for p in self.parameters())
        for lim, label in thresholds:
            if n < lim:
                return label
        return "xl"

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Inicialización de pesos estilo LLaMA/GPT-2."""
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
        """
        Forward pass.

        Parameters
        ----------
        input_ids : (B, T) tensor de IDs de tokens
        labels    : (B, T) tensor de labels (si es training)

        Returns
        -------
        Si labels: (loss, logits)
        Si no:     logits
        """
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
        """
        Genera tokens usando nucleus (top-p) y/o top-k sampling.

        Parameters
        ----------
        input_ids          : (1, T) prompt tokenizado
        max_new_tokens     : máximo de tokens a generar
        temperature        : controla la aleatoriedad (>1 más random, <1 más greedy)
        top_p              : probabilidad acumulada máxima (nucleus sampling)
        top_k              : si >0, solo top-k tokens más probables
        eos_token_id       : token de parada
        repetition_penalty : penaliza la repetición de tokens ya generados
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            ctx    = generated[:, -self.cfg.n_positions:]
            logits = self(ctx)[:, -1, :]

            # Penalización de repetición
            if repetition_penalty != 1.0:
                # Sólo penalizar los tokens que ha generado el modelo
                # No penalizamos el prompt inicial, para que no "prohíba"
                # las palabras que le hemos dado en la pregunta
                prompt_len = input_ids.shape[1]
                generated_tokens = generated[:, prompt_len:]
                
                # Check if generated_tokens is not empty
                if generated_tokens.shape[1] > 0:
                    for tid in generated_tokens[0].unique():
                        if logits[0, tid] > 0:
                            logits[0, tid] /= repetition_penalty
                        else:
                            logits[0, tid] *= repetition_penalty

            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits[logits < kth] = float("-inf")

            # Nucleus (top-p) sampling
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
        """Guarda la configuración y pesos en save_dir."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.save(save_dir / "config.json")
        torch.save(self.state_dict(), save_dir / "model.pt")
        print(f"✅ Modelo guardado en: {save_dir}")

    @classmethod
    def load(cls, load_dir: Path | str, map_location: str = "cpu") -> "SmallLM":
        """Carga un modelo guardado desde load_dir."""
        load_dir = Path(load_dir)
        cfg   = SLMConfig.load(load_dir / "config.json")
        model = cls(cfg)
        state = torch.load(load_dir / "model.pt",
                           map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model
