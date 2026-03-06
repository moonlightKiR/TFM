"""
transformer/attention.py — Mecanismo de atención para el Transformer.

Contiene:
  - RotaryEmbedding (RoPE): codificación posicional rotacional
  - GroupedQueryAttention (GQA): atención eficiente con cabezas K/V agrupadas

Ambos componentes siguen el diseño de LLaMA 2/3 y Mistral.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import SLMConfig


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) — Su et al., 2021.

    Codifica la posición de cada token rotando los vectores Q y K en el
    espacio complejo. Ventajas sobre embeddings posicionales aprendidos:
      - No añade parámetros entrenables
      - Generaliza mejor a longitudes no vistas
      - El producto escalar Q·K codifica distancia relativa, no absoluta

    Cómo funciona:
      1. Pre-computa frecuencias: θ_i = 1 / (10000^(2i/d))
      2. Para posición t: rotación = e^(i·t·θ_i)
      3. Multiplica Q y K (como números complejos) por las rotaciones
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096,
                 theta: float = 10_000.0):
        super().__init__()
        # Frecuencias base: una por cada par de dimensiones
        freqs = 1.0 / (theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        ))
        # Posiciones: 0, 1, 2, ..., max_seq_len-1
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # Producto externo: (max_seq_len, head_dim//2)
        freqs = torch.outer(t, freqs)
        # Convertir a forma polar compleja: e^(i·θ)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Aplica RoPE a x de shape (B, n_head, T, head_dim).

        Parameters
        ----------
        x      : tensor Q o K con shape (B, H, T, D)
        offset : posición inicial (para generación incremental)
        """
        B, H, T, D = x.shape
        freqs = self.freqs_cis[offset: offset + T]
        # Interpretar pares consecutivos como números complejos
        xc = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
        # Multiplicar por rotaciones posicionales
        out = torch.view_as_real(xc * freqs.unsqueeze(0).unsqueeze(0))
        return out.reshape(B, H, T, D).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) con RoPE y Flash Attention.

    Tres modos según la relación entre n_head y n_kv_head:
      - n_kv_head == n_head   →  Multi-Head Attention (MHA) estándar
      - n_kv_head == 1        →  Multi-Query Attention (MQA)
      - 1 < n_kv_head < n_head → Grouped Query Attention (GQA)
        Cada grupo de (n_head // n_kv_head) cabezas Q comparte 1 cabeza K/V.

    GQA reduce el uso de memoria ~50% vs MHA con mínima pérdida de calidad.
    Usado en LLaMA 2/3, Mistral, Gemma, Phi-3.

    Flash Attention se activa automáticamente via F.scaled_dot_product_attention
    en dispositivos MPS (Apple Silicon) y CUDA.
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, \
            f"n_embd ({cfg.n_embd}) debe ser divisible por n_head ({cfg.n_head})"
        assert cfg.n_head % cfg.n_kv_head == 0, \
            f"n_head ({cfg.n_head}) debe ser divisible por n_kv_head ({cfg.n_kv_head})"

        self.n_head    = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep     = cfg.n_head // cfg.n_kv_head   # repeticiones K/V
        self.head_dim  = cfg.n_embd // cfg.n_head
        self.dropout_p = cfg.dropout

        # Proyecciones lineales (sin bias — estilo LLaMA)
        self.q_proj  = nn.Linear(cfg.n_embd, cfg.n_head    * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj  = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

        self.proj_drop = nn.Dropout(cfg.dropout)

        # RoPE pre-computado — no necesita máscara causal manual
        self.rope = RotaryEmbedding(self.head_dim, cfg.n_positions, cfg.rope_theta)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repite las cabezas K/V para igualar el número de cabezas Q."""
        if self.n_rep == 1:
            return x   # MHA estándar — sin repetición
        B, H, T, D = x.shape
        return (x.unsqueeze(3)
                  .expand(B, H, T, self.n_rep, D)
                  .reshape(B, H * self.n_rep, T, D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_embd) → (B, T, n_embd)

        Flujo:
          1. Proyectar a Q, K, V
          2. Aplicar RoPE a Q y K
          3. Repetir K/V para GQA
          4. Flash Attention con máscara causal
          5. Proyección de salida
        """
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Aplicar codificación posicional rotacional
        q = self.rope(q)
        k = self.rope(k)

        # Repetir K/V para igualar cabezas Q (GQA)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Flash Attention — optimizado para CUDA y MPS
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_drop(self.o_proj(out))
