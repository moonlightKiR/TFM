"""
rnn/gru.py — Bloque GRU básico con pre-normalización.

GRU (Gated Recurrent Unit) captura dependencias secuenciales locales
con O(T) de memoria (vs O(T²) del Transformer).

IMPORTANTE — Compatibilidad MPS (Apple Silicon):
  nn.GRU no soporta bfloat16 en MPS (error de tipos en
  MetalPerformanceShadersGraph). Se fuerza float32 en el GRU
  con conversión automática en forward().
"""

import torch
import torch.nn as nn

from model.config import SLMConfig
from model.shared import RMSNorm


class GRUBlock(nn.Module):
    """
    Bloque GRU simple con Pre-RMSNorm y conexión residual.

    Flujo:
        x → RMSNorm → GRU(float32) → Linear → residual

    El GRU procesa la secuencia de forma causal (unidireccional),
    token a token. Mantiene un estado oculto h_t que acumula
    información secuencial.

    Ecuaciones del GRU (para cada token t):
        z_t = σ(W_z·[h_{t-1}, x_t])      ← puerta de actualización
        r_t = σ(W_r·[h_{t-1}, x_t])      ← puerta de reset
        ĥ_t = tanh(W·[r_t * h_{t-1}, x_t])  ← candidato
        h_t = (1 - z_t) * h_{t-1} + z_t * ĥ_t  ← nuevo estado

    La puerta z decide cuánto del estado anterior conservar.
    La puerta r decide qué parte del pasado es relevante.
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.n_embd)
        self.gru  = nn.GRU(
            input_size=cfg.n_embd,
            hidden_size=cfg.n_embd,
            num_layers=1,
            batch_first=True,
            bidirectional=False,  # Causal: solo ve tokens anteriores
            dropout=0.0,
        )
        # Proyección de salida (mezcla la representación de vuelta)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop     = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------
    # Mantener GRU en float32 (fix MPS bfloat16)
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs) -> "GRUBlock":
        super().to(*args, **kwargs)
        self.gru.float()
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
        gru_out    = gru_out.to(x.dtype)
        out = self.drop(self.out_proj(gru_out))
        return residual + out
