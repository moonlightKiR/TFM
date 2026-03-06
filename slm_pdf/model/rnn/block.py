"""
rnn/block.py — RecurrentBlock mejorado (GRU + SwiGLU FFN).

Inspirado en Griffin/Hawk (Google DeepMind, 2024).

Diferencia con GRUBlock simple:
  - GRUBlock:      x → RMSNorm → GRU → proj → residual
  - RecurrentBlock: x → RMSNorm → GRU → proj → residual → RMSNorm → SwiGLU → residual

El RecurrentBlock tiene la misma estructura dual que el TransformerBlock:
  TransformerBlock = Attention + FFN
  RecurrentBlock   = GRU + FFN

Esto le da al bloque recurrente mayor capacidad de transformación no-lineal
de las representaciones, mejorando la calidad especialmente en textos con
estructura compleja (código, fórmulas matemáticas, documentación técnica).
"""

import torch
import torch.nn as nn

from model.config import SLMConfig
from model.shared import RMSNorm
from model.transformer.ffn import SwiGLU
from model.rnn.gru import GRUBlock


class RecurrentBlock(nn.Module):
    """
    Bloque recurrente completo: GRU + SwiGLU Feed-Forward.

    Flujo completo (equivalente al TransformerBlock pero con GRU):
        x → [+ GRU(RMSNorm(x))] → [+ SwiGLU(RMSNorm(x))] → out

    La adición del FFN (SwiGLU) es clave:
      - GRU solo mezcla información a lo largo de la secuencia (temporal)
      - SwiGLU transforma cada posición de forma no-lineal (feature mixing)
      - Juntos cubren las mismas funciones que el TransformerBlock

    Número de parámetros:
      RecurrentBlock ≈ TransformerBlock × 0.65
      (GRU es más compacto que la atención multi-cabeza)
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        # Rama recurrente: RMSNorm → GRU → proj → residual
        self.gru_block = GRUBlock(cfg)
        # Rama FFN: RMSNorm → SwiGLU → residual
        self.ffn_norm  = RMSNorm(cfg.n_embd)
        self.ffn       = SwiGLU(cfg)

    def to(self, *args, **kwargs) -> "RecurrentBlock":
        super().to(*args, **kwargs)
        # Propagar fix float32 al GRU interno
        self.gru_block.gru.float()
        return self

    def bfloat16(self) -> "RecurrentBlock":
        super().bfloat16()
        self.gru_block.gru.float()
        return self

    def half(self) -> "RecurrentBlock":
        super().half()
        self.gru_block.gru.float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_embd) → (B, T, n_embd)"""
        # Paso 1: GRU (mezcla secuencial + residual)
        x = self.gru_block(x)
        # Paso 2: FFN (transformación no-lineal + residual)
        x = x + self.ffn(self.ffn_norm(x))
        return x
