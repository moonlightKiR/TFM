"""
shared.py — Componentes compartidos entre Transformer y RNN.

Contiene:
  - RMSNorm: normalización usada en ambas ramas.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Más simple y rápido que LayerNorm clásico:
      - No calcula media (solo RMS)
      - Solo tiene parámetro de escala γ (sin sesgo β)
      - ~10-15% más rápido que LayerNorm

    Usado en LLaMA, Mistral, Gemma y prácticamente todos los LLM modernos.

    Fórmula: y = x / sqrt(mean(x²) + ε) * γ
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
