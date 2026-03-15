"""
transformer/block.py — Bloque Transformer completo.

Un TransformerBlock combina:
  1. Pre-RMSNorm → Grouped Query Attention → Residual
  2. Pre-RMSNorm → SwiGLU FFN → Residual

Este es el patrón Pre-Norm usado en GPT-2+, LLaMA, Mistral, etc.
(a diferencia del Post-Norm original de Vaswani 2017).

Pre-Norm es más estable durante el entrenamiento y permite eliminar
el warmup en muchos casos.
"""

import torch
import torch.nn as nn

from model.config import SLMConfig
from model.shared import RMSNorm
from model.transformer.attention import GroupedQueryAttention
from model.transformer.ffn import SwiGLU


class TransformerBlock(nn.Module):
    """
    Bloque Transformer Decoder con Pre-RMSNorm.

    Flujo:
        x → [+ Attn(RMSNorm(x))] → [+ FFN(RMSNorm(x))] → out

    Componentes:
      - RMSNorm: normalización pre-atención y pre-FFN
      - GQA: Grouped Query Attention con RoPE y Flash Attention
      - SwiGLU: Feed-Forward gated
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn      = GroupedQueryAttention(cfg)
        self.ffn_norm  = RMSNorm(cfg.n_embd)
        self.ffn       = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))   # Self-attention + residual
        x = x + self.ffn(self.ffn_norm(x))     # FFN + residual
        return x