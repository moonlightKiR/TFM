"""
transformer/ffn.py — Feed-Forward Network (SwiGLU) del Transformer.

SwiGLU es la variante gated de SiLU usada en LLaMA, PaLM, Gemma y otros.
Reemplaza al MLP clásico (Linear → GELU → Linear) con:

    SwiGLU(x) = SiLU(x @ W_gate) * (x @ W_up) → @ W_down

Tiene ~13% más parámetros que un MLP clásico de igual dimensión, pero
produce mejores resultados por parámetro.
"""

import torch.nn as nn
import torch.nn.functional as F

from model.config import SLMConfig


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Arquitectura de 3 matrices (vs 2 del MLP clásico):
      gate_proj: n_embd → hidden  (rama gate, pasa por SiLU)
      up_proj:   n_embd → hidden  (rama linear)
      down_proj: hidden → n_embd  (reduce de vuelta)

    hidden = round_up(n_embd * ffn_mult, 256)
    El redondeo a múltiplo de 256 optimiza el uso de tensor cores en GPU.
    """

    def __init__(self, cfg: SLMConfig):
        super().__init__()
        hidden = int(cfg.n_embd * cfg.ffn_mult)
        hidden = (hidden + 255) // 256 * 256  # múltiplo de 256

        self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up_proj   = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop      = nn.Dropout(cfg.dropout)

    def forward(self, x):
        """x: (B, T, n_embd) → (B, T, n_embd)"""
        gate = F.silu(self.gate_proj(x))   # SiLU activation (gating)
        up   = self.up_proj(x)             # Linear (sin activación)
        return self.drop(self.down_proj(gate * up))
