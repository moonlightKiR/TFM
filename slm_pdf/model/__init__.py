"""
model — Paquete modular del Small Language Model (SLM).

Estructura:
    model/
    ├── config.py           ← SLMConfig (tamaños y hiperparámetros)
    ├── shared.py           ← RMSNorm (compartido Transformer ↔ RNN)
    ├── base.py             ← SmallLM (modelo solo Transformer)
    ├── hybrid.py           ← HybridSmallLM + build_model() + load_model()
    ├── transformer/
    │   ├── attention.py    ← RotaryEmbedding + GroupedQueryAttention
    │   ├── ffn.py          ← SwiGLU Feed-Forward Network
    │   └── block.py        ← TransformerBlock
    └── rnn/
        ├── gru.py          ← GRUBlock (GRU con fix MPS float32)
        └── block.py        ← RecurrentBlock (GRU + SwiGLU FFN)

Uso:
    from model import SLMConfig, SmallLM, HybridSmallLM, build_model
    cfg = SLMConfig.small(vocab_size=8000)
    model = SmallLM(cfg)               # Solo Transformer
    cfg.architecture = "hybrid"
    model = build_model(cfg)           # Híbrido Transformer + GRU
"""

from model.config import SLMConfig
from model.shared import RMSNorm
from model.base import SmallLM
from model.hybrid import HybridSmallLM, build_model, load_model

# Submódulos para acceso directo
from model.transformer import TransformerBlock, GroupedQueryAttention, RotaryEmbedding, SwiGLU
from model.rnn import GRUBlock, RecurrentBlock

__all__ = [
    # Configuración
    "SLMConfig",
    # Componentes compartidos
    "RMSNorm",
    # Transformer
    "RotaryEmbedding",
    "GroupedQueryAttention",
    "SwiGLU",
    "TransformerBlock",
    # RNN
    "GRUBlock",
    "RecurrentBlock",
    # Modelos
    "SmallLM",
    "HybridSmallLM",
    # Factory
    "build_model",
    "load_model",
]
