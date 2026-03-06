"""
model.py — Módulo de compatibilidad.

Este archivo re-exporta todo desde el paquete model/ para que los imports
existentes (train.py, qa_pipeline.py) sigan funcionando sin cambios:

    from model import SmallLM, SLMConfig, build_model

La implementación real está en model/ (ver model/__init__.py).
"""

# Re-exports desde el paquete model/
from model import (         # noqa: F401
    SLMConfig,
    RMSNorm,
    SmallLM,
    HybridSmallLM,
    build_model,
    load_model,
    # Transformer components
    RotaryEmbedding,
    GroupedQueryAttention,
    SwiGLU,
    TransformerBlock,
    # RNN components
    GRUBlock,
    RecurrentBlock,
)
