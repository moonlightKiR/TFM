"""
config.py — Configuración del modelo SLM.

Define SLMConfig con 6 tamaños predefinidos (micro→xl) y soporte para
arquitectura transformer | hybrid.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path


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
        """~2M params — smoke-test en segundos, cualquier CPU."""
        return cls(vocab_size=vocab_size,
                   n_layer=2, n_head=4, n_kv_head=4, n_embd=128,
                   n_positions=256)

    @classmethod
    def tiny(cls, vocab_size: int) -> "SLMConfig":
        """~7M params — CPU en minutos, ideal para 1-5 PDFs."""
        return cls(vocab_size=vocab_size,
                   n_layer=4, n_head=4, n_kv_head=4, n_embd=256,
                   n_positions=512)

    @classmethod
    def small(cls, vocab_size: int) -> "SLMConfig":
        """~22M params — Mac M1/M2/M3 MPS, 5-20 PDFs."""
        return cls(vocab_size=vocab_size,
                   n_layer=6, n_head=6, n_kv_head=6, n_embd=384,
                   n_positions=512)

    @classmethod
    def medium(cls, vocab_size: int) -> "SLMConfig":
        """~85M params — GPU >=8 GB, 20-100 PDFs."""
        return cls(vocab_size=vocab_size,
                   n_layer=12, n_head=8, n_kv_head=8, n_embd=512,
                   n_positions=1024)

    @classmethod
    def large(cls, vocab_size: int) -> "SLMConfig":
        """~250M params — GPU >=16 GB, corpus grande."""
        return cls(vocab_size=vocab_size,
                   n_layer=16, n_head=12, n_kv_head=4, n_embd=768,
                   n_positions=2048)

    @classmethod
    def xl(cls, vocab_size: int) -> "SLMConfig":
        """~500M params — GPU >=24 GB, corpus muy grande."""
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
