"""
hybrid.py — HybridSmallLM y factory build_model().

HybridSmallLM alterna bloques Transformer y RecurrentBlock (GRU + SwiGLU),
inspirado en Griffin/Hawk (Google DeepMind, 2024).

Ventajas del híbrido vs Transformer puro:
  - GRU captura dependencias secuenciales locales en O(T) de memoria
  - Transformer captura relaciones globales pero con O(T²)
  - La combinación aprovecha lo mejor de ambos
  - RecurrentBlock (GRU + SwiGLU) es más compacto que TransformerBlock
    (~35% menos parámetros por bloque), permitiendo modelos más eficientes
  - Converge más rápido con datasets pequeños/medianos
"""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from model.config import SLMConfig
from model.shared import RMSNorm
from model.base import SmallLM
from model.transformer.block import TransformerBlock
from model.rnn.block import RecurrentBlock


class HybridSmallLM(SmallLM):
    """
    Arquitectura híbrida Transformer + GRU.

    Alterna bloques de atención y recurrencia:
        Token emb → [Attn] → [GRU+FFN] → [Attn] → [GRU+FFN] → ... → LM Head

    El parámetro `rnn_ratio` (en SLMConfig) controla qué fracción de las
    capas son RecurrentBlock. Ejemplos con n_layer=6:
        rnn_ratio=0.5  →  [Attn, GRU, Attn, GRU, Attn, GRU]    (3 de cada)
        rnn_ratio=0.33 →  [Attn, Attn, GRU, Attn, Attn, GRU]    (2 GRU)
        rnn_ratio=0.66 →  [GRU, Attn, GRU, GRU, Attn, GRU]      (4 GRU)

    Notas sobre la distribución de bloques:
      - Las posiciones GRU se distribuyen uniformemente
      - Se fuerza al menos 1 bloque Transformer (necesario para atención global)
      - Se fuerza al menos 1 bloque GRU (si no, sería un SmallLM normal)
    """

    def __init__(self, cfg: SLMConfig):
        # Bypass SmallLM.__init__ (reconstruimos self.blocks)
        nn.Module.__init__(self)
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)

        # Calcular cuántos bloques de cada tipo
        n_rnn = max(1, min(cfg.n_layer - 1, round(cfg.n_layer * cfg.rnn_ratio)))

        # Distribuir GRU uniformemente a lo largo del modelo
        if n_rnn > 1:
            gru_positions = set(
                round(i * (cfg.n_layer - 1) / max(n_rnn - 1, 1))
                for i in range(n_rnn)
            )
        else:
            gru_positions = {cfg.n_layer // 2}

        # Construir bloques alternados
        blocks = []
        self.block_types: list[str] = []
        for i in range(cfg.n_layer):
            if i in gru_positions:
                blocks.append(RecurrentBlock(cfg))
                self.block_types.append("RNN")
            else:
                blocks.append(TransformerBlock(cfg))
                self.block_types.append("Attn")

        self.blocks  = nn.ModuleList(blocks)
        self.norm_f  = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight   # weight tying

        # Inicialización de pesos
        self.apply(self._init_weights)
        scale = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight", "out_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=scale)

        # Log
        n_params = sum(p.numel() for p in self.parameters())
        n_emb    = self.tok_emb.weight.numel()
        layout   = " → ".join(self.block_types)
        n_attn   = self.block_types.count("Attn")
        n_rnn_   = self.block_types.count("RNN")
        print(
            f"🤖 HybridSmallLM [{self._size_label()}] inicializado:\n"
            f"   Layout: {layout}\n"
            f"   Bloques: {n_attn} Transformer + {n_rnn_} RecurrentBlock(GRU+SwiGLU)\n"
            f"   Heads Q: {cfg.n_head} | Heads KV: {cfg.n_kv_head} | "
            f"Hidden: {cfg.n_embd}\n"
            f"   Vocab: {cfg.vocab_size:,} | Contexto: {cfg.n_positions}\n"
            f"   Parámetros totales: {n_params/1e6:.2f}M "
            f"(únicos: {(n_params - n_emb)/1e6:.2f}M con weight tying)"
        )

    @classmethod
    def load(cls, load_dir: Path | str, map_location: str = "cpu") -> "HybridSmallLM":
        """Carga un modelo híbrido guardado desde load_dir."""
        load_dir = Path(load_dir)
        cfg   = SLMConfig.load(load_dir / "config.json")
        model = cls(cfg)
        state = torch.load(load_dir / "model.pt",
                           map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model


# ============================================================================
# Factory
# ============================================================================

def build_model(cfg: SLMConfig) -> SmallLM | HybridSmallLM:
    """
    Construye el modelo según cfg.architecture.

    Parameters
    ----------
    cfg : SLMConfig con architecture="transformer" o "hybrid"

    Returns
    -------
    SmallLM o HybridSmallLM configurado
    """
    if cfg.architecture == "hybrid":
        return HybridSmallLM(cfg)
    return SmallLM(cfg)


def load_model(load_dir: Path | str, map_location: str = "cpu") -> SmallLM | HybridSmallLM:
    """
    Carga cualquier modelo (transformer o hybrid) detectando el tipo
    desde config.json.
    """
    load_dir = Path(load_dir)
    cfg = SLMConfig.load(load_dir / "config.json")
    if cfg.architecture == "hybrid":
        return HybridSmallLM.load(load_dir, map_location)
    return SmallLM.load(load_dir, map_location)
