"""
test_hybrid.py — Tests del modelo híbrido Transformer + GRU.

Ejecutar: python3 -m pytest tests/test_hybrid.py -v
"""

import sys
import tempfile
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import SLMConfig
from model.hybrid import HybridSmallLM, build_model, load_model
from model.base import SmallLM
from model.transformer.block import TransformerBlock
from model.rnn.block import RecurrentBlock


@pytest.fixture
def cfg():
    return SLMConfig(
        vocab_size=500, n_positions=32, n_layer=6,
        n_head=4, n_kv_head=4, n_embd=64, dropout=0.0,
        architecture="hybrid", rnn_ratio=0.5,
    )


@pytest.fixture
def model(cfg):
    return HybridSmallLM(cfg)


# ============================================================================
# Build model factory
# ============================================================================

class TestBuildModel:
    def test_build_transformer(self):
        cfg = SLMConfig.micro(vocab_size=500)
        cfg.architecture = "transformer"
        m = build_model(cfg)
        assert isinstance(m, SmallLM)
        assert not isinstance(m, HybridSmallLM)

    def test_build_hybrid(self):
        cfg = SLMConfig.micro(vocab_size=500)
        cfg.architecture = "hybrid"
        m = build_model(cfg)
        assert isinstance(m, HybridSmallLM)


# ============================================================================
# Layout de bloques
# ============================================================================

class TestHybridLayout:
    def test_has_both_block_types(self, model):
        has_transformer = any(isinstance(b, TransformerBlock) for b in model.blocks)
        has_recurrent   = any(isinstance(b, RecurrentBlock) for b in model.blocks)
        assert has_transformer, "El modelo híbrido debe tener al menos 1 TransformerBlock"
        assert has_recurrent,   "El modelo híbrido debe tener al menos 1 RecurrentBlock"

    def test_block_types_match_ratio(self, cfg, model):
        n_rnn  = sum(1 for b in model.blocks if isinstance(b, RecurrentBlock))
        n_attn = sum(1 for b in model.blocks if isinstance(b, TransformerBlock))
        assert n_rnn + n_attn == cfg.n_layer
        # Con rnn_ratio=0.5 y n_layer=6: debería haber 3 de cada tipo
        assert n_rnn == 3
        assert n_attn == 3

    def test_different_ratios(self):
        for ratio, expected_rnn in [(0.33, 2), (0.5, 3), (0.66, 4)]:
            cfg = SLMConfig(
                vocab_size=500, n_positions=32, n_layer=6,
                n_head=4, n_kv_head=4, n_embd=64,
                architecture="hybrid", rnn_ratio=ratio,
            )
            m = HybridSmallLM(cfg)
            n_rnn = sum(1 for b in m.blocks if isinstance(b, RecurrentBlock))
            assert n_rnn == expected_rnn, f"rnn_ratio={ratio}: expected {expected_rnn} RNN blocks, got {n_rnn}"

    def test_minimum_one_transformer(self):
        """Incluso con rnn_ratio muy alto, debe haber al menos 1 Transformer."""
        cfg = SLMConfig(
            vocab_size=500, n_positions=32, n_layer=4,
            n_head=4, n_kv_head=4, n_embd=64,
            architecture="hybrid", rnn_ratio=0.99,
        )
        m = HybridSmallLM(cfg)
        n_attn = sum(1 for b in m.blocks if isinstance(b, TransformerBlock))
        assert n_attn >= 1


# ============================================================================
# Forward y Generate
# ============================================================================

class TestHybridForward:
    def test_forward_logits(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        logits = model(x)
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_forward_with_labels(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        loss, logits = model(x, labels=x.clone())
        assert loss.item() > 0
        assert logits.shape == (2, 16, cfg.vocab_size)

    def test_generate(self, model, cfg):
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        gen = model.generate(prompt, max_new_tokens=10, temperature=0.8)
        assert gen.shape[1] >= 4
        assert gen.shape[1] <= 14

    def test_gradient_backprop(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        loss, _ = model(x, labels=x.clone())
        loss.backward()

        # Verificar gradientes en AMBOS tipos de bloques
        grad_found_attn = False
        grad_found_rnn  = False
        for block in model.blocks:
            if isinstance(block, TransformerBlock):
                if block.attn.q_proj.weight.grad is not None:
                    grad_found_attn = True
            elif isinstance(block, RecurrentBlock):
                if block.gru_block.out_proj.weight.grad is not None:
                    grad_found_rnn = True

        assert grad_found_attn, "No se encontraron gradientes en bloques Transformer"
        assert grad_found_rnn,  "No se encontraron gradientes en bloques RNN"


# ============================================================================
# Save / Load
# ============================================================================

class TestHybridPersistence:
    def test_save_and_load(self, model, cfg):
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            # Cargar con método de clase
            loaded = HybridSmallLM.load(tmpdir)
            assert loaded.cfg.architecture == "hybrid"

            # Verificar pesos
            x = torch.randint(0, cfg.vocab_size, (1, 8))
            model.eval()
            loaded.eval()
            assert torch.allclose(model(x), loaded(x), atol=1e-5)

    def test_load_model_factory(self, model, cfg):
        """load_model() detecta automáticamente el tipo de modelo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = load_model(tmpdir)
            assert isinstance(loaded, HybridSmallLM)


# ============================================================================
# Comparación Transformer vs Híbrido
# ============================================================================

class TestHybridVsTransformer:
    def test_hybrid_comparable_params(self):
        """
        RecurrentBlock (GRU+SwiGLU) y TransformerBlock tienen parámetros
        comparables. El híbrido no debería tener más del doble de params.
        """
        cfg_t = SLMConfig.small(vocab_size=1000)
        cfg_h = SLMConfig.small(vocab_size=1000)
        cfg_h.architecture = "hybrid"
        cfg_h.rnn_ratio = 0.5

        t_model = SmallLM(cfg_t)
        h_model = HybridSmallLM(cfg_h)

        t_params = sum(p.numel() for p in t_model.parameters())
        h_params = sum(p.numel() for p in h_model.parameters())

        # Los params deberían estar en rango comparable (dentro de 2x)
        ratio = h_params / t_params
        assert 0.5 < ratio < 2.0, \
            f"Ratio params híbrido/transformer = {ratio:.2f} (esperado ~1.0)"
