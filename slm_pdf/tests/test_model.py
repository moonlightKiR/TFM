"""
test_model.py — Tests end-to-end del modelo SmallLM.

Ejecutar: python3 -m pytest tests/test_model.py -v
"""

import sys
import tempfile
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import SLMConfig
from model.base import SmallLM


@pytest.fixture
def cfg():
    return SLMConfig(
        vocab_size=500, n_positions=32, n_layer=2,
        n_head=4, n_kv_head=4, n_embd=64, dropout=0.0,
    )


@pytest.fixture
def model(cfg):
    return SmallLM(cfg)


# ============================================================================
# SLMConfig
# ============================================================================

class TestSLMConfig:
    def test_predefined_sizes(self):
        for size in ["micro", "tiny", "small", "medium", "large", "xl"]:
            cfg = getattr(SLMConfig, size)(vocab_size=1000)
            assert cfg.vocab_size == 1000
            assert cfg.n_layer > 0
            assert cfg.n_embd > 0

    def test_save_load(self, cfg, tmp_path):
        path = tmp_path / "config.json"
        cfg.save(path)
        loaded = SLMConfig.load(path)
        assert loaded.vocab_size == cfg.vocab_size
        assert loaded.n_layer == cfg.n_layer
        assert loaded.n_embd == cfg.n_embd
        assert loaded.architecture == cfg.architecture


# ============================================================================
# SmallLM
# ============================================================================

class TestSmallLM:
    def test_forward_logits(self, model, cfg):
        B, T = 2, 16
        x = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(x)
        assert logits.shape == (B, T, cfg.vocab_size)

    def test_forward_with_labels(self, model, cfg):
        B, T = 2, 16
        x = torch.randint(0, cfg.vocab_size, (B, T))
        labels = x.clone()
        loss, logits = model(x, labels=labels)
        assert loss.shape == ()
        assert loss.item() > 0
        assert logits.shape == (B, T, cfg.vocab_size)

    def test_gradient_backprop(self, model, cfg):
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        labels = x.clone()
        loss, _ = model(x, labels=labels)
        loss.backward()

        # Verificar que los gradientes fluyen a las capas clave
        assert model.tok_emb.weight.grad is not None
        assert model.blocks[0].attn.q_proj.weight.grad is not None

    def test_generate(self, model, cfg):
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        gen = model.generate(prompt, max_new_tokens=10, temperature=0.8)
        assert gen.shape[0] == 1
        assert gen.shape[1] >= 4   # al menos el prompt
        assert gen.shape[1] <= 14  # prompt + max_new_tokens

    def test_generate_with_eos(self, model, cfg):
        prompt = torch.randint(0, cfg.vocab_size, (1, 4))
        gen = model.generate(
            prompt, max_new_tokens=100, temperature=1.0,
            eos_token_id=1,  # muy probable que se genere rápido
        )
        assert gen.shape[1] >= 4

    def test_weight_tying(self, model):
        """Embedding y LM head deberían compartir pesos."""
        assert model.tok_emb.weight is model.lm_head.weight

    def test_save_and_load(self, model, cfg):
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = SmallLM.load(tmpdir)

            assert loaded.cfg.vocab_size == cfg.vocab_size
            assert loaded.cfg.n_layer == cfg.n_layer

            # Verificar que los pesos son iguales
            x = torch.randint(0, cfg.vocab_size, (1, 8))
            model.eval()
            loaded.eval()
            out1 = model(x)
            out2 = loaded(x)
            assert torch.allclose(out1, out2, atol=1e-5)

    def test_sequence_too_long(self, cfg):
        model = SmallLM(cfg)
        x = torch.randint(0, cfg.vocab_size, (1, cfg.n_positions + 1))
        with pytest.raises(AssertionError):
            model(x)

    def test_parameter_count(self, model):
        """Verificar que el modelo tiene parámetros razonables."""
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        assert n_params < 1e9  # < 1B params (sanity check)
