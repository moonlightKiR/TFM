"""
test_rnn.py — Tests de los componentes RNN (GRU).

Ejecutar: python3 -m pytest tests/test_rnn.py -v
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import SLMConfig
from model.rnn.gru import GRUBlock
from model.rnn.block import RecurrentBlock


@pytest.fixture
def cfg():
    return SLMConfig(
        vocab_size=1000, n_positions=32, n_layer=2,
        n_head=4, n_kv_head=4, n_embd=64, dropout=0.0,
    )


# ============================================================================
# GRUBlock
# ============================================================================

class TestGRUBlock:
    def test_output_shape(self, cfg):
        block = GRUBlock(cfg)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self, cfg):
        block = GRUBlock(cfg)
        x = torch.randn(2, 16, 64)
        out = block(x)
        diff = (out - x).abs().mean()
        assert diff < 10.0

    def test_gradient_flows(self, cfg):
        block = GRUBlock(cfg)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_causal_behavior(self, cfg):
        """El GRU es causal: el output del token t solo depende de tokens ≤ t."""
        block = GRUBlock(cfg)
        block.eval()
        x = torch.randn(1, 8, 64)

        out_full = block(x)
        out_partial = block(x[:, :4, :])

        # Los primeros 4 tokens deberían dar el mismo resultado
        assert torch.allclose(out_full[:, :4, :], out_partial, atol=1e-5)

    def test_bfloat16_conversion(self, cfg):
        """El GRU interno se mantiene float32 tras conversión a bfloat16."""
        block = GRUBlock(cfg)
        block.bfloat16()

        # El GRU debería seguir en float32
        for p in block.gru.parameters():
            assert p.dtype == torch.float32

        # Pero norm y out_proj deberían estar en bfloat16
        assert block.norm.weight.dtype == torch.bfloat16
        assert block.out_proj.weight.dtype == torch.bfloat16

    def test_forward_with_bfloat16_input(self, cfg):
        """El bloque debería funcionar con input bfloat16."""
        block = GRUBlock(cfg)
        block.bfloat16()
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        out = block(x)
        assert out.shape == x.shape
        assert out.dtype == torch.bfloat16


# ============================================================================
# RecurrentBlock (GRU + SwiGLU FFN)
# ============================================================================

class TestRecurrentBlock:
    def test_output_shape(self, cfg):
        block = RecurrentBlock(cfg)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_more_expressive_than_gru_alone(self, cfg):
        """RecurrentBlock tiene SwiGLU FFN, así que debería tener más params."""
        gru_block = GRUBlock(cfg)
        rec_block = RecurrentBlock(cfg)

        gru_params = sum(p.numel() for p in gru_block.parameters())
        rec_params = sum(p.numel() for p in rec_block.parameters())

        # RecurrentBlock = GRUBlock + RMSNorm + SwiGLU → más parámetros
        assert rec_params > gru_params

    def test_gradient_flows(self, cfg):
        block = RecurrentBlock(cfg)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_bfloat16_forward(self, cfg):
        """RecurrentBlock debería funcionar con bfloat16."""
        block = RecurrentBlock(cfg)
        block.bfloat16()
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        out = block(x)
        assert out.shape == x.shape
        assert out.dtype == torch.bfloat16

    def test_causal_behavior(self, cfg):
        block = RecurrentBlock(cfg)
        block.eval()
        x = torch.randn(1, 8, 64)

        out_full = block(x)
        out_partial = block(x[:, :4, :])

        assert torch.allclose(out_full[:, :4, :], out_partial, atol=1e-5)
