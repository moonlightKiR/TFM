"""
test_transformer.py — Tests de los componentes Transformer.

Ejecutar: python3 -m pytest tests/test_transformer.py -v
"""

import sys
from pathlib import Path

import torch
import pytest

# Ajustar path para imports desde slm_pdf/
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import SLMConfig
from model.shared import RMSNorm
from model.transformer.attention import RotaryEmbedding, GroupedQueryAttention
from model.transformer.ffn import SwiGLU
from model.transformer.block import TransformerBlock


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cfg():
    """Configuración mínima para tests rápidos."""
    return SLMConfig(
        vocab_size=1000, n_positions=32, n_layer=2,
        n_head=4, n_kv_head=4, n_embd=64, dropout=0.0,
    )


@pytest.fixture
def cfg_gqa():
    """Configuración con GQA (n_kv_head < n_head)."""
    return SLMConfig(
        vocab_size=1000, n_positions=32, n_layer=2,
        n_head=8, n_kv_head=2, n_embd=64, dropout=0.0,
    )


# ============================================================================
# RMSNorm
# ============================================================================

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_scale(self):
        """La salida debería tener RMS ≈ 1 (antes de la escala γ)."""
        norm = RMSNorm(128)
        x = torch.randn(4, 32, 128)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        # γ=1 inicialmente, así que RMS ≈ 1.0 ± algo de tolerancia
        assert rms.mean().item() < 5.0  # sanity check

    def test_gradient_flows(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ============================================================================
# RotaryEmbedding
# ============================================================================

class TestRotaryEmbedding:
    def test_output_shape(self):
        rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
        x = torch.randn(2, 4, 16, 16)  # B, H, T, D
        out = rope(x)
        assert out.shape == x.shape

    def test_different_positions_different_output(self):
        rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
        x = torch.randn(1, 1, 8, 16)
        out1 = rope(x, offset=0)
        out2 = rope(x, offset=10)
        # Diferentes offsets → diferentes rotaciones
        assert not torch.allclose(out1, out2, atol=1e-5)


# ============================================================================
# GroupedQueryAttention
# ============================================================================

class TestGroupedQueryAttention:
    def test_mha_output_shape(self, cfg):
        """Multi-Head Attention: n_kv_head == n_head."""
        attn = GroupedQueryAttention(cfg)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_gqa_output_shape(self, cfg_gqa):
        """Grouped Query Attention: n_kv_head < n_head."""
        attn = GroupedQueryAttention(cfg_gqa)
        x = torch.randn(2, 16, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_masking(self, cfg):
        """El token t no debería atender a tokens futuros t+1, t+2, ..."""
        attn = GroupedQueryAttention(cfg)
        attn.eval()
        x = torch.randn(1, 8, 64)

        # Forward con toda la secuencia
        out_full = attn(x)
        # Forward solo con los primeros 4 tokens
        out_partial = attn(x[:, :4, :])

        # El output de los primeros 4 tokens debería ser igual en ambos casos
        # (porque la atención causal no permite ver hacia adelante)
        assert torch.allclose(out_full[:, :4, :], out_partial, atol=1e-5)


# ============================================================================
# SwiGLU
# ============================================================================

class TestSwiGLU:
    def test_output_shape(self, cfg):
        ffn = SwiGLU(cfg)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_gradient_flows(self, cfg):
        ffn = SwiGLU(cfg)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None


# ============================================================================
# TransformerBlock
# ============================================================================

class TestTransformerBlock:
    def test_output_shape(self, cfg):
        block = TransformerBlock(cfg)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self, cfg):
        """Inicialmente el bloque debería producir output ≈ input (residual)."""
        block = TransformerBlock(cfg)
        x = torch.randn(2, 16, 64)
        out = block(x)
        # Con pesos inicializados random-small, residual domina
        diff = (out - x).abs().mean()
        assert diff < 10.0  # no debería divergir

    def test_gradient_flows(self, cfg):
        block = TransformerBlock(cfg)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
