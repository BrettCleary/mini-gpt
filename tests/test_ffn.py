import torch
import torch.nn.functional as F

from config import GPTConfig
from model import GELUFFN, SwiGLU, build_ffn


def test_swiglu_shape(cfg):
    ffn = SwiGLU(cfg)
    x = torch.randn(2, 9, cfg.d_model)
    assert ffn(x).shape == x.shape


def test_swiglu_matches_the_formula(cfg):
    ffn = SwiGLU(cfg).eval()
    x = torch.randn(2, 5, cfg.d_model)
    expected = ffn.down_proj(F.silu(ffn.gate_proj(x)) * ffn.up_proj(x))
    assert torch.allclose(ffn(x), expected, atol=1e-6)


def test_swiglu_gate_can_zero_the_output(cfg):
    """The gate really gates: a zeroed up_proj kills everything."""
    ffn = SwiGLU(cfg).eval()
    with torch.no_grad():
        ffn.up_proj.weight.zero_()
    x = torch.randn(2, 5, cfg.d_model)
    assert ffn(x).abs().max().item() == 0.0


def test_hidden_dim_is_wider_than_the_residual_stream(cfg):
    ffn = SwiGLU(cfg)
    assert ffn.gate_proj.out_features > cfg.d_model


def test_swiglu_param_count(cfg):
    ffn = SwiGLU(cfg)
    assert sum(p.numel() for p in ffn.parameters()) == 3 * cfg.d_model * cfg.d_ff


def test_gelu_ffn_shape_and_params(cfg):
    ffn = GELUFFN(cfg)
    x = torch.randn(2, 6, cfg.d_model)
    assert ffn(x).shape == x.shape
    assert sum(p.numel() for p in ffn.parameters()) == 2 * cfg.d_model * 4 * cfg.d_model


def test_build_ffn_dispatch(cfg):
    from dataclasses import replace
    assert isinstance(build_ffn(cfg), SwiGLU)
    assert isinstance(build_ffn(replace(cfg, ffn_type="gelu")), GELUFFN)


def test_swiglu_is_position_wise(cfg):
    """The FFN must not mix positions -- only attention does that."""
    ffn = SwiGLU(cfg).eval()
    x = torch.randn(1, 8, cfg.d_model)
    full = ffn(x)
    per_pos = torch.cat([ffn(x[:, t:t + 1]) for t in range(8)], dim=1)
    assert torch.allclose(full, per_pos, atol=1e-6)
