import math

import torch

from model import apply_rope, build_rope_cache


def _q(B=2, H=3, T=16, D=8):
    return torch.randn(B, H, T, D)


def test_shapes_preserved():
    x = _q()
    cos, sin = build_rope_cache(16, 8)
    assert apply_rope(x, cos, sin).shape == x.shape


def test_cache_shapes():
    cos, sin = build_rope_cache(64, 16)
    assert cos.shape == (64, 8) and sin.shape == (64, 8)


def test_position_zero_is_identity():
    """Angle at position 0 is 0, so the rotation is the identity there."""
    x = _q(T=5)
    cos, sin = build_rope_cache(5, 8)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(y[:, :, 0], x[:, :, 0], atol=1e-6)


def test_deterministic():
    x = _q()
    cos, sin = build_rope_cache(16, 8)
    assert torch.equal(apply_rope(x, cos, sin), apply_rope(x, cos, sin))


def test_norm_is_preserved():
    """Rotation is orthogonal, so per-token vector norms are unchanged."""
    x = _q(T=32, D=16)
    cos, sin = build_rope_cache(32, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_different_positions_rotate_differently():
    """The same vector placed at two positions comes out different."""
    v = torch.randn(1, 1, 1, 8)
    x = v.expand(1, 1, 12, 8).contiguous()
    cos, sin = build_rope_cache(12, 8)
    y = apply_rope(x, cos, sin)
    for t in range(1, 12):
        assert not torch.allclose(y[0, 0, t], y[0, 0, 0], atol=1e-4)


def test_dot_product_depends_only_on_relative_position():
    """The defining property of RoPE: <R_m q, R_n k> is a function of (m - n)."""
    torch.manual_seed(0)
    D, T = 8, 40
    cos, sin = build_rope_cache(T, D)
    q = torch.randn(1, 1, 1, D)
    k = torch.randn(1, 1, 1, D)

    def dot(m, n):
        qi = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        ki = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        return (qi * ki).sum().item()

    for offset in (0, 1, 5, 13):
        vals = [dot(m, m - offset) for m in range(offset, offset + 8)]
        assert max(vals) - min(vals) < 1e-4, f"offset {offset} not translation-invariant"


def test_matches_explicit_2d_rotation():
    """Check one pair by hand against the textbook rotation matrix."""
    D, theta = 4, 10000.0
    cos, sin = build_rope_cache(8, D, theta)
    x = torch.zeros(1, 1, 8, D)
    x[0, 0, :, 0] = 1.0            # unit vector along the first axis of pair 0
    y = apply_rope(x, cos, sin)
    for t in range(8):
        angle = t * theta ** (-0.0 / (D // 2))   # pair 0: inv_freq = 1
        assert math.isclose(y[0, 0, t, 0].item(), math.cos(angle), abs_tol=1e-5)
        assert math.isclose(y[0, 0, t, 1].item(), math.sin(angle), abs_tol=1e-5)


def test_rejects_wrong_rank():
    cos, sin = build_rope_cache(8, 8)
    try:
        apply_rope(torch.randn(8, 8), cos, sin)
    except AssertionError:
        return
    raise AssertionError("apply_rope should reject a non-4D tensor")
