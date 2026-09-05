import torch

from model import RMSNorm


def test_shape_preserved():
    norm = RMSNorm(16)
    x = torch.randn(3, 5, 16)
    assert norm(x).shape == x.shape


def test_output_has_unit_rms():
    """With the scale left at 1, the output's RMS over the last dim is 1."""
    norm = RMSNorm(64, eps=1e-8)
    x = torch.randn(8, 7, 64) * 17.0 + 4.0        # arbitrary scale and offset
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_scale_parameter_is_applied():
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.copy_(torch.arange(8, dtype=torch.float32))
    x = torch.randn(2, 8)
    y = norm(x)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps) * norm.weight
    assert torch.allclose(y, expected, atol=1e-5)


def test_scale_invariance():
    """RMSNorm removes overall magnitude: f(a*x) == f(x) for a > 0."""
    norm = RMSNorm(32, eps=1e-12)
    x = torch.randn(4, 32)
    assert torch.allclose(norm(x), norm(x * 100.0), atol=1e-4)


def test_no_bias_parameter():
    norm = RMSNorm(8)
    names = [n for n, _ in norm.named_parameters()]
    assert names == ["weight"]


def test_does_not_subtract_the_mean():
    """Unlike LayerNorm, a constant input stays constant (does not go to 0)."""
    norm = RMSNorm(8, eps=1e-12)
    y = norm(torch.full((1, 8), 3.0))
    assert torch.allclose(y, torch.ones(1, 8), atol=1e-5)
