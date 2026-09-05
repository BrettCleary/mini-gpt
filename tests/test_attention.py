import math

import torch

from config import GPTConfig
from model import CausalSelfAttention, TransformerBlock, build_rope_cache


def _rope(cfg, T):
    return build_rope_cache(T, cfg.d_head, cfg.rope_theta)


def test_output_shape(cfg):
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 16, cfg.d_model)
    cos, sin = _rope(cfg, 16)
    assert attn(x, cos, sin).shape == x.shape


def test_mask_is_lower_triangular(cfg):
    attn = CausalSelfAttention(cfg)
    m = attn.causal_mask[0, 0]
    assert m.shape == (cfg.context_length, cfg.context_length)
    assert torch.equal(m, torch.tril(torch.ones_like(m)))


def test_attention_weights_are_causal_and_normalised(cfg):
    attn = CausalSelfAttention(cfg)
    attn.store_attn = True
    T = 12
    cos, sin = _rope(cfg, T)
    attn(torch.randn(2, T, cfg.d_model), cos, sin)
    a = attn.last_attn                       # [B, H, T, T]
    assert a.shape == (2, cfg.n_heads, T, T)
    # every row sums to 1
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
    # nothing above the diagonal
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    assert a[:, :, upper].abs().max().item() == 0.0


def test_first_token_attends_only_to_itself(cfg):
    attn = CausalSelfAttention(cfg)
    attn.store_attn = True
    cos, sin = _rope(cfg, 8)
    attn(torch.randn(1, 8, cfg.d_model), cos, sin)
    assert torch.allclose(attn.last_attn[0, :, 0, 0],
                          torch.ones(cfg.n_heads), atol=1e-6)


def test_future_tokens_do_not_change_the_past(cfg):
    """Perturbing position t only affects outputs at positions >= t."""
    attn = CausalSelfAttention(cfg).eval()
    T = 10
    cos, sin = _rope(cfg, T)
    x = torch.randn(1, T, cfg.d_model)
    y1 = attn(x, cos, sin)

    x2 = x.clone()
    x2[0, 6:] = torch.randn(T - 6, cfg.d_model)     # scramble the tail
    y2 = attn(x2, cos, sin)

    assert torch.allclose(y1[0, :6], y2[0, :6], atol=1e-5)
    assert not torch.allclose(y1[0, 6:], y2[0, 6:], atol=1e-5)


def test_gradient_of_early_output_wrt_later_input_is_zero(cfg):
    """The strongest causality check: d out[t] / d x[t'] == 0 for t' > t."""
    attn = CausalSelfAttention(cfg).eval()
    T = 8
    cos, sin = _rope(cfg, T)
    x = torch.randn(1, T, cfg.d_model, requires_grad=True)
    out = attn(x, cos, sin)
    out[0, 3].sum().backward()
    assert x.grad[0, 4:].abs().max().item() == 0.0
    assert x.grad[0, :4].abs().max().item() > 0.0


def test_scaling_by_sqrt_d_head(cfg):
    """Reproduce the score matrix by hand and compare to the stored softmax."""
    attn = CausalSelfAttention(cfg).eval()
    attn.store_attn = True
    B, T = 1, 6
    H, Dh = cfg.n_heads, cfg.d_head
    x = torch.randn(B, T, cfg.d_model)
    cos, sin = _rope(cfg, T)
    attn(x, cos, sin)

    from model import apply_rope
    q = attn.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
    k = attn.k_proj(x).view(B, T, H, Dh).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
    scores = scores.masked_fill(torch.triu(torch.ones(T, T), 1).bool(), float("-inf"))
    assert torch.allclose(torch.softmax(scores, -1), attn.last_attn, atol=1e-5)


def test_rejects_sequences_longer_than_context(cfg):
    attn = CausalSelfAttention(cfg)
    T = cfg.context_length + 1
    cos, sin = _rope(cfg, T)
    try:
        attn(torch.randn(1, T, cfg.d_model), cos, sin)
    except AssertionError:
        return
    raise AssertionError("attention should refuse T > context_length")


def test_transformer_block_is_shape_preserving(cfg):
    block = TransformerBlock(cfg)
    x = torch.randn(3, 11, cfg.d_model)
    cos, sin = _rope(cfg, 11)
    y = block(x, cos, sin)
    assert y.shape == x.shape == (3, 11, cfg.d_model)


def test_block_residual_path_exists(cfg):
    """Zero both sublayer output projections: the block becomes the identity."""
    block = TransformerBlock(cfg).eval()
    with torch.no_grad():
        block.attn.out_proj.weight.zero_()
        block.ffn.down_proj.weight.zero_()
    x = torch.randn(2, 7, cfg.d_model)
    cos, sin = _rope(cfg, 7)
    assert torch.allclose(block(x, cos, sin), x, atol=1e-6)
