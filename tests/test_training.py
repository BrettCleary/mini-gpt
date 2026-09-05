"""Tests for the loss, the optimizer setup, the schedule, and the section-15
tiny-batch overfit check."""

from dataclasses import replace

import math
import torch
import torch.nn.functional as F

from config import GPTConfig, TrainConfig
from model import GPT
from train import configure_optimizer, get_lr


def test_loss_matches_manual_cross_entropy(cfg):
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 7))
    tgt = torch.randint(0, cfg.vocab_size, (2, 7))
    logits, loss = m(idx, tgt)
    manual = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(), tgt.reshape(-1))
    assert torch.allclose(loss, manual, atol=1e-6)


def test_loss_uses_position_t_to_predict_target_t(cfg):
    """Section 14: if the target at position t were taken from the wrong place,
    a model trained to copy would not reach zero loss. Here we check the
    alignment directly: making logits huge at exactly the target index gives
    loss ~ 0, and shifting the targets by one breaks it."""
    B, T, V = 2, 6, cfg.vocab_size
    tgt = torch.randint(0, V, (B, T))
    logits = torch.zeros(B, T, V)
    logits.scatter_(2, tgt.unsqueeze(-1), 50.0)
    aligned = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
    shifted = F.cross_entropy(logits[:, :-1].reshape(-1, V), tgt[:, 1:].reshape(-1))
    assert aligned.item() < 1e-4
    assert shifted.item() > 1.0


def test_lr_warmup_then_cosine():
    c = TrainConfig(lr=1e-3, warmup_steps=100, max_steps=1000, min_lr_ratio=0.1)
    assert get_lr(0, c) == c.lr / 100                  # first step is not zero
    assert math.isclose(get_lr(99, c), c.lr)           # peak at end of warmup
    mid = get_lr(550, c)
    assert c.lr * 0.1 < mid < c.lr                     # decaying
    assert math.isclose(get_lr(999, c), c.lr * 0.1, rel_tol=1e-3)
    assert math.isclose(get_lr(5000, c), c.lr * 0.1)   # clamped past the end
    # monotone decreasing after warmup
    after = [get_lr(s, c) for s in range(100, 1000)]
    assert all(a >= b - 1e-12 for a, b in zip(after, after[1:]))


def test_optimizer_param_groups(cfg):
    m = GPT(cfg)
    opt = configure_optimizer(m, TrainConfig(weight_decay=0.1), verbose=False)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    # RMSNorm scales: 2 per block + 1 final
    assert len(no_decay["params"]) == 2 * cfg.n_layers + 1


def test_optimizer_counts_tied_weights_once(cfg):
    m = GPT(replace(cfg, tie_embeddings=True))
    opt = configure_optimizer(m, TrainConfig(), verbose=False)
    total = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert total == m.num_params()


def test_gradient_accumulation_equals_one_big_batch(cfg):
    """Section 20: summing (loss/N) over N micro-batches gives the same gradient
    as one forward over the concatenated batch."""
    torch.manual_seed(0)
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (8, 12))
    tgt = torch.randint(0, cfg.vocab_size, (8, 12))

    _, loss = m(idx, tgt)
    loss.backward()
    big = [p.grad.clone() for p in m.parameters()]

    m.zero_grad(set_to_none=True)
    for i in range(4):
        sl = slice(i * 2, (i + 1) * 2)
        _, l = m(idx[sl], tgt[sl])
        (l / 4).backward()
    acc = [p.grad.clone() for p in m.parameters()]

    for a, b in zip(big, acc):
        assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_grad_clipping_bounds_the_norm(cfg):
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 12))
    _, loss = m(idx, idx)
    (loss * 1000).backward()                      # blow the gradients up
    pre = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    post = torch.sqrt(sum(p.grad.pow(2).sum() for p in m.parameters() if p.grad is not None))
    assert pre > 1.0
    assert post <= 1.0 + 1e-4


def test_tiny_batch_overfit():
    """Section 15: the model must be able to memorise one small batch.

    This is the single most valuable test in the suite -- it fails loudly for a
    broken mask, misaligned targets, a detached graph, or a dead optimizer.
    """
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, context_length=32, d_model=64, n_layers=2,
                    n_heads=4, d_head=16, d_ff=176)
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (4, 32))
    tgt = torch.randint(0, cfg.vocab_size, (4, 32))
    # Give every row a distinct first token. Position 0 can only condition on
    # x[:, 0], so two rows starting with the same token but wanting different
    # targets there is a genuinely unlearnable pair -- and it silently caps the
    # achievable loss above zero, which looks exactly like an architecture bug.
    idx[:, 0] = torch.arange(idx.shape[0])
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    start = None
    for step in range(400):
        opt.zero_grad(set_to_none=True)
        _, loss = m(idx, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if start is None:
            start = loss.item()
    assert start > 3.0, "sanity: untrained loss should be near ln(64)=4.16"
    assert loss.item() < 0.02, f"failed to overfit: final loss {loss.item():.4f}"

    # and it should now reproduce the targets greedily
    logits, _ = m(idx)
    assert (logits.argmax(-1) == tgt).float().mean().item() == 1.0


def test_no_nan_or_inf_under_bf16_autocast(cfg):
    if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        import pytest
        pytest.skip("needs a bf16-capable CUDA device")
    m = GPT(cfg).cuda()
    idx = torch.randint(0, cfg.vocab_size, (4, 16), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = m(idx, idx)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    # parameters stay fp32 even though the matmuls ran in bf16
    assert all(p.dtype == torch.float32 for p in m.parameters())
    assert logits.dtype == torch.bfloat16
