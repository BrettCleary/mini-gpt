import os
from dataclasses import replace

import torch

from checkpoint import (latest_checkpoint, load_checkpoint, model_from_checkpoint,
                        prune_checkpoints, restore, save_checkpoint)
from config import GPTConfig, TrainConfig
from model import GPT
from train import configure_optimizer


def _train_a_few(m, opt, idx, tgt, n):
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        _, loss = m(idx, tgt)
        loss.backward()
        opt.step()
    return loss.item()


def test_save_and_load_roundtrip(cfg, tmp_path):
    m = GPT(cfg)
    opt = configure_optimizer(m, TrainConfig(), verbose=False)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, m, opt, 42, cfg, TrainConfig(), val_loss=1.23,
                    best_val_loss=1.23, tokens_seen=999)
    ckpt = load_checkpoint(path)
    assert ckpt["step"] == 42 and ckpt["tokens_seen"] == 999
    assert ckpt["val_loss"] == 1.23
    assert ckpt["model_cfg"]["d_model"] == cfg.d_model


def test_model_rebuilt_from_checkpoint_is_identical(cfg, tmp_path):
    m = GPT(cfg).eval()
    opt = configure_optimizer(m, TrainConfig(), verbose=False)
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, m, opt, 1, cfg, TrainConfig())
    m2, cfg2 = model_from_checkpoint(load_checkpoint(path))
    m2.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    assert cfg2 == cfg
    assert torch.allclose(m(idx)[0], m2(idx)[0], atol=1e-6)


def test_resume_reproduces_uninterrupted_training(cfg, tmp_path):
    """Train 20 steps straight through; train 10, save, reload, train 10 more.
    The two runs must land on identical weights."""
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    tgt = torch.randint(0, cfg.vocab_size, (4, 16))

    torch.manual_seed(1)
    ref = GPT(cfg)
    ref_opt = configure_optimizer(ref, TrainConfig(), verbose=False)
    _train_a_few(ref, ref_opt, idx, tgt, 20)

    torch.manual_seed(1)
    m = GPT(cfg)
    opt = configure_optimizer(m, TrainConfig(), verbose=False)
    _train_a_few(m, opt, idx, tgt, 10)
    path = str(tmp_path / "mid.pt")
    save_checkpoint(path, m, opt, 10, cfg, TrainConfig())

    torch.manual_seed(1)
    m2 = GPT(cfg)
    opt2 = configure_optimizer(m2, TrainConfig(), verbose=False)
    step = restore(load_checkpoint(path), m2, opt2)
    assert step == 10
    _train_a_few(m2, opt2, idx, tgt, 10)

    for (n, a), b in zip(ref.named_parameters(), m2.parameters()):
        assert torch.allclose(a, b, atol=1e-6), f"{n} diverged after resume"


def test_optimizer_moments_are_restored(cfg, tmp_path):
    m = GPT(cfg)
    opt = configure_optimizer(m, TrainConfig(), verbose=False)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    _train_a_few(m, opt, idx, idx, 5)
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, m, opt, 5, cfg, TrainConfig())

    m2 = GPT(cfg)
    opt2 = configure_optimizer(m2, TrainConfig(), verbose=False)
    restore(load_checkpoint(path), m2, opt2)
    s1 = list(opt.state.values())[0]
    s2 = list(opt2.state.values())[0]
    assert torch.allclose(s1["exp_avg"], s2["exp_avg"])
    assert torch.allclose(s1["exp_avg_sq"], s2["exp_avg_sq"])
    assert int(s1["step"]) == int(s2["step"])


def test_rng_state_restored(cfg, tmp_path):
    m = GPT(cfg)
    opt = configure_optimizer(m, TrainConfig(), verbose=False)
    path = str(tmp_path / "c.pt")
    save_checkpoint(path, m, opt, 0, cfg, TrainConfig())
    a = torch.randn(5)
    restore(load_checkpoint(path), m, opt)
    assert torch.equal(a, torch.randn(5))


def test_latest_and_prune(cfg, tmp_path):
    d = str(tmp_path)
    m, opt = GPT(cfg), None
    opt = configure_optimizer(m, TrainConfig(), verbose=False)
    for s in (100, 200, 300):
        save_checkpoint(os.path.join(d, f"ckpt_{s:07d}.pt"), m, opt, s, cfg, TrainConfig())
    assert latest_checkpoint(d).endswith("ckpt_0000300.pt")
    prune_checkpoints(d, keep_last_k=2)
    remaining = sorted(f for f in os.listdir(d) if f.startswith("ckpt_"))
    assert remaining == ["ckpt_0000200.pt", "ckpt_0000300.pt"]
