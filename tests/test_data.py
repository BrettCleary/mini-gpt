import json
import os

import numpy as np
import torch

from data import DataLoader, make_synthetic_data


def _fixture_dir(tmp_path):
    d = str(tmp_path / "toy")
    os.makedirs(d, exist_ok=True)
    for split, n in (("train", 5000), ("val", 1000)):
        np.arange(n, dtype=np.uint16).tofile(os.path.join(d, f"{split}.bin"))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"tokenizer": "synthetic", "vocab_size": 65536,
                   "counts": {"train": 5000, "val": 1000}}, f)
    return d


def test_batch_shapes_and_dtype(tmp_path):
    d = _fixture_dir(tmp_path)
    dl = DataLoader(d, context_length=16, batch_size=4, device="cpu")
    x, y = dl.get_batch("train")
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == torch.int64


def test_targets_are_inputs_shifted_by_exactly_one(tmp_path):
    """y[t] must be the token that actually followed x[t]."""
    d = _fixture_dir(tmp_path)
    dl = DataLoader(d, context_length=16, batch_size=8, device="cpu")
    x, y = dl.get_batch("train")
    assert torch.equal(x[:, 1:], y[:, :-1])
    # our fixture stream is 0,1,2,3,... so y == x + 1 exactly
    assert torch.equal(y, x + 1)


def test_windows_are_contiguous(tmp_path):
    d = _fixture_dir(tmp_path)
    dl = DataLoader(d, context_length=32, batch_size=4, device="cpu")
    x, _ = dl.get_batch("train")
    steps = x[:, 1:] - x[:, :-1]
    assert torch.all(steps == 1)


def test_never_reads_past_the_end(tmp_path):
    d = _fixture_dir(tmp_path)
    dl = DataLoader(d, context_length=64, batch_size=64, device="cpu")
    n = dl.n_tokens("val")
    for _ in range(20):
        x, y = dl.get_batch("val")
        assert int(y.max()) <= n - 1


def test_both_splits_load(tmp_path):
    d = _fixture_dir(tmp_path)
    dl = DataLoader(d, context_length=8, batch_size=2, device="cpu")
    assert dl.n_tokens("train") == 5000 and dl.n_tokens("val") == 1000


def test_sampler_state_is_restorable(tmp_path):
    d = _fixture_dir(tmp_path)
    dl = DataLoader(d, context_length=8, batch_size=2, device="cpu", seed=7)
    state = dl.state_dict()
    a, _ = dl.get_batch("train")
    dl.load_state_dict(state)
    b, _ = dl.get_batch("train")
    assert torch.equal(a, b)


def test_synthetic_corpus_helper(tmp_path):
    d = str(tmp_path / "syn")
    make_synthetic_data(d, vocab_size=256, n_train=5000, n_val=1000)
    dl = DataLoader(d, context_length=32, batch_size=4, device="cpu")
    x, y = dl.get_batch("train")
    assert x.max() < 256 and torch.equal(x[:, 1:], y[:, :-1])


def test_missing_data_raises_a_useful_error(tmp_path):
    try:
        DataLoader(str(tmp_path / "nope"), 8, 2, device="cpu")
    except FileNotFoundError as e:
        assert "data.py" in str(e)
        return
    raise AssertionError("expected FileNotFoundError")


def test_tokenizer_roundtrip():
    from tokenizer import get_tokenizer
    tok = get_tokenizer("gpt2")
    s = "Once upon a time, there was a little girl named Lily."
    assert tok.decode(tok.encode(s)) == s
    assert tok.vocab_size == 50257 and tok.eot_id == 50256
