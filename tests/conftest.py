import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GPTConfig  # noqa: E402


@pytest.fixture
def cfg():
    """A tiny model: fast, but exercises every code path."""
    return GPTConfig(vocab_size=97, context_length=32, d_model=32,
                     n_layers=2, n_heads=4, d_head=8, d_ff=88)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
