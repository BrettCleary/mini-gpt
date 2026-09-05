"""Checkpoint save / load.

A checkpoint is everything needed to continue a run bit-for-bit:
model weights, optimizer moments, the step counter, the LR schedule position
(derived from the step), the configs, the best validation loss, and RNG state
for python/numpy/torch/cuda plus the data loader's sampler.
"""

import glob
import json
import os
import random
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from config import GPTConfig, TrainConfig


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        except (RuntimeError, ValueError):
            pass  # different GPU count than the run that saved it


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_cfg: GPTConfig,
    train_cfg: TrainConfig,
    val_loss: Optional[float] = None,
    best_val_loss: Optional[float] = None,
    tokens_seen: int = 0,
    loader_state: Optional[dict] = None,
    scaler: Optional[object] = None,
    tokenizer_name: str = "gpt2",
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "rng": _rng_state(),
        "loader": loader_state,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "tokenizer_name": tokenizer_name,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)   # atomic: a crash mid-save never corrupts the old file
    return path


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def restore(
    ckpt: dict,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loader=None,
    scaler=None,
    restore_rng: bool = True,
) -> int:
    """Load state into live objects. Returns the step to resume from."""
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if loader is not None and ckpt.get("loader") is not None:
        loader.load_state_dict(ckpt["loader"])
    if restore_rng and ckpt.get("rng") is not None:
        _set_rng_state(ckpt["rng"])
    return int(ckpt["step"])


def model_from_checkpoint(ckpt: dict, device: str = "cpu"):
    """Rebuild the exact architecture the checkpoint was trained with."""
    from model import GPT
    cfg = GPTConfig(**ckpt["model_cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg


def latest_checkpoint(out_dir: str) -> Optional[str]:
    cands = sorted(glob.glob(os.path.join(out_dir, "ckpt_*.pt")),
                   key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    return cands[-1] if cands else None


def prune_checkpoints(out_dir: str, keep_last_k: int):
    """Delete old step checkpoints, keeping the newest k. `best.pt` is never touched."""
    if keep_last_k <= 0:
        return
    cands = sorted(glob.glob(os.path.join(out_dir, "ckpt_*.pt")),
                   key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    for p in cands[:-keep_last_k]:
        os.remove(p)
