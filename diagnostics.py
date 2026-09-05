"""Optional training diagnostics: per-layer activation RMS, gradient RMS, and
attention entropy.

These are cheap enough to run every few hundred steps and they answer the three
questions that come up when training misbehaves:

  activation RMS  - is the residual stream exploding or collapsing with depth?
  gradient RMS    - are gradients reaching the early layers at all?
  attention entropy - are heads diffuse (entropy ~ ln T) or sharp (entropy ~ 0)?
                      Heads that stay at ln T are doing nothing.
"""

import math
from typing import Dict

import torch

from model import CausalSelfAttention, TransformerBlock


@torch.no_grad()
def activation_rms(model, x: torch.Tensor) -> Dict[str, float]:
    """RMS of the residual stream entering each block, plus the final norm output.

    Uses forward hooks so the model's forward code stays clean.
    """
    stats: Dict[str, float] = {}
    handles = []

    def make_hook(name):
        def hook(_module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            stats[name] = t.float().pow(2).mean().sqrt().item()
        return hook

    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(make_hook(f"resid_after_block{i}")))
        handles.append(block.attn.register_forward_hook(make_hook(f"attn_out{i}")))
        handles.append(block.ffn.register_forward_hook(make_hook(f"ffn_out{i}")))
    handles.append(model.final_norm.register_forward_hook(make_hook("final_norm")))

    was_training = model.training
    model.eval()
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    return stats


def gradient_rms(model) -> Dict[str, float]:
    """RMS of .grad, grouped per block (call after backward, before zero_grad).

    Called from the training loop after the optimizer step, so these are
    *post-clipping* gradients: useful for comparing layers against each other,
    not for reading off the raw gradient scale.
    """
    per_block: Dict[str, list] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if name.startswith("blocks."):
            key = "block" + name.split(".")[1]
        else:
            key = name.split(".")[0]
        per_block.setdefault(key, []).append(p.grad.detach().float().pow(2).sum())
        per_block.setdefault(key + "__n", []).append(p.grad.numel())

    out = {}
    for key in [k for k in per_block if not k.endswith("__n")]:
        sq = torch.stack(per_block[key]).sum()
        n = sum(per_block[key + "__n"])
        out[key] = (sq / n).sqrt().item()
    return out


@torch.no_grad()
def attention_entropy(model, x: torch.Tensor) -> Dict[str, float]:
    """Mean entropy (nats) of each layer's attention distribution.

    Reference points: a uniform distribution over the i+1 visible tokens has
    entropy ln(i+1); attending to a single token has entropy 0.
    """
    attns = []
    for m in model.modules():
        if isinstance(m, CausalSelfAttention):
            m.store_attn = True
            attns.append(m)

    was_training = model.training
    model.eval()
    try:
        model(x)
        out = {}
        for i, m in enumerate(attns):
            a = m.last_attn.float()                      # [B, H, T, T]
            ent = -(a * torch.log(a.clamp_min(1e-9))).sum(-1)   # [B, H, T]
            # Per-head mean over batch and query position.
            per_head = ent.mean(dim=(0, 2))              # [H]
            out[f"layer{i}"] = per_head.mean().item()
            for h in range(per_head.numel()):
                out[f"layer{i}_head{h}"] = per_head[h].item()
            m.last_attn = None
        T = x.shape[1]
        out["uniform_reference"] = float(torch.log(torch.arange(1, T + 1).float()).mean())
        return out
    finally:
        for m in attns:
            m.store_attn = False
            m.last_attn = None
        if was_training:
            model.train()


def collect(model, x: torch.Tensor, include_attention: bool = True) -> Dict[str, dict]:
    d = {"activation_rms": activation_rms(model, x)}
    if include_attention:
        d["attention_entropy"] = attention_entropy(model, x)
    return d
