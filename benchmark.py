"""Throughput / memory benchmark.

    python benchmark.py                       # sweep batch sizes and context lengths
    python benchmark.py --context-lengths 128 256 512 1024 --batch-sizes 8

Reports, per configuration: ms/step, tokens/sec, peak VRAM, and the share of
memory that the explicit [B, H, T, T] attention score matrices account for.

The point of the context-length sweep is to see the quadratic term arrive:
doubling T roughly doubles the linear (projection/FFN) work but quadruples the
attention work and its activation memory.  That crossover is the argument for
FlashAttention, which never materialises the T x T matrix at all.
"""

import argparse
import gc
import json
import os
import time
from dataclasses import replace

import torch

from config import PRESETS, GPTConfig
from model import GPT


def attention_bytes(cfg: GPTConfig, B: int, T: int, dtype_bytes: int = 2) -> int:
    """Bytes of *stored* attention probabilities across the whole model.

    Each layer saves one [B, H, T, T] softmax output for the backward pass.
    """
    return cfg.n_layers * B * cfg.n_heads * T * T * dtype_bytes


def flop_breakdown(cfg: GPTConfig, B: int, T: int) -> dict:
    """Forward+backward FLOPs, counting 2 FLOPs per multiply-add and 3x for bwd.

    Split so the quadratic term is visible on its own: `attn` is the only entry
    that grows as T^2; everything else is linear in T.
    """
    C, H, Dh, L = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.n_layers
    proj = L * 4 * 2 * B * T * C * (H * Dh)        # q, k, v, out projections
    attn = L * 2 * 2 * B * H * T * T * Dh          # scores (QK^T) and A@V  <- T^2
    ffn = L * 3 * 2 * B * T * C * cfg.d_ff         # swiglu gate/up/down
    head = 2 * B * T * C * cfg.vocab_size          # LM head
    total = 3.0 * (proj + attn + ffn + head)
    return {"proj": 3.0 * proj, "attn": 3.0 * attn, "ffn": 3.0 * ffn,
            "head": 3.0 * head, "total": total, "attn_share": 3.0 * attn / total}


def bench_one(cfg: GPTConfig, B: int, T: int, steps: int = 8, warmup: int = 3,
              dtype=torch.bfloat16, device="cuda", backward=True):
    torch.cuda.empty_cache()
    gc.collect()
    cfg = replace(cfg, context_length=T)
    model = GPT(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    def one_step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(x, y)
        if backward:
            loss.backward()
            opt.step()
        return loss

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps

    peak = torch.cuda.max_memory_allocated()
    res = {
        "batch_size": B, "context_length": T, "tokens_per_step": B * T,
        "ms_per_step": dt * 1e3, "tokens_per_sec": B * T / dt,
        "peak_vram_gb": peak / 1e9,
        "attn_matrix_gb": attention_bytes(cfg, B, T) / 1e9,
        "tflops": flop_breakdown(cfg, B, T)["total"] / dt / 1e12,
        "attn_flop_share": flop_breakdown(cfg, B, T)["attn_share"],
    }
    res["attn_share_of_vram"] = res["attn_matrix_gb"] / max(res["peak_vram_gb"], 1e-9)
    del model, opt, x, y
    torch.cuda.empty_cache()
    gc.collect()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small", choices=list(PRESETS))
    p.add_argument("--batch-sizes", dest="batch_sizes", type=int, nargs="+",
                   default=[4, 8, 16, 32])
    p.add_argument("--context-lengths", dest="context_lengths", type=int, nargs="+",
                   default=[128, 256, 512, 1024])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--out", default="runs/benchmark.json")
    args = p.parse_args()

    assert torch.cuda.is_available(), "benchmark needs a CUDA device"
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = PRESETS[args.preset]
    dtype = getattr(torch, args.dtype)

    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print(f"model : {args.preset}  d_model={cfg.d_model} layers={cfg.n_layers} "
          f"heads={cfg.n_heads}  dtype={args.dtype}\n")

    results = []
    hdr = (f"{'B':>4} {'T':>6} {'tok/step':>9} {'ms/step':>9} {'tok/s':>11} "
           f"{'TFLOP/s':>8} {'VRAM GB':>8} {'attnGB':>8} {'attn%mem':>8} {'attn%flop':>9}")

    print("=== batch-size sweep (T fixed at %d) ===" % cfg.context_length)
    print(hdr)
    for B in args.batch_sizes:
        try:
            r = bench_one(cfg, B, cfg.context_length, args.steps, dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>4} {cfg.context_length:>6}  OOM")
            torch.cuda.empty_cache(); gc.collect()
            continue
        r["sweep"] = "batch"
        results.append(r)
        print(f"{r['batch_size']:>4} {r['context_length']:>6} {r['tokens_per_step']:>9,} "
              f"{r['ms_per_step']:>9.1f} {r['tokens_per_sec']:>11,.0f} {r['tflops']:>8.1f} "
              f"{r['peak_vram_gb']:>8.2f} {r['attn_matrix_gb']:>8.2f} "
              f"{100*r['attn_share_of_vram']:>7.1f}% {100*r['attn_flop_share']:>8.1f}%")

    print("\n=== context-length sweep (B fixed at %d) ===" % args.batch_sizes[0])
    print(hdr)
    B0 = args.batch_sizes[0]
    prev = None
    for T in args.context_lengths:
        try:
            r = bench_one(cfg, B0, T, args.steps, dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"{B0:>4} {T:>6}  OOM  <- the T^2 attention matrix no longer fits")
            torch.cuda.empty_cache(); gc.collect()
            continue
        r["sweep"] = "context"
        results.append(r)
        growth = ""
        if prev is not None:
            growth = (f"   (T x{T/prev['context_length']:.0f}: "
                      f"ms x{r['ms_per_step']/prev['ms_per_step']:.2f}, "
                      f"attn mem x{r['attn_matrix_gb']/max(prev['attn_matrix_gb'],1e-9):.2f})")
        print(f"{r['batch_size']:>4} {r['context_length']:>6} {r['tokens_per_step']:>9,} "
              f"{r['ms_per_step']:>9.1f} {r['tokens_per_sec']:>11,.0f} {r['tflops']:>8.1f} "
              f"{r['peak_vram_gb']:>8.2f} {r['attn_matrix_gb']:>8.2f} "
              f"{100*r['attn_share_of_vram']:>7.1f}% {100*r['attn_flop_share']:>8.1f}%{growth}")
        prev = r

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "dtype": args.dtype,
                   "model": cfg.to_dict(), "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
