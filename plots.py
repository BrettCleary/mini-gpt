"""Plot training curves from a run's log.jsonl (and, optionally, benchmark.json).

    python plots.py --run runs/tinystories
    python plots.py --run runs/tinystories --benchmark runs/benchmark.json

Writes PNGs into <run>/plots/.  Deliberately just matplotlib over a JSONL file --
no experiment-tracking framework.
"""

import argparse
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_log(path):
    train, evals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            (evals if "val_loss" in rec else train).append(rec)
    return train, evals


def _series(records, key):
    return [r[key] for r in records if key in r]


def plot_training(run_dir, out_dir):
    train, evals = read_log(os.path.join(run_dir, "log.jsonl"))
    if not train:
        print("  no training records found")
        return
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # --- loss vs tokens -----------------------------------------------------
    a = ax[0][0]
    a.plot([r["tokens"] / 1e6 for r in train], [r["train_loss"] for r in train],
           lw=0.8, alpha=0.55, label="train (per step)")
    if evals:
        a.plot([r["tokens"] / 1e6 for r in evals], [r["eval_train_loss"] for r in evals],
               "o-", ms=3, lw=1.5, label="train (eval)")
        a.plot([r["tokens"] / 1e6 for r in evals], [r["val_loss"] for r in evals],
               "o-", ms=3, lw=1.5, label="validation")
        best = min(evals, key=lambda r: r["val_loss"])
        a.axhline(best["val_loss"], ls=":", c="grey", lw=1)
        a.annotate(f"best val {best['val_loss']:.3f}\n(ppl {math.exp(min(best['val_loss'],20)):.1f})",
                   xy=(best["tokens"] / 1e6, best["val_loss"]),
                   xytext=(0.55, 0.75), textcoords="axes fraction", fontsize=9)
    a.axhline(math.log(50257), ls="--", c="tab:red", lw=1, alpha=0.6)
    a.annotate("ln(V) = 10.83  (uniform guess)", xy=(0, math.log(50257)),
               xytext=(0.32, 0.88), textcoords="axes fraction", fontsize=8,
               color="tab:red")
    # Log scale: without it the drop from 10.8 to ~2 in the first 20M tokens
    # squashes the entire rest of the run into a flat line.
    a.set_yscale("log")
    a.set_yticks([1.5, 2, 3, 5, 8, 11])
    a.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a.set_xlabel("tokens processed (millions)")
    a.set_ylabel("cross-entropy loss (nats/token, log scale)")
    a.set_title("Loss vs tokens")
    a.legend(); a.grid(alpha=0.3, which="both")

    # --- learning rate ------------------------------------------------------
    a = ax[0][1]
    a.plot(_series(train, "step"), _series(train, "lr"), lw=1.5)
    a.set_xlabel("optimizer step"); a.set_ylabel("learning rate")
    a.set_title("LR schedule: linear warmup + cosine decay")
    a.grid(alpha=0.3)

    # --- gradient norm ------------------------------------------------------
    a = ax[1][0]
    steps, gn = _series(train, "step"), _series(train, "grad_norm")
    a.plot(steps, gn, lw=0.7, alpha=0.6)
    if len(gn) > 20:                       # running mean
        w = max(len(gn) // 50, 1)
        sm = [sum(gn[max(0, i - w):i + 1]) / len(gn[max(0, i - w):i + 1])
              for i in range(len(gn))]
        a.plot(steps, sm, lw=1.8, label=f"running mean ({w})")
        a.legend()
    a.set_yscale("log")
    a.set_xlabel("optimizer step"); a.set_ylabel("global grad norm (pre-clip)")
    a.set_title("Gradient norm")
    a.grid(alpha=0.3)

    # --- throughput ---------------------------------------------------------
    a = ax[1][1]
    a.plot(_series(train, "step"), _series(train, "tokens_per_sec"), lw=0.8)
    a.set_xlabel("optimizer step"); a.set_ylabel("tokens / second")
    a.set_title("Throughput")
    a.grid(alpha=0.3)
    a2 = a.twinx()
    a2.plot(_series(train, "step"), _series(train, "vram_gb"), lw=1.0, c="tab:red", alpha=0.6)
    a2.set_ylabel("peak VRAM (GB)", color="tab:red")

    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  wrote {path}")


def plot_diagnostics(run_dir, out_dir):
    _, evals = read_log(os.path.join(run_dir, "log.jsonl"))
    evals = [r for r in evals if "diagnostics" in r]
    if not evals:
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    # activation RMS by layer, over training
    series = defaultdict(list)
    for r in evals:
        for k, v in r["diagnostics"]["activation_rms"].items():
            if k.startswith("resid_after_block"):
                series[k].append((r["step"], v))
    for k in sorted(series, key=lambda s: int(s.split("block")[1])):
        pts = series[k]
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], label=k.replace("resid_after_", ""))
    ax[0].set_title("Residual-stream RMS by depth"); ax[0].set_xlabel("step")
    ax[0].set_ylabel("RMS"); ax[0].set_yscale("log"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    # gradient RMS by block
    series = defaultdict(list)
    for r in evals:
        for k, v in r["diagnostics"].get("gradient_rms", {}).items():
            series[k].append((r["step"], v))
    for k in sorted(series):
        pts = series[k]
        ax[1].plot([p[0] for p in pts], [p[1] for p in pts], label=k)
    ax[1].set_title("Gradient RMS by block"); ax[1].set_xlabel("step")
    ax[1].set_ylabel("RMS"); ax[1].set_yscale("log"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)

    # attention entropy per layer, plus the uniform reference
    series = defaultdict(list)
    ref = None
    for r in evals:
        ent = r["diagnostics"].get("attention_entropy", {})
        ref = ent.get("uniform_reference", ref)
        for k, v in ent.items():
            if k.startswith("layer") and "head" not in k:
                series[k].append((r["step"], v))
    for k in sorted(series, key=lambda s: int(s.replace("layer", ""))):
        pts = series[k]
        ax[2].plot([p[0] for p in pts], [p[1] for p in pts], label=k)
    if ref:
        ax[2].axhline(ref, ls="--", c="k", lw=1, label="uniform (ln T avg)")
    ax[2].set_title("Attention entropy (nats)"); ax[2].set_xlabel("step")
    ax[2].set_ylabel("mean entropy"); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "diagnostics.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  wrote {path}")


def plot_benchmark(bench_path, out_dir):
    with open(bench_path) as f:
        data = json.load(f)
    res = data["results"]
    ctx = [r for r in res if r.get("sweep") == "context"]
    bat = [r for r in res if r.get("sweep") == "batch"]
    if not (ctx or bat):
        return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    if ctx:
        T = [r["context_length"] for r in ctx]
        ax[0].plot(T, [r["ms_per_step"] for r in ctx], "o-", label="measured")
        base = ctx[0]
        ax[0].plot(T, [base["ms_per_step"] * (t / base["context_length"]) for t in T],
                   "--", label="linear in T")
        ax[0].plot(T, [base["ms_per_step"] * (t / base["context_length"]) ** 2 for t in T],
                   ":", label="quadratic in T")
        ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
        ax[0].set_xlabel("context length T"); ax[0].set_ylabel("ms / step")
        ax[0].set_title("Step time vs context length"); ax[0].legend(); ax[0].grid(alpha=0.3)

        ax[1].plot(T, [r["peak_vram_gb"] for r in ctx], "o-", label="peak VRAM")
        ax[1].plot(T, [r["attn_matrix_gb"] for r in ctx], "s--",
                   label="stored T x T attention")
        ax[1].set_xscale("log", base=2)
        ax[1].set_xlabel("context length T"); ax[1].set_ylabel("GB")
        ax[1].set_title("Memory vs context length"); ax[1].legend(); ax[1].grid(alpha=0.3)

    if bat:
        B = [r["batch_size"] for r in bat]
        ax[2].plot(B, [r["tokens_per_sec"] for r in bat], "o-")
        # Anchor at zero: the spread here is a few percent, and an autoscaled
        # axis would dramatise noise into an apparent trend.
        ax[2].set_ylim(0, max(r["tokens_per_sec"] for r in bat) * 1.15)
        ax[2].set_xlabel("batch size"); ax[2].set_ylabel("tokens / second")
        ax[2].set_title(f"Throughput vs batch size (T={bat[0]['context_length']})")
        ax[2].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "benchmark.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  wrote {path}")


def plot_lr_schedule(out_dir, train_cfg=None):
    """Standalone check of the schedule shape (section 18)."""
    from config import TrainConfig
    from train import get_lr
    cfg = TrainConfig(**train_cfg) if train_cfg else TrainConfig()
    steps = list(range(cfg.max_steps))
    fig, a = plt.subplots(figsize=(7, 4))
    a.plot(steps, [get_lr(s, cfg) for s in steps])
    a.axvline(cfg.warmup_steps, ls="--", c="grey", lw=1)
    a.annotate("end of warmup", xy=(cfg.warmup_steps, cfg.lr),
               xytext=(cfg.warmup_steps * 1.4, cfg.lr * 0.85), fontsize=9)
    a.set_xlabel("optimizer step"); a.set_ylabel("learning rate")
    a.set_title(f"warmup {cfg.warmup_steps} steps -> cosine to {cfg.min_lr_ratio:g} x peak")
    a.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "lr_schedule.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/tinystories")
    p.add_argument("--benchmark", default=None)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.run, "plots")
    os.makedirs(out_dir, exist_ok=True)

    cfg_path = os.path.join(args.run, "config.json")
    train_cfg = json.load(open(cfg_path))["train"] if os.path.exists(cfg_path) else None

    if os.path.exists(os.path.join(args.run, "log.jsonl")):
        plot_training(args.run, out_dir)
        plot_diagnostics(args.run, out_dir)
    plot_lr_schedule(out_dir, train_cfg)
    if args.benchmark and os.path.exists(args.benchmark):
        plot_benchmark(args.benchmark, out_dir)


if __name__ == "__main__":
    main()
