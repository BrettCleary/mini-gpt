"""Training loop.

Plain PyTorch: our own step function, our own LR schedule, our own AMP handling,
our own checkpointing.  Nothing here is hidden inside a Trainer class.

    python train.py --preset small --max-steps 6000
    python train.py --resume auto
    python train.py --overfit           # section-15 sanity check: memorise one batch
"""

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Optional

import numpy as np
import torch

import diagnostics as diag
from checkpoint import (latest_checkpoint, load_checkpoint, prune_checkpoints,
                        restore, save_checkpoint)
from config import PRESETS, GPTConfig, TrainConfig
from data import DataLoader
from model import GPT
from tokenizer import get_tokenizer


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup, then cosine decay to min_lr.
# ---------------------------------------------------------------------------
def get_lr(step: int, cfg: TrainConfig) -> float:
    min_lr = cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        # Linear from ~0 to lr. Warmup exists because Adam's second-moment
        # estimate is garbage for the first few dozen steps; taking full-size
        # steps then is how runs blow up.
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    if step >= cfg.max_steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Optimizer: two parameter groups, weight decay only on matrices.
# ---------------------------------------------------------------------------
def configure_optimizer(model: torch.nn.Module, cfg: TrainConfig, verbose: bool = True):
    """Decay every >=2D parameter (all the Linear/Embedding matrices);
    do not decay 1D parameters (RMSNorm scales and any biases).

    Shrinking a norm scale toward zero would fight the layer's whole purpose,
    and decaying a bias just adds a constant pull with no regularising effect.
    """
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue           # tied weights appear twice; count them once
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # Fused AdamW keeps the whole update on-GPU in one kernel.
    use_fused = torch.cuda.is_available() and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            eps=1e-8, fused=use_fused)
    if verbose:
        print(f"  optimizer: AdamW(fused={use_fused}) "
              f"decay={sum(p.numel() for p in decay):,} params in {len(decay)} tensors | "
              f"no_decay={sum(p.numel() for p in no_decay):,} in {len(no_decay)} tensors")
    return opt


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
def setup_precision(cfg: TrainConfig):
    """Return (autocast_ctx_factory, scaler, resolved_dtype_name).

    Parameters and the optimizer always stay in fp32.  autocast runs the matmuls
    in bf16/fp16 and leaves the reductions (softmax, RMSNorm, cross-entropy) in
    fp32.  bf16 has the same exponent range as fp32, so it needs no loss scaler;
    fp16 does not, so it needs one.
    """
    want = cfg.dtype
    if want == "bfloat16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        print("  [warn] bf16 unsupported on this device, falling back to float16")
        want = "float16"
    if not torch.cuda.is_available():
        want = "float32"

    if want == "float32":
        ctx = lambda: nullcontext()
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    else:
        dt = torch.bfloat16 if want == "bfloat16" else torch.float16
        ctx = lambda: torch.amp.autocast(device_type="cuda", dtype=dt)
        scaler = torch.amp.GradScaler("cuda", enabled=(want == "float16"))
    return ctx, scaler, want


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, cfg: TrainConfig, autocast_ctx, splits=("train", "val")):
    model.eval()
    out = {}
    for split in splits:
        losses = torch.zeros(cfg.eval_batches)
        for i in range(cfg.eval_batches):
            x, y = loader.get_batch(split)
            with autocast_ctx():
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Startup summary
# ---------------------------------------------------------------------------
def print_summary(model, mcfg: GPTConfig, tcfg: TrainConfig, loader, dtype_name):
    n = model.num_params()
    print("=" * 72)
    print("  model")
    print(f"    parameters        {n:,}  ({n/1e6:.2f}M)")
    print(f"    non-embedding     {model.num_params(non_embedding=True):,}")
    print(f"    layers            {mcfg.n_layers}")
    print(f"    d_model           {mcfg.d_model}")
    print(f"    heads             {mcfg.n_heads} x d_head {mcfg.d_head}")
    print(f"    ffn               {mcfg.ffn_type} (d_ff={mcfg.d_ff})")
    print(f"    context length    {mcfg.context_length}")
    print(f"    vocab size        {mcfg.vocab_size}")
    print(f"    tied embeddings   {mcfg.tie_embeddings}")
    print("  training")
    print(f"    micro batch       {tcfg.micro_batch_size}")
    print(f"    grad accum        {tcfg.grad_accum_steps}")
    print(f"    effective batch   {tcfg.effective_batch_size}")
    print(f"    tokens / step     {tcfg.tokens_per_optim_step(mcfg.context_length):,}")
    print(f"    max steps         {tcfg.max_steps}  "
          f"(~{tcfg.max_steps * tcfg.tokens_per_optim_step(mcfg.context_length)/1e6:.0f}M tokens)")
    print(f"    peak lr           {tcfg.lr}  warmup {tcfg.warmup_steps}  "
          f"min {tcfg.lr*tcfg.min_lr_ratio:g}")
    print(f"    weight decay      {tcfg.weight_decay}   grad clip {tcfg.grad_clip}")
    print(f"    precision         params fp32 / autocast {dtype_name}")
    if loader is not None:
        print(f"    train tokens      {loader.n_tokens('train'):,}")
        print(f"    val tokens        {loader.n_tokens('val'):,}")
        epochs = (tcfg.max_steps * tcfg.tokens_per_optim_step(mcfg.context_length)
                  / max(loader.n_tokens('train'), 1))
        print(f"    epochs over data  {epochs:.2f}")
    if torch.cuda.is_available():
        print(f"    device            {torch.cuda.get_device_name(0)}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def train(mcfg: GPTConfig, tcfg: TrainConfig, overfit: bool = False):
    set_seed(tcfg.seed)
    os.makedirs(tcfg.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = tcfg.device if torch.cuda.is_available() else "cpu"

    loader = DataLoader(tcfg.data_dir, mcfg.context_length, tcfg.micro_batch_size,
                        device=device, seed=tcfg.seed)
    meta_vocab = loader.meta.get("vocab_size")
    if meta_vocab and meta_vocab != mcfg.vocab_size:
        print(f"  [info] adopting vocab_size={meta_vocab} from {tcfg.data_dir}/meta.json")
        mcfg = replace(mcfg, vocab_size=meta_vocab)

    model = GPT(mcfg).to(device)
    optimizer = configure_optimizer(model, tcfg)
    autocast_ctx, scaler, dtype_name = setup_precision(tcfg)

    start_step, tokens_seen, best_val = 0, 0, float("inf")
    resume_path = None
    if tcfg.resume == "auto":
        resume_path = latest_checkpoint(tcfg.out_dir)
    elif tcfg.resume:
        resume_path = tcfg.resume
    if resume_path:
        print(f"  resuming from {resume_path}")
        ckpt = load_checkpoint(resume_path, map_location=device)
        start_step = restore(ckpt, model, optimizer, loader, scaler)
        tokens_seen = ckpt.get("tokens_seen", 0)
        best_val = ckpt.get("best_val_loss") or float("inf")
        print(f"  resumed at step {start_step}, {tokens_seen:,} tokens seen")

    raw_model = model
    if tcfg.compile:
        print("  compiling model (first step will be slow) ...")
        model = torch.compile(model)

    print_summary(raw_model, mcfg, tcfg, loader, dtype_name)

    tokenizer_name = loader.meta.get("tokenizer", "gpt2")
    tokenizer = None if tokenizer_name == "synthetic" else get_tokenizer(tokenizer_name)

    log_path = os.path.join(tcfg.out_dir, "log.jsonl")
    sample_path = os.path.join(tcfg.out_dir, "samples.txt")
    log_file = open(log_path, "a")
    with open(os.path.join(tcfg.out_dir, "config.json"), "w") as f:
        json.dump({"model": mcfg.to_dict(), "train": tcfg.to_dict()}, f, indent=2)

    tokens_per_step = tcfg.tokens_per_optim_step(mcfg.context_length)

    def _restart_timer(step):
        """Begin a fresh timing / memory interval after an eval, sample or save.

        Resetting the CUDA high-water mark here keeps the logged peak VRAM a
        measure of the *training* step; without it the diagnostics forward pass,
        which holds a [B, H, T, T] attention tensor per layer, sets a peak that
        every later reading inherits.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return time.perf_counter(), step

    # --overfit: one fixed batch, reused forever. Loss must go to ~0.
    fixed_batch = loader.get_batch("train") if overfit else None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.train()
    # Timing bookkeeping: `t0` starts an interval and `last_timed_step` records
    # the step it starts after, so an interval cut short by an eval or a
    # checkpoint is divided by the steps that actually ran in it rather than by
    # log_interval. Getting this wrong reports 100x throughput spikes.
    t0 = time.perf_counter()
    last_timed_step = start_step - 1

    for step in range(start_step, tcfg.max_steps):
        lr = get_lr(step, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        # ---- gradient accumulation ------------------------------------------
        # Dividing each micro-batch loss by grad_accum_steps makes the summed
        # gradient equal to the gradient of the mean loss over the full
        # effective batch -- identical to running one big batch.
        loss_accum = 0.0
        for _ in range(tcfg.grad_accum_steps):
            x, y = fixed_batch if overfit else loader.get_batch("train")
            with autocast_ctx():
                _, loss = model(x, y)
                loss = loss / tcfg.grad_accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # ---- clip, step -----------------------------------------------------
        scaler.unscale_(optimizer)   # gradients must be unscaled before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens_seen += tokens_per_step

        # ---- logging --------------------------------------------------------
        if step % tcfg.log_interval == 0 or step == tcfg.max_steps - 1:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            dt = now - t0
            steps_done = max(step - last_timed_step, 1)
            t0, last_timed_step = now, step
            ms_per_step = dt / steps_done * 1e3
            tok_per_sec = tokens_per_step * steps_done / dt
            mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            gn = float(grad_norm)
            rec = {"step": step, "tokens": tokens_seen, "train_loss": loss_accum,
                   "lr": lr, "grad_norm": gn, "ms_per_step": ms_per_step,
                   "tokens_per_sec": tok_per_sec, "vram_gb": mem}
            log_file.write(json.dumps(rec) + "\n"); log_file.flush()
            print(f"step {step:6d} | loss {loss_accum:7.4f} | lr {lr:.2e} | "
                  f"gnorm {gn:6.3f} | {ms_per_step:7.1f} ms | "
                  f"{tok_per_sec:9,.0f} tok/s | {mem:.2f} GB")
            if not math.isfinite(loss_accum):
                raise RuntimeError(f"loss is {loss_accum} at step {step} -- training diverged")
            if not math.isfinite(gn):
                # bf16 should never get here; fp16 can, and the scaler will have
                # skipped the step, but a persistent NaN grad means a real bug.
                raise RuntimeError(f"gradient norm is {gn} at step {step}")

        # ---- validation -----------------------------------------------------
        if (step + 1) % tcfg.eval_interval == 0 or step == tcfg.max_steps - 1:
            losses = evaluate(raw_model, loader, tcfg, autocast_ctx)
            rec = {"step": step, "tokens": tokens_seen,
                   "eval_train_loss": losses["train"], "val_loss": losses["val"],
                   "val_ppl": math.exp(min(losses["val"], 20))}
            if tcfg.diagnostics:
                # A few rows only: attention_entropy holds a [B, H, T, T]
                # tensor per layer, which is the biggest allocation in the run.
                xd, _ = loader.get_batch("val")
                xd = xd[:4]
                rec["diagnostics"] = {
                    "activation_rms": diag.activation_rms(raw_model, xd),
                    "attention_entropy": diag.attention_entropy(raw_model, xd),
                    "gradient_rms": diag.gradient_rms(raw_model),
                }
            log_file.write(json.dumps(rec) + "\n"); log_file.flush()
            print(f"  eval @ {step:6d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | ppl {math.exp(min(losses['val'],20)):.2f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(os.path.join(tcfg.out_dir, "best.pt"), raw_model,
                                optimizer, step + 1, mcfg, tcfg, losses["val"],
                                best_val, tokens_seen, loader.state_dict(), scaler,
                                tokenizer_name)
            t0, last_timed_step = _restart_timer(step)

        # ---- periodic sample ------------------------------------------------
        if tokenizer is not None and ((step + 1) % tcfg.sample_interval == 0
                                      or step == tcfg.max_steps - 1):
            prompt_ids = torch.tensor([tokenizer.encode(tcfg.sample_prompt)],
                                      dtype=torch.long, device=device)
            with autocast_ctx():
                out = raw_model.generate(prompt_ids, tcfg.sample_tokens,
                                         temperature=0.8, top_k=50)
            text = tokenizer.decode(out[0].tolist())
            with open(sample_path, "a") as f:
                f.write(f"\n===== step {step+1} (tokens={tokens_seen:,}) =====\n{text}\n")
            print(f"  sample @ {step+1}: {text[:300]}...")
            t0, last_timed_step = _restart_timer(step)

        # ---- checkpoint -----------------------------------------------------
        if (step + 1) % tcfg.ckpt_interval == 0 or step == tcfg.max_steps - 1:
            path = os.path.join(tcfg.out_dir, f"ckpt_{step+1:07d}.pt")
            save_checkpoint(path, raw_model, optimizer, step + 1, mcfg, tcfg,
                            None, best_val, tokens_seen, loader.state_dict(), scaler,
                            tokenizer_name)
            prune_checkpoints(tcfg.out_dir, tcfg.keep_last_k)
            print(f"  saved {path}")
            t0, last_timed_step = _restart_timer(step)

        if overfit and loss_accum < 1e-3:
            print(f"  overfit target reached at step {step}: loss {loss_accum:.6f}")
            break

    log_file.close()
    print(f"done. best val loss {best_val:.4f}")
    return raw_model, best_val


# ---------------------------------------------------------------------------
def build_configs(args):
    mcfg = PRESETS[args.preset]
    overrides = {}
    for k in ("context_length", "d_model", "n_layers", "n_heads", "d_head",
              "d_ff", "dropout", "ffn_type"):
        v = getattr(args, k, None)
        if v is not None:
            overrides[k] = v
    if args.no_tie:
        overrides["tie_embeddings"] = False
    if overrides:
        mcfg = replace(mcfg, **overrides)

    tcfg = TrainConfig()
    for k in ("data_dir", "micro_batch_size", "grad_accum_steps", "max_steps",
              "warmup_steps", "lr", "weight_decay", "grad_clip", "dtype",
              "eval_interval", "eval_batches", "log_interval", "sample_interval",
              "out_dir", "ckpt_interval", "seed", "resume"):
        v = getattr(args, k, None)
        if v is not None:
            tcfg = replace(tcfg, **{k: v})
    tcfg = replace(tcfg, compile=args.compile, diagnostics=args.diagnostics)
    return mcfg, tcfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="small", choices=list(PRESETS))
    p.add_argument("--data-dir", dest="data_dir", default=None)
    p.add_argument("--out-dir", dest="out_dir", default=None)
    # model overrides
    p.add_argument("--context-length", dest="context_length", type=int)
    p.add_argument("--d-model", dest="d_model", type=int)
    p.add_argument("--n-layers", dest="n_layers", type=int)
    p.add_argument("--n-heads", dest="n_heads", type=int)
    p.add_argument("--d-head", dest="d_head", type=int)
    p.add_argument("--d-ff", dest="d_ff", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--ffn-type", dest="ffn_type", choices=["swiglu", "gelu"])
    p.add_argument("--no-tie", action="store_true", help="untied LM head (ablation)")
    # training overrides
    p.add_argument("--micro-batch-size", dest="micro_batch_size", type=int)
    p.add_argument("--grad-accum-steps", dest="grad_accum_steps", type=int)
    p.add_argument("--max-steps", dest="max_steps", type=int)
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", dest="weight_decay", type=float)
    p.add_argument("--grad-clip", dest="grad_clip", type=float)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--eval-interval", dest="eval_interval", type=int)
    p.add_argument("--eval-batches", dest="eval_batches", type=int)
    p.add_argument("--log-interval", dest="log_interval", type=int)
    p.add_argument("--sample-interval", dest="sample_interval", type=int)
    p.add_argument("--ckpt-interval", dest="ckpt_interval", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", default=None, help="'auto' or a checkpoint path")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--overfit", action="store_true",
                   help="train on one fixed batch until the loss collapses")
    args = p.parse_args()

    # Line-buffer stdout so `python train.py > log.txt` streams progress instead
    # of sitting in a 8 KB block buffer for minutes at a time.
    sys.stdout.reconfigure(line_buffering=True)

    mcfg, tcfg = build_configs(args)
    if args.overfit:
        tcfg = replace(tcfg, micro_batch_size=4, grad_accum_steps=1, max_steps=400,
                       warmup_steps=20, eval_interval=10**9, sample_interval=10**9,
                       ckpt_interval=10**9, log_interval=20,
                       out_dir=tcfg.out_dir + "_overfit")
        mcfg = replace(mcfg, context_length=min(mcfg.context_length, 64))
    train(mcfg, tcfg, overfit=args.overfit)


if __name__ == "__main__":
    main()
