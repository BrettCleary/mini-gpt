"""Sample text from a trained checkpoint.

    python generate.py --ckpt runs/tinystories/best.pt --prompt "Once upon a time"
    python generate.py --ckpt runs/tinystories/best.pt --greedy
    python generate.py --ckpt runs/tinystories/best.pt --temperature 0.8 --top-k 50 -n 5

Sampling knobs:
  greedy        always take argmax. Deterministic, and repetitive: it collapses
                into loops because the highest-probability continuation of a
                loop is more of the loop.
  temperature   divides the logits before softmax. <1 sharpens the distribution
                (safer, blander), >1 flattens it (more surprising, less coherent).
  top-k         zero out everything outside the k most likely tokens, then
                renormalise. Stops the long tail of ~50k tokens from
                contributing, in aggregate, a lot of probability to nonsense.
"""

import argparse
import time

import torch

from checkpoint import load_checkpoint, model_from_checkpoint
from tokenizer import get_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", dest="top_k", type=int, default=50)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--num-samples", "-n", dest="num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--stop-at-eot", action="store_true",
                   help="stop when the model emits <|endoftext|>")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt = load_checkpoint(args.ckpt, map_location=args.device)
    model, cfg = model_from_checkpoint(ckpt, args.device)
    model.eval()

    tokenizer = get_tokenizer(ckpt.get("tokenizer_name", "gpt2"))

    print(f"checkpoint : {args.ckpt}")
    print(f"step       : {ckpt.get('step')}   val_loss: {ckpt.get('val_loss')}")
    print(f"params     : {model.num_params():,}")
    print(f"decoding   : {'greedy' if args.greedy else f'temp={args.temperature} top_k={args.top_k}'}")
    print("-" * 72)

    ids = tokenizer.encode(args.prompt)
    if not ids:
        ids = [tokenizer.eot_id]
    x = torch.tensor([ids], dtype=torch.long, device=args.device)

    amp = (torch.amp.autocast(device_type="cuda", dtype=getattr(torch, args.dtype))
           if args.device.startswith("cuda") and args.dtype != "float32"
           else torch.amp.autocast(device_type="cpu", enabled=False))

    for i in range(args.num_samples):
        t0 = time.perf_counter()
        with amp:
            out = model.generate(
                x, args.max_new_tokens,
                temperature=args.temperature,
                top_k=None if args.greedy else args.top_k,
                greedy=args.greedy,
                eos_id=tokenizer.eot_id if args.stop_at_eot else None,
            )
        dt = time.perf_counter() - t0
        n_new = out.shape[1] - x.shape[1]
        text = tokenizer.decode(out[0].tolist())
        print(text)
        print("-" * 72)
        print(f"[sample {i+1}/{args.num_samples}: {n_new} tokens in {dt:.2f}s "
              f"= {n_new/dt:.1f} tok/s]")
        print("-" * 72)


if __name__ == "__main__":
    main()
