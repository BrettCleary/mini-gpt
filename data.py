"""Data pipeline: raw text -> flat token stream on disk -> [B, T] batches.

Layout on disk (per dataset directory):
    train.bin   uint16 token ids, one long concatenated stream
    val.bin     same, held out
    meta.json   tokenizer name, vocab size, token counts

uint16 is enough because the GPT-2 vocabulary (50257) fits in 16 bits, and it
halves both the file size and the bytes we page in per batch.

Run `python data.py --dataset tinystories` to build the .bin files.
"""

import argparse
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from tokenizer import get_tokenizer

DTYPE = np.uint16

DATASETS = {
    # (name) -> {split: url}
    "tinystories": {
        "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
        "val": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
    },
}

# TinyStories separates stories with a line containing exactly this.
TINYSTORIES_SEP = "<|endoftext|>"


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------
def _download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  [skip] {dest} already present ({os.path.getsize(dest)/1e6:.1f} MB)")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    print(f"  downloading {url}")

    def hook(blocks, block_size, total):
        if total > 0 and blocks % 200 == 0:
            done = blocks * block_size
            print(f"    {done/1e6:8.1f} / {total/1e6:.1f} MB", end="\r", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    os.replace(tmp, dest)
    print(f"\n  saved {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")


def _iter_documents(path: str, sep: str = TINYSTORIES_SEP):
    """Yield documents from a text file split on the separator line."""
    buf = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == sep:
                doc = "".join(buf).strip()
                if doc:
                    yield doc
                buf = []
            else:
                buf.append(line)
    doc = "".join(buf).strip()
    if doc:
        yield doc


def prepare(
    dataset: str = "tinystories",
    data_dir: Optional[str] = None,
    tokenizer_name: str = "gpt2",
    max_docs: Optional[int] = None,
    raw_dir: str = "data/raw",
):
    """Download (if needed), tokenize, and write flat uint16 token streams."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {list(DATASETS)}")
    data_dir = data_dir or f"data/{dataset}"
    os.makedirs(data_dir, exist_ok=True)

    tok = get_tokenizer(tokenizer_name)
    meta = {"tokenizer": tokenizer_name, "vocab_size": tok.vocab_size, "counts": {}}

    for split, url in DATASETS[dataset].items():
        raw = os.path.join(raw_dir, os.path.basename(url))
        _download(url, raw)

        out_path = os.path.join(data_dir, f"{split}.bin")
        print(f"  tokenizing {split} -> {out_path}")

        n_tokens = 0
        n_docs = 0
        batch, batch_chars = [], 0
        with open(out_path, "wb") as out:
            def flush(batch):
                nonlocal n_tokens
                if not batch:
                    return
                for ids in tok.encode_batch(batch):
                    ids.append(tok.eot_id)          # document boundary
                    arr = np.asarray(ids, dtype=DTYPE)
                    out.write(arr.tobytes())
                    n_tokens += arr.size

            for doc in _iter_documents(raw):
                batch.append(doc)
                batch_chars += len(doc)
                n_docs += 1
                if batch_chars > 8_000_000:
                    flush(batch)
                    batch, batch_chars = [], 0
                    print(f"    {n_docs:,} docs, {n_tokens:,} tokens", end="\r", flush=True)
                if max_docs is not None and n_docs >= max_docs:
                    break
            flush(batch)

        meta["counts"][split] = n_tokens
        print(f"\n  {split}: {n_docs:,} documents, {n_tokens:,} tokens "
              f"({os.path.getsize(out_path)/1e6:.1f} MB)")

    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {data_dir}/meta.json")
    return meta


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class DataLoader:
    """Samples random contiguous windows from a flat token stream.

    For a start index i and context length T:
        x = tokens[i     : i + T    ]
        y = tokens[i + 1 : i + T + 1]

    so y is x shifted left by exactly one token, and the model's prediction at
    position t is scored against the token that actually followed it.

    Both come back as [B, T] int64 tensors.
    """

    data_dir: str
    context_length: int
    batch_size: int
    device: str = "cuda"
    seed: int = 1337

    def __post_init__(self):
        self.splits = {}
        for split in ("train", "val"):
            path = os.path.join(self.data_dir, f"{split}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found. Run: python data.py --dataset <name>"
                )
            # memmap: the token stream can be far larger than RAM.
            self.splits[split] = np.memmap(path, dtype=DTYPE, mode="r")
        meta_path = os.path.join(self.data_dir, "meta.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.rng = np.random.default_rng(self.seed)

    def n_tokens(self, split: str) -> int:
        return int(self.splits[split].size)

    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.splits[split]
        T, B = self.context_length, self.batch_size
        # +1 because y needs one token past the end of x.
        hi = data.size - T - 1
        assert hi > 0, f"{split} split is shorter than context_length+1"
        ix = self.rng.integers(0, hi, size=B)

        x = np.stack([data[i: i + T] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1: i + 1 + T] for i in ix]).astype(np.int64)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        if self.device.startswith("cuda"):
            # pin + non_blocking lets the H2D copy overlap with compute
            x_t = x_t.pin_memory().to(self.device, non_blocking=True)
            y_t = y_t.pin_memory().to(self.device, non_blocking=True)
        else:
            x_t = x_t.to(self.device)
            y_t = y_t.to(self.device)
        return x_t, y_t

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]


def make_synthetic_data(data_dir: str, vocab_size: int = 256, n_train: int = 200_000,
                        n_val: int = 20_000, seed: int = 0):
    """A tiny fake corpus with learnable structure, for tests and CI.

    The stream is a repeating arithmetic pattern with noise, so a working model
    should be able to get the loss well below ln(vocab_size).
    """
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        base = (np.arange(n) * 7) % (vocab_size - 8)
        noise = rng.integers(0, 8, size=n)
        stream = ((base + noise) % vocab_size).astype(DTYPE)
        stream.tofile(os.path.join(data_dir, f"{split}.bin"))
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump({"tokenizer": "synthetic", "vocab_size": vocab_size,
                   "counts": {"train": n_train, "val": n_val}}, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download and tokenize a corpus.")
    ap.add_argument("--dataset", default="tinystories", choices=list(DATASETS))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="limit documents per split (for a quick smaller corpus)")
    args = ap.parse_args()
    prepare(args.dataset, args.data_dir, args.tokenizer, args.max_docs)
