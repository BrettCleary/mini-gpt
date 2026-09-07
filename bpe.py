"""Byte-level BPE: train a tokenizer on a corpus instead of borrowing GPT-2's.

Why bother, when `tokenizer.py` already wraps tiktoken?  Because GPT-2's 50257
merges were fitted to WebText, and on a narrow corpus most of them never fire.
TinyStories has roughly eleven thousand distinct word types, so a 50257-entry
embedding table spends two thirds of the model's parameters on rows that are
nearly dead.  Fitting the vocabulary to the data moves that budget into layers.

The algorithm, in four stages:

  1. pre-tokenize    split text on a regex so merges can never cross a word
                     boundary (without this, BPE cheerfully learns "the cat")
  2. count           collapse the corpus into unique chunks with frequencies;
                     the merge loop then runs over ~1e4 items, not ~1e9
  3. merge           repeatedly fuse the most frequent adjacent pair, 256
                     byte tokens growing into `vocab_size - n_special`
  4. serialize       merges in rank order; the id of merge i is 256 + i

Encoding replays the merges in the order they were learned, which is what makes
`decode(encode(s)) == s` for any string: the floor is raw bytes, so nothing is
ever out-of-vocabulary.

    python bpe.py --input data/raw/TinyStories-valid.txt --vocab-size 8192 \
        --out data/tinystories/tokenizer.json
"""

import argparse
import heapq
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# GPT-2's pre-tokenization pattern, restricted to ASCII letter/digit classes so
# it runs on the standard library's `re` (the original needs `regex` for \p{L}).
# The leading " ?" is why " the" is one token and "the" is another: whitespace
# is carried into the following word rather than tokenized on its own.
SPLIT_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)

END_OF_TEXT = "<|endoftext|>"
N_BYTES = 256          # the base alphabet: every byte is already a token


def _merge(seq: Sequence[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of `pair` in `seq` with `new_id`."""
    out: List[int] = []
    i, n = 0, len(seq)
    while i < n:
        if i < n - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_bpe(
    texts: Iterable[str],
    vocab_size: int = 8192,
    specials: Sequence[str] = (END_OF_TEXT,),
    verbose: bool = True,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """Learn merges from a corpus.  Returns (merges in rank order, specials).

    The corpus is consumed once into a frequency table of pre-tokens, so memory
    scales with the number of distinct words, not with the size of the text.
    """
    n_merges = vocab_size - N_BYTES - len(specials)
    if n_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size} is too small: need at least "
            f"{N_BYTES + len(specials)} for the byte alphabet plus specials"
        )

    # -- stages 1 and 2: pre-tokenize and count -----------------------------
    freqs: Counter = Counter()
    for text in texts:
        freqs.update(SPLIT_PATTERN.findall(text))

    words: List[List[int]] = []
    counts: List[int] = []
    for chunk, c in freqs.items():
        words.append(list(chunk.encode("utf-8")))
        counts.append(c)
    if verbose:
        total = sum(counts)
        print(f"  {len(words):,} distinct pre-tokens from {total:,} occurrences")

    # -- stage 3: the merge loop --------------------------------------------
    # A naive implementation recounts every pair in the corpus on every
    # iteration, which is O(merges x corpus) and takes minutes.  Instead keep a
    # running count per pair plus an index from pair -> the words containing it,
    # so a merge only touches the words that actually changed.
    pair_counts: Counter = Counter()
    pair_words: Dict[Tuple[int, int], set] = {}
    # Lazy-deletion heap: entries go stale when a count changes, so validate on
    # pop.  Ordering on (-count, pair) makes ties break on the smaller pair,
    # which keeps training deterministic across runs.
    heap: List[Tuple[int, Tuple[int, int]]] = []

    def _index(idx: int, sign: int):
        w, c = words[idx], counts[idx] * sign
        for p in zip(w, w[1:]):
            pair_counts[p] += c
            if pair_counts[p] <= 0:
                del pair_counts[p]
                s = pair_words.get(p)
                if s is not None:
                    s.discard(idx)
            else:
                if sign > 0:
                    pair_words.setdefault(p, set()).add(idx)
                elif idx in pair_words.get(p, ()):
                    pair_words[p].discard(idx)
                heapq.heappush(heap, (-pair_counts[p], p))

    for i in range(len(words)):
        _index(i, +1)   # the pushes here leave a valid entry for every pair

    merges: List[Tuple[int, int]] = []
    for i in range(n_merges):
        best: Optional[Tuple[int, int]] = None
        while heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) == -neg_count:   # not stale
                best = pair
                break
        if best is None:
            if verbose:
                print(f"\n  corpus exhausted after {i:,} merges "
                      f"(vocab {N_BYTES + i + len(specials):,})")
            break

        new_id = N_BYTES + i
        merges.append(best)
        affected = list(pair_words.get(best, ()))
        for idx in affected:
            _index(idx, -1)
        for idx in affected:
            words[idx] = _merge(words[idx], best, new_id)
        for idx in affected:
            _index(idx, +1)

        if verbose and (i % 500 == 0 or i == n_merges - 1):
            print(f"    merge {i + 1:,}/{n_merges:,}", end="\r", flush=True)

    if verbose:
        print(f"\n  learned {len(merges):,} merges -> vocab "
              f"{N_BYTES + len(merges) + len(specials):,}")
    return merges, list(specials)


# ---------------------------------------------------------------------------
# The tokenizer
# ---------------------------------------------------------------------------
class BPETokenizer:
    """Interface-compatible with `tokenizer.Tokenizer`, backed by learned merges.

    Ids are laid out as:  [0, 256) raw bytes | merges in rank order | specials
    """

    def __init__(self, merges: Sequence[Tuple[int, int]],
                 specials: Sequence[str] = (END_OF_TEXT,),
                 name: str = "bpe"):
        self.name = name
        self.merges = [tuple(m) for m in merges]
        self.specials = list(specials)
        # rank == position in training order; the id of rank r is 256 + r, and
        # encoding must apply the *lowest* available rank first to replay it.
        self._ranks: Dict[Tuple[int, int], int] = {p: r for r, p in enumerate(self.merges)}

        self._vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(N_BYTES)}
        for r, (a, b) in enumerate(self.merges):
            self._vocab[N_BYTES + r] = self._vocab[a] + self._vocab[b]

        base = N_BYTES + len(self.merges)
        self.special_ids: Dict[str, int] = {s: base + i for i, s in enumerate(self.specials)}
        for s, i in self.special_ids.items():
            self._vocab[i] = s.encode("utf-8")

        self.vocab_size: int = base + len(self.specials)
        self.eot_id: int = self.special_ids.get(END_OF_TEXT, self.vocab_size - 1)
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in self.specials) + ")")
            if self.specials else None
        )
        self._cache: Dict[str, List[int]] = {}

    # -- encoding -----------------------------------------------------------
    def _encode_chunk(self, chunk: str) -> List[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = self._ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            ids = _merge(ids, best, N_BYTES + best_rank)
        if len(self._cache) < 500_000:      # pre-tokens repeat heavily; bound it anyway
            self._cache[chunk] = ids
        return list(ids)

    def _encode_ordinary(self, text: str) -> List[int]:
        out: List[int] = []
        for chunk in SPLIT_PATTERN.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allow_special: bool = False) -> List[int]:
        """`allow_special=False` treats "<|endoftext|>" in the text as literal
        characters, so caller-supplied text can never inject a document break."""
        if not allow_special or self._special_re is None:
            return self._encode_ordinary(text)
        out: List[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            if part in self.special_ids:
                out.append(self.special_ids[part])
            else:
                out.extend(self._encode_ordinary(part))
        return out

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [self._encode_ordinary(t) for t in texts]

    # -- decoding -----------------------------------------------------------
    def decode(self, ids: Sequence[int]) -> str:
        parts = [self._vocab[int(i)] for i in ids]
        # errors="replace" because generation can stop part-way through a
        # multi-byte character, leaving a byte sequence that is not valid UTF-8.
        return b"".join(parts).decode("utf-8", errors="replace")

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "kind": "byte-bpe",
            "pattern": SPLIT_PATTERN.pattern,
            "specials": self.specials,
            "merges": [list(m) for m in self.merges],
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. A checkpoint trained with a fitted tokenizer "
                f"stores its path; rebuild it with: python data.py --train-tokenizer"
            )
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("kind") != "byte-bpe":
            raise ValueError(f"{path} is not a byte-bpe tokenizer file")
        if payload.get("pattern") != SPLIT_PATTERN.pattern:
            raise ValueError(
                f"{path} was trained with a different pre-tokenization pattern; "
                "its merges would not replay correctly"
            )
        return cls([tuple(m) for m in payload["merges"]],
                   payload.get("specials", [END_OF_TEXT]), name=path)

    def __repr__(self) -> str:
        return (f"BPETokenizer(name={self.name!r}, vocab_size={self.vocab_size}, "
                f"merges={len(self.merges)}, eot_id={self.eot_id})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_sample(path: str, max_bytes: Optional[int]) -> Iterable[str]:
    """Stream a prefix of a text file in chunks, stopping at max_bytes."""
    read = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                return
            read += len(block)
            yield block
            if max_bytes is not None and read >= max_bytes:
                return


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    ap.add_argument("--input", required=True, help="raw text file to fit on")
    ap.add_argument("--out", required=True, help="destination .json")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--sample-mb", type=int, default=200,
                    help="fit on at most this many MB of the input (0 = all)")
    args = ap.parse_args()

    limit = args.sample_mb * 1_000_000 if args.sample_mb else None
    print(f"training byte-level BPE on {args.input}")
    merges, specials = train_bpe(_read_sample(args.input, limit), args.vocab_size)
    tok = BPETokenizer(merges, specials, name=args.out)
    tok.save(args.out)
    print(f"  wrote {args.out}  ({tok!r})")
