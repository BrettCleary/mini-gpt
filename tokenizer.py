"""Tokenizer wrapper.

A thin, explicit wrapper around tiktoken's GPT-2 encoding so the rest of the
code never has to know which library produced the ids.

GPT-2 BPE: 50257 tokens = 50000 merges + 256 byte tokens + 1 special token
(<|endoftext|>, id 50256), which we reuse as a document separator.

`get_tokenizer` also accepts a path to a tokenizer trained by `bpe.py`, which
exposes the same four methods.  That is the interesting comparison on a narrow
corpus: GPT-2's merges were fitted to WebText, and most of them never fire on
TinyStories, so the embedding table carries thousands of nearly dead rows.
"""

import os
from functools import lru_cache
from typing import List, Sequence

import tiktoken


class Tokenizer:
    def __init__(self, name: str = "gpt2"):
        self.name = name
        self._enc = tiktoken.get_encoding(name)
        # n_vocab already includes the special tokens.
        self.vocab_size: int = self._enc.n_vocab
        self.eot_id: int = self._enc.eot_token

    def encode(self, text: str, allow_special: bool = False) -> List[int]:
        if allow_special:
            return self._enc.encode(text, allowed_special="all")
        return self._enc.encode_ordinary(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._enc.decode(list(ids))

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return self._enc.encode_ordinary_batch(list(texts))

    def __repr__(self) -> str:
        return f"Tokenizer(name={self.name!r}, vocab_size={self.vocab_size}, eot_id={self.eot_id})"


@lru_cache(maxsize=4)
def get_tokenizer(name: str = "gpt2"):
    """Cached so repeated calls don't re-read the merge table.

    `name` is either a tiktoken encoding ("gpt2") or a path to a .json written
    by `bpe.py`.  Both satisfy the same interface: encode / decode /
    encode_batch / vocab_size / eot_id.
    """
    if name.endswith(".json") or os.path.sep in name:
        from bpe import BPETokenizer      # imported lazily: tiktoken users never need it
        return BPETokenizer.load(name)
    return Tokenizer(name)
