"""Tokenizer wrapper.

We deliberately do *not* implement BPE here: the point of this project is the
transformer, and a hand-rolled tokenizer would only slow that down.  This is a
thin, explicit wrapper around tiktoken's GPT-2 encoding so the rest of the code
never has to know which library produced the ids.

GPT-2 BPE: 50257 tokens = 50000 merges + 256 byte tokens + 1 special token
(<|endoftext|>, id 50256), which we reuse as a document separator.
"""

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
def get_tokenizer(name: str = "gpt2") -> Tokenizer:
    """Cached so repeated calls don't re-read the merge table."""
    return Tokenizer(name)
