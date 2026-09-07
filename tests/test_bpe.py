import json

import pytest

from bpe import (END_OF_TEXT, N_BYTES, SPLIT_PATTERN, BPETokenizer, _merge,
                 train_bpe)

CORPUS = [
    "the cat sat on the mat. the cat ran.\n",
    "a little girl said hello to the little cat.\n",
    "running runners run. the runner ran away happily.\n",
] * 40


@pytest.fixture
def tok():
    merges, specials = train_bpe(CORPUS, vocab_size=300, verbose=False)
    return BPETokenizer(merges, specials)


# -- the merge primitive ----------------------------------------------------
def test_merge_replaces_non_overlapping_occurrences():
    assert _merge([1, 2, 1, 2, 3], (1, 2), 99) == [99, 99, 3]
    assert _merge([1, 1, 1], (1, 1), 99) == [99, 1]      # left-to-right, no overlap
    assert _merge([4, 5], (1, 2), 99) == [4, 5]          # absent pair is a no-op


# -- pre-tokenization -------------------------------------------------------
def test_split_keeps_leading_space_with_the_word():
    assert SPLIT_PATTERN.findall("the cat") == ["the", " cat"]


def test_merges_never_cross_a_pre_token_boundary(tok):
    """If a merge spanned the boundary, "the cat" would encode to fewer tokens
    than "the" and " cat" encoded separately."""
    assert tok.encode("the cat") == tok.encode("the") + tok.encode(" cat")


# -- training ---------------------------------------------------------------
def test_vocab_size_is_exact():
    """This corpus supports more merges than 300 needs, so the budget is hit
    exactly rather than running out early."""
    merges, specials = train_bpe(CORPUS, vocab_size=300, verbose=False)
    assert len(merges) == 300 - N_BYTES - len(specials)
    assert BPETokenizer(merges, specials).vocab_size == 300


def test_training_is_deterministic():
    a, _ = train_bpe(CORPUS, vocab_size=300, verbose=False)
    b, _ = train_bpe(CORPUS, vocab_size=300, verbose=False)
    assert a == b


def test_merges_are_learned_in_frequency_order(tok):
    """Rank 0 must be the most frequent adjacent byte pair in the corpus."""
    from collections import Counter
    pairs = Counter()
    for chunk, c in Counter(w for t in CORPUS for w in SPLIT_PATTERN.findall(t)).items():
        b = chunk.encode("utf-8")
        for p in zip(b, b[1:]):
            pairs[p] += c
    assert pairs[tok.merges[0]] == max(pairs.values())


def test_indexed_trainer_matches_brute_force():
    """The trainer keeps incremental pair counts for speed.  This pins it to the
    obvious O(merges x corpus) implementation, merge for merge."""
    def brute_force(texts, vocab_size, n_special=1):
        from collections import Counter
        freqs = Counter()
        for t in texts:
            freqs.update(SPLIT_PATTERN.findall(t))
        words = {tuple(c.encode("utf-8")): n for c, n in freqs.items()}
        merges = []
        for i in range(vocab_size - N_BYTES - n_special):
            pairs = Counter()
            for w, c in words.items():
                for p in zip(w, w[1:]):
                    pairs[p] += c
            if not pairs:
                break
            # most frequent, ties broken on the smaller pair (as the heap does)
            best = min(pairs, key=lambda p: (-pairs[p], p))
            merges.append(best)
            merged = Counter()
            for w, c in words.items():
                merged[tuple(_merge(list(w), best, N_BYTES + i))] += c
            words = merged
        return merges

    for vocab_size in (270, 290, 300):
        fast, _ = train_bpe(CORPUS, vocab_size=vocab_size, verbose=False)
        assert fast == brute_force(CORPUS, vocab_size)


def test_vocab_smaller_than_byte_alphabet_is_rejected():
    with pytest.raises(ValueError, match="too small"):
        train_bpe(CORPUS, vocab_size=100, verbose=False)


def test_stops_early_when_corpus_has_no_pairs_left():
    """A tiny corpus runs out of merges before an oversized vocab is filled."""
    merges, specials = train_bpe(["ab ab ab"], vocab_size=1000, verbose=False)
    assert len(merges) < 1000 - N_BYTES - len(specials)


# -- round-trip -------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "the cat sat on the mat.",
    "unseen vocabulary: xylophone quixotic",
    "",
    " ",
    "\n\n\t  ",
    "Ünïcödé — 🐈 日本語",
    "digits 12345 and 007",
    "MiXeD CaSe",
    "a" * 300,
])
def test_decode_encode_round_trip_is_exact(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_every_byte_is_representable(tok):
    """The byte alphabet is the floor, so nothing is ever out-of-vocabulary."""
    text = bytes(range(256)).decode("latin-1")
    assert tok.decode(tok.encode(text)) == text


def test_ids_are_in_range(tok):
    ids = tok.encode("the cat sat on the mat.")
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_encoding_is_deterministic(tok):
    assert tok.encode("the little cat") == tok.encode("the little cat")


def test_encode_batch_matches_encode(tok):
    texts = ["the cat", "a little girl", "running"]
    assert tok.encode_batch(texts) == [tok.encode(t) for t in texts]


def test_cache_does_not_alias_returned_lists(tok):
    """A caller mutating the result must not corrupt the chunk cache."""
    first = tok.encode("the cat")
    first.append(12345)
    assert tok.encode("the cat") != first


# -- special tokens ---------------------------------------------------------
def test_eot_id_is_the_last_id(tok):
    assert tok.eot_id == tok.vocab_size - 1


def test_specials_are_literal_unless_allowed(tok):
    text = "a" + END_OF_TEXT + "b"
    assert tok.eot_id not in tok.encode(text)
    assert tok.eot_id in tok.encode(text, allow_special=True)
    # either way the text survives a round trip
    assert tok.decode(tok.encode(text)) == text
    assert tok.decode(tok.encode(text, allow_special=True)) == text


# -- compression ------------------------------------------------------------
def test_learned_merges_beat_raw_bytes(tok):
    text = "the cat sat on the mat. the little girl said hello."
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


# -- persistence ------------------------------------------------------------
def test_save_load_round_trip(tmp_path, tok):
    p = str(tmp_path / "tok.json")
    tok.save(p)
    other = BPETokenizer.load(p)
    assert other.merges == tok.merges
    assert other.vocab_size == tok.vocab_size
    assert other.eot_id == tok.eot_id
    text = "the little cat ran away."
    assert other.encode(text) == tok.encode(text)


def test_load_rejects_a_different_pretokenization_pattern(tmp_path, tok):
    """Merges are only replayable under the pattern they were learned with."""
    p = str(tmp_path / "tok.json")
    tok.save(p)
    payload = json.load(open(p))
    payload["pattern"] = r"\w+"
    json.dump(payload, open(p, "w"))
    with pytest.raises(ValueError, match="pre-tokenization pattern"):
        BPETokenizer.load(p)


def test_load_rejects_a_foreign_file(tmp_path):
    p = str(tmp_path / "nope.json")
    json.dump({"kind": "sentencepiece"}, open(p, "w"))
    with pytest.raises(ValueError, match="not a byte-bpe"):
        BPETokenizer.load(p)


def test_missing_file_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="--train-tokenizer"):
        BPETokenizer.load(str(tmp_path / "absent.json"))


def test_get_tokenizer_dispatches_on_path(tmp_path, tok):
    from tokenizer import get_tokenizer
    p = str(tmp_path / "tok.json")
    tok.save(p)
    loaded = get_tokenizer(p)
    assert isinstance(loaded, BPETokenizer)
    assert loaded.encode("the cat") == tok.encode("the cat")
