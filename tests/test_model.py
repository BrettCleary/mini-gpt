from dataclasses import replace

import pytest
import torch

from config import GPTConfig
from model import GPT


def test_forward_shapes(cfg):
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (3, 12))
    logits, loss = m(idx)
    assert logits.shape == (3, 12, cfg.vocab_size)
    assert loss is None


def test_loss_is_finite_and_near_ln_vocab_at_init(cfg):
    """An untrained model should be about as good as a uniform guess."""
    import math
    m = GPT(replace(cfg, vocab_size=1000))
    idx = torch.randint(0, 1000, (8, 16))
    tgt = torch.randint(0, 1000, (8, 16))
    _, loss = m(idx, tgt)
    assert torch.isfinite(loss)
    assert abs(loss.item() - math.log(1000)) < 0.5


def test_backward_produces_gradients_everywhere(cfg):
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    _, loss = m(idx, idx)
    loss.backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not missing, f"no gradient reached: {missing}"


def test_last_token_only_matches_full_logits(cfg):
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 9))
    full, _ = m(idx)
    last, _ = m(idx, last_token_only=True)
    assert last.shape == (2, 1, cfg.vocab_size)
    assert torch.allclose(full[:, -1:], last, atol=1e-5)


def test_weight_tying(cfg):
    tied = GPT(replace(cfg, tie_embeddings=True))
    assert tied.lm_head.weight is tied.token_embedding.weight
    untied = GPT(replace(cfg, tie_embeddings=False))
    assert untied.lm_head.weight is not untied.token_embedding.weight
    # Tying removes exactly one vocab_size x d_model matrix.
    assert (untied.num_params() - tied.num_params()
            == cfg.vocab_size * cfg.d_model)


def test_no_positional_embedding_table(cfg):
    m = GPT(cfg)
    names = [n for n, _ in m.named_parameters()]
    assert not any("pos" in n for n in names), names


def test_rope_cache_is_a_buffer_not_a_parameter(cfg):
    m = GPT(cfg)
    assert "rope_cos" in dict(m.named_buffers())
    assert "rope_cos" not in dict(m.named_parameters())
    # non-persistent: it is recomputed, not carried in the checkpoint
    assert "rope_cos" not in m.state_dict()


def test_causality_end_to_end_shared_prefix(cfg):
    """Section 10: two sequences sharing a prefix must agree on that prefix.

        A B C D E
        A B C X Y
    """
    m = GPT(cfg).eval()
    prefix = [5, 9, 13]
    a = torch.tensor([prefix + [21, 34]])
    b = torch.tensor([prefix + [77, 88]])
    la, _ = m(a)
    lb, _ = m(b)
    # positions 0..2 see only the shared prefix
    assert torch.allclose(la[:, :3], lb[:, :3], atol=1e-5)
    # position 3 onward differs, because the inputs differ there
    assert not torch.allclose(la[:, 3:], lb[:, 3:], atol=1e-5)


def test_causality_via_gradients(cfg):
    """d logits[t] / d embedding[t'] must be zero for t' > t."""
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    emb = m.token_embedding(idx).detach().requires_grad_(True)

    # run the stack manually on a leaf embedding so we can take gradients
    x = emb
    cos, sin = m.rope_cos[:8], m.rope_sin[:8]
    for blk in m.blocks:
        x = blk(x, cos, sin)
    logits = m.lm_head(m.final_norm(x))
    logits[0, 2].sum().backward()
    assert emb.grad[0, 3:].abs().max().item() == 0.0
    assert emb.grad[0, :3].abs().max().item() > 0.0


def test_variable_sequence_length(cfg):
    m = GPT(cfg).eval()
    for T in (1, 5, cfg.context_length):
        logits, _ = m(torch.randint(0, cfg.vocab_size, (2, T)))
        assert logits.shape == (2, T, cfg.vocab_size)


def test_rejects_too_long_sequence(cfg):
    m = GPT(cfg)
    with pytest.raises(AssertionError):
        m(torch.randint(0, cfg.vocab_size, (1, cfg.context_length + 1)))


def test_batch_independence(cfg):
    """Row b of the output must not depend on row b' of the input."""
    m = GPT(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (1, 10))
    b = torch.randint(0, cfg.vocab_size, (1, 10))
    la, _ = m(a)
    together, _ = m(torch.cat([a, b], dim=0))
    assert torch.allclose(la[0], together[0], atol=1e-5)


def test_generation_shapes_and_greedy_determinism(cfg):
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 4))
    out = m.generate(idx, max_new_tokens=6, greedy=True)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, :4], idx)                  # prompt preserved
    assert torch.equal(out, m.generate(idx, 6, greedy=True))  # deterministic


def test_generation_respects_context_window(cfg):
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
    out = m.generate(idx, max_new_tokens=3, greedy=True)
    assert out.shape == (1, cfg.context_length + 3)


def test_top_k_restricts_the_sampled_set(cfg):
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, _ = m(idx, last_token_only=True)
    allowed = set(logits[0, -1].topk(3).indices.tolist())
    torch.manual_seed(0)
    for _ in range(20):
        out = m.generate(idx, max_new_tokens=1, top_k=3, temperature=1.0)
        assert out[0, -1].item() in allowed


def test_temperature_zero_ish_is_greedy(cfg):
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    g = m.generate(idx, 5, greedy=True)
    t = m.generate(idx, 5, temperature=1e-6)
    assert torch.equal(g, t)


def test_gelu_variant_still_works(cfg):
    m = GPT(replace(cfg, ffn_type="gelu"))
    logits, loss = m(torch.randint(0, cfg.vocab_size, (2, 8)),
                     torch.randint(0, cfg.vocab_size, (2, 8)))
    assert logits.shape == (2, 8, cfg.vocab_size) and torch.isfinite(loss)


def test_param_count_matches_hand_calculation():
    cfg = GPTConfig(vocab_size=1000, context_length=16, d_model=32, n_layers=2,
                    n_heads=4, d_head=8, d_ff=88, tie_embeddings=True, bias=False)
    m = GPT(cfg)
    per_block = 4 * cfg.d_model * cfg.d_model + 3 * cfg.d_model * cfg.d_ff + 2 * cfg.d_model
    expected = cfg.vocab_size * cfg.d_model + cfg.n_layers * per_block + cfg.d_model
    assert m.num_params() == expected
