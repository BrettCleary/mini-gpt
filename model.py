"""A decoder-only transformer (GPT) written from primitives.

Nothing in this file comes from a transformer library: no `nn.Transformer`,
no `nn.MultiheadAttention`, no fused attention kernel.  Attention builds the
full [B, H, T, T] score matrix explicitly so that the cost of the operation is
visible in both the code and the profiler.

Shape notation used throughout:
    B  = batch size
    T  = sequence length (number of tokens)
    C  = d_model, the width of the residual stream
    H  = n_heads
    Dh = d_head        (here H * Dh == C)
    V  = vocab_size
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

        RMS(x) = sqrt(mean(x_i^2) + eps)
        y      = x / RMS(x) * g

    Compared with LayerNorm it drops the mean subtraction and the bias, which
    is both cheaper and empirically just as good.  The learned scale `g` lets
    each channel of the residual stream keep its own gain.

    Input :  [..., C]
    Output:  [..., C]   (same shape)
    Params:  C          (the scale vector)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in fp32 even under bf16/fp16 autocast: the sum of squares
        # is the one place in this layer where low precision actually hurts.
        dtype = x.dtype
        x32 = x.float()
        rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # [..., 1]
        return (x32 / rms).to(dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary position embeddings (RoPE)
# ---------------------------------------------------------------------------
def build_rope_cache(
    context_length: int, d_head: int, theta: float = 10000.0, device=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for every position up to `context_length`.

    Dimension pair i (i = 0 .. Dh/2-1) is rotated at angular frequency
        inv_freq[i] = theta ** (-2i / Dh)
    so pair 0 spins once per token and the last pair spins extremely slowly.
    A query at position m and a key at position n end up with a dot product
    that depends only on (m - n): RoPE encodes *relative* position.

    Returns cos, sin each of shape [T, Dh/2] (fp32).
    """
    half = d_head // 2
    # inv_freq: [Dh/2]
    inv_freq = theta ** (-torch.arange(0, half, dtype=torch.float32, device=device) / half)
    # positions: [T]
    pos = torch.arange(context_length, dtype=torch.float32, device=device)
    # angles: [T, Dh/2]
    angles = torch.outer(pos, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs of channels of `x` by the position-dependent angle.

    x   : [B, H, T, Dh]
    cos : [T, Dh/2]
    sin : [T, Dh/2]
    out : [B, H, T, Dh]

    Each pair (x_{2i}, x_{2i+1}) is treated as a 2-D vector and rotated:

        [ x'_{2i}   ]   [ cos  -sin ] [ x_{2i}   ]
        [ x'_{2i+1} ] = [ sin   cos ] [ x_{2i+1} ]

    Rotation is orthogonal, so ||x'|| == ||x|| exactly (up to fp error).
    """
    assert x.ndim == 4, f"expected [B, H, T, Dh], got {tuple(x.shape)}"
    B, H, T, Dh = x.shape
    assert cos.shape == (T, Dh // 2), f"cos {tuple(cos.shape)} != {(T, Dh // 2)}"
    assert sin.shape == (T, Dh // 2)

    # Broadcast the tables over batch and head: [1, 1, T, Dh/2]
    cos = cos.view(1, 1, T, Dh // 2).to(x.dtype)
    sin = sin.view(1, 1, T, Dh // 2).to(x.dtype)

    x_even = x[..., 0::2]                       # [B, H, T, Dh/2]
    x_odd = x[..., 1::2]                        # [B, H, T, Dh/2]

    out_even = x_even * cos - x_odd * sin       # [B, H, T, Dh/2]
    out_odd = x_even * sin + x_odd * cos        # [B, H, T, Dh/2]

    # Interleave the pairs back together: [B, H, T, Dh/2, 2] -> [B, H, T, Dh]
    out = torch.stack((out_even, out_odd), dim=-1).flatten(-2)
    return out


# ---------------------------------------------------------------------------
# Causal self-attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an explicit T x T score matrix.

    Params: 4 * C * (H*Dh)  (q, k, v, out projections; no biases by default)
    Compute: O(B * H * T^2 * Dh) FLOPs and O(B * H * T^2) memory for the scores.
    That quadratic memory term is exactly what FlashAttention removes.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model

        # Separate projections rather than one fused matrix: slower, but the
        # roles of Q, K and V stay legible.
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular mask: position i may attend to j <= i.
        # [1, 1, T_max, T_max] so it broadcasts over batch and heads.
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, *mask.shape), persistent=False)

        # Diagnostics hook: when enabled, the last attention probabilities are
        # stashed (detached) so analysis code can compute e.g. attention entropy.
        self.store_attn = False
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        B, T, C = x.shape
        H, Dh = self.n_heads, self.d_head
        assert C == self.d_model, f"expected d_model={self.d_model}, got {C}"
        assert T <= self.causal_mask.shape[-1], (
            f"sequence length {T} exceeds context_length {self.causal_mask.shape[-1]}"
        )

        q = self.q_proj(x)                              # q: [B, T, H*Dh]
        k = self.k_proj(x)                              # k: [B, T, H*Dh]
        v = self.v_proj(x)                              # v: [B, T, H*Dh]

        # Split the channel axis into heads and move the head axis next to batch,
        # so every head is an independent [T, Dh] attention problem.
        q = q.view(B, T, H, Dh).transpose(1, 2)         # q: [B, H, T, Dh]
        k = k.view(B, T, H, Dh).transpose(1, 2)         # k: [B, H, T, Dh]
        v = v.view(B, T, H, Dh).transpose(1, 2)         # v: [B, H, T, Dh]

        # RoPE goes on Q and K only: it must change *where* tokens look, not
        # *what* they carry.  V is the payload and stays untouched.
        q = apply_rope(q, cos, sin)                     # q: [B, H, T, Dh]
        k = apply_rope(k, cos, sin)                     # k: [B, H, T, Dh]

        # scores[b,h,i,j] = <q_i, k_j> / sqrt(Dh)
        # [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        # Causality: kill every j > i before the softmax.
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))

        # Softmax in fp32; the exponential is the numerically delicate step.
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)   # attn: [B, H, T, T]
        if self.store_attn:
            self.last_attn = attn.detach()
        attn = self.attn_dropout(attn)

        # Weighted sum of values: [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
        y = torch.matmul(attn, v)

        # Merge heads back into the residual width:
        # [B, H, T, Dh] -> [B, T, H, Dh] -> [B, T, H*Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, H * Dh)

        y = self.out_proj(y)                            # y: [B, T, C]
        return self.resid_dropout(y)


# ---------------------------------------------------------------------------
# Feed-forward networks
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """Gated feed-forward network.

        SwiGLU(x) = ( SiLU(x W_gate) * (x W_up) ) W_down

    The elementwise product makes one branch a *gate* on the other, so the
    layer can suppress or pass features conditionally instead of applying one
    fixed pointwise nonlinearity.  Three matrices instead of two is why d_ff is
    set to ~8/3*d_model rather than 4*d_model: it keeps the parameter count of
    the block roughly the same.

    The hidden dimension is larger than d_model because this is where the model
    does its per-token computation; attention only *moves* information between
    positions, the FFN is the only place a token transforms its own content, and
    it needs room to do that in a higher-dimensional space before projecting
    back down to the narrow residual stream.

    Input : [B, T, C]  ->  Output: [B, T, C]
    Params: 3 * C * d_ff
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        gate = F.silu(self.gate_proj(x))     # gate: [B, T, d_ff]
        up = self.up_proj(x)                 # up:   [B, T, d_ff]
        h = gate * up                        # h:    [B, T, d_ff]
        return self.dropout(self.down_proj(h))   # [B, T, C]


class GELUFFN(nn.Module):
    """The classic two-matrix FFN, kept as a phase-1 baseline / ablation.

    Input : [B, T, C]  ->  Output: [B, T, C]
    Params: 2 * C * d_ff_effective, with d_ff_effective = 4*C by convention.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = 4 * cfg.d_model
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x))))


def build_ffn(cfg: GPTConfig) -> nn.Module:
    return SwiGLU(cfg) if cfg.ffn_type == "swiglu" else GELUFFN(cfg)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Pre-norm block:

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    Pre-norm (normalise the *input* of each sublayer, not the output of the
    residual add) leaves an unnormalised identity path from the embeddings all
    the way to the LM head, which is what makes deep stacks trainable without
    a warmup-heavy schedule.

    Invariant: the residual stream is [B, T, C] at every point in this block.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = build_ffn(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] in, [B, T, C] out
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    """tokens -> embeddings -> N x TransformerBlock -> RMSNorm -> LM head -> logits"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # No learned positional embedding table: position enters only through RoPE
        # inside attention.
        cos, sin = build_rope_cache(cfg.context_length, cfg.d_head, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        if cfg.tie_embeddings:
            # One matrix used twice: as a lookup table on the way in and as a
            # similarity score against every vocabulary item on the way out.
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Scale down the projections that write into the residual stream, so the
        # variance of the stream does not grow linearly with depth.
        residual_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.out_proj.weight, mean=0.0,
                                  std=cfg.init_std * residual_scale)
            torch.nn.init.normal_(block.ffn.down_proj.weight, mean=0.0,
                                  std=cfg.init_std * residual_scale)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    # -- introspection ------------------------------------------------------
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def param_breakdown(self) -> dict:
        out = {}
        for name, p in self.named_parameters():
            top = name.split(".")[0]
            if top == "blocks":
                top = "blocks." + ".".join(name.split(".")[2:4])
            out[top] = out.get(top, 0) + p.numel()
        return out

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        last_token_only: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        idx     : [B, T]  int64 token ids
        targets : [B, T]  int64 next-token ids (already shifted by the data loader)
        returns : (logits, loss)
                  logits [B, T, V]   (or [B, 1, V] when last_token_only)
                  loss   scalar or None
        """
        B, T = idx.shape
        assert T <= self.cfg.context_length, (
            f"sequence length {T} > context_length {self.cfg.context_length}"
        )

        x = self.token_embedding(idx)                 # x: [B, T, C]
        x = self.embed_dropout(x)

        cos = self.rope_cos[:T]                       # cos: [T, Dh/2]
        sin = self.rope_sin[:T]                       # sin: [T, Dh/2]

        for block in self.blocks:
            x = block(x, cos, sin)                    # x: [B, T, C] throughout

        x = self.final_norm(x)                        # x: [B, T, C]

        if last_token_only and targets is None:
            # Generation only needs the distribution after the final token.
            logits = self.lm_head(x[:, -1:, :])       # logits: [B, 1, V]
            return logits, None

        logits = self.lm_head(x)                      # logits: [B, T, V]

        loss = None
        if targets is not None:
            # Flatten every (batch, position) pair into one big classification
            # problem: [B*T, V] predictions against [B*T] labels.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
            )
        return logits, loss

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        greedy: bool = False,
        eos_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling, recomputing the whole context every step.

        idx: [B, T0] -> returns [B, T0 + max_new_tokens] (or shorter if eos hit)

        No KV cache on purpose: step t redoes the work of steps 0..t-1, which is
        O(T^2) per token and O(T^3) for the whole sequence.  Feeling that cost is
        the motivation for caching later.
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            # Never feed more than context_length tokens; keep the most recent.
            idx_cond = idx[:, -self.cfg.context_length:]
            logits, _ = self(idx_cond, last_token_only=True)   # [B, 1, V]
            logits = logits[:, -1, :].float()                  # [B, V]

            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)   # [B, 1]
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth = logits.topk(k, dim=-1).values[:, -1:]    # [B, 1]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)              # [B, V]
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)

            idx = torch.cat((idx, next_token), dim=1)              # [B, T+1]
            if eos_id is not None and (next_token == eos_id).all():
                break
        if was_training:
            self.train()
        return idx
