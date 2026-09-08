"""Configuration objects for the model and for training.

Everything that defines the architecture lives in `GPTConfig`; everything that
defines a training run lives in `TrainConfig`.  Both are plain dataclasses so
they can be serialised into a checkpoint and reconstructed exactly.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class GPTConfig:
    # --- vocabulary / sequence -------------------------------------------------
    vocab_size: int = 50257          # GPT-2 BPE vocabulary (tiktoken "gpt2")
    context_length: int = 512        # maximum T the model can attend over

    # --- residual stream -------------------------------------------------------
    d_model: int = 384               # width of the residual stream

    # --- depth / attention -----------------------------------------------------
    n_layers: int = 6
    n_heads: int = 6
    d_head: int = 64                 # n_heads * d_head == d_model in this config

    # --- feed-forward ----------------------------------------------------------
    ffn_type: str = "swiglu"         # "swiglu" (default) or "gelu" (phase-1 FFN)
    d_ff: int = 1024                 # SwiGLU hidden size. 8/3 * d_model = 1024,
                                     # which matches the parameter count of a
                                     # classic 4*d_model two-matrix GELU FFN.

    # --- normalisation / position ----------------------------------------------
    norm_eps: float = 1e-6           # epsilon inside RMSNorm
    rope_theta: float = 10000.0      # RoPE base frequency

    # --- regularisation --------------------------------------------------------
    dropout: float = 0.0             # 0.0 is right when we are data-rich

    # --- misc ------------------------------------------------------------------
    bias: bool = False               # biases in Linear layers (modern LLMs: off)
    tie_embeddings: bool = True      # share nn.Embedding weight with the LM head
    init_std: float = 0.02

    def __post_init__(self):
        assert self.d_head % 2 == 0, "RoPE rotates pairs of dims, so d_head must be even"
        assert self.n_heads * self.d_head == self.d_model, (
            f"n_heads * d_head ({self.n_heads} * {self.d_head} = "
            f"{self.n_heads * self.d_head}) must equal d_model ({self.d_model}). "
            "Keeping them equal makes the attention block shape-preserving."
        )
        assert self.ffn_type in ("swiglu", "gelu")

    # Number of parameters is computed on the real module (see model.GPT.num_params),
    # this is only for a startup estimate before the model is built.
    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    # --- data ------------------------------------------------------------------
    data_dir: str = "data/tinystories"
    dataset: str = "tinystories"

    # --- batching --------------------------------------------------------------
    micro_batch_size: int = 24       # sequences per forward pass (benchmarked sweet spot)
    grad_accum_steps: int = 2        # micro-batches per optimizer step
    # effective_batch_size    = micro_batch_size * grad_accum_steps
    # tokens_per_optim_step   = effective_batch_size * context_length

    # --- schedule --------------------------------------------------------------
    max_steps: int = 40000           # optimizer steps
    warmup_steps: int = 400
    lr: float = 1e-3                 # peak learning rate
    min_lr_ratio: float = 0.1        # cosine decays to min_lr_ratio * lr
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- precision -------------------------------------------------------------
    dtype: str = "bfloat16"          # "bfloat16" | "float16" | "float32"
    compile: bool = False            # torch.compile the model

    # --- evaluation / logging --------------------------------------------------
    eval_interval: int = 1000
    eval_batches: int = 40
    log_interval: int = 50
    sample_interval: int = 2500
    sample_prompt: str = "Once upon a time"
    sample_tokens: int = 120

    # --- checkpointing ---------------------------------------------------------
    out_dir: str = "runs/tinystories"
    ckpt_interval: int = 5000
    keep_last_k: int = 2
    resume: Optional[str] = None     # path to a checkpoint, or "auto"

    # --- misc ------------------------------------------------------------------
    seed: int = 1337
    device: str = "cuda"
    diagnostics: bool = False        # per-layer activation/gradient RMS + attn entropy

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps

    def tokens_per_optim_step(self, context_length: int) -> int:
        return self.effective_batch_size * context_length

    def to_dict(self):
        return asdict(self)


# A couple of ready-made model presets.
PRESETS = {
    # ~30M parameters; the configuration every number in the README came from.
    "small": GPTConfig(),
    # A tiny model used by the tests / tiny-batch overfit check.
    "debug": GPTConfig(
        vocab_size=256, context_length=64, d_model=64,
        n_layers=2, n_heads=4, d_head=16, d_ff=176,
    ),
    # Bigger, if VRAM permits.
    "medium": GPTConfig(
        context_length=512, d_model=640, n_layers=10, n_heads=10, d_head=64, d_ff=1728,
    ),
}
