"""Core data model.

The distinction this module encodes, and that the whole package is organised around: some
numbers about fine-tuning are **computed** and some are **measured**. Parameter counts and
memory footprints are exact arithmetic over an architecture definition -- no weights, no
GPU, no uncertainty. Loss curves and perplexities are measurements of a particular run on
particular hardware.

Conflating the two is how a fine-tuning claim ends up unfalsifiable, so they have separate
types and the reports label them separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # torch is a hard dependency, but this module is pure data
    import torch

Architecture = Literal["llama", "mistral"]
Precision = Literal["fp32", "fp16", "bf16"]
Quantization = Literal["none", "int8", "nf4"]
TemplateName = Literal["alpaca", "chatml", "llama2-chat", "raw"]

#: Bytes per parameter, by storage precision. nf4 is 4-bit plus a small amount of
#: per-block scaling metadata, which the QLoRA paper puts at roughly 0.5 bits per
#: parameter; 4.5/8 is that total.
BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "nf4": 4.5 / 8.0,
}


class CausalLMOutput(Protocol):
    """The part of a model's output that evaluation and generation read."""

    @property
    def logits(self) -> torch.Tensor: ...


class CausalLM(Protocol):
    """What evaluation and generation actually require of a model.

    A base ``PreTrainedModel`` and a LoRA-wrapped ``PeftModel`` both satisfy this, and neither
    is a subtype of the other -- merging an adapter turns the second back into the first. Typing
    against the concrete classes would mean a cast at every call site and would claim a
    requirement that is not real. The protocol names what is: switch to inference mode, and map
    tensors to logits.
    """

    def eval(self) -> object: ...

    def __call__(self, **kwargs: torch.Tensor) -> CausalLMOutput: ...


class ModelSpec(BaseModel):
    """An architecture definition, sufficient to build a model with no weights.

    Every field here comes from a published model card or config.json. Holding them as data
    means parameter arithmetic for a 70B model costs nothing and needs no download.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    architecture: Architecture
    vocab_size: int = Field(gt=0)
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    max_position_embeddings: int = Field(gt=0)
    tie_word_embeddings: bool = False
    #: True when this spec is a real published model rather than a tiny test architecture.
    published: bool = True

    @model_validator(mode="after")
    def _heads_divide_hidden_size(self) -> ModelSpec:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"{self.name}: hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"{self.name}: num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        return self

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def uses_grouped_query_attention(self) -> bool:
        return self.num_key_value_heads < self.num_attention_heads


class LoraSpec(BaseModel):
    """LoRA hyperparameters.

    ``alpha`` is a scaling numerator, not a second rank: the adapter contributes
    ``B @ A * (alpha / r)``. Doubling r while holding alpha fixed therefore *halves* the
    effective update scale, which is why the convention alpha = 2r exists and why this
    model exposes :attr:`scaling` rather than leaving it implicit.
    """

    model_config = ConfigDict(extra="forbid")

    r: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    #: Which linear projections get adapters. "attention" is q/k/v/o; "all-linear" adds the
    #: MLP projections.
    target: Literal["attention", "mlp", "all-linear"] = "attention"
    bias: Literal["none", "all", "lora_only"] = "none"

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


class DatasetStats(BaseModel):
    """Measured properties of a prepared dataset."""

    model_config = ConfigDict(extra="forbid")

    n_examples: int = Field(ge=0)
    n_train: int = Field(ge=0)
    n_eval: int = Field(ge=0)
    n_dropped_empty: int = 0
    n_dropped_duplicate: int = 0
    n_dropped_too_long: int = 0
    token_length_mean: float = 0.0
    token_length_p50: int = 0
    token_length_p95: int = 0
    token_length_max: int = 0
    total_train_tokens: int = 0
    #: Examples whose text appears in both splits. Should always be zero.
    n_leaked: int = 0
    leaked_ids: list[str] = Field(default_factory=list)


class ParameterBudget(BaseModel):
    """**Computed**, not measured: exact parameter counts for a spec and a LoRA setting."""

    model_config = ConfigDict(extra="forbid")

    model: str
    base_parameters: int = Field(ge=0)
    adapter_parameters: int = Field(ge=0)
    target_modules: list[str] = Field(default_factory=list)
    n_adapted_layers: int = Field(ge=0)
    lora: LoraSpec

    @property
    def trainable_fraction(self) -> float:
        return self.adapter_parameters / self.base_parameters if self.base_parameters else 0.0

    @property
    def total_parameters(self) -> int:
        return self.base_parameters + self.adapter_parameters


class MemoryBudget(BaseModel):
    """**Computed**, not measured: a footprint estimate in gigabytes.

    Weights, gradients and optimizer state are exact given the precision. Activation memory
    is deliberately excluded because it depends on sequence length, batch size, whether
    gradient checkpointing is on, and the attention implementation -- an estimate here would
    look authoritative while being wrong by a factor of two.
    """

    model_config = ConfigDict(extra="forbid")

    model: str
    base_weights_gb: float = Field(ge=0.0)
    adapter_weights_gb: float = Field(ge=0.0)
    adapter_gradients_gb: float = Field(ge=0.0)
    optimizer_state_gb: float = Field(ge=0.0)
    base_precision: str
    adapter_precision: str
    quantization: Quantization
    optimizer: str

    @property
    def total_gb(self) -> float:
        return round(
            self.base_weights_gb
            + self.adapter_weights_gb
            + self.adapter_gradients_gb
            + self.optimizer_state_gb,
            4,
        )

    @property
    def excludes(self) -> str:
        return "activations, KV cache, CUDA context, fragmentation"


class TrainingConfig(BaseModel):
    """Everything that determines a training run's result."""

    model_config = ConfigDict(extra="forbid")

    model: str
    lora: LoraSpec = Field(default_factory=LoraSpec)
    template: TemplateName = "alpaca"
    max_length: int = Field(default=512, gt=0)
    batch_size: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    epochs: int = Field(default=3, gt=0)
    seed: int = 0
    weight_decay: float = Field(default=0.0, ge=0.0)
    warmup_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    quantization: Quantization = "none"
    precision: Precision = "fp32"


class StepMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=0)
    epoch: float = Field(ge=0.0)
    loss: float
    learning_rate: float
    grad_norm: float | None = None


class EvalMetrics(BaseModel):
    """**Measured**: held-out performance of one checkpoint."""

    model_config = ConfigDict(extra="forbid")

    n_examples: int = Field(ge=0)
    loss: float
    perplexity: float
    #: Token-level accuracy on the label positions, which moves earlier than perplexity on
    #: short runs and makes a stalled run obvious sooner.
    token_accuracy: float = Field(ge=0.0, le=1.0)


class TrainingRun(BaseModel):
    """A complete record of one run: config, environment, curve and results."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: str
    ftlab_version: str
    config: TrainingConfig
    config_hash: str
    #: Hardware and library versions, because a loss curve is only reproducible on a
    #: comparable stack.
    environment: dict[str, Any] = Field(default_factory=dict)
    dataset: DatasetStats | None = None
    budget: ParameterBudget | None = None
    steps: list[StepMetric] = Field(default_factory=list)
    eval_before: EvalMetrics | None = None
    eval_after: EvalMetrics | None = None
    wall_seconds: float = 0.0
    notes: str = ""

    @property
    def final_loss(self) -> float | None:
        return self.steps[-1].loss if self.steps else None

    @property
    def initial_loss(self) -> float | None:
        return self.steps[0].loss if self.steps else None

    @property
    def perplexity_delta(self) -> float | None:
        """Held-out perplexity change. Negative is an improvement."""
        if self.eval_before is None or self.eval_after is None:
            return None
        return self.eval_after.perplexity - self.eval_before.perplexity


class Example(BaseModel):
    """One instruction-tuning example."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    input: str = ""
    output: str = Field(min_length=1)
    split: Literal["train", "eval"] = "train"

    @property
    def fingerprint_source(self) -> str:
        """The text used for duplicate and leakage detection."""
        return f"{self.instruction}\u0000{self.input}\u0000{self.output}"
