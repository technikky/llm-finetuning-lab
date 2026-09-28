"""Parameter and memory arithmetic.

This is the part of fine-tuning you can know exactly before renting anything. Given an
architecture and a LoRA setting, the number of trainable parameters is determined, and so is
the footprint of weights, gradients and optimizer state. The only unknown is activation
memory, which is excluded rather than guessed at -- see :class:`MemoryBudget`.

The adapter count is computed in closed form here **and** cross-checked against what ``peft``
actually builds, in ``tests/test_budget.py``. A closed form nobody checks against the library
is a formula that was right once.

For a LoRA adapter on ``Linear(in_features, out_features)`` at rank ``r``:

    A has shape (r, in_features), B has shape (out_features, r)
    parameters = r * (in_features + out_features)

The per-projection shapes for both architectures, where ``kv_dim = num_key_value_heads *
head_dim``:

    q_proj     hidden -> hidden            2 * r * hidden
    k_proj     hidden -> kv_dim            r * (hidden + kv_dim)
    v_proj     hidden -> kv_dim            r * (hidden + kv_dim)
    o_proj     hidden -> hidden            2 * r * hidden
    gate_proj  hidden -> intermediate      r * (hidden + intermediate)
    up_proj    hidden -> intermediate      r * (hidden + intermediate)
    down_proj  intermediate -> hidden      r * (intermediate + hidden)

Grouped-query attention is why k and v are cheaper than q and o on Mistral and Llama-3: their
output dimension is ``kv_dim``, not ``hidden``.
"""

from __future__ import annotations

from typing import Literal

from ftlab.specs import get_spec, linear_module_names
from ftlab.types import (
    BYTES_PER_PARAM,
    LoraSpec,
    MemoryBudget,
    ModelSpec,
    ParameterBudget,
    Precision,
    Quantization,
)

OptimizerName = Literal["adamw", "adamw_8bit", "sgd", "sgd_momentum"]

#: Optimizer state, as (number of state tensors per parameter, bytes per state element).
#: AdamW keeps two moments and keeps them in fp32 even for a bf16 model, because fp16
#: moments lose the small updates that are the point of having them.
OPTIMIZER_STATE: dict[str, tuple[int, int]] = {
    "adamw": (2, 4),
    "adamw_8bit": (2, 1),
    "sgd": (0, 0),
    "sgd_momentum": (1, 4),
}

GIB = 1024**3


def _projection_parameters(spec: ModelSpec, module: str, r: int) -> int:
    """Adapter parameters for one projection in one layer."""
    hidden = spec.hidden_size
    intermediate = spec.intermediate_size
    kv_dim = spec.num_key_value_heads * spec.head_dim

    shapes: dict[str, tuple[int, int]] = {
        "q_proj": (hidden, hidden),
        "k_proj": (hidden, kv_dim),
        "v_proj": (hidden, kv_dim),
        "o_proj": (hidden, hidden),
        "gate_proj": (hidden, intermediate),
        "up_proj": (hidden, intermediate),
        "down_proj": (intermediate, hidden),
    }
    if module not in shapes:
        raise KeyError(f"unknown projection {module!r}")
    in_features, out_features = shapes[module]
    return r * (in_features + out_features)


def adapter_parameters(spec: ModelSpec, lora: LoraSpec) -> tuple[int, list[str]]:
    """Total adapter parameters, and the projections they attach to."""
    modules = linear_module_names(spec, lora.target)
    per_layer = sum(_projection_parameters(spec, module, lora.r) for module in modules)
    return per_layer * spec.num_hidden_layers, modules


def base_parameters(spec: ModelSpec) -> int:
    """Base model parameters, in closed form.

    Derived rather than measured so this works for an architecture too large to instantiate
    even on the meta device. Cross-checked against a real build in the tests.

    Per decoder layer: q/k/v/o projections, three MLP projections, two RMSNorm weights.
    Plus the embedding, the final norm, and the language-model head.
    """
    hidden = spec.hidden_size
    intermediate = spec.intermediate_size
    kv_dim = spec.num_key_value_heads * spec.head_dim

    attention = hidden * hidden * 2 + hidden * kv_dim * 2  # q, o, then k, v
    mlp = hidden * intermediate * 3  # gate, up, down
    norms = hidden * 2  # input_layernorm, post_attention_layernorm
    per_layer = attention + mlp + norms

    embedding = spec.vocab_size * hidden
    head = 0 if spec.tie_word_embeddings else spec.vocab_size * hidden
    final_norm = hidden

    return per_layer * spec.num_hidden_layers + embedding + head + final_norm


def parameter_budget(model: str | ModelSpec, lora: LoraSpec | None = None) -> ParameterBudget:
    """Exact parameter counts for a model and LoRA setting."""
    spec = get_spec(model) if isinstance(model, str) else model
    settings = lora or LoraSpec()
    adapter, modules = adapter_parameters(spec, settings)

    return ParameterBudget(
        model=spec.name,
        base_parameters=base_parameters(spec),
        adapter_parameters=adapter,
        target_modules=modules,
        n_adapted_layers=spec.num_hidden_layers,
        lora=settings,
    )


def memory_budget(
    model: str | ModelSpec,
    lora: LoraSpec | None = None,
    *,
    precision: Precision = "bf16",
    quantization: Quantization = "none",
    optimizer: OptimizerName = "adamw",
    adapter_precision: Precision = "fp32",
) -> MemoryBudget:
    """Footprint of weights, gradients and optimizer state, in GiB.

    The asymmetry that makes LoRA work is visible in the result: only the adapter carries
    gradients and optimizer state, so those terms shrink by three orders of magnitude while
    the base weights stay the same size. QLoRA then attacks the remaining term by storing the
    frozen base in 4 bits.

    Activation memory is **not** included. It depends on sequence length, batch size,
    gradient checkpointing and the attention kernel, and a number here would look
    authoritative while being wrong by a factor of two.
    """
    spec = get_spec(model) if isinstance(model, str) else model
    settings = lora or LoraSpec()
    budget = parameter_budget(spec, settings)

    base_bytes_per_param = (
        BYTES_PER_PARAM[quantization] if quantization != "none" else BYTES_PER_PARAM[precision]
    )
    adapter_bytes_per_param = BYTES_PER_PARAM[adapter_precision]

    if optimizer not in OPTIMIZER_STATE:
        raise KeyError(
            f"unknown optimizer {optimizer!r}; expected one of {', '.join(OPTIMIZER_STATE)}"
        )
    n_states, state_bytes = OPTIMIZER_STATE[optimizer]

    return MemoryBudget(
        model=spec.name,
        base_weights_gb=round(budget.base_parameters * base_bytes_per_param / GIB, 4),
        adapter_weights_gb=round(budget.adapter_parameters * adapter_bytes_per_param / GIB, 4),
        adapter_gradients_gb=round(budget.adapter_parameters * adapter_bytes_per_param / GIB, 4),
        optimizer_state_gb=round(budget.adapter_parameters * n_states * state_bytes / GIB, 4),
        base_precision=quantization if quantization != "none" else precision,
        adapter_precision=adapter_precision,
        quantization=quantization,
        optimizer=optimizer,
    )


def full_finetune_memory_gb(
    model: str | ModelSpec,
    *,
    precision: Precision = "bf16",
    optimizer: OptimizerName = "adamw",
) -> float:
    """The comparison LoRA is measured against: every parameter trainable.

    Weights, plus a gradient per parameter, plus optimizer state per parameter. This is the
    number that makes a 7B model need far more than its 14GB of weights to train, and why a
    24GB card cannot do it without offloading.
    """
    spec = get_spec(model) if isinstance(model, str) else model
    total = base_parameters(spec)
    n_states, state_bytes = OPTIMIZER_STATE[optimizer]

    weights = total * BYTES_PER_PARAM[precision]
    gradients = total * BYTES_PER_PARAM[precision]
    state = total * n_states * state_bytes
    return round((weights + gradients + state) / GIB, 4)


def fits_within(budget: MemoryBudget, vram_gb: float, *, headroom: float = 0.15) -> bool:
    """Whether a budget leaves room for the activations this module refuses to estimate.

    ``headroom`` is the fraction of the card reserved for everything excluded from the
    budget. 15% is a working default for short sequences, not a guarantee.
    """
    if not 0.0 <= headroom < 1.0:
        raise ValueError(f"headroom must be in [0, 1), got {headroom}")
    return budget.total_gb <= vram_gb * (1.0 - headroom)
