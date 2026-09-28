"""Model architecture specs, and building models from them without weights.

Every published spec below is transcribed from that model's ``config.json``. Two properties
follow, and they are what makes this package useful without a GPU:

* Parameter counts are **exact**, not estimated, because they are determined entirely by the
  architecture.
* Building a model on PyTorch's ``meta`` device allocates **no memory at all** -- only
  shapes -- so the arithmetic for a 70B model costs the same as for a 1M one.

The counts this produces are checkable against the published figures. Llama-2-7B comes out
at 6,738,415,616 parameters, which is the number in the paper.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    PreTrainedModel,
)

from ftlab.types import ModelSpec

#: Attention projections. Present in every decoder layer of both architectures.
ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
#: The gated-MLP projections.
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")

SPECS: dict[str, ModelSpec] = {
    # --- published architectures -------------------------------------------------
    "llama-2-7b": ModelSpec(
        name="llama-2-7b",
        architecture="llama",
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_position_embeddings=4096,
    ),
    "llama-2-13b": ModelSpec(
        name="llama-2-13b",
        architecture="llama",
        vocab_size=32000,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=40,
        max_position_embeddings=4096,
    ),
    "llama-3-8b": ModelSpec(
        name="llama-3-8b",
        architecture="llama",
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=8192,
    ),
    "mistral-7b": ModelSpec(
        name="mistral-7b",
        architecture="mistral",
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=32768,
    ),
    # --- tiny architectures, for real runs that finish in seconds ----------------
    # Not scaled-down versions of anything published: they exist so the training,
    # evaluation and merge code paths are exercised for real, on CPU, with no download.
    "tiny-llama": ModelSpec(
        name="tiny-llama",
        architecture="llama",
        vocab_size=259,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        published=False,
    ),
    "tiny-mistral": ModelSpec(
        name="tiny-mistral",
        architecture="mistral",
        vocab_size=259,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        published=False,
    ),
    "nano-llama": ModelSpec(
        name="nano-llama",
        architecture="llama",
        vocab_size=259,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        published=False,
    ),
}


class SpecError(KeyError):
    """Raised for an unknown model spec."""


def get_spec(name: str) -> ModelSpec:
    try:
        return SPECS[name]
    except KeyError:
        known = ", ".join(sorted(SPECS))
        raise SpecError(f"unknown model spec {name!r}; known specs: {known}") from None


def published_specs() -> list[ModelSpec]:
    return [spec for spec in SPECS.values() if spec.published]


def tiny_specs() -> list[ModelSpec]:
    return [spec for spec in SPECS.values() if not spec.published]


def to_hf_config(
    spec: ModelSpec,
    **overrides: Any,  # noqa: ANN401 - passed straight to an untyped transformers config
) -> LlamaConfig | MistralConfig:
    """Build the transformers config for a spec."""
    kwargs: dict[str, Any] = {
        "vocab_size": spec.vocab_size,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_hidden_layers": spec.num_hidden_layers,
        "num_attention_heads": spec.num_attention_heads,
        "num_key_value_heads": spec.num_key_value_heads,
        "max_position_embeddings": spec.max_position_embeddings,
        "tie_word_embeddings": spec.tie_word_embeddings,
        **overrides,
    }
    if spec.architecture == "llama":
        return LlamaConfig(**kwargs)
    if spec.architecture == "mistral":
        return MistralConfig(**kwargs)
    raise SpecError(f"no config class for architecture {spec.architecture!r}")


def init_seed(spec: ModelSpec) -> int:
    """The seed a spec's weights are initialised from.

    Derived from the name with blake2b rather than ``hash()``, which is salted per
    interpreter and would give a different base model on every process.
    """
    digest = hashlib.blake2b(spec.name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def build_model(
    spec: ModelSpec,
    *,
    on_meta: bool = False,
    seed: int | None = None,
    **overrides: Any,  # noqa: ANN401 - passed straight to an untyped transformers config
) -> PreTrainedModel:
    """Instantiate a model from a spec, with initialisation determined by the spec.

    None of these specs has published weights -- they are architectures, built fresh. An
    adapter, though, is only meaningful against one specific base: if two builds of the same
    spec differed, ``ftlab evaluate --adapter`` would load the adapter onto weights it was
    never trained against and report the number as though it meant something. Seeding the
    initialisation from the spec name is what makes the name a checkpoint. Pass ``seed`` to
    override it.

    The global RNG state is restored afterwards, so building a model does not silently
    reposition a caller's own seeded stream.

    With ``on_meta=True`` the model has shapes but no storage, which is what makes counting
    the parameters of a 7B architecture free. A meta model cannot be run; it exists to be
    measured.
    """
    config = to_hf_config(spec, **overrides)
    model_class = LlamaForCausalLM if spec.architecture == "llama" else MistralForCausalLM

    if on_meta:
        with torch.device("meta"):
            return model_class(config)

    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(init_seed(spec) if seed is None else seed)
        return model_class(config)
    finally:
        torch.random.set_rng_state(state)


def count_parameters(model: PreTrainedModel) -> int:
    """Total parameters, including those on the meta device."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def linear_module_names(spec: ModelSpec, target: str) -> list[str]:
    """The projection names a LoRA target selects, for this architecture."""
    if target == "attention":
        return list(ATTENTION_MODULES)
    if target == "mlp":
        return list(MLP_MODULES)
    if target == "all-linear":
        return [*ATTENTION_MODULES, *MLP_MODULES]
    raise SpecError(f"unknown LoRA target {target!r}; expected attention, mlp or all-linear")
