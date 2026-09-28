"""Applying LoRA, and verifying the properties everyone assumes.

LoRA replaces a frozen weight ``W`` with ``W + (alpha/r) * B @ A``, where ``A`` is
initialised randomly and ``B`` is initialised to **zero**. Three consequences follow, and
each is a property that can be checked rather than trusted:

1. **At initialisation the adapter is a mathematical no-op.** ``B = 0`` means ``B @ A = 0``,
   so a freshly adapted model produces bit-identical outputs to the base. If it does not, the
   adapter was attached wrong, and every subsequent number is measuring the wrong model.
2. **Only adapter parameters receive gradients.** If a base weight has a gradient, the run is
   not parameter-efficient regardless of what the trainable-parameter count says.
3. **Merging is exact.** ``merge_and_unload`` should produce a plain model whose weights equal
   ``W + (alpha/r) * B @ A`` and whose outputs match the adapted model to floating-point
   tolerance. A merge that drifts means deployed weights differ from the ones evaluated.

:mod:`tests.test_lora` asserts all three against real models. They are cheap to check and
expensive to get wrong.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel

from ftlab.specs import get_spec, linear_module_names
from ftlab.types import LoraSpec, ModelSpec


def to_peft_config(spec: ModelSpec, lora: LoraSpec) -> LoraConfig:
    """Translate a :class:`LoraSpec` into a ``peft`` config for this architecture."""
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=linear_module_names(spec, lora.target),
        bias=lora.bias,
        task_type="CAUSAL_LM",
    )


def apply_lora(
    model: PreTrainedModel, model_spec: str | ModelSpec, lora: LoraSpec | None = None
) -> PeftModel:
    """Attach LoRA adapters, freezing the base."""
    spec = get_spec(model_spec) if isinstance(model_spec, str) else model_spec
    return get_peft_model(model, to_peft_config(spec, lora or LoraSpec()))


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def base_parameters_are_frozen(model: torch.nn.Module) -> bool:
    """Whether every trainable parameter belongs to an adapter.

    The check is on names rather than counts: a count can look right while a base weight is
    also trainable and an adapter is not.
    """
    return all("lora_" in name for name in trainable_parameter_names(model))


def adapter_is_zero_initialised(model: torch.nn.Module) -> bool:
    """Whether every ``lora_B`` is still zero, making the adapter an identity."""
    return all(
        torch.all(parameter == 0).item()
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    )


def expected_delta(lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float) -> torch.Tensor:
    """The weight update LoRA should produce: ``scaling * B @ A``.

    Written out so the merge test compares against the definition rather than against
    whatever the library happened to compute.
    """
    return scaling * (lora_b @ lora_a)


def merge_adapter(model: PeftModel) -> PreTrainedModel:
    """Fold the adapters into the base weights and return a plain model.

    This is what makes LoRA free at inference time: after merging there is no adapter, no
    extra matmul and no runtime dependency on ``peft``. The parameter count returns to the
    base model's exactly.
    """
    # `merge_and_unload` is untyped in peft; the cast records what it returns
    # rather than letting Any leak into every caller.
    merged: PreTrainedModel = model.merge_and_unload()
    return merged


@torch.no_grad()
def max_logit_difference(
    left: torch.nn.Module, right: torch.nn.Module, input_ids: torch.Tensor
) -> float:
    """Largest absolute logit difference between two models on the same input.

    The measurement behind properties 1 and 3. Exactly zero at initialisation; a small
    floating-point residual after a merge.
    """
    left.eval()
    right.eval()
    left_logits = left(input_ids=input_ids).logits
    right_logits = right(input_ids=input_ids).logits
    return float((left_logits - right_logits).abs().max().item())
