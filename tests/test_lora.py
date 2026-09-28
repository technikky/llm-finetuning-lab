"""The LoRA invariants.

Three properties that everyone assumes and nobody checks. Each is cheap to verify and
expensive to have wrong: if the adapter is not an identity at initialisation, every
"before" measurement is of the wrong model; if the merge drifts, the deployed weights are
not the ones that were evaluated.
"""

from __future__ import annotations

import copy

import pytest
import torch

from ftlab.lora import (
    adapter_is_zero_initialised,
    apply_lora,
    base_parameters_are_frozen,
    expected_delta,
    max_logit_difference,
    merge_adapter,
    to_peft_config,
    trainable_parameter_names,
)
from ftlab.specs import build_model, count_parameters, get_spec
from ftlab.types import LoraSpec, ModelSpec


def _perturb_adapter(model: torch.nn.Module, std: float = 0.02) -> None:
    """Give ``lora_B`` non-zero values, so the adapter actually changes the output."""
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=std)


# --- invariant 1: identity at initialisation --------------------------------------


def test_lora_b_is_zero_initialised(nano_spec: ModelSpec, nano_model, small_lora: LoraSpec) -> None:
    assert adapter_is_zero_initialised(apply_lora(nano_model, nano_spec, small_lora))


def test_a_freshly_adapted_model_is_bit_identical_to_the_base(
    nano_spec: ModelSpec, nano_model, small_lora: LoraSpec, token_ids: torch.Tensor
) -> None:
    """Exactly zero, not approximately.

    B = 0 means B @ A = 0, so the adapted forward pass adds a literal zero. A non-zero
    difference here means the adapter was attached to the wrong thing, and every subsequent
    number would be measuring a different model than intended.
    """
    reference = copy.deepcopy(nano_model)
    adapted = apply_lora(nano_model, nano_spec, small_lora)
    assert max_logit_difference(reference, adapted, token_ids) == 0.0


def test_only_adapter_parameters_are_trainable(
    nano_spec: ModelSpec, nano_model, small_lora: LoraSpec
) -> None:
    adapted = apply_lora(nano_model, nano_spec, small_lora)
    names = trainable_parameter_names(adapted)

    assert names, "no parameters are trainable at all"
    assert base_parameters_are_frozen(adapted)
    assert all("lora_" in name for name in names)


def test_a_base_weight_gets_no_gradient(
    nano_spec: ModelSpec, nano_model, small_lora: LoraSpec, token_ids: torch.Tensor
) -> None:
    """The check that a trainable-parameter count cannot make: run a real backward pass."""
    adapted = apply_lora(nano_model, nano_spec, small_lora)
    adapted(input_ids=token_ids, labels=token_ids).loss.backward()

    for name, parameter in adapted.named_parameters():
        if "lora_" in name:
            continue
        assert parameter.grad is None, f"base parameter {name} received a gradient"


# --- invariant 2: the merge is exact ----------------------------------------------


def test_merged_weight_equals_the_definition(
    nano_spec: ModelSpec, nano_model, small_lora: LoraSpec
) -> None:
    """Compared against ``W + (alpha/r) * B @ A``, not against whatever peft computed."""
    adapted = apply_lora(nano_model, nano_spec, small_lora)
    _perturb_adapter(adapted)

    projection = adapted.base_model.model.model.layers[0].self_attn.q_proj
    before = projection.base_layer.weight.detach().clone()
    delta = expected_delta(
        projection.lora_A["default"].weight.detach(),
        projection.lora_B["default"].weight.detach(),
        small_lora.scaling,
    )

    merged = merge_adapter(adapted)
    after = merged.model.layers[0].self_attn.q_proj.weight.detach()

    assert torch.allclose(after, before + delta, atol=1e-6)


def test_merging_preserves_the_output(
    nano_spec: ModelSpec, nano_model, small_lora: LoraSpec, token_ids: torch.Tensor
) -> None:
    adapted = apply_lora(nano_model, nano_spec, small_lora)
    _perturb_adapter(adapted)

    with torch.no_grad():
        before = adapted(input_ids=token_ids).logits.clone()
    merged = merge_adapter(adapted)
    with torch.no_grad():
        after = merged(input_ids=token_ids).logits

    # Floating point, not bit-exact: the merge changes the order of operations.
    assert torch.allclose(before, after, atol=1e-5)


def test_merging_returns_the_parameter_count_to_the_base(
    nano_spec: ModelSpec, nano_model, small_lora: LoraSpec
) -> None:
    """Why LoRA is free at inference: after merging there is no adapter left."""
    base_count = count_parameters(nano_model)
    adapted = apply_lora(copy.deepcopy(nano_model), nano_spec, small_lora)

    assert count_parameters(adapted) > base_count
    assert count_parameters(merge_adapter(adapted)) == base_count


def test_merging_a_zero_adapter_is_a_no_op(
    nano_spec: ModelSpec, nano_model, small_lora: LoraSpec, token_ids: torch.Tensor
) -> None:
    reference = copy.deepcopy(nano_model)
    merged = merge_adapter(apply_lora(nano_model, nano_spec, small_lora))
    assert max_logit_difference(reference, merged, token_ids) == pytest.approx(0.0, abs=1e-6)


# --- scaling ---------------------------------------------------------------------


def test_alpha_scales_the_delta(nano_spec: ModelSpec, small_lora: LoraSpec) -> None:
    """Doubling alpha at fixed r doubles the weight update."""
    lora_a = torch.randn(small_lora.r, 8)
    lora_b = torch.randn(8, small_lora.r)

    single = expected_delta(lora_a, lora_b, 1.0)
    double = expected_delta(lora_a, lora_b, 2.0)
    assert torch.allclose(double, 2 * single)


def test_the_peft_config_carries_the_spec(nano_spec: ModelSpec) -> None:
    lora = LoraSpec(r=8, alpha=16, target="all-linear", dropout=0.05)
    config = to_peft_config(nano_spec, lora)

    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == pytest.approx(0.05)
    assert set(config.target_modules) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }


def test_mlp_only_targeting_skips_attention(nano_spec: ModelSpec) -> None:
    config = to_peft_config(nano_spec, LoraSpec(r=4, alpha=8, target="mlp"))
    assert set(config.target_modules) == {"gate_proj", "up_proj", "down_proj"}


def test_a_mistral_model_adapts_too(small_lora: LoraSpec, token_ids: torch.Tensor) -> None:
    """Both architectures, so the module names are not silently Llama-specific."""
    spec = get_spec("tiny-mistral")
    model = build_model(spec)
    reference = copy.deepcopy(model)
    adapted = apply_lora(model, spec, small_lora)

    assert base_parameters_are_frozen(adapted)
    assert max_logit_difference(reference, adapted, token_ids) == 0.0
