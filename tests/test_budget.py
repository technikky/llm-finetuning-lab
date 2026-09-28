"""Tests for the computed budgets.

The important ones cross-check the closed forms in :mod:`ftlab.budget` against what
``transformers`` and ``peft`` actually build. A formula nobody checks against the library is a
formula that was right once, and these are the numbers the README publishes.
"""

from __future__ import annotations

import pytest
import torch
from peft import get_peft_model

from ftlab.budget import (
    OPTIMIZER_STATE,
    adapter_parameters,
    base_parameters,
    fits_within,
    full_finetune_memory_gb,
    memory_budget,
    parameter_budget,
)
from ftlab.lora import to_peft_config
from ftlab.specs import (
    SPECS,
    SpecError,
    build_model,
    count_parameters,
    count_trainable,
    get_spec,
    linear_module_names,
    published_specs,
)
from ftlab.types import LoraSpec, ModelSpec

# --- the cross-checks -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SPECS))
def test_closed_form_base_count_matches_transformers(name: str) -> None:
    """Every spec, including the 13B one, built on the meta device for free."""
    spec = get_spec(name)
    built = count_parameters(build_model(spec, on_meta=True))
    assert base_parameters(spec) == built, name


@pytest.mark.parametrize("name", ["llama-2-7b", "mistral-7b", "llama-3-8b", "tiny-llama"])
@pytest.mark.parametrize("target", ["attention", "mlp", "all-linear"])
@pytest.mark.parametrize("r", [1, 8, 16])
def test_closed_form_adapter_count_matches_peft(name: str, target: str, r: int) -> None:
    spec = get_spec(name)
    lora = LoraSpec(r=r, alpha=2 * r, target=target)  # type: ignore[arg-type]

    formula, modules = adapter_parameters(spec, lora)
    model = get_peft_model(build_model(spec, on_meta=True), to_peft_config(spec, lora))

    assert formula == count_trainable(model), f"{name} r={r} {target}"
    assert modules == linear_module_names(spec, target)


def test_published_counts_match_the_literature() -> None:
    """Llama-2-7B is 6,738,415,616 parameters in the paper. If this drifts, a spec is wrong."""
    assert base_parameters(get_spec("llama-2-7b")) == 6_738_415_616
    assert base_parameters(get_spec("llama-2-13b")) == 13_015_864_320


def test_meta_device_allocates_nothing() -> None:
    """The property that makes 13B arithmetic free."""
    model = build_model(get_spec("llama-2-13b"), on_meta=True)
    assert all(p.device.type == "meta" for p in model.parameters())


# --- parameter budget ------------------------------------------------------------


def test_trainable_fraction_is_a_small_fraction_of_base() -> None:
    budget = parameter_budget("llama-2-7b", LoraSpec(r=16, alpha=32, target="attention"))
    assert budget.adapter_parameters == 16_777_216
    assert 0.002 < budget.trainable_fraction < 0.003


def test_rank_scales_the_adapter_linearly() -> None:
    small = parameter_budget("mistral-7b", LoraSpec(r=8, alpha=16))
    large = parameter_budget("mistral-7b", LoraSpec(r=16, alpha=32))
    assert large.adapter_parameters == 2 * small.adapter_parameters


def test_all_linear_is_larger_than_attention_only() -> None:
    attention = parameter_budget("llama-2-7b", LoraSpec(r=16, target="attention"))
    everything = parameter_budget("llama-2-7b", LoraSpec(r=16, target="all-linear"))
    assert everything.adapter_parameters > attention.adapter_parameters


def test_grouped_query_attention_makes_k_and_v_cheaper() -> None:
    """Mistral has 8 KV heads to Llama-2's 32, so its k/v adapters are smaller.

    Both models have the same hidden size, so a naive formula that ignored KV grouping would
    give them identical attention-adapter counts.
    """
    llama = parameter_budget("llama-2-7b", LoraSpec(r=16, target="attention"))
    mistral = parameter_budget("mistral-7b", LoraSpec(r=16, target="attention"))

    assert get_spec("llama-2-7b").hidden_size == get_spec("mistral-7b").hidden_size
    assert mistral.adapter_parameters < llama.adapter_parameters
    assert get_spec("mistral-7b").uses_grouped_query_attention
    assert not get_spec("llama-2-7b").uses_grouped_query_attention


# --- memory budget ---------------------------------------------------------------


def test_quantisation_shrinks_only_the_base_term() -> None:
    plain = memory_budget("llama-2-7b", precision="bf16")
    quantised = memory_budget("llama-2-7b", precision="bf16", quantization="nf4")

    assert quantised.base_weights_gb < plain.base_weights_gb
    # The adapter is what is being trained, so QLoRA leaves it alone.
    assert quantised.adapter_weights_gb == plain.adapter_weights_gb
    assert quantised.optimizer_state_gb == plain.optimizer_state_gb


def test_nf4_compression_is_not_quite_four_times() -> None:
    """The per-block scaling metadata is real and counted."""
    plain = memory_budget("llama-2-7b", precision="bf16")
    quantised = memory_budget("llama-2-7b", precision="bf16", quantization="nf4")
    ratio = plain.base_weights_gb / quantised.base_weights_gb
    assert 3.4 < ratio < 3.6


def test_an_eight_bit_optimizer_shrinks_only_its_own_term() -> None:
    adamw = memory_budget("llama-2-7b", optimizer="adamw")
    eight_bit = memory_budget("llama-2-7b", optimizer="adamw_8bit")
    assert eight_bit.optimizer_state_gb < adamw.optimizer_state_gb
    assert eight_bit.base_weights_gb == adamw.base_weights_gb


def test_sgd_without_momentum_has_no_optimizer_state() -> None:
    assert memory_budget("llama-2-7b", optimizer="sgd").optimizer_state_gb == 0.0


def test_total_is_the_sum_of_its_parts() -> None:
    budget = memory_budget("mistral-7b")
    assert budget.total_gb == pytest.approx(
        budget.base_weights_gb
        + budget.adapter_weights_gb
        + budget.adapter_gradients_gb
        + budget.optimizer_state_gb,
        abs=1e-4,
    )


def test_full_finetune_is_far_larger_than_lora() -> None:
    """The comparison LoRA exists to win: a gradient and optimizer state per parameter."""
    lora = memory_budget("llama-2-7b", precision="bf16")
    full = full_finetune_memory_gb("llama-2-7b", precision="bf16")
    assert full > 5 * lora.total_gb


def test_qlora_fits_a_seven_b_on_a_24gb_card_and_bf16_lora_also_does() -> None:
    quantised = memory_budget("llama-2-7b", LoraSpec(r=16), quantization="nf4")
    plain = memory_budget("llama-2-7b", LoraSpec(r=16), precision="bf16")

    assert fits_within(quantised, 24.0)
    assert fits_within(plain, 24.0)
    # A 13B in bf16 does not, which is the decision the tool exists to inform.
    assert not fits_within(memory_budget("llama-2-13b", LoraSpec(r=16), precision="bf16"), 24.0)


def test_headroom_is_validated() -> None:
    budget = memory_budget("llama-2-7b")
    with pytest.raises(ValueError, match=r"headroom must be in \[0, 1\)"):
        fits_within(budget, 24.0, headroom=1.0)


def test_an_unknown_optimizer_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown optimizer"):
        memory_budget("llama-2-7b", optimizer="adagrad")  # type: ignore[arg-type]


def test_every_declared_optimizer_is_usable() -> None:
    for optimizer in OPTIMIZER_STATE:
        assert memory_budget("llama-2-7b", optimizer=optimizer).total_gb > 0  # type: ignore[arg-type]


# --- specs -----------------------------------------------------------------------


def test_an_unknown_spec_lists_what_exists() -> None:
    with pytest.raises(SpecError, match="unknown model spec"):
        get_spec("llama-4-900b")


def test_published_and_tiny_specs_are_separated() -> None:
    assert all(spec.published for spec in published_specs())
    assert {spec.name for spec in published_specs()} == {
        "llama-2-7b",
        "llama-2-13b",
        "llama-3-8b",
        "mistral-7b",
    }


def test_a_spec_with_inconsistent_heads_is_rejected() -> None:
    with pytest.raises(ValueError, match="divisible by num_attention_heads"):
        ModelSpec(
            name="bad",
            architecture="llama",
            vocab_size=32,
            hidden_size=10,
            intermediate_size=20,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            max_position_embeddings=32,
        )


def test_kv_heads_must_divide_attention_heads() -> None:
    with pytest.raises(ValueError, match="divisible by num_key_value_heads"):
        ModelSpec(
            name="bad",
            architecture="llama",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=3,
            max_position_embeddings=32,
        )


def test_an_unknown_lora_target_is_rejected() -> None:
    with pytest.raises(SpecError, match="unknown LoRA target"):
        linear_module_names(get_spec("nano-llama"), "everything")


def test_lora_scaling_is_alpha_over_r() -> None:
    # Not a second rank: doubling r at fixed alpha halves the update scale.
    assert LoraSpec(r=8, alpha=16).scaling == 2.0
    assert LoraSpec(r=16, alpha=16).scaling == 1.0


def test_a_real_build_is_not_on_meta() -> None:
    model = build_model(get_spec("nano-llama"))
    assert all(p.device.type != "meta" for p in model.parameters())
    assert isinstance(next(model.parameters()), torch.Tensor)
