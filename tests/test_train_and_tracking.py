"""Training, evaluation, tracking and quantisation.

The training tests run real optimizer steps against the nano architecture, so they exercise
masking, batching, clipping, scheduling and the loss path for real rather than by mock. The run
finishes in well under a second, which is why it can be a per-commit gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ftlab.data import PreparedDataset, prepare
from ftlab.evaluate import MAX_PERPLEXITY, evaluate, generate
from ftlab.lora import merge_adapter
from ftlab.quantization import (
    bnb_config_kwargs,
    build_bnb_config,
    compression_ratio,
    quantization_available,
    weight_bytes_per_parameter,
)
from ftlab.specs import build_model, get_spec
from ftlab.tokenizer import ByteTokenizer
from ftlab.tracking import RegistryError, RunRegistry, comparable
from ftlab.train import (
    _learning_rate_at,
    config_hash,
    environment_info,
    load_adapter,
    save_adapter,
    set_seed,
    train,
)
from ftlab.types import Example, LoraSpec, TrainingConfig

# --- training --------------------------------------------------------------------


def test_a_run_produces_steps_and_both_evaluations(
    dataset: PreparedDataset, nano_config: TrainingConfig
) -> None:
    _, run = train(dataset, nano_config)

    assert run.steps
    assert run.eval_before is not None
    assert run.eval_after is not None
    assert run.budget is not None
    assert run.dataset is not None
    assert run.wall_seconds > 0.0


def test_the_loss_decreases_over_a_short_run(dataset: PreparedDataset) -> None:
    """A weak assertion on purpose.

    This checks the optimisation loop is wired up -- gradients flow, the optimizer steps, the
    masking does not zero everything. It is not a claim that the model is any good.
    """
    config = TrainingConfig(
        model="nano-llama",
        lora=LoraSpec(r=8, alpha=16),
        template="raw",
        max_length=256,
        batch_size=2,
        learning_rate=5e-3,
        epochs=4,
        seed=0,
    )
    _, run = train(dataset, config)

    assert run.initial_loss is not None
    assert run.final_loss is not None
    assert run.final_loss < run.initial_loss


def test_held_out_perplexity_improves(dataset: PreparedDataset) -> None:
    config = TrainingConfig(
        model="nano-llama",
        lora=LoraSpec(r=8, alpha=16),
        template="raw",
        max_length=256,
        batch_size=2,
        learning_rate=5e-3,
        epochs=6,
        seed=0,
    )
    _, run = train(dataset, config)

    assert run.eval_before is not None and run.eval_after is not None
    assert run.eval_after.perplexity < run.eval_before.perplexity
    assert run.perplexity_delta is not None and run.perplexity_delta < 0


def test_the_same_seed_gives_the_same_curve(
    dataset: PreparedDataset, nano_config: TrainingConfig
) -> None:
    """Without this, no two runs can be compared and no ablation means anything."""
    _, first = train(dataset, nano_config)
    _, second = train(dataset, nano_config)

    assert [s.loss for s in first.steps] == [s.loss for s in second.steps]
    assert first.config_hash == second.config_hash


def test_a_different_seed_gives_a_different_curve(
    dataset: PreparedDataset, nano_config: TrainingConfig
) -> None:
    _, first = train(dataset, nano_config)
    _, other = train(dataset, nano_config.model_copy(update={"seed": 99}))
    assert [s.loss for s in first.steps] != [s.loss for s in other.steps]


def test_training_refuses_a_leaking_dataset(tokenizer: ByteTokenizer) -> None:
    """Refusing is the point: a result from a leaking split measures memorisation."""
    examples = [
        Example(id="t1", instruction="What is it?", output="A thing.", split="train"),
        Example(id="t2", instruction="Other?", output="No.", split="train"),
        Example(id="e1", instruction="What is it?", output="A thing.", split="eval"),
    ]
    leaking = prepare(examples, tokenizer, template="raw")

    with pytest.raises(ValueError, match="also appear in training"):
        train(leaking, TrainingConfig(model="nano-llama", template="raw", epochs=1))


def test_training_on_an_empty_split_is_an_error(nano_config: TrainingConfig) -> None:
    with pytest.raises(ValueError, match="empty training split"):
        train(PreparedDataset(), nano_config)


def test_the_run_records_its_environment(
    dataset: PreparedDataset, nano_config: TrainingConfig
) -> None:
    _, run = train(dataset, nano_config)
    assert run.environment["torch"]
    assert run.environment["peft"]
    assert "device" in run.environment


def test_grad_norm_is_recorded_and_clipped(
    dataset: PreparedDataset, nano_config: TrainingConfig
) -> None:
    _, run = train(dataset, nano_config)
    assert all(step.grad_norm is not None for step in run.steps)


def test_config_hash_is_stable_and_sensitive(nano_config: TrainingConfig) -> None:
    assert config_hash(nano_config) == config_hash(nano_config)
    assert config_hash(nano_config) != config_hash(nano_config.model_copy(update={"seed": 1}))
    assert config_hash(nano_config) != config_hash(
        nano_config.model_copy(update={"lora": LoraSpec(r=32, alpha=64)})
    )


def test_warmup_then_decay(nano_config: TrainingConfig) -> None:
    config = nano_config.model_copy(update={"warmup_ratio": 0.2, "learning_rate": 1.0})
    schedule = [_learning_rate_at(step, 10, config) for step in range(10)]

    assert schedule[0] < schedule[1]  # warming up
    assert max(schedule) == pytest.approx(1.0)
    assert schedule[-1] < schedule[2]  # decaying


def test_no_warmup_starts_at_the_full_rate(nano_config: TrainingConfig) -> None:
    config = nano_config.model_copy(update={"warmup_ratio": 0.0, "learning_rate": 1.0})
    assert _learning_rate_at(0, 10, config) == pytest.approx(1.0)


def test_environment_info_is_serialisable() -> None:
    import json

    json.dumps(environment_info())


def test_set_seed_makes_torch_deterministic() -> None:
    set_seed(5)
    first = torch.randn(4)
    set_seed(5)
    assert torch.equal(first, torch.randn(4))


# --- adapter persistence ---------------------------------------------------------


def test_the_adapter_round_trips_and_keeps_its_outputs(
    dataset: PreparedDataset, nano_config: TrainingConfig, tmp_path: Path, token_ids: torch.Tensor
) -> None:
    """Saving only megabytes and reconstructing the base from its spec."""
    model, _ = train(dataset, nano_config)
    with torch.no_grad():
        before = model(input_ids=token_ids).logits.clone()

    save_adapter(model, tmp_path / "adapter")
    reloaded = load_adapter(nano_config.model, tmp_path / "adapter")
    with torch.no_grad():
        after = reloaded(input_ids=token_ids).logits

    assert torch.allclose(before, after, atol=1e-5)


def test_a_reloaded_adapter_can_be_merged(
    dataset: PreparedDataset, nano_config: TrainingConfig, tmp_path: Path, token_ids: torch.Tensor
) -> None:
    model, _ = train(dataset, nano_config)
    save_adapter(model, tmp_path / "adapter")

    reloaded = load_adapter(nano_config.model, tmp_path / "adapter")
    with torch.no_grad():
        before = reloaded(input_ids=token_ids).logits.clone()
    merged = merge_adapter(reloaded)
    with torch.no_grad():
        after = merged(input_ids=token_ids).logits

    assert torch.allclose(before, after, atol=1e-5)


def test_loading_from_a_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no adapter at"):
        load_adapter("nano-llama", tmp_path / "nope")


# --- evaluation -----------------------------------------------------------------


def test_evaluation_reports_loss_perplexity_and_accuracy(dataset: PreparedDataset) -> None:
    model = build_model(get_spec("nano-llama"))
    metrics = evaluate(model, dataset.eval, batch_size=2)

    assert metrics.n_examples == len(dataset.eval)
    assert metrics.loss > 0.0
    assert metrics.perplexity > 1.0
    assert 0.0 <= metrics.token_accuracy <= 1.0


def test_perplexity_is_the_exponential_of_the_loss(dataset: PreparedDataset) -> None:
    import math

    model = build_model(get_spec("nano-llama"))
    metrics = evaluate(model, dataset.eval, batch_size=2)
    assert metrics.perplexity == pytest.approx(math.exp(metrics.loss), rel=1e-3)


def test_perplexity_is_capped_rather_than_overflowing() -> None:
    assert MAX_PERPLEXITY == 1e6


def test_evaluating_nothing_returns_zeros() -> None:
    model = build_model(get_spec("nano-llama"))
    metrics = evaluate(model, [], batch_size=2)
    assert metrics.n_examples == 0
    assert metrics.perplexity == 0.0


def test_generation_is_deterministic_at_temperature_zero(tokenizer: ByteTokenizer) -> None:
    """Greedy by default so two evaluations of one checkpoint agree."""
    model = build_model(get_spec("nano-llama"))
    first = generate(model, tokenizer, "Question: ", max_new_tokens=8)
    second = generate(model, tokenizer, "Question: ", max_new_tokens=8)
    assert first == second


def test_generation_rejects_nonsense_arguments(tokenizer: ByteTokenizer) -> None:
    model = build_model(get_spec("nano-llama"))
    with pytest.raises(ValueError, match="max_new_tokens must be >= 1"):
        generate(model, tokenizer, "x", max_new_tokens=0)
    with pytest.raises(ValueError, match="temperature must be >= 0"):
        generate(model, tokenizer, "x", temperature=-1.0)


# --- tracking -------------------------------------------------------------------


def test_a_run_round_trips_through_the_registry(
    dataset: PreparedDataset, nano_config: TrainingConfig, tmp_path: Path
) -> None:
    _, run = train(dataset, nano_config)
    registry = RunRegistry(tmp_path / "runs")
    registry.save(run)

    restored = registry.load(run.run_id)
    assert restored.config_hash == run.config_hash
    assert [s.loss for s in restored.steps] == [s.loss for s in run.steps]
    assert registry.run_ids() == [run.run_id]


def test_the_index_summarises_each_run(
    dataset: PreparedDataset, nano_config: TrainingConfig, tmp_path: Path
) -> None:
    _, run = train(dataset, nano_config)
    registry = RunRegistry(tmp_path / "runs")
    registry.save(run)

    summaries = registry.summaries()
    assert len(summaries) == 1
    assert summaries[0]["run_id"] == run.run_id
    assert summaries[0]["trainable_parameters"] == run.budget.adapter_parameters  # type: ignore[union-attr]


def test_saving_the_same_run_twice_is_refused(
    dataset: PreparedDataset, nano_config: TrainingConfig, tmp_path: Path
) -> None:
    _, run = train(dataset, nano_config)
    registry = RunRegistry(tmp_path / "runs")
    registry.save(run)

    with pytest.raises(RegistryError, match="already exists"):
        registry.save(run)


def test_an_empty_registry_is_empty(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "nothing")
    assert registry.run_ids() == []
    assert registry.summaries() == []


def test_loading_an_unknown_run_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no run"):
        RunRegistry(tmp_path).load("nope")


def test_runs_on_the_same_setup_are_comparable(
    dataset: PreparedDataset, nano_config: TrainingConfig
) -> None:
    _, first = train(dataset, nano_config)
    _, second = train(dataset, nano_config.model_copy(update={"lora": LoraSpec(r=8, alpha=16)}))

    ok, differences = comparable(first, second)
    assert ok, differences


def test_runs_on_different_models_are_flagged_as_incomparable(
    dataset: PreparedDataset, nano_config: TrainingConfig
) -> None:
    """A comparison across models is not an ablation of the hyperparameter that also changed."""
    _, first = train(dataset, nano_config)
    _, second = train(dataset, nano_config.model_copy(update={"model": "tiny-llama"}))

    ok, differences = comparable(first, second)
    assert not ok
    assert any("model" in difference for difference in differences)


def test_runs_with_different_templates_are_flagged(
    dataset: PreparedDataset, nano_config: TrainingConfig, tokenizer: ByteTokenizer, examples
) -> None:
    other = prepare(examples, tokenizer, template="alpaca", max_length=512)
    _, first = train(dataset, nano_config)
    _, second = train(
        other, nano_config.model_copy(update={"template": "alpaca", "max_length": 512})
    )

    ok, differences = comparable(first, second)
    assert not ok
    assert any("template" in difference for difference in differences)


# --- quantisation ---------------------------------------------------------------


def test_nf4_config_kwargs_are_complete() -> None:
    kwargs = bnb_config_kwargs(quantization="nf4", compute_dtype="bfloat16")
    assert kwargs["load_in_4bit"] is True
    assert kwargs["bnb_4bit_quant_type"] == "nf4"
    assert kwargs["bnb_4bit_use_double_quant"] is True
    assert kwargs["bnb_4bit_compute_dtype"] == "bfloat16"


def test_int8_and_none_configs() -> None:
    assert bnb_config_kwargs(quantization="int8") == {"load_in_8bit": True}
    assert bnb_config_kwargs(quantization="none") == {}


def test_an_unknown_quantisation_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown quantization"):
        bnb_config_kwargs(quantization="int3")  # type: ignore[arg-type]


def test_double_quant_can_be_disabled() -> None:
    kwargs = bnb_config_kwargs(quantization="nf4", double_quant=False)
    assert kwargs["bnb_4bit_use_double_quant"] is False


def test_support_is_probed_without_raising() -> None:
    """On a CPU-only machine this reports unusable with a reason rather than failing."""
    support = quantization_available()
    assert isinstance(support.bitsandbytes_installed, bool)
    assert isinstance(support.cuda_available, bool)
    if not support.usable:
        assert support.reason


def test_building_a_real_config_fails_with_the_reason_when_unsupported() -> None:
    support = quantization_available()
    if support.bitsandbytes_installed:
        pytest.skip("bitsandbytes is installed; the unsupported path cannot be exercised")
    with pytest.raises(RuntimeError, match="needs bitsandbytes"):
        build_bnb_config(**bnb_config_kwargs(quantization="nf4"))


def test_compression_ratios_are_honest_about_metadata() -> None:
    # "4-bit" is 3.56x against bf16, not 4x: the block scales are counted.
    assert compression_ratio("nf4", "bf16") == pytest.approx(3.5556, abs=1e-3)
    assert compression_ratio("int8", "bf16") == pytest.approx(2.0)
    assert compression_ratio("none", "bf16") == pytest.approx(1.0)


def test_bytes_per_parameter_are_ordered() -> None:
    assert weight_bytes_per_parameter("nf4") < weight_bytes_per_parameter("int8")
    assert weight_bytes_per_parameter("int8") < weight_bytes_per_parameter("none", "bf16")
