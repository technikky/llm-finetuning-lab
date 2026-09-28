"""Rendering budgets, runs and comparisons.

Computed figures and measured figures are labelled as such in every table. A report that
prints an exact parameter count beside a loss from one run on one machine, with no indication
which is which, invites the reader to treat both as equally solid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ftlab.budget import full_finetune_memory_gb, memory_budget, parameter_budget
from ftlab.specs import published_specs
from ftlab.types import (
    DatasetStats,
    LoraSpec,
    MemoryBudget,
    ParameterBudget,
    Precision,
    Quantization,
    TrainingRun,
)


def _pct(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def abbreviate(count: int) -> str:
    """A parameter count at a readable scale.

    One rule, used everywhere, because the tiny specs and the 13B one appear in the same
    tables: a fixed "B" suffix printed the 660k-parameter model as 0.001B.
    """
    if count >= 1_000_000_000:
        return f"{count / 1e9:.3f}B"
    if count >= 1_000_000:
        return f"{count / 1e6:.2f}M"
    if count >= 1_000:
        return f"{count / 1e3:.1f}k"
    return str(count)


def parameter_table(budgets: list[ParameterBudget]) -> str:
    lines = [
        "| Model | Base parameters | LoRA r | Target | Trainable | % of base |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for budget in budgets:
        lines.append(
            f"| `{budget.model}` | {abbreviate(budget.base_parameters)} | {budget.lora.r} "
            f"| {budget.lora.target} | {abbreviate(budget.adapter_parameters)} "
            f"| {_pct(budget.trainable_fraction)} |"
        )
    return "\n".join(lines)


def memory_table(budgets: list[MemoryBudget]) -> str:
    lines = [
        "| Model | Base | Adapter | Grads | Optimizer | Total | Quantization |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for budget in budgets:
        lines.append(
            f"| `{budget.model}` | {budget.base_weights_gb:.2f} GiB "
            f"| {budget.adapter_weights_gb:.3f} GiB | {budget.adapter_gradients_gb:.3f} GiB "
            f"| {budget.optimizer_state_gb:.3f} GiB | **{budget.total_gb:.2f} GiB** "
            f"| {budget.quantization} |"
        )
    lines.append("")
    lines.append(f"Excludes: {budgets[0].excludes}." if budgets else "")
    return "\n".join(lines)


def budget_report(
    lora: LoraSpec | None = None,
    *,
    precision: Precision = "bf16",
    quantization: Quantization = "none",
    optimizer: str = "adamw",
) -> str:
    """A full computed report over every published spec."""
    settings = lora or LoraSpec()
    specs = published_specs()

    parameters = [parameter_budget(spec, settings) for spec in specs]
    memories = [
        memory_budget(
            spec,
            settings,
            precision=precision,
            quantization=quantization,
            optimizer=optimizer,  # type: ignore[arg-type]
        )
        for spec in specs
    ]

    lines = [
        "# Parameter and memory budget",
        "",
        "**Computed, not measured.** These follow from the architecture definitions and the",
        "LoRA setting; no weights are loaded and no GPU is involved.",
        "",
        f"LoRA: r={settings.r}, alpha={settings.alpha}, target={settings.target}, "
        f"scaling={settings.scaling:.2f}",
        f"Precision: {precision}, quantization: {quantization}, optimizer: {optimizer}",
        "",
        "## Trainable parameters",
        "",
        parameter_table(parameters),
        "",
        "## Memory",
        "",
        memory_table(memories),
        "",
        "## Against a full fine-tune",
        "",
        "| Model | LoRA total | Full fine-tune | Ratio |",
        "| --- | --- | --- | --- |",
    ]
    for spec, memory in zip(specs, memories, strict=True):
        full = full_finetune_memory_gb(spec, precision=precision, optimizer=optimizer)  # type: ignore[arg-type]
        ratio = full / memory.total_gb if memory.total_gb else 0.0
        lines.append(
            f"| `{spec.name}` | {memory.total_gb:.2f} GiB | {full:.2f} GiB | {ratio:.1f}x |"
        )
    lines.append("")
    lines.append(
        "The full fine-tune column is weights plus a gradient and optimizer state for every "
        "parameter. It is why a 7B model needs far more than its weight footprint to train, "
        "and activations are excluded from both columns equally."
    )
    lines.append("")
    return "\n".join(lines)


def dataset_table(stats: DatasetStats) -> str:
    lines = [
        "| Property | Value |",
        "| --- | --- |",
        f"| Examples in | {stats.n_examples} |",
        f"| Train | {stats.n_train} |",
        f"| Eval | {stats.n_eval} |",
        f"| Dropped: empty | {stats.n_dropped_empty} |",
        f"| Dropped: duplicate | {stats.n_dropped_duplicate} |",
        f"| Dropped: over max_length | {stats.n_dropped_too_long} |",
        f"| Token length mean | {stats.token_length_mean} |",
        f"| Token length p50 / p95 / max | {stats.token_length_p50} / "
        f"{stats.token_length_p95} / {stats.token_length_max} |",
        f"| Train tokens | {stats.total_train_tokens:,} |",
        f"| **Train/eval leakage** | **{stats.n_leaked}** |",
    ]
    if stats.leaked_ids:
        lines.append(f"| Leaked ids | {', '.join(stats.leaked_ids[:10])} |")
    return "\n".join(lines)


def run_report(run: TrainingRun) -> str:
    """A single run: config, dataset, computed budget and measured results."""
    lines = [
        f"# Run `{run.run_id}`",
        "",
        f"- Model: `{run.config.model}`",
        f"- LoRA: r={run.config.lora.r}, alpha={run.config.lora.alpha}, "
        f"target={run.config.lora.target}",
        f"- Template: {run.config.template}, max_length={run.config.max_length}",
        f"- Epochs {run.config.epochs}, batch {run.config.batch_size}, "
        f"lr {run.config.learning_rate}, seed {run.config.seed}",
        f"- Config hash: `{run.config_hash[:16]}`",
        f"- Wall time: {run.wall_seconds:.2f}s on {run.environment.get('device', 'unknown')}",
        f"- Stack: torch {run.environment.get('torch')}, "
        f"transformers {run.environment.get('transformers')}, peft {run.environment.get('peft')}",
        "",
    ]

    if run.budget:
        lines += [
            "## Computed budget",
            "",
            parameter_table([run.budget]),
            "",
        ]
    if run.dataset:
        lines += ["## Dataset (measured)", "", dataset_table(run.dataset), ""]

    lines += [
        "## Measured results",
        "",
        "| Metric | Before | After | Change |",
        "| --- | --- | --- | --- |",
    ]
    if run.eval_before and run.eval_after:
        for label, before, after in (
            ("Held-out loss", run.eval_before.loss, run.eval_after.loss),
            ("Held-out perplexity", run.eval_before.perplexity, run.eval_after.perplexity),
            (
                "Held-out token accuracy",
                run.eval_before.token_accuracy,
                run.eval_after.token_accuracy,
            ),
        ):
            delta = after - before
            lines.append(f"| {label} | {_num(before)} | {_num(after)} | {delta:+.4f} |")
    else:
        lines.append("| - | n/a | n/a | n/a |")

    lines += [
        "",
        f"Training loss: {_num(run.initial_loss)} -> {_num(run.final_loss)} "
        f"over {len(run.steps)} steps.",
        "",
    ]
    return "\n".join(lines)


def comparison_table(runs: list[TrainingRun]) -> str:
    lines = [
        "| Run | Model | r | Target | Trainable | Final loss | Eval ppl before | after | Change |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        trainable = abbreviate(run.budget.adapter_parameters) if run.budget else "n/a"
        before = run.eval_before.perplexity if run.eval_before else None
        after = run.eval_after.perplexity if run.eval_after else None
        delta = run.perplexity_delta
        lines.append(
            f"| `{run.run_id[-12:]}` | {run.config.model} | {run.config.lora.r} "
            f"| {run.config.lora.target} | {trainable} | {_num(run.final_loss)} "
            f"| {_num(before)} | {_num(after)} "
            f"| {'n/a' if delta is None else f'{delta:+.4f}'} |"
        )
    return "\n".join(lines)


def budget_payload(
    parameters: ParameterBudget, memory: MemoryBudget, full_finetune_gb: float
) -> dict[str, Any]:
    """The JSON shape for a budget.

    ``total_gb`` and ``trainable_fraction`` are properties rather than fields, so a plain
    ``model_dump`` leaves out the two numbers a reader came for. They are not promoted to
    computed fields because :class:`~ftlab.types.TrainingRun` embeds these models and is
    round-tripped through JSON by the run registry, where an extra key is rejected.
    """
    return {
        "parameters": {
            **parameters.model_dump(mode="json"),
            "trainable_fraction": round(parameters.trainable_fraction, 6),
            "total_parameters": parameters.total_parameters,
        },
        "memory": {**memory.model_dump(mode="json"), "total_gb": memory.total_gb},
        "full_finetune_gb": round(full_finetune_gb, 4),
        "full_finetune_ratio": round(full_finetune_gb / memory.total_gb, 2),
    }


def write_json(payload: Any, path: str | Path) -> Path:  # noqa: ANN401 - any JSON-able value
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    data = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    file.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file


def write_text(text: str, path: str | Path) -> Path:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")
    return file
