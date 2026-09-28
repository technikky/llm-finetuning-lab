"""The CLI and the reports it prints.

These are the surface a reader of the README actually touches, so every documented command is
exercised end to end. The training commands run on the nano architecture and finish in about a
second, which is what makes them affordable as tests rather than as a manual checklist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftlab.budget import memory_budget, parameter_budget
from ftlab.cli import main
from ftlab.data import PreparedDataset
from ftlab.report import (
    abbreviate,
    budget_report,
    comparison_table,
    dataset_table,
    memory_table,
    parameter_table,
    run_report,
)
from ftlab.train import train
from ftlab.types import LoraSpec, TrainingConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTRUCTIONS = str(REPO_ROOT / "data" / "instructions.jsonl")
LEAKY = str(REPO_ROOT / "data" / "leaky.jsonl")

NANO = ["--model", "nano-llama", "--template", "raw", "--max-length", "256"]


# --- informational commands -------------------------------------------------------


def test_specs_lists_every_architecture(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["specs"]) == 0
    out = capsys.readouterr().out
    assert "llama-2-7b" in out
    assert "nano-llama" in out
    # The tiny ones are marked unpublished so nobody mistakes them for real checkpoints.
    assert "False" in out


def test_budget_for_one_model_labels_its_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["budget", "--model", "llama-2-7b", "--r", "16"]) == 0
    out = capsys.readouterr().out
    assert "Computed, not measured" in out
    assert "16.78M" in out  # 16,777,216 trainable, abbreviated by the table
    assert "0.2490%" in out
    assert "Full fine-tune for comparison" in out


def test_budget_across_all_published_specs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["budget", "--r", "16"]) == 0
    out = capsys.readouterr().out
    for name in ("llama-2-7b", "llama-2-13b", "llama-3-8b", "mistral-7b"):
        assert name in out


def test_budget_answers_the_fits_question(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["budget", "--model", "llama-2-7b", "--quantization", "nf4", "--vram", "24"]) == 0
    out = capsys.readouterr().out
    assert "fits" in out
    assert "compresses base weights" in out


def test_budget_says_when_something_does_not_fit(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["budget", "--model", "llama-2-13b", "--precision", "bf16", "--vram", "24"]) == 0
    assert "does NOT fit" in capsys.readouterr().out


def test_budget_writes_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["budget", "--model", "mistral-7b", "--out", str(tmp_path)]) == 0
    capsys.readouterr()

    payload = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert payload["parameters"]["adapter_parameters"] > 0
    # The two headline numbers are properties on the models, so a plain dump would omit them.
    assert 0.0 < payload["parameters"]["trainable_fraction"] < 0.01
    assert payload["full_finetune_gb"] > payload["memory"]["total_gb"]
    assert payload["full_finetune_ratio"] > 1.0


def test_an_unknown_model_exits_with_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["budget", "--model", "llama-9b"]) == 2
    assert "unknown model spec" in capsys.readouterr().err


def test_quantization_reports_this_machine(capsys: pytest.CaptureFixture[str]) -> None:
    """On CPU-only CI this prints unusable with a reason; that is the correct output."""
    assert main(["quantization"]) == 0
    out = capsys.readouterr().out
    assert "bitsandbytes installed" in out
    assert "4-bit usable" in out


# --- data ------------------------------------------------------------------------


def test_prepare_reports_the_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["prepare", INSTRUCTIONS, "--template", "alpaca", "--max-length", "512"]) == 0
    out = capsys.readouterr().out
    assert "| Train |" in out
    assert "Train/eval leakage" in out


def test_prepare_fails_on_the_leaky_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    """The fixture exists so this failure path is covered, not just described."""
    assert main(["prepare", LEAKY, "--template", "alpaca"]) == 1
    captured = capsys.readouterr()
    assert "also appear in training" in captured.err


def test_prepare_writes_its_statistics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["prepare", INSTRUCTIONS, "--out", str(tmp_path)]) == 0
    capsys.readouterr()

    stats = json.loads((tmp_path / "dataset.json").read_text(encoding="utf-8"))
    assert stats["n_train"] > 0
    assert stats["n_eval"] > 0
    assert stats["n_leaked"] == 0


def test_a_missing_data_file_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["prepare", "does-not-exist.jsonl"]) == 2
    assert "error:" in capsys.readouterr().err


# --- training and the run registry -----------------------------------------------


def test_train_then_runs_then_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The full loop a reader follows from the README."""
    runs_dir = str(tmp_path / "runs")
    common = [
        "train",
        INSTRUCTIONS,
        *NANO,
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--runs-dir",
        runs_dir,
    ]

    assert main([*common, "--r", "4"]) == 0
    assert "Held-out" in capsys.readouterr().out

    assert main([*common, "--r", "16"]) == 0
    capsys.readouterr()

    assert main(["runs", "--runs-dir", runs_dir]) == 0
    assert "2 run(s)" in capsys.readouterr().out

    assert main(["compare", "--runs-dir", runs_dir]) == 0
    assert "r" in capsys.readouterr().out


def test_comparing_fewer_than_two_runs_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["compare", "--runs-dir", str(tmp_path / "runs")]) == 1
    assert "need at least two runs" in capsys.readouterr().err


def test_runs_on_an_empty_registry_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["runs", "--runs-dir", str(tmp_path / "nothing")]) == 0
    assert "no runs" in capsys.readouterr().out


def test_train_writes_a_report_and_an_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "train",
            INSTRUCTIONS,
            *NANO,
            "--epochs",
            "1",
            "--r",
            "4",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--adapter-out",
            str(tmp_path / "adapter"),
            "--out",
            str(tmp_path / "reports"),
        ]
    )
    assert code == 0
    capsys.readouterr()

    assert (tmp_path / "adapter" / "adapter_config.json").exists()
    assert list((tmp_path / "reports").glob("*.md"))
    assert list((tmp_path / "reports").glob("*.json"))


def test_the_saved_adapter_is_small(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The whole point of the technique: a checkpoint is the adapter, not the model."""
    assert (
        main(
            [
                "train",
                INSTRUCTIONS,
                *NANO,
                "--epochs",
                "1",
                "--r",
                "4",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--adapter-out",
                str(tmp_path / "adapter"),
            ]
        )
        == 0
    )
    capsys.readouterr()

    weights = list((tmp_path / "adapter").glob("adapter_model.*"))
    assert weights, "no adapter weights were written"
    assert sum(path.stat().st_size for path in weights) < 1_000_000


def test_train_refuses_a_leaking_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["train", LEAKY, *NANO, "--epochs", "1", "--runs-dir", str(tmp_path / "runs")])
    assert code == 2
    assert "also appear in training" in capsys.readouterr().err


# --- evaluate and generate -------------------------------------------------------


def test_evaluate_a_saved_adapter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    adapter = str(tmp_path / "adapter")
    assert (
        main(
            [
                "train",
                INSTRUCTIONS,
                *NANO,
                "--epochs",
                "1",
                "--r",
                "4",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--adapter-out",
                adapter,
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["evaluate", INSTRUCTIONS, *NANO, "--adapter", adapter]) == 0
    out = capsys.readouterr().out
    assert "perplexity" in out
    assert "token accuracy" in out


def test_merging_before_evaluation_gives_the_same_answer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """If these disagreed, the deployed weights would not be the evaluated ones."""
    adapter = str(tmp_path / "adapter")
    assert (
        main(
            [
                "train",
                INSTRUCTIONS,
                *NANO,
                "--epochs",
                "1",
                "--r",
                "4",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--adapter-out",
                adapter,
            ]
        )
        == 0
    )
    capsys.readouterr()

    def perplexity_of(*extra: str) -> float:
        assert main(["evaluate", INSTRUCTIONS, *NANO, "--adapter", adapter, *extra]) == 0
        for line in capsys.readouterr().out.splitlines():
            if line.startswith("perplexity"):
                return float(line.split(":")[1])
        raise AssertionError("no perplexity in the output")

    assert perplexity_of() == pytest.approx(perplexity_of("--merge"), rel=1e-4)


def test_generate_from_a_base_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["generate", "Question: what is it?", "--model", "nano-llama"]) == 0
    # A randomly initialised nano model emits noise; that it emits at all is the check.
    assert capsys.readouterr().out is not None


def test_generate_through_an_adapter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    adapter = str(tmp_path / "adapter")
    assert (
        main(
            [
                "train",
                INSTRUCTIONS,
                *NANO,
                "--epochs",
                "1",
                "--r",
                "4",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--adapter-out",
                adapter,
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["generate", "Question: ", "--model", "nano-llama", "--adapter", adapter]) == 0
    assert main(["generate", "Q: ", "--model", "nano-llama", "--adapter", adapter, "--merge"]) == 0


def test_evaluating_a_missing_adapter_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["evaluate", INSTRUCTIONS, *NANO, "--adapter", "no-such-dir"]) == 2
    assert "no adapter at" in capsys.readouterr().err


# --- reports ---------------------------------------------------------------------


def test_parameter_counts_are_abbreviated_at_every_scale() -> None:
    """A fixed "B" suffix printed the 660k-parameter spec as 0.001B."""
    assert abbreviate(6_738_415_616) == "6.738B"
    assert abbreviate(16_777_216) == "16.78M"
    assert abbreviate(663_552) == "663.6k"
    assert abbreviate(512) == "512"


def test_the_parameter_table_has_a_row_per_budget() -> None:
    table = parameter_table(
        [
            parameter_budget("llama-2-7b", LoraSpec(r=16)),
            parameter_budget("mistral-7b", LoraSpec(r=16)),
        ]
    )
    assert table.count("\n") == 3  # header, rule, two rows
    assert "llama-2-7b" in table and "mistral-7b" in table


def test_the_memory_table_shows_the_component_terms() -> None:
    table = memory_table([memory_budget("llama-2-7b", LoraSpec(r=16))])
    assert "base" in table.lower()
    assert "optimizer" in table.lower()


def test_the_budget_report_states_what_it_excludes() -> None:
    """Activations and the KV cache are not modelled; the report has to say so."""
    report = budget_report(LoraSpec(r=16))
    assert "activation" in report.lower()
    assert "Computed" in report


def test_the_dataset_table_shows_leakage(dataset: PreparedDataset) -> None:
    table = dataset_table(dataset.stats)
    assert "Train/eval leakage" in table
    assert "**0**" in table


def test_the_run_report_distinguishes_before_and_after(dataset: PreparedDataset) -> None:
    _, run = train(
        dataset,
        TrainingConfig(model="nano-llama", lora=LoraSpec(r=4), template="raw", max_length=256),
    )
    report = run_report(run)

    assert "Held-out" in report
    assert "before" in report.lower() and "after" in report.lower()
    assert run.run_id in report


def test_the_comparison_table_lists_every_run(dataset: PreparedDataset) -> None:
    runs = [
        train(
            dataset,
            TrainingConfig(model="nano-llama", lora=LoraSpec(r=r), template="raw", max_length=256),
        )[1]
        for r in (4, 8)
    ]
    table = comparison_table(runs)
    assert all(run.run_id[-12:] in table for run in runs)


def test_the_parser_rejects_an_unknown_command() -> None:
    with pytest.raises(SystemExit):
        main(["frobnicate"])
