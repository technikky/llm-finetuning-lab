"""Experiment tracking: a local run registry.

Deliberately a directory of JSON files rather than a hosted tracker. Three reasons, in order
of how much they matter here:

* A run record must be **readable without a service**. A result nobody can inspect in six
  months is not a result.
* Every run carries its config hash and its environment, so two curves can be compared with a
  claim about *why* they differ rather than a guess.
* It works offline, which means CI can write and read runs on every commit.

A hosted tracker is the right tool once runs cost money and several people share them. This
one is the right tool for making a fine-tuning claim checkable.
"""

from __future__ import annotations

import json
from pathlib import Path

from ftlab.types import TrainingRun

RUNS_FILE = "runs.jsonl"


class RegistryError(RuntimeError):
    """Raised for registry consistency problems."""


class RunRegistry:
    """An append-only JSONL log of training runs, plus per-run detail files."""

    def __init__(self, directory: str | Path = "runs") -> None:
        self.directory = Path(directory)

    @property
    def index_path(self) -> Path:
        return self.directory / RUNS_FILE

    def _run_path(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.json"

    def save(self, run: TrainingRun) -> Path:
        """Write the full run and append a summary line to the index."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._run_path(run.run_id)
        if path.exists():
            raise RegistryError(f"run {run.run_id!r} already exists at {path}")

        path.write_text(
            json.dumps(run.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.index_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(self._summary(run), sort_keys=True) + "\n")
        return path

    def load(self, run_id: str) -> TrainingRun:
        path = self._run_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"no run {run_id!r} in {self.directory}")
        return TrainingRun.model_validate_json(path.read_text(encoding="utf-8"))

    def run_ids(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(path.stem for path in self.directory.glob("*.json") if path.name != RUNS_FILE)

    def load_all(self) -> list[TrainingRun]:
        return [self.load(run_id) for run_id in self.run_ids()]

    def summaries(self) -> list[dict[str, object]]:
        """Index rows, newest last. Reads the index rather than every full run."""
        if not self.index_path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _summary(run: TrainingRun) -> dict[str, object]:
        return {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "model": run.config.model,
            "r": run.config.lora.r,
            "alpha": run.config.lora.alpha,
            "target": run.config.lora.target,
            "template": run.config.template,
            "epochs": run.config.epochs,
            "learning_rate": run.config.learning_rate,
            "seed": run.config.seed,
            "config_hash": run.config_hash[:16],
            "trainable_parameters": run.budget.adapter_parameters if run.budget else None,
            "initial_loss": run.initial_loss,
            "final_loss": run.final_loss,
            "eval_perplexity_before": run.eval_before.perplexity if run.eval_before else None,
            "eval_perplexity_after": run.eval_after.perplexity if run.eval_after else None,
            "wall_seconds": run.wall_seconds,
        }


def comparable(left: TrainingRun, right: TrainingRun) -> tuple[bool, list[str]]:
    """Whether two runs differ in a way that makes their metrics comparable.

    Returns the verdict and the fields that differ. A comparison across different models or
    datasets is not an ablation of the hyperparameter that also changed, and this is what
    stops a report claiming it is.
    """
    differences: list[str] = []
    if left.config.model != right.config.model:
        differences.append(f"model: {left.config.model} vs {right.config.model}")
    if (left.dataset and right.dataset) and (
        left.dataset.n_train != right.dataset.n_train
        or left.dataset.total_train_tokens != right.dataset.total_train_tokens
    ):
        differences.append("dataset: different size or token count")
    if left.config.template != right.config.template:
        differences.append(f"template: {left.config.template} vs {right.config.template}")

    left_env = left.environment.get("torch")
    right_env = right.environment.get("torch")
    if left_env != right_env:
        differences.append(f"torch: {left_env} vs {right_env}")

    return not differences, differences
