"""Shared fixtures.

Everything here is offline. The tiny architectures in :mod:`ftlab.specs` are built from a
config with random initialisation, so there is no download, no network and no GPU -- which is
what lets CI run real training, evaluation and merge code on every commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ftlab.data import PreparedDataset, prepare
from ftlab.specs import build_model, get_spec
from ftlab.tokenizer import ByteTokenizer
from ftlab.types import Example, LoraSpec, ModelSpec, TrainingConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
INSTRUCTIONS = DATA_DIR / "instructions.jsonl"
LEAKY = DATA_DIR / "leaky.jsonl"


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed torch before every test.

    Autouse because several tests build randomly initialised models and compare them; an
    unseeded run would make those comparisons depend on test ordering.
    """
    torch.manual_seed(0)


@pytest.fixture
def nano_spec() -> ModelSpec:
    """The smallest architecture: ~107k parameters, two layers."""
    return get_spec("nano-llama")


@pytest.fixture
def nano_model(nano_spec: ModelSpec):
    return build_model(nano_spec)


@pytest.fixture
def tokenizer() -> ByteTokenizer:
    return ByteTokenizer()


@pytest.fixture
def small_lora() -> LoraSpec:
    return LoraSpec(r=4, alpha=8, target="attention")


@pytest.fixture
def examples() -> list[Example]:
    """Six examples with declared splits and no duplicates."""
    return [
        Example(
            id="t1", instruction="What is the rate limit?", output="600 per minute.", split="train"
        ),
        Example(id="t2", instruction="How long do tokens last?", output="One hour.", split="train"),
        Example(
            id="t3", instruction="Are webhooks once?", output="No, at least once.", split="train"
        ),
        Example(
            id="t4", instruction="Which formats export?", output="CSV and Parquet.", split="train"
        ),
        Example(
            id="e1",
            instruction="What is the burst limit?",
            output="1000 for ten seconds.",
            split="eval",
        ),
        Example(
            id="e2", instruction="How long are logs kept?", output="Thirty days.", split="eval"
        ),
    ]


@pytest.fixture
def dataset(examples: list[Example], tokenizer: ByteTokenizer) -> PreparedDataset:
    return prepare(examples, tokenizer, template="raw", max_length=256)


@pytest.fixture
def nano_config(small_lora: LoraSpec) -> TrainingConfig:
    """A config that trains in well under a second."""
    return TrainingConfig(
        model="nano-llama",
        lora=small_lora,
        template="raw",
        max_length=256,
        batch_size=2,
        learning_rate=3e-3,
        epochs=1,
        seed=0,
    )


@pytest.fixture
def token_ids() -> torch.Tensor:
    return torch.randint(0, 259, (2, 12))
