"""Dataset preparation: loading, masking, deduplication, splitting and leakage detection.

Two things in here are load-bearing for whether a fine-tuning number means anything.

**Prompt masking.** Label positions covering the prompt are set to ``IGNORE_INDEX``, so loss
is computed on the completion only. Without it, most of the gradient goes into reproducing
instruction boilerplate.

**Leakage detection.** The most common reason a reported fine-tuning result is wrong is that
evaluation examples were also in training. :func:`prepare` checks for it by content
fingerprint and reports the count, and the CLI treats a non-zero count as a failure. An
evaluation set that overlaps training measures memorisation.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from ftlab.templates import get_template
from ftlab.tokenizer import ByteTokenizer
from ftlab.types import DatasetStats, Example, TemplateName

#: PyTorch's cross-entropy ignore index. Label positions set to this contribute no loss.
IGNORE_INDEX = -100


@dataclass
class TokenizedExample:
    """One example, tokenised, with the prompt masked out of the labels."""

    id: str
    input_ids: list[int]
    labels: list[int]
    n_prompt_tokens: int
    n_completion_tokens: int

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise ValueError(
                f"{self.id}: input_ids ({len(self.input_ids)}) and labels "
                f"({len(self.labels)}) must be the same length"
            )

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def n_supervised_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE_INDEX)


@dataclass
class PreparedDataset:
    """Tokenised train and eval splits, plus the statistics behind them."""

    train: list[TokenizedExample] = field(default_factory=list)
    eval: list[TokenizedExample] = field(default_factory=list)
    stats: DatasetStats = field(
        default_factory=lambda: DatasetStats(n_examples=0, n_train=0, n_eval=0)
    )

    def __len__(self) -> int:
        return len(self.train) + len(self.eval)


def fingerprint(example: Example) -> str:
    """Content hash used for duplicate and leakage detection.

    Whitespace-normalised and case-folded, so two examples differing only in formatting count
    as the same example. An exact-bytes hash would miss most real duplicates.
    """
    normalised = " ".join(example.fingerprint_source.split()).casefold()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _dedupe(examples: list[Example]) -> tuple[list[Example], int]:
    """Drop later examples whose fingerprint has already been seen. Returns (kept, dropped)."""
    seen: set[str] = set()
    kept: list[Example] = []
    dropped = 0
    for example in examples:
        digest = fingerprint(example)
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        kept.append(example)
    return kept, dropped


def load_examples(path: str | Path) -> list[Example]:
    """Load examples from JSONL, reporting the failing line."""
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"dataset not found: {file}")

    examples: list[Example] = []
    for lineno, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file}:{lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"{file}:{lineno}: expected a JSON object, got {type(record).__name__}"
            )
        try:
            examples.append(Example.model_validate(record))
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first["loc"]) or "<root>"
            raise ValueError(f"{file}:{lineno}: field {location}: {first['msg']}") from exc

    if not examples:
        raise ValueError(f"{file}: contains no examples")
    return examples


def tokenize_example(
    example: Example,
    tokenizer: ByteTokenizer,
    *,
    template: TemplateName | str = "alpaca",
    max_length: int = 512,
) -> TokenizedExample | None:
    """Tokenise one example with the prompt masked. ``None`` if it does not fit.

    Truncation would silently remove the end of the answer -- the part the model is supposed
    to learn -- so an over-long example is dropped and counted instead.
    """
    prompt, completion = get_template(template).render(example)

    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    completion_ids = tokenizer.encode(completion, add_eos=True)
    input_ids = prompt_ids + completion_ids

    if len(input_ids) > max_length:
        return None

    labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
    return TokenizedExample(
        id=example.id,
        input_ids=input_ids,
        labels=labels,
        n_prompt_tokens=len(prompt_ids),
        n_completion_tokens=len(completion_ids),
    )


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(int(percentile / 100.0 * len(ordered)), len(ordered) - 1)
    return ordered[index]


def prepare(
    examples: list[Example],
    tokenizer: ByteTokenizer | None = None,
    *,
    template: TemplateName | str = "alpaca",
    max_length: int = 512,
    eval_fraction: float = 0.2,
    seed: int = 0,
    respect_declared_split: bool = True,
) -> PreparedDataset:
    """Deduplicate, tokenise, split and check for leakage.

    Order matters: duplicates are removed **before** splitting, because a duplicate that
    lands on both sides of the split is leakage created by the preparation step itself.
    """
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in [0, 1), got {eval_fraction}")
    if not examples:
        raise ValueError("cannot prepare an empty dataset")

    tokens = tokenizer or ByteTokenizer()

    n_dropped_empty = 0
    non_empty: list[Example] = []
    for example in examples:
        if not example.instruction.strip() or not example.output.strip():
            n_dropped_empty += 1
            continue
        non_empty.append(example)

    declared = {example.split for example in non_empty}
    use_declared = respect_declared_split and declared == {"train", "eval"}

    if use_declared:
        # Deduplicate *within* each split, never across them. A repeated example inside one
        # split is waste; the same example on both sides is leakage, and a global dedup pass
        # would silently delete the eval copy and report zero leakage -- hiding the serious
        # problem by "fixing" it.
        train_examples, n_train_dupes = _dedupe([e for e in non_empty if e.split == "train"])
        eval_examples, n_eval_dupes = _dedupe([e for e in non_empty if e.split == "eval"])
        n_dropped_duplicate = n_train_dupes + n_eval_dupes
    else:
        # No declared splits, so there is nothing to leak across yet: deduplicating before
        # the split is what stops the split from creating leakage of its own.
        unique, n_dropped_duplicate = _dedupe(non_empty)
        shuffled = list(unique)
        random.Random(seed).shuffle(shuffled)
        n_eval = round(len(shuffled) * eval_fraction)
        eval_examples = shuffled[:n_eval]
        train_examples = shuffled[n_eval:]

    train_prints = {fingerprint(e) for e in train_examples}
    leaked = [e.id for e in eval_examples if fingerprint(e) in train_prints]

    n_dropped_too_long = 0
    train_tokenized: list[TokenizedExample] = []
    eval_tokenized: list[TokenizedExample] = []

    for source, destination in ((train_examples, train_tokenized), (eval_examples, eval_tokenized)):
        for example in source:
            tokenized = tokenize_example(example, tokens, template=template, max_length=max_length)
            if tokenized is None:
                n_dropped_too_long += 1
                continue
            destination.append(tokenized)

    lengths = [item.length for item in train_tokenized + eval_tokenized]
    stats = DatasetStats(
        n_examples=len(examples),
        n_train=len(train_tokenized),
        n_eval=len(eval_tokenized),
        n_dropped_empty=n_dropped_empty,
        n_dropped_duplicate=n_dropped_duplicate,
        n_dropped_too_long=n_dropped_too_long,
        token_length_mean=round(statistics.fmean(lengths), 2) if lengths else 0.0,
        token_length_p50=_percentile(lengths, 50),
        token_length_p95=_percentile(lengths, 95),
        token_length_max=max(lengths) if lengths else 0,
        total_train_tokens=sum(item.length for item in train_tokenized),
        n_leaked=len(leaked),
        leaked_ids=sorted(leaked),
    )
    return PreparedDataset(train=train_tokenized, eval=eval_tokenized, stats=stats)


def collate(
    batch: list[TokenizedExample], *, pad_id: int
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Right-pad a batch, returning (input_ids, labels, attention_mask).

    Padding positions get ``IGNORE_INDEX`` in the labels as well as a zero attention mask.
    Both are needed: the mask stops the model attending to padding, and the ignore index
    stops the loss counting it. Setting only one of them is a common and quiet bug -- the
    model still trains, just on a slightly wrong objective.
    """
    if not batch:
        raise ValueError("cannot collate an empty batch")

    width = max(item.length for item in batch)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_mask: list[list[int]] = []

    for item in batch:
        padding = width - item.length
        input_ids.append(item.input_ids + [pad_id] * padding)
        labels.append(item.labels + [IGNORE_INDEX] * padding)
        attention_mask.append([1] * item.length + [0] * padding)

    return input_ids, labels, attention_mask


def batches(
    items: list[TokenizedExample], batch_size: int, *, seed: int | None = None
) -> list[list[TokenizedExample]]:
    """Split into fixed-size batches, optionally shuffled with a given seed."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    ordered = list(items)
    if seed is not None:
        random.Random(seed).shuffle(ordered)
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
