"""Tests for the tokenizer, templates and dataset preparation.

The two that matter most are prompt masking and leakage detection. Both are silent when
wrong: a model still trains with an unmasked prompt, and an evaluation still produces a
number when the eval set overlaps training. The number is just meaningless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftlab.data import (
    IGNORE_INDEX,
    batches,
    collate,
    fingerprint,
    load_examples,
    prepare,
    tokenize_example,
)
from ftlab.templates import TEMPLATES, get_template
from ftlab.tokenizer import PAD_ID, ByteTokenizer
from ftlab.types import Example

# --- tokenizer --------------------------------------------------------------------


def test_encoding_round_trips_exactly(tokenizer: ByteTokenizer) -> None:
    """Byte-level is lossless, which is what makes this assertion exact rather than fuzzy."""
    for text in ("hello", "", "café — naïve", "你好", "emoji \U0001f600"):
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_special_tokens_are_added_and_stripped(tokenizer: ByteTokenizer) -> None:
    ids = tokenizer.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "hi"


def test_special_tokens_can_be_kept(tokenizer: ByteTokenizer) -> None:
    ids = [tokenizer.bos_id, *tokenizer.encode("x")]
    assert tokenizer.decode(ids, skip_special=False) != "x"


def test_a_truncated_multibyte_sequence_decodes_without_raising(tokenizer: ByteTokenizer) -> None:
    """Generation can stop mid-codepoint; raising there would make output uninspectable."""
    ids = tokenizer.encode("café")[:-1]
    assert isinstance(tokenizer.decode(ids), str)


def test_vocab_size_matches_the_tiny_specs(tokenizer: ByteTokenizer) -> None:
    from ftlab.specs import get_spec

    assert tokenizer.vocab_size == get_spec("tiny-llama").vocab_size


# --- templates --------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_splits_prompt_from_completion(name: str) -> None:
    example = Example(id="x", instruction="Do the thing", output="Done.")
    prompt, completion = get_template(name).render(example)

    assert prompt, name
    assert completion, name
    # The completion must not be inside the prompt, or masking would hide the answer.
    assert "Done." not in prompt, name
    assert "Done." in completion, name


def test_alpaca_uses_its_input_variant() -> None:
    template = get_template("alpaca")
    without, _ = template.render(Example(id="a", instruction="Sum them", output="3"))
    with_input, _ = template.render(
        Example(id="b", instruction="Sum them", input="1 and 2", output="3")
    )

    assert "### Input:" not in without
    assert "### Input:" in with_input
    assert "1 and 2" in with_input


def test_llama2_chat_keeps_its_leading_space() -> None:
    """Llama-2-chat was trained with it; dropping it changes the first token."""
    _, completion = get_template("llama2-chat").render(
        Example(id="a", instruction="Hi", output="Hello")
    )
    assert completion.startswith(" ")


def test_raw_is_the_control_with_no_framing() -> None:
    prompt, _ = get_template("raw").render(Example(id="a", instruction="Hi", output="Hello"))
    assert "###" not in prompt
    assert "[INST]" not in prompt


def test_an_unknown_template_lists_what_exists() -> None:
    with pytest.raises(KeyError, match="unknown template"):
        get_template("vicuna")


# --- tokenising and masking -------------------------------------------------------


def test_the_prompt_is_masked_out_of_the_labels(tokenizer: ByteTokenizer) -> None:
    """The load-bearing detail: loss is computed on the answer, not the boilerplate."""
    example = Example(id="x", instruction="Say hello", output="Hello.")
    tokenized = tokenize_example(example, tokenizer, template="alpaca", max_length=512)

    assert tokenized is not None
    assert all(label == IGNORE_INDEX for label in tokenized.labels[: tokenized.n_prompt_tokens])
    assert all(label != IGNORE_INDEX for label in tokenized.labels[tokenized.n_prompt_tokens :])
    assert tokenized.n_supervised_tokens == tokenized.n_completion_tokens


def test_input_ids_and_labels_are_the_same_length(tokenizer: ByteTokenizer) -> None:
    tokenized = tokenize_example(
        Example(id="x", instruction="Say hello", output="Hello."), tokenizer, max_length=512
    )
    assert tokenized is not None
    assert len(tokenized.input_ids) == len(tokenized.labels)


def test_an_over_long_example_is_dropped_not_truncated(tokenizer: ByteTokenizer) -> None:
    """Truncation would remove the end of the answer, which is the part being learned."""
    example = Example(id="x", instruction="Repeat", output="word " * 500)
    assert tokenize_example(example, tokenizer, template="raw", max_length=64) is None


def test_mismatched_lengths_are_rejected() -> None:
    from ftlab.data import TokenizedExample

    with pytest.raises(ValueError, match="same length"):
        TokenizedExample(
            id="x", input_ids=[1, 2, 3], labels=[1, 2], n_prompt_tokens=0, n_completion_tokens=2
        )


# --- fingerprints and deduplication ----------------------------------------------


def test_fingerprints_ignore_whitespace_and_case() -> None:
    left = Example(id="a", instruction="What  is  it?", output="A thing.")
    right = Example(id="b", instruction="what is it?", output="a   THING.")
    assert fingerprint(left) == fingerprint(right)


def test_fingerprints_distinguish_different_content() -> None:
    left = Example(id="a", instruction="What is it?", output="A thing.")
    right = Example(id="b", instruction="What is it?", output="Another thing.")
    assert fingerprint(left) != fingerprint(right)


def test_duplicates_within_a_split_are_dropped(tokenizer: ByteTokenizer) -> None:
    examples = [
        Example(id="t1", instruction="What is it?", output="A thing.", split="train"),
        Example(id="t2", instruction="what is   it?", output="a thing.", split="train"),
        Example(id="e1", instruction="Something else?", output="Yes.", split="eval"),
    ]
    stats = prepare(examples, tokenizer, template="raw").stats

    assert stats.n_dropped_duplicate == 1
    assert stats.n_train == 1
    assert stats.n_leaked == 0


def test_cross_split_duplicates_are_reported_as_leakage_not_deduplicated(
    tokenizer: ByteTokenizer,
) -> None:
    """The bug this design avoids.

    A global dedup pass would delete the eval copy and report zero leakage, "fixing" the
    serious problem by hiding it. Deduplication is within a split; across splits it is
    leakage and it is named.
    """
    examples = [
        Example(id="t1", instruction="What is it?", output="A thing.", split="train"),
        Example(id="t2", instruction="Other?", output="No.", split="train"),
        Example(id="e1", instruction="What is it?", output="A thing.", split="eval"),
    ]
    stats = prepare(examples, tokenizer, template="raw").stats

    assert stats.n_leaked == 1
    assert stats.leaked_ids == ["e1"]
    assert stats.n_eval == 1, "the leaking example must survive to be reported"


def test_empty_examples_are_counted_separately(tokenizer: ByteTokenizer) -> None:
    examples = [
        Example(id="t1", instruction="Real", output="Answer", split="train"),
        Example(id="t2", instruction="   ", output="Answer", split="train"),
        Example(id="e1", instruction="Eval", output="Answer", split="eval"),
    ]
    stats = prepare(examples, tokenizer, template="raw").stats
    assert stats.n_dropped_empty == 1


def test_a_random_split_is_used_when_none_is_declared(tokenizer: ByteTokenizer) -> None:
    examples = [
        Example(id=f"x{i}", instruction=f"Question {i}", output=f"Answer {i}") for i in range(10)
    ]
    dataset = prepare(examples, tokenizer, template="raw", eval_fraction=0.3, seed=1)

    assert dataset.stats.n_eval == 3
    assert dataset.stats.n_train == 7
    # Deduplication happened before the split, so the split cannot have created leakage.
    assert dataset.stats.n_leaked == 0


def test_the_random_split_is_reproducible(tokenizer: ByteTokenizer) -> None:
    examples = [
        Example(id=f"x{i}", instruction=f"Question {i}", output=f"Answer {i}") for i in range(10)
    ]
    first = prepare(examples, tokenizer, template="raw", seed=3)
    second = prepare(examples, tokenizer, template="raw", seed=3)
    assert [e.id for e in first.eval] == [e.id for e in second.eval]


def test_an_invalid_eval_fraction_is_rejected(tokenizer: ByteTokenizer) -> None:
    with pytest.raises(ValueError, match=r"eval_fraction must be in \[0, 1\)"):
        prepare([Example(id="a", instruction="x", output="y")], tokenizer, eval_fraction=1.0)


def test_an_empty_dataset_is_rejected(tokenizer: ByteTokenizer) -> None:
    with pytest.raises(ValueError, match="cannot prepare an empty dataset"):
        prepare([], tokenizer)


# --- loading ---------------------------------------------------------------------


def test_loading_reports_the_failing_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a", "instruction": "x", "output": "y"}\n{oops}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: invalid JSON"):
        load_examples(path)


def test_loading_reports_a_schema_violation_with_its_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    payload = json.dumps({"id": "a", "instruction": "x", "output": ""})
    path.write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r":1: field output"):
        load_examples(path)


def test_an_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": "a", "instruction": "x", "output": "y", "outpout": "typo"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r":1: field outpout"):
        load_examples(path)


def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "ok.jsonl"
    path.write_text(
        '# a comment\n\n{"id": "a", "instruction": "x", "output": "y"}\n', encoding="utf-8"
    )
    assert len(load_examples(path)) == 1


def test_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        load_examples(tmp_path / "nope.jsonl")


# --- batching --------------------------------------------------------------------


def test_collate_pads_and_masks_both_ways(tokenizer: ByteTokenizer) -> None:
    """Padding needs the attention mask *and* the ignore index. One alone is a quiet bug."""
    short = tokenize_example(
        Example(id="a", instruction="Hi", output="Yes"), tokenizer, template="raw"
    )
    long = tokenize_example(
        Example(id="b", instruction="A longer question here", output="A longer answer"),
        tokenizer,
        template="raw",
    )
    assert short is not None and long is not None

    input_ids, labels, attention_mask = collate([short, long], pad_id=PAD_ID)
    width = max(short.length, long.length)

    assert all(len(row) == width for row in input_ids)
    assert input_ids[0][short.length :] == [PAD_ID] * (width - short.length)
    assert labels[0][short.length :] == [IGNORE_INDEX] * (width - short.length)
    assert attention_mask[0] == [1] * short.length + [0] * (width - short.length)


def test_collating_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="cannot collate an empty batch"):
        collate([], pad_id=PAD_ID)


def test_batches_cover_every_item_once(tokenizer: ByteTokenizer) -> None:
    items = [
        tokenize_example(
            Example(id=f"x{i}", instruction=f"Q{i}", output=f"A{i}"),
            tokenizer,
            template="raw",
        )
        for i in range(7)
    ]
    kept = [item for item in items if item is not None]
    grouped = batches(kept, 3)

    assert [len(group) for group in grouped] == [3, 3, 1]
    assert sorted(item.id for group in grouped for item in group) == sorted(i.id for i in kept)


def test_batch_shuffling_is_seeded(tokenizer: ByteTokenizer) -> None:
    items = [
        tokenize_example(
            Example(id=f"x{i}", instruction=f"Q{i}", output=f"A{i}"),
            tokenizer,
            template="raw",
        )
        for i in range(8)
    ]
    kept = [item for item in items if item is not None]

    first = [i.id for group in batches(kept, 2, seed=1) for i in group]
    same = [i.id for group in batches(kept, 2, seed=1) for i in group]
    other = [i.id for group in batches(kept, 2, seed=2) for i in group]

    assert first == same
    assert first != other


def test_a_nonsense_batch_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        batches([], 0)
