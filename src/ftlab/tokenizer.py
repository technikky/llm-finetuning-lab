"""A byte-level tokenizer.

Deliberately not a downloaded BPE. A real tokenizer would make every test depend on a
network fetch and a licence, and it would contribute nothing to what this package
demonstrates: the training, masking, evaluation and merge machinery is identical whatever
produced the ids.

Byte-level has two properties that matter here. It is **lossless** -- every byte sequence
round-trips exactly, so a decode/encode identity test is meaningful rather than approximate.
And its vocabulary is fixed at 259, which is why the tiny specs in :mod:`ftlab.specs` use
that number.

The trade-off is a much longer sequence per unit of text than BPE, so token-count statistics
from this tokenizer are not comparable to a real model's. The docs say so.
"""

from __future__ import annotations

BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259

SPECIAL_IDS = frozenset({BOS_ID, EOS_ID, PAD_ID})
SPECIAL_NAMES = {BOS_ID: "<bos>", EOS_ID: "<eos>", PAD_ID: "<pad>"}


class ByteTokenizer:
    """Maps text to UTF-8 byte ids, plus three special tokens."""

    def __init__(self) -> None:
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        """Decode ids back to text.

        ``errors="replace"`` because a partially generated sequence can end mid-codepoint,
        and raising there would make it impossible to inspect a model's output while it is
        still bad at producing valid UTF-8.

        With ``skip_special=False`` the special ids are rendered by name. They are 256-258 and
        so are not bytes: decoding them as bytes is not possible, and the interesting case --
        inspecting where a model emitted ``<eos>`` -- is exactly the one that needs them shown.
        Byte runs are decoded together rather than one at a time, because a multi-byte
        codepoint split across separate decode calls would come back as replacement
        characters.
        """
        parts: list[str] = []
        run: list[int] = []

        for token in ids:
            if token in SPECIAL_IDS:
                if skip_special:
                    continue
                parts.append(bytes(run).decode("utf-8", errors="replace"))
                run.clear()
                parts.append(SPECIAL_NAMES[token])
                continue
            if not 0 <= token < 256:
                raise ValueError(
                    f"id {token} is outside the vocabulary of {VOCAB_SIZE}; "
                    "ids are UTF-8 bytes plus 256-258"
                )
            run.append(token)

        parts.append(bytes(run).decode("utf-8", errors="replace"))
        return "".join(parts)

    def encode_batch(self, texts: list[str], **kwargs: bool) -> list[list[int]]:
        return [self.encode(text, **kwargs) for text in texts]

    def __repr__(self) -> str:
        return f"ByteTokenizer(vocab_size={self.vocab_size})"
