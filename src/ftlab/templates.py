"""Instruction templates, and the prompt/completion split that makes masking possible.

Every template returns the prompt and the completion **separately**. That is the whole point
of this module, and it is the detail most often got wrong: if loss is computed over the
prompt as well as the answer, the model spends most of its gradient learning to reproduce the
instruction boilerplate it was given, which is both wasted capacity and a silently worse
result.

:func:`ftlab.data.tokenize_example` uses the split to set label positions for the prompt to
``-100``, the ignore index, so loss is computed on the completion only.
"""

from __future__ import annotations

from typing import Protocol

from ftlab.types import Example, TemplateName

ALPACA_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
ALPACA_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)


class Template(Protocol):
    """Renders an example into a prompt and the completion that should follow it."""

    @property
    def name(self) -> str: ...

    def render(self, example: Example) -> tuple[str, str]: ...


class AlpacaTemplate:
    """The Stanford Alpaca format, including its two variants for with/without input."""

    @property
    def name(self) -> str:
        return "alpaca"

    def render(self, example: Example) -> tuple[str, str]:
        if example.input.strip():
            prompt = ALPACA_WITH_INPUT.format(
                instruction=example.instruction.strip(), input=example.input.strip()
            )
        else:
            prompt = ALPACA_NO_INPUT.format(instruction=example.instruction.strip())
        return prompt, example.output.strip()


class ChatMLTemplate:
    """The ChatML format used by several instruct models."""

    @property
    def name(self) -> str:
        return "chatml"

    def render(self, example: Example) -> tuple[str, str]:
        user = example.instruction.strip()
        if example.input.strip():
            user = f"{user}\n\n{example.input.strip()}"
        prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        return prompt, f"{example.output.strip()}<|im_end|>"


class Llama2ChatTemplate:
    """The Llama-2-chat format.

    The leading space before the completion is not decoration: Llama-2-chat was trained with
    it, and dropping it shifts the tokenisation of the first word.
    """

    @property
    def name(self) -> str:
        return "llama2-chat"

    def render(self, example: Example) -> tuple[str, str]:
        user = example.instruction.strip()
        if example.input.strip():
            user = f"{user}\n\n{example.input.strip()}"
        return f"<s>[INST] {user} [/INST]", f" {example.output.strip()}</s>"


class RawTemplate:
    """No framing at all. The control: any gain a template shows is measured against this."""

    @property
    def name(self) -> str:
        return "raw"

    def render(self, example: Example) -> tuple[str, str]:
        prompt = example.instruction.strip()
        if example.input.strip():
            prompt = f"{prompt}\n{example.input.strip()}"
        return f"{prompt}\n", example.output.strip()


TEMPLATES: dict[str, Template] = {
    "alpaca": AlpacaTemplate(),
    "chatml": ChatMLTemplate(),
    "llama2-chat": Llama2ChatTemplate(),
    "raw": RawTemplate(),
}


def get_template(name: TemplateName | str) -> Template:
    try:
        return TEMPLATES[name]
    except KeyError:
        known = ", ".join(sorted(TEMPLATES))
        raise KeyError(f"unknown template {name!r}; known templates: {known}") from None
