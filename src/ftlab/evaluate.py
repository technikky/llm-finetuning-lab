"""Held-out evaluation and generation.

Perplexity is reported over **completion tokens only**, because the labels are masked. That is
the number worth comparing across runs: perplexity computed over the prompt as well is
dominated by how predictable the template boilerplate is, which is not what changed.

Token accuracy is reported alongside because it moves earlier than perplexity on short runs.
A run whose accuracy is flat at chance has not started learning, and that is visible well
before perplexity says anything.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from ftlab.data import IGNORE_INDEX, TokenizedExample, batches, collate
from ftlab.tokenizer import ByteTokenizer
from ftlab.types import CausalLM, EvalMetrics

#: Perplexity is exp(loss) and overflows for an untrained model on a large vocabulary. Capped
#: so a report stays readable, and the cap is documented rather than silently applied.
MAX_PERPLEXITY = 1e6


def _shift_for_causal_lm(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align logits with the tokens they predict.

    A causal model's logits at position ``i`` predict token ``i + 1``, so comparing them
    unshifted measures the model against its own input and reports an accuracy that looks
    excellent and means nothing.
    """
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


@torch.no_grad()
def evaluate(
    model: CausalLM,
    examples: list[TokenizedExample],
    *,
    batch_size: int = 4,
    pad_id: int = 258,
) -> EvalMetrics:
    """Mean loss, perplexity and token accuracy over supervised positions."""
    if not examples:
        return EvalMetrics(n_examples=0, loss=0.0, perplexity=0.0, token_accuracy=0.0)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in batches(examples, batch_size):
        input_ids, labels, attention_mask = collate(batch, pad_id=pad_id)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        logits = model(input_ids=input_tensor, attention_mask=mask_tensor).logits
        shifted_logits, shifted_labels = _shift_for_causal_lm(logits, label_tensor)

        # Summed, not averaged, then divided by the token count at the end: averaging
        # per-batch means would weight a short batch the same as a long one.
        loss = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        supervised = shifted_labels != IGNORE_INDEX
        n_supervised = int(supervised.sum().item())
        if n_supervised == 0:
            continue

        predictions = shifted_logits.argmax(dim=-1)
        correct = int(((predictions == shifted_labels) & supervised).sum().item())

        total_loss += float(loss.item())
        total_tokens += n_supervised
        total_correct += correct

    if total_tokens == 0:
        return EvalMetrics(n_examples=len(examples), loss=0.0, perplexity=0.0, token_accuracy=0.0)

    mean_loss = total_loss / total_tokens
    return EvalMetrics(
        n_examples=len(examples),
        loss=round(mean_loss, 6),
        perplexity=round(min(math.exp(mean_loss), MAX_PERPLEXITY), 4),
        token_accuracy=round(total_correct / total_tokens, 6),
    )


@torch.no_grad()
def generate(
    model: CausalLM,
    tokenizer: ByteTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    seed: int | None = None,
) -> str:
    """Generate a continuation. Greedy by default.

    Greedy because a fine-tuning comparison needs the decode to be deterministic: sampling
    makes two evaluations of the same checkpoint disagree, and then the difference between two
    checkpoints is unmeasurable.
    """
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits[:, -1, :]
        if temperature == 0.0:
            next_id = int(logits.argmax(dim=-1).item())
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())

        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

    return tokenizer.decode(generated)
