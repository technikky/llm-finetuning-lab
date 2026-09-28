# llm-finetuning-lab

LoRA and QLoRA fine-tuning where the arithmetic is exact, the invariants are tested, and every
published number came from a command in this README.

[![CI](https://github.com/technikky/llm-finetuning-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/technikky/llm-finetuning-lab/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/technikky/llm-finetuning-lab)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why this exists

Most fine-tuning write-ups report a loss curve and a vibe. Two things go wrong, and both are
avoidable:

**The budget is guessed.** "Can I fine-tune a 13B on a 24 GB card?" is not a question that needs
an experiment. Parameter counts and memory footprints follow from the architecture and the LoRA
setting by arithmetic. This repository computes them in closed form and cross-checks every
formula against what `transformers` and `peft` actually build — 7 architectures and 48
rank/target combinations, in CI, on every commit.

**The invariants are assumed.** Three properties everyone relies on and nobody checks: that a
freshly attached adapter changes nothing, that only the adapter trains, and that merging the
adapter is exact. If the first is false, every "before" measurement is of the wrong model. If
the third drifts, the weights you deploy are not the weights you evaluated. Each is cheap to
verify and expensive to have wrong, so each is a test.

Separating the two kinds of claim is the organising idea. Parameter and memory numbers are
**computed** — exact, hardware-free, reproducible by anyone with a calculator. Loss curves and
perplexities are **measured** — a specific run, on specific hardware, with a config hash. The
types are separate, the reports label them separately, and nothing in this README blurs them.

---

## Quickstart

```bash
pip install -e ".[dev]"
```

No downloads, no tokens, no GPU. Every model here is built from an architecture definition with
seeded random initialisation, which is exactly what makes real training runs affordable as
tests.

```bash
# What can I fine-tune, and what will it cost?
ftlab budget --r 16 --target attention

# Will a 7B in 4-bit fit my 24 GB card?
ftlab budget --model llama-2-7b --quantization nf4 --vram 24

# Is my dataset sound before I spend anything on it?
ftlab prepare data/instructions.jsonl --template alpaca

# Fine-tune, evaluate held-out, record the run
ftlab train data/instructions.jsonl --model tiny-llama --r 16 --epochs 4 \
  --template alpaca --lr 3e-3 --adapter-out adapters/tiny-r16

# Compare what you have run
ftlab runs
ftlab compare
```

---

## Computed: what fine-tuning costs

`ftlab budget --r 16 --target attention` — no weights loaded, no GPU touched:

| Model | Base parameters | Trainable | % of base |
| --- | --- | --- | --- |
| `llama-2-7b` | 6.738B | 16.78M | 0.2490% |
| `llama-2-13b` | 13.016B | 26.21M | 0.2014% |
| `llama-3-8b` | 8.030B | 13.63M | 0.1698% |
| `mistral-7b` | 7.242B | 13.63M | 0.1882% |

Llama-3-8B and Mistral-7B have the same hidden size as Llama-2-7B but adapt *fewer* parameters,
because grouped-query attention makes their `k_proj` and `v_proj` outputs narrower. A formula
that ignored KV grouping would give all three the same count. That distinction is a test.

Memory, bf16 weights with AdamW:

| Model | Base | Adapter | Grads | Optimizer | **Total** | Full fine-tune | Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `llama-2-7b` | 12.55 | 0.062 | 0.062 | 0.125 | **12.80 GiB** | 75.31 GiB | 5.9x |
| `llama-2-13b` | 24.24 | 0.098 | 0.098 | 0.195 | **24.63 GiB** | 145.46 GiB | 5.9x |
| `llama-3-8b` | 14.96 | 0.051 | 0.051 | 0.102 | **15.16 GiB** | 89.75 GiB | 5.9x |
| `mistral-7b` | 13.49 | 0.051 | 0.051 | 0.102 | **13.69 GiB** | 80.93 GiB | 5.9x |

**Activation memory is excluded, not estimated.** It depends on sequence length, batch size,
whether gradient checkpointing is on, and the attention implementation. An estimate would look
authoritative while being wrong by a factor of two, so the budget reserves 15% headroom when
answering "does it fit" and says what it left out.

With that headroom, `ftlab budget --model llama-2-7b --quantization nf4 --vram 24` reports
**3.78 GiB total, fits** — and `--model llama-2-13b --precision bf16 --vram 24` reports
**24.63 GiB, does NOT fit**. That is the decision the tool exists to inform.

### nf4 is 3.56x, not 4x

"4-bit" quantisation stores per-block scaling metadata alongside the weights, which the QLoRA
paper puts at roughly 0.5 bits per parameter. So base weights compress by 2 / 0.5625 ≈ **3.56x**
against bf16, not 4x. Rounding that up is a small lie that becomes an out-of-memory error at the
top of a 24 GB card, so it is counted.

Quantising shrinks the base term only. The adapter is what is being trained, so QLoRA leaves the
adapter, its gradients and its optimizer state exactly where bf16 LoRA had them.

---

## Measured: a real run

Every number below came from `ftlab train` on this machine, CPU-only. The model is
`tiny-llama` — a 657k-parameter architecture built from a config, not a checkpoint — so **these
losses say nothing about model quality.** They are here to show that the machinery measures
something, that the measurement moves, and that it reproduces.

```bash
ftlab train data/instructions.jsonl --model tiny-llama --r 16 --target attention \
  --template alpaca --max-length 512 --epochs 4 --batch-size 4 --lr 3e-3
```

| Metric | Before | After | Change |
| --- | --- | --- | --- |
| Held-out loss | 5.5745 | 4.9958 | -0.5787 |
| Held-out perplexity | 263.6200 | 147.7881 | -115.8319 |
| Held-out token accuracy | 0.0015 | 0.2023 | +0.2009 |

48 steps, 1.82 s wall on CPU. torch 2.14.0+cpu, transformers 5.17.0, peft 0.21.0.
Config hash `0c77b649e7b93b2c`.

**Before and after, not just after.** A final loss on its own says nothing: it could be the
value an untrained model already had. The pair is the measurement, and the "before" is taken on
the same held-out split with the adapter attached and zero-initialised.

**Rank sweep**, same data, same seed, same everything else:

| r | Trainable | Final train loss | Eval ppl before | after | Change |
| --- | --- | --- | --- | --- | --- |
| 4 | 14.3k | 5.0581 | 263.6200 | 155.2226 | -108.3974 |
| 16 | 57.3k | 5.0259 | 263.6200 | 147.7881 | -115.8319 |
| 64 | 229.4k | 4.9866 | 263.6200 | 141.1987 | -122.4213 |

The "before" column being identical across all three runs is not a coincidence and not a copy
error — it is the point of seeding base initialisation from the spec (see below). Without it,
three runs would start from three different models and the comparison would be noise.

**The adapter round-trips.** The r=16 checkpoint is 233,504 bytes of weights against the base
model's 2,629,120 — and reloading it onto a base rebuilt from the spec reproduces the run's
held-out number exactly, merged or not:

```
$ ftlab evaluate data/instructions.jsonl --model tiny-llama \
    --template alpaca --max-length 512 --adapter adapters/tiny-r16
perplexity        : 147.7881      # matches the run above
merged            : False

$ ftlab evaluate ... --adapter adapters/tiny-r16 --merge
perplexity        : 147.7881
merged            : True
```

That equality is the payoff of the third invariant. If base initialisation were not seeded from
the spec, the adapter would land on different weights and this number would quietly differ.

**Benchmark on a real published checkpoint: pending.** Running Llama-2-7B needs hardware this
was not developed on. The budget arithmetic for it is exact and cross-checked against
`transformers`; the loss numbers are not claimed.

---

## The three invariants

Reproduce all of them with `pytest tests/test_lora.py -v`. The values below are from this
machine:

**1. A freshly attached adapter is a bit-exact no-op.**
`lora_B` is zero-initialised, so `B @ A = 0` and the adapted forward pass adds a literal zero.

```
max |logit difference| base vs adapted : 0.0        (exactly, not approximately)
lora_B is zero                         : True
base parameters frozen                 : True
```

Zero is the right assertion here, not `allclose`. A non-zero difference means the adapter was
attached to something other than what was intended, and every number downstream would describe
a different model. There is also a test that runs a real backward pass and asserts no base
parameter received a gradient — a trainable-parameter count cannot catch a leaked gradient.

**2. The merge equals the definition.** Checked against `W + (alpha/r) · B @ A` written out
independently, not against whatever `peft` computed:

```
max |W_merged - (W + s·B@A)|            : 0.0
max |logits adapted - logits merged|    : 6.6e-07   (float, not bit-exact: the merge reorders ops)
parameters base / adapted / merged      : 657,280 / 714,624 / 657,280
```

The parameter count returning exactly to the base is why LoRA is free at inference: after
merging there is no adapter, no extra matmul and no runtime dependency on `peft`.

**3. A rebuilt base is the same base.**

```
max |logit difference| build vs rebuild : 0.0
```

None of the tiny specs has published weights — they are architectures, built fresh. But an
adapter is only meaningful against one specific base. If two builds of `tiny-llama` differed,
`ftlab evaluate --adapter` would load an adapter onto weights it was never trained against and
print a perplexity as though it meant something. So initialisation is seeded from the spec name
via blake2b (not `hash()`, which is salted per interpreter), and the global RNG state is
restored afterwards so building a model does not silently reposition a caller's seeded stream.

---

## Honesty machinery

These are the parts that exist to stop the repository from producing a number that looks good
and means nothing.

**Leakage is named, not fixed.** `data/leaky.jsonl` is deliberately broken: a whitespace-and-case
near-duplicate inside the train split, and one example (`l05`) that appears in both train and
eval. An earlier version of `prepare`
deduplicated globally *before* splitting, which silently deleted the eval copy and reported zero
leakage — "fixing" the serious problem by hiding it. Deduplication now happens *within* each
declared split; an overlap *across* splits is leakage, is counted, is listed by id, and
`ftlab train` refuses to run:

```
$ ftlab prepare data/leaky.jsonl
error: 1 evaluation example(s) also appear in training: l05
```

A result from a leaking split measures memorisation, so the useful behaviour is to stop.

**Over-long examples are dropped, not truncated.** Truncating an instruction example cuts off
the completion, leaving a prompt supervised against nothing. That trains the model on a
malformed target while the example count stays reassuringly unchanged. Dropped examples are
counted and reported (`Dropped: over max_length | 1` on the bundled data at 512 tokens with the
alpaca template).

**Perplexity is over completion tokens only.** Prompt positions are masked with
`IGNORE_INDEX = -100`. Perplexity computed over the prompt as well is dominated by how
predictable the template boilerplate is, which is not what fine-tuning changed.

**Logits are shifted before scoring.** A causal model's logits at position `i` predict token
`i + 1`. Comparing them unshifted scores the model against its own input and reports an accuracy
that looks excellent and means nothing.

**Padding is masked twice.** Pad positions get both the attention mask and `IGNORE_INDEX`.
Either one alone is a quiet bug: the first lets the model attend to padding, the second lets
padding into the loss.

**Runs that are not comparable say so.** `ftlab compare` warns when two runs differ in model,
template, dataset or max length — those are not an ablation of the hyperparameter that also
changed, and a table that presents them side by side is misleading.

**Determinism is a test, not a hope.** Same seed, identical loss curve and identical config
hash; different seed, a different curve. Without that, no two runs can be compared and no
ablation means anything.

**4-bit support is probed, not assumed.** `ftlab quantization` reports what this machine can
actually do. On the CPU-only machine these numbers came from:

```
bitsandbytes installed : False
CUDA available         : False
4-bit usable           : False
reason                 : bitsandbytes is not installed; no CUDA device is available
```

The nf4 memory arithmetic is still exact — it is arithmetic — but no 4-bit run is claimed here.

---

## What is in the box

```
src/ftlab/
  types.py         Computed vs measured, as separate types. The CausalLM protocol.
  specs.py         Architecture definitions; seeded model construction; meta-device builds.
  budget.py        Closed-form parameter and memory arithmetic.
  tokenizer.py     Byte-level tokenizer: lossless, offline, vocab 259.
  templates.py     alpaca / chatml / llama2-chat / raw, each returning prompt and completion
                   separately so the prompt can be masked.
  data.py          Fingerprinting, within-split dedup, cross-split leakage, collation.
  lora.py          Adapter application, the merge, and the invariant checks.
  quantization.py  nf4 / int8 config, honest support probing, real compression ratios.
  train.py         Seeding, config hashing, warmup-then-decay, gradient clipping.
  evaluate.py      Shifted loss, perplexity, token accuracy, greedy generation.
  tracking.py      Run registry and comparability checks.
  report.py        Markdown and JSON reports that label their own provenance.
  cli.py           specs / budget / prepare / train / evaluate / generate / runs / compare /
                   quantization
```

- [docs/methodology.md](docs/methodology.md) — what is computed, what is measured, how each is
  produced, and what is not claimed.
- [docs/budgets.md](docs/budgets.md) — the arithmetic, derived, with the per-projection shapes.
- [docs/architecture.md](docs/architecture.md) — module boundaries and the data flow.

---

## Tests

```bash
pytest -q                 # 184 tests
ruff check . && ruff format --check .
mypy                      # strict
```

Real training, real merging and real evaluation run in CI on every commit, on Python 3.10, 3.11
and 3.12. That is possible because the models are 107k and 657k parameters: the whole suite,
including several complete fine-tuning runs, finishes in well under a minute on CPU. Coverage is
95% of statements and branches.

---

## Limitations

Stated plainly, because they bound what the numbers above mean.

- **No published-checkpoint results.** Loss and perplexity come from tiny randomly initialised
  architectures. They demonstrate correct machinery, not model quality.
- **No 4-bit run.** nf4 arithmetic is exact; `bitsandbytes` needs CUDA, which was unavailable.
- **Byte-level tokenization.** Lossless and offline, which is what makes the tests hermetic, but
  it produces far more tokens per unit of text than BPE. Token-count statistics from this
  tokenizer are not comparable to a real model's.
- **Activation memory is out of scope.** See above; the budget reserves headroom instead.
- **Single-device only.** No FSDP, no DeepSpeed, no tensor parallelism, and the memory model
  does not describe them.
- **No instruction-following evaluation.** Perplexity and token accuracy are proxies. Judging
  whether a fine-tune actually follows instructions better needs a rubric and a judge — that is
  [llm-evaluation-framework](https://github.com/technikky/llm-evaluation-framework), not this
  repository.

---

## License

MIT. See [LICENSE](LICENSE).
