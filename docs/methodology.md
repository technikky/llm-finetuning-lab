# Methodology

What this repository claims, how each claim is produced, and what it does not claim.

## Two kinds of number

Every number here is one of two kinds, and conflating them is how a fine-tuning claim becomes
unfalsifiable.

**Computed.** Parameter counts and memory footprints. These follow from an architecture
definition and a LoRA setting by arithmetic — no weights, no GPU, no run, no uncertainty. They
are the same on any machine and reproducible with a calculator. `ParameterBudget` and
`MemoryBudget` carry them, and every report that prints them says "Computed, not measured".

**Measured.** Loss curves, perplexities, token accuracies, wall times. These describe one run of
one configuration on one machine. `TrainingRun` carries them together with the config hash, the
library versions and the device, because without those the number is not reproducible and
therefore not a measurement.

The types are separate on purpose. It is not possible to accidentally print a measured loss in a
computed-budget table.

## Computed: how the arithmetic is checked

A closed form nobody checks against the library is a formula that was right once. Both formulas
are cross-checked in `tests/test_budget.py` against what `transformers` and `peft` actually
build:

- `base_parameters(spec)` is compared with `count_parameters(build_model(spec))` for all 7
  specs, including the 13B one. This is affordable because the comparison model is built on the
  **meta device** — shapes but no storage — so counting a 13-billion-parameter architecture
  allocates nothing.
- `adapter_parameters(spec, lora)` is compared with `count_trainable(get_peft_model(...))` across
  4 architectures × 3 targets × 3 ranks. The returned module list is compared too, so a formula
  cannot be right by adapting the wrong set of projections.
- `base_parameters(get_spec("llama-2-7b")) == 6_738_415_616`, the figure in the Llama 2 paper. If
  a spec field drifts, this fails.

See [budgets.md](budgets.md) for the derivation.

### What the memory model includes

Weights, gradients and optimizer state, each at an explicit precision. Nothing else.

**Activation memory is excluded deliberately.** It depends on sequence length, batch size,
whether gradient checkpointing is enabled, and the attention implementation. Any single number
would look authoritative while being wrong by a factor of two for a plausible configuration. So
`MemoryBudget` carries an `excludes` string that the reports print, and `fits_within` reserves
15% headroom by default rather than pretending the excluded terms are zero.

Also excluded: the CUDA context, allocator fragmentation, the KV cache during generation, and
anything a distributed strategy adds. This is a single-device model.

### Precisions

`BYTES_PER_PARAM` gives fp32 4, fp16 and bf16 2, int8 1, and nf4 `4.5 / 8`. The nf4 figure is
4 bits for the weight plus roughly 0.5 bits of per-block scaling metadata, which is the QLoRA
paper's accounting. That is why compression against bf16 is 2 / 0.5625 ≈ 3.56x rather than the
4x the name suggests — a difference that matters at the top of a card.

AdamW keeps two moment tensors and keeps them in fp32 even for a bf16 model, because fp16
moments lose the small updates that are the reason for keeping moments at all. `adamw_8bit` keeps
two at one byte. `sgd` keeps none, which is why it has no optimizer-state term at all.

## Measured: how a run is produced

### The dataset comes first

`ftlab prepare` runs before anything is spent on compute, and reports:

- examples in, and how many survived
- dropped as empty, as duplicates, and as over `max_length`
- token length mean, p50, p95, max
- **train/eval leakage**, with the leaking ids

**Deduplication is within a split. Overlap across splits is leakage.** This distinction was a bug
before it was a feature. An earlier `prepare` deduplicated globally and *then* split, which
deleted the eval copy of a leaking example and reported zero leakage — hiding the serious problem
under a cosmetic fix. Now each declared split is deduplicated on its own, and a fingerprint
present in both is counted, listed and refused: `train` raises rather than producing a number
that measures memorisation.

Fingerprints are blake2b over normalised instruction + input + output. Not Python's `hash()`,
which is salted per interpreter and would make leakage detection depend on which process ran it.

**Over-long examples are dropped, not truncated.** Truncating an instruction example removes the
end of the completion, which leaves a supervised prompt with nothing correct to predict. The
example count would stay reassuringly unchanged while the target became malformed. Drops are
counted and reported.

### Masking

Templates return the prompt and the completion separately — `render()` gives a 2-tuple — so the
prompt can be masked with `IGNORE_INDEX = -100` rather than recovered by string surgery
afterwards.

Perplexity is therefore over **completion tokens only**. Perplexity computed over the prompt as
well is dominated by how predictable the template boilerplate is, which is not what fine-tuning
changed.

Padding is masked twice: the attention mask *and* `IGNORE_INDEX`. Either alone is a quiet bug —
the first lets the model attend to pad tokens, the second lets pad tokens into the loss.

### Scoring

Logits are shifted before comparison: a causal model's logits at position `i` predict token
`i + 1`. Unshifted comparison scores the model against its own input and reports an accuracy that
looks excellent and means nothing.

Loss is the summed cross-entropy over supervised positions divided by the number of supervised
positions, not the mean of per-batch means, which would weight a short final batch as heavily as
a full one.

Perplexity is `exp(loss)`, capped at 1e6. An untrained model on any real vocabulary overflows;
the cap is documented rather than silently applied.

Token accuracy is reported alongside perplexity because it moves earlier on short runs. A run
whose accuracy sits at chance has not started learning, and that is visible well before
perplexity says anything.

### Before *and* after

Every run evaluates the held-out split twice: once with the adapter attached and
zero-initialised, once after training. A final loss on its own could be the value the model
already had. The pair is the measurement.

Because `lora_B` is zero at initialisation, the "before" number is the base model's — measured,
not assumed, on the same split with the same code path.

### Reproducibility

- `set_seed` seeds Python, torch and CUDA.
- `config_hash` is a blake2b digest of the full training config. Two runs with the same hash
  should produce the same curve, and there is a test that they do.
- Base model initialisation is seeded from the spec name (blake2b again). None of the tiny specs
  has published weights, so without this an adapter would be loaded onto weights it was never
  trained against. The global RNG state is saved and restored around the build so that seeding a
  model does not silently advance a caller's stream.
- `environment_info` records torch, transformers, peft, Python and device in every run.
- `RunRegistry` stores each run as JSON plus a JSONL index line.
- `comparable(left, right)` reports whether two runs differ in model, template, dataset
  fingerprint or max length. `ftlab compare` prints the warning. Two runs that differ in the
  model *and* the rank are not an ablation of the rank.

## The invariants

`tests/test_lora.py`. Each is cheap to check and expensive to have wrong.

1. **Identity at initialisation.** `lora_B` is zero, so `B @ A = 0` and the adapted forward pass
   adds a literal zero. Asserted as exactly `0.0`, not `allclose`: a non-zero difference means
   the adapter was attached to something other than intended, and every downstream number
   describes a different model than the report claims.
2. **Only the adapter trains.** Checked two ways — the trainable-parameter names are all
   `lora_*`, and a real backward pass leaves every base parameter with `grad is None`. A count
   alone cannot catch a leaked gradient.
3. **The merge is exact.** `W_merged` is compared against `W + (alpha/r) · B @ A` written out
   independently in `expected_delta`, not against whatever `peft` computed. Logits agree to
   float tolerance rather than bit-exactly, because merging changes the order of operations. The
   parameter count returns exactly to the base, which is what makes LoRA free at inference.

Both architectures are exercised, so the projection names cannot be silently Llama-specific.

## Quantisation

`quantization_available()` probes for `bitsandbytes` and CUDA and returns a reason when 4-bit is
unusable, rather than failing at the point of use. `bnb_config_kwargs` builds the configuration
dictionary without importing anything, so the config can be tested on a machine that cannot run
it — and `build_bnb_config` raises with the install instruction when actually called.

The nf4 memory arithmetic is exact regardless, because it is arithmetic. **No 4-bit training run
is claimed here**; the machine these numbers came from has no CUDA device.

## What is not claimed

- **No published-checkpoint results.** The measured losses come from 107k- and 657k-parameter
  architectures built from configs. They demonstrate that the machinery measures something real
  and reproduces; they say nothing about model quality.
- **No 4-bit run.** See above.
- **No instruction-following evaluation.** Perplexity and token accuracy are proxies for the
  thing anyone actually wants to know. Judging whether a fine-tune follows instructions better
  needs a rubric and a judge, which is a different repository.
- **Token statistics are not comparable to a real model's.** The byte-level tokenizer is
  lossless and needs no download, which is what makes the tests hermetic, but it produces far
  more tokens per unit of text than BPE.
- **Single device only.** No FSDP, no DeepSpeed, no tensor parallelism, and the memory model does
  not describe them.
