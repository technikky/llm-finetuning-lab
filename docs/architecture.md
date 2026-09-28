# Architecture

## Data flow

```
data/*.jsonl
     |
     |  load_examples      pydantic validation, one error per line with the field named
     v
  [Example]
     |
     |  prepare            within-split dedup -> cross-split leakage -> tokenize -> stats
     |    templates.py     render() returns (prompt, completion) so the prompt can be masked
     |    tokenizer.py     byte-level, vocab 259
     v
PreparedDataset(train, eval, stats)
     |
     |  train              refuses if stats.n_leaked
     |    specs.py         build_model, seeded from the spec name
     |    lora.py          apply_lora -> adapter attached, zero-initialised
     |    evaluate.py      eval "before" on the held-out split
     |    <optimizer loop> batches -> collate -> shifted loss -> clip -> step -> schedule
     |    evaluate.py      eval "after" on the held-out split
     v
(PeftModel, TrainingRun)
     |                              |
     |  save_adapter                |  tracking.py  RunRegistry: run JSON + JSONL index
     v                              v
adapters/<name>/                 runs/
     |                              |
     |  load_adapter                |  comparable(left, right)
     |  merge_adapter               v
     v                          report.py  markdown and JSON, labelled computed vs measured
plain PreTrainedModel
```

## Module boundaries

| Module | Owns | Depends on |
| --- | --- | --- |
| `types.py` | the data model; `Computed` vs `Measured` as separate types; the `CausalLM` protocol | pydantic only (torch under `TYPE_CHECKING`) |
| `specs.py` | architecture definitions, seeded construction, meta builds | `types`, transformers |
| `budget.py` | closed-form parameter and memory arithmetic | `types`, `specs` |
| `tokenizer.py` | byte-level encode/decode | nothing |
| `templates.py` | prompt/completion rendering per format | `types` |
| `data.py` | fingerprinting, dedup, leakage, tokenization, collation | `types`, `templates`, `tokenizer` |
| `lora.py` | adapter application, merge, invariant predicates | `types`, `specs`, peft |
| `quantization.py` | bnb config, support probing, compression ratios | `types`, torch |
| `train.py` | seeding, config hashing, the optimizer loop, adapter IO | everything above |
| `evaluate.py` | shifted loss, perplexity, accuracy, greedy generation | `data`, `tokenizer`, `types` |
| `tracking.py` | run registry, comparability | `types` |
| `report.py` | markdown and JSON rendering | `types`, `budget`, `specs` |
| `cli.py` | argument parsing, exit codes, stdout/stderr | everything |

The dependency graph is acyclic and one-directional: `types` knows nothing about the rest,
`budget` knows nothing about training, and only `cli` knows about output formatting *and*
execution.

## Three decisions worth explaining

### `types.py` stays torch-free at runtime

It imports torch only under `TYPE_CHECKING`, which keeps the data model — the part that describes
budgets, configs and results — importable and inspectable without pulling in a deep-learning
stack. The `CausalLM` protocol lives there because it is part of the interface, not the
implementation.

### `CausalLM` is a protocol, not a base class

A base `PreTrainedModel` and a LoRA-wrapped `PeftModel` are both valid inputs to `evaluate` and
`generate`, and neither is a subtype of the other — merging an adapter turns the second back into
the first. Typing against the concrete classes would mean a cast at every call site and would
claim a requirement that is not real. The protocol names what is actually needed: switch to
inference mode, and map tensors to logits.

```python
class CausalLM(Protocol):
    def eval(self) -> object: ...
    def __call__(self, **kwargs: torch.Tensor) -> CausalLMOutput: ...
```

### Models are built, not downloaded

`specs.py` holds architecture definitions as data, and `build_model` instantiates from them. Three
consequences:

1. **The test suite is hermetic.** No network, no gated repositories, no tokens, no multi-gigabyte
   cache. Real training, real merging and real evaluation run in CI on every commit.
2. **13B arithmetic is free.** `build_model(spec, on_meta=True)` produces shapes with no storage,
   so the closed-form parameter counts can be cross-checked against a real `transformers` build
   for every spec including the largest.
3. **Initialisation has to be deterministic.** A spec is only useful as a checkpoint if two builds
   agree, or `load_adapter` would put an adapter onto weights it never saw. `init_seed(spec)` is
   a blake2b digest of the name — not `hash()`, which is salted per interpreter — and the global
   RNG state is saved and restored around the build.

## Where the honesty machinery lives

| Guard | Where | What it stops |
| --- | --- | --- |
| cross-split leakage detection | `data.prepare` | reporting memorisation as generalisation |
| `train` refuses a leaking dataset | `train.train` | producing that number at all |
| drop, don't truncate | `data.tokenize_example` | supervising a prompt against a cut-off target |
| double padding mask | `data.collate` | pad tokens in the loss, or attended to |
| logit shift | `evaluate._shift_for_causal_lm` | scoring a model against its own input |
| before *and* after evaluation | `train.train` | a final loss that was already the initial one |
| config hash + environment | `train.config_hash`, `environment_info` | an unreproducible measurement |
| seeded base construction | `specs.build_model` | comparing runs that started from different models |
| `comparable()` warnings | `tracking.comparable` | a side-by-side table of two different experiments |
| computed vs measured types | `types.py` | a guess printed where an exact figure belongs |
| honest capability probing | `quantization.quantization_available` | claiming a 4-bit run that never happened |

## Testing strategy

| File | Tests | Covers |
| --- | --- | --- |
| `test_budget.py` | 66 | the closed forms against `transformers` and `peft`, spec validation, memory arithmetic |
| `test_lora.py` | 12 | the three invariants, on both architectures |
| `test_data.py` | 36 | tokenizer round-trip, every template, masking, dedup, leakage, collation |
| `test_train_and_tracking.py` | 39 | real optimizer steps, adapter IO, evaluation, the registry, quantisation config |
| `test_cli.py` | 31 | every documented command end to end, plus report formatting |

184 tests, 95% statement and branch coverage, well under a minute on CPU. The training tests run
real optimizer steps rather than mocks, which is the only way masking, batching, clipping,
scheduling and the loss path get exercised together.
