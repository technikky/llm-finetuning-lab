# Budgets, derived

Everything here is arithmetic over an architecture definition. No weights, no GPU, no run. It is
worth writing out because the alternative — renting a machine to find out whether something fits
— is slow and expensive, and because a formula that is only ever checked against itself stays
wrong indefinitely.

Both formulas below are cross-checked against what `transformers` and `peft` actually build, in
`tests/test_budget.py`, on every commit.

## Notation

From `ModelSpec`:

| Symbol | Field | Llama-2-7B |
| --- | --- | --- |
| `V` | `vocab_size` | 32,000 |
| `H` | `hidden_size` | 4,096 |
| `I` | `intermediate_size` | 11,008 |
| `L` | `num_hidden_layers` | 32 |
| `A` | `num_attention_heads` | 32 |
| `K` | `num_key_value_heads` | 32 |
| `d` | `head_dim` = `H / A` | 128 |
| `kv` | `K · d` | 4,096 |

`kv` is the width of the key and value projections' output. When `K < A` the model uses
grouped-query attention and `kv < H`. Llama-3-8B and Mistral-7B both have `K = 8`, so
`kv = 8 · 128 = 1,024` against `H = 4,096`.

## Base parameters

Per decoder layer:

```
attention = H·H  (q_proj)
          + H·kv (k_proj)
          + H·kv (v_proj)
          + H·H  (o_proj)
          = 2·H² + 2·H·kv

mlp       = H·I   (gate_proj)
          + H·I   (up_proj)
          + I·H   (down_proj)
          = 3·H·I

norms     = H     (input_layernorm)
          + H     (post_attention_layernorm)
          = 2·H
```

RMSNorm has a weight and no bias, and these architectures have no biases on the linear layers at
all, which is why nothing is added for them.

Whole model:

```
total = L · (2·H² + 2·H·kv + 3·H·I + 2·H)
      + V·H                                  (embed_tokens)
      + H                                    (final norm)
      + V·H         if not tie_word_embeddings   (lm_head)
```

For Llama-2-7B that is **6,738,415,616** — the figure in the Llama 2 paper, and a test asserts it
so that a drifting spec field fails loudly.

`base_parameters` is derived rather than measured so it works for an architecture too large to
instantiate even with shapes only. The verification goes the other way: a real build on the
**meta device** — shapes, no storage — is counted and compared. Building a 13B architecture that
way allocates nothing, which is what makes the cross-check free.

## Adapter parameters

A LoRA adapter on `Linear(in_features, out_features)` at rank `r` adds

```
A : (r, in_features)
B : (out_features, r)
parameters = r · (in_features + out_features)
```

`B` is zero-initialised, which is what makes a fresh adapter a no-op. Per projection:

| Projection | in → out | Parameters |
| --- | --- | --- |
| `q_proj` | `H → H` | `2·r·H` |
| `k_proj` | `H → kv` | `r·(H + kv)` |
| `v_proj` | `H → kv` | `r·(H + kv)` |
| `o_proj` | `H → H` | `2·r·H` |
| `gate_proj` | `H → I` | `r·(H + I)` |
| `up_proj` | `H → I` | `r·(H + I)` |
| `down_proj` | `I → H` | `r·(I + H)` |

Multiply by `L`, since the adapter attaches in every layer.

**Grouped-query attention is why `k` and `v` are cheaper.** Their output dimension is `kv`, not
`H`. At `r = 16` on the attention projections:

| | Llama-2-7B (`kv = 4,096`) | Mistral-7B (`kv = 1,024`) |
| --- | --- | --- |
| `q_proj` | `2·16·4096` = 131,072 | 131,072 |
| `k_proj` | `16·(4096+4096)` = 131,072 | `16·(4096+1024)` = 81,920 |
| `v_proj` | 131,072 | 81,920 |
| `o_proj` | 131,072 | 131,072 |
| per layer | 524,288 | 425,984 |
| × 32 layers | **16,777,216** | **13,631,488** |

Both models have `H = 4,096`, so a formula that ignored KV grouping would report the same number
for both. A test asserts they differ.

Scaling is `alpha / r`, so `alpha` is not a second rank: doubling `r` at fixed `alpha` *halves*
the update scale. `LoraSpec.scaling` exposes it, and a test pins `r=8, alpha=16 → 2.0` and
`r=16, alpha=16 → 1.0`.

## Memory

```
base_weights      = base_parameters      · bytes_per_param(quantization or precision)
adapter_weights   = adapter_parameters   · bytes_per_param(precision)
adapter_gradients = adapter_parameters   · bytes_per_param(precision)
optimizer_state   = adapter_parameters   · n_state_tensors · bytes_per_state
total             = sum of the above
```

`bytes_per_param`:

| | Bytes |
| --- | --- |
| fp32 | 4 |
| fp16, bf16 | 2 |
| int8 | 1 |
| nf4 | 4.5 / 8 = 0.5625 |

The nf4 figure is 4 bits for the weight plus about 0.5 bits of per-block scaling metadata, which
is the QLoRA paper's accounting. Compression against bf16 is therefore `2 / 0.5625 ≈ 3.56x`, not
4x. That gap is ~1 GiB on a 7B model, which is the difference between fitting and not.

Optimizer state, as (tensors per parameter, bytes each):

| Optimizer | Tensors | Bytes | Note |
| --- | --- | --- | --- |
| `adamw` | 2 | 4 | moments in fp32 even for a bf16 model |
| `adamw_8bit` | 2 | 1 | |
| `sgd` | 0 | 0 | no state at all |
| `sgd_momentum` | 1 | 4 | |

AdamW's moments stay fp32 because fp16 moments lose the small updates that are the reason for
keeping moments.

### Gradients are per *trainable* parameter

This is the whole point of the technique. A full fine-tune needs a gradient and optimizer state
for every parameter:

```
full_finetune = base_parameters · (bytes_per_param + bytes_per_param + optimizer_bytes)
```

For Llama-2-7B in bf16 with AdamW: 12.55 + 12.55 + 50.21 = **75.31 GiB**, against **12.80 GiB**
for LoRA at `r = 16` — a 5.9x difference that comes almost entirely from the optimizer.

### What is excluded

Activations, the KV cache, the CUDA context, allocator fragmentation, and anything a distributed
strategy adds.

Activation memory is excluded rather than estimated because it depends on sequence length, batch
size, gradient checkpointing and the attention implementation; a single number would be wrong by
a factor of two for a plausible configuration while looking authoritative. `MemoryBudget.excludes`
says so, every report prints it, and `fits_within` reserves 15% headroom by default:

```python
fits_within(budget, vram_gb) == budget.total_gb <= vram_gb * (1 - headroom)
```

## Worked answers

```bash
$ ftlab budget --model llama-2-7b --quantization nf4 --vram 24
... 3.78 GiB ... fits
$ ftlab budget --model llama-2-13b --precision bf16 --vram 24
... 24.63 GiB ... does NOT fit
```

A 13B in bf16 does not fit a 24 GiB card even under LoRA, because the *base weights* alone are
24.24 GiB. Quantise it and it does. That is a decision made by arithmetic in under a second
instead of by an out-of-memory error twenty minutes into a rented hour.
