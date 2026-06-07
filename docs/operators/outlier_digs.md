---
title: Outlier mechanism digs
---

# Outlier mechanism digs

Targeted follow-ups on the two recurring outliers the cross-model dossier surfaced. Provisional.

## Why is Llama-3.2-1B's MLP0 context-determined?

MLP0's input is the token embedding **plus** the layer-0 attention write. If the early attention injects a large, context-determined component, MLP0 processes a context-mixed residual and its output is no longer token-determined. Per model: the L0-attention write size relative to the embedding, and the token-determinism (η²) of the embedding, the L0 attention output, and MLP0's output.

| model | ‖L0 attn‖ / ‖embedding‖ | η² embedding | η² L0 attn-out | η² MLP0-out |
|---|---|---|---|---|
| gpt2 | 6.60 | 0.61 | 0.64 | 0.63 |
| gpt2-large | 3.14 | 0.66 | 0.72 | 0.74 |
| gemma-2-2b | 0.18 | 1.00 | 0.69 | 0.91 |
| Qwen2.5-1.5B | 11.88 | 1.00 | 0.84 | 0.64 |
| Llama-3.2-1B | 0.90 | 1.00 | 0.49 | -0.14 |

_Reading: it is the **determinism of the L0 attention output**, not its size, that distinguishes Llama. Qwen's L0 attention is far *larger* (≈12× the embedding) yet token-determined (η² 0.84), so its MLP0 stays token-ish; Gemma's L0 attention is tiny (0.18×) so MLP0 ≈ the pure embedding (0.91). **Llama's L0 attention is both comparable in size to the embedding and the most context-determined (η² 0.49)** — its layer-0 induction-mass head cluster does genuine context-mixing — so MLP0 ingests a large context-laden component and its output carries ~no token-determinism (η²≈0; small negatives are estimation noise). The context-dependence is inherited from the early heads, not intrinsic to MLP0. (GPT-2's embedding η²<1 is its absolute positional embedding adding position variance to the residual; the RoPE models read 1.00 — no positional component in the residual.)_

## Compensatory induction: which head triggers the recovery?

For each model's top-k induction heads (by induction-mass), the **leave-one-out marginal** of head h = effect(ablate-all) − effect(ablate-all-except-h). A **negative** marginal means ablating h *reduces* induction damage — a net suppressor / self-repair trigger; a distributed op has all-positive marginals.

| model | induction top-k | full ΔNLL | most-negative LOO marginal (head) | distributed? |
|---|---|---|---|---|
| gpt2 | `5.1`, `6.9`, `5.5`, `7.10`, `7.2` | +5.27 | +0.44 (none<0) | yes (all marginals ≥0) |
| gpt2-large | `16.0`, `19.4`, `16.9`, `18.7`, `20.1` | +0.84 | -0.66 (`16.0`) | no — has a suppressor |
| gemma-2-2b | `6.3`, `6.2`, `22.3`, `4.4`, `22.4` | +3.77 | -1.10 (`4.4`) | no — has a suppressor |

_Per-head solo + LOO marginals are in the JSON. The suppressor head is the one whose removal lets a backup carry induction (the non-monotonic cumulative curve in the [operator catalog](induction.md))._

_Data: [outlier_digs_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/outlier_digs_summary.json). Regenerate: [outlier_dig.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/outlier_dig.py)._