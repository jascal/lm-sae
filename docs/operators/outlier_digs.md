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

_Per-head solo + LOO marginals are in the JSON. The suppressor head is the one whose removal lets a backup carry induction (the non-monotonic cumulative curve in the [operator catalog](induction.md)). **Caveat (see the next section):** the apparent suppression is largely a *synthetic repeated-random probe artifact* — these heads have positive OV and are ~neutral on natural-text induction._

## Is the suppressor a genuine negative head, or a synthetic-probe artifact?

For each identified suppressor (+ the workhorse for contrast): the **OV copy-score** sign (+ve copies the attended token → a real copy/induction head; −ve suppresses it → a copy-suppression / negative head), and the head's ablation effect on induction measured over **natural**-repeated text (a real passage + itself) vs **synthetic**-repeated (random tokens + itself). A probe artifact would help natural induction (ablation ΔNLL > 0) but hurt synthetic (< 0); a genuine suppressor is −ve OV and helps both (ablation ΔNLL < 0).

| model | head | role | OV copy-score | ablate ΔNLL natural | ablate ΔNLL synthetic |
|---|---|---|---|---|---|
| gemma-2-2b | `4.4` | suppressor | +0.036 | -0.04 | -0.60 |
| gemma-2-2b | `22.4` | workhorse (contrast) | +0.068 | +0.07 | +2.95 |
| gpt2-large | `16.0` | suppressor | +0.087 | +0.11 | -0.01 |
| gpt2-large | `16.9` | workhorse (contrast) | +0.039 | +0.33 | +0.70 |

_+ve ablation ΔNLL = the head HELPS induction (removing it hurts); −ve = the head SUPPRESSES it (removing it helps). **Finding:** both suppressors have **positive** OV copy-scores — they are copy/induction heads, **not** copy-suppression / negative heads. The suppression shows up **only on the synthetic repeated-random probe** (Gemma 4.4: ΔNLL synthetic −0.60 but natural ≈0; gpt2-large 16.0 likewise marginal/positive on natural). So the *compensatory* redundancy is substantially a **repeated-random probe artifact** — these heads interfere with the degenerate synthetic-induction task but are ~neutral on real-text induction — not a genuine negative-head self-repair mechanism._

## Do Llama's layer-0 heads do single-layer (RoPE-enabled) induction?

GPT-2 needs a two-layer chain (a prev-token head feeds an induction head's key). Llama has induction-load-bearing heads at **layer 0** — where there is no prior layer to supply a prev-token signal. RoPE puts relative position in the key, so a single head can match *token-after-previous-occurrence* directly. If these heads carry **induction-mass ≫ duplicate-mass** and a **+ve OV copy-score**, they are single-layer inductors (no upstream writer needed).

| model | head | induction-mass | duplicate-mass | OV copy-score | single-layer inductor? |
|---|---|---|---|---|---|
| Llama-3.2-1B | `0.31` | 0.034 | 0.021 | +0.023 | **no** — enabler, not inductor |
| Llama-3.2-1B | `0.29` | 0.025 | 0.014 | -0.060 | **no** — enabler, not inductor |
| Llama-3.2-1B | `0.13` | 0.025 | 0.028 | -0.005 | **no** — enabler, not inductor |
| Llama-3.2-1B | `0.14` | 0.030 | 0.045 | -0.009 | **no** — enabler, not inductor |
| Llama-3.2-1B | `1.31` | 0.033 | 0.033 | -0.077 | **no** — enabler, not inductor |
| Llama-3.2-1B | `1.29` | 0.026 | 0.025 | +0.017 | **no** — enabler, not inductor |

_**Finding (hypothesis not supported):** these layer-0 heads do **not** behave as single-layer inductors — their induction-mass is weak (~0.03) and ≈ their duplicate-mass, even though 0.31 is strongly induction-*causal* (+7.99 when ablated, per the [discovered candidates](discovered_xmodel.md)). So they are induction **enablers**, not inductors: they don't attend induction-style themselves, but their early context-mixing (Dig 1 — Llama's L0 attention is the most context-determined) sets up the residual that later heads read. Llama's actual induction *reader* is a later head (10.23 in the [dossier](induction.md)). A clean reminder that high causal effect ≠ doing the named operation._

_Data: [outlier_digs_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/outlier_digs_summary.json). Regenerate: [outlier_dig.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/outlier_dig.py)._