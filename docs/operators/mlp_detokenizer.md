---
title: MLP extended-embedding test
---

# Is the early MLP an "extended embedding" / detokenizer?

MLP0/MLP1 are the single most causally load-bearing components in every model ([discovered components](discovered.md)), with **mechanism unverified**. The canonical reading is that the early MLP's output is largely a function of the **current token identity** (an *extended embedding* / detokenizer) rather than the broader context. Two measurements per MLP layer:

1. **Token-determinism** (clean, no entropy confound) — the fraction of the layer's output variance explained by the current token identity (η²: 1 − within-token var / total var, over frequent tokens). **≈1 = token-determined (embedding-like); ≈0 = context-determined.** The extended-embedding claim predicts high at MLP0, decaying with depth.

2. **Category-split ablation** (supporting) — mean-ablate the layer, next-token-NLL damage split by the **target** token's category, shown *with the per-category baseline NLL*. Word-starts are inherently higher-entropy, so read ΔNLL **relative to its baseline**, not in absolute terms.

Provisional, single corpus (Shakespeare prose).

## gpt2 (GPT-2/absolute, 12 layers)

Target-token mix: word-start 50%, continuation 34%, other 16%. Baseline NLL by category: word-start 5.62, continuation 4.90, other 2.64. MLP0 token-determinism **0.63**; across probed layers it decays with depth.

| layer | depth | **token-determinism** | ΔNLL word-start | ΔNLL continuation | ΔNLL other |
|---|---|---|---|---|---|
| 0 | 0.00 | **0.63** | +2.167 | +1.656 | +1.667 |
| 1 | 0.09 | **0.07** | +0.068 | +0.136 | +0.024 |
| 2 | 0.18 | **0.04** | +0.097 | +0.138 | +0.031 |
| 6 | 0.55 | **0.30** | -0.017 | -0.881 | +0.038 |
| 10 | 0.91 | **0.38** | +0.057 | -0.670 | +0.014 |

## gpt2-medium (GPT-2/absolute, 24 layers)

Target-token mix: word-start 50%, continuation 34%, other 16%. Baseline NLL by category: word-start 5.37, continuation 4.24, other 2.48. MLP0 token-determinism **0.61**; across probed layers it is not monotonic.

| layer | depth | **token-determinism** | ΔNLL word-start | ΔNLL continuation | ΔNLL other |
|---|---|---|---|---|---|
| 0 | 0.00 | **0.61** | +7.667 | +9.574 | +3.295 |
| 1 | 0.04 | **0.68** | +0.022 | +0.273 | +0.020 |
| 2 | 0.09 | **0.18** | +0.038 | +0.026 | +0.009 |
| 12 | 0.52 | **0.30** | +0.014 | -0.403 | -0.016 |
| 22 | 0.96 | **0.12** | +0.036 | +0.052 | +0.007 |

## gpt2-large (GPT-2/absolute, 36 layers)

Target-token mix: word-start 50%, continuation 34%, other 16%. Baseline NLL by category: word-start 5.28, continuation 4.55, other 2.36. MLP0 token-determinism **0.75**; across probed layers it decays with depth.

| layer | depth | **token-determinism** | ΔNLL word-start | ΔNLL continuation | ΔNLL other |
|---|---|---|---|---|---|
| 0 | 0.00 | **0.75** | +4.171 | +4.735 | +2.439 |
| 1 | 0.03 | **0.68** | +0.005 | +0.037 | -0.012 |
| 2 | 0.06 | **0.68** | -0.001 | +0.013 | -0.013 |
| 18 | 0.51 | **0.24** | +0.015 | -0.047 | +0.001 |
| 34 | 0.97 | **0.34** | +0.046 | -0.390 | -0.001 |

## gemma-2-2b (RoPE, 26 layers)

Target-token mix: word-start 52%, continuation 20%, other 28%. Baseline NLL by category: word-start 8.46, continuation 7.83, other 3.46. MLP0 token-determinism **0.91**; across probed layers it decays with depth.

| layer | depth | **token-determinism** | ΔNLL word-start | ΔNLL continuation | ΔNLL other |
|---|---|---|---|---|---|
| 0 | 0.00 | **0.91** | +0.304 | +1.140 | -0.075 |
| 1 | 0.04 | **0.63** | +0.016 | -0.030 | -0.048 |
| 2 | 0.08 | **0.56** | +0.318 | +0.536 | +0.076 |
| 13 | 0.52 | **0.16** | +0.033 | +0.253 | +0.235 |
| 24 | 0.96 | **0.46** | +0.263 | +0.976 | +0.245 |

## Llama-3.2-1B (RoPE, 16 layers)

Target-token mix: word-start 54%, continuation 28%, other 18%. Baseline NLL by category: word-start 4.77, continuation 2.90, other 2.82. MLP0 token-determinism **0.01**; across probed layers it is not monotonic.

| layer | depth | **token-determinism** | ΔNLL word-start | ΔNLL continuation | ΔNLL other |
|---|---|---|---|---|---|
| 0 | 0.00 | **0.01** | +3.998 | +6.719 | +3.888 |
| 1 | 0.07 | **-0.02** | +3.337 | +3.479 | +3.639 |
| 2 | 0.13 | **0.42** | +0.222 | +0.156 | +0.100 |
| 8 | 0.53 | **0.34** | +0.166 | +0.255 | +0.076 |
| 14 | 0.93 | **0.43** | +0.161 | +0.625 | +0.165 |

## Qwen2.5-1.5B (RoPE, 28 layers)

Target-token mix: word-start 54%, continuation 28%, other 18%. Baseline NLL by category: word-start 4.63, continuation 2.82, other 2.58. MLP0 token-determinism **0.65**; across probed layers it decays with depth.

| layer | depth | **token-determinism** | ΔNLL word-start | ΔNLL continuation | ΔNLL other |
|---|---|---|---|---|---|
| 0 | 0.00 | **0.65** | +3.040 | +3.990 | +2.842 |
| 1 | 0.04 | **0.05** | +5.484 | +6.924 | +3.431 |
| 2 | 0.07 | **0.06** | +5.966 | +7.974 | +3.419 |
| 14 | 0.52 | **0.22** | +0.035 | +0.044 | +0.053 |
| 26 | 0.96 | **0.05** | +0.285 | +0.627 | +0.254 |

_Token-determinism = η² of the MLP-layer output on current-token identity (frequent tokens). Data: [mlp_detokenizer_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/mlp_detokenizer_summary.json). Regenerate: [mlp_detokenizer.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/mlp_detokenizer.py). See the [MLP / COMPUTE catalog](mlp_compute.md)._