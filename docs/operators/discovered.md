---
title: Discovered components
---

# Discovered components — the debugger run across every model

A **working catalog** (amateur, exploratory, provisional) of the load-bearing components the [discovery engine](../DECOMPILATION.md) surfaces — ranked by causal effect on **induction** (multi-seed) and **generic** prose — with each flagged **named** (already in the behavioural operator catalog for that model) or **UNNAMED** (a candidate new operator). The UNNAMED load-bearing components are the leads to dossier next.

_6 models · every head + every MLP mean-ablated · induction over 3 seeds._

## gpt2 (GPT-2/absolute, 12L × 12H) — 2 unnamed load-bearing

Top components by induction ΔNLL (named in *italics*; **UNNAMED** = candidate op):

| component | kind | induction ΔNLL (μ±σ) | generic ΔNLL | status |
|---|---|---|---|---|
| `mlp0` | mlp | +11.75 ± 0.14 | +1.72 | *mlp* |
| `5.1` | head | +1.23 ± 0.03 | +0.02 | *induction* |
| `mlp1` | mlp | +0.97 ± 0.27 | +1.03 | *mlp* |
| `0.3` | head | +0.73 ± 0.04 | +0.08 | *self* |
| `7.6` | head | +0.53 ± 0.02 | -0.03 | *sink* |
| `4.11` | head | +0.40 ± 0.01 | +0.02 | *prevtok* |
| `0.9` | head | +0.37 ± 0.01 | +0.03 | **UNNAMED — candidate** |
| `0.10` | head | +0.29 ± 0.03 | +0.02 | *self* |
| `mlp11` | mlp | +0.25 ± 0.03 | +0.04 | *mlp* |
| `0.8` | head | +0.23 ± 0.01 | +0.04 | **UNNAMED — candidate** |

**Candidate operators (UNNAMED, load-bearing):** `0.9` (+0.37), `0.8` (+0.23)

## gpt2-medium (GPT-2/absolute, 24L × 16H) — 0 unnamed load-bearing

Top components by induction ΔNLL (named in *italics*; **UNNAMED** = candidate op):

| component | kind | induction ΔNLL (μ±σ) | generic ΔNLL | status |
|---|---|---|---|---|
| `mlp0` | mlp | +21.80 ± 0.28 | +7.64 | *mlp* |
| `2.13` | head | +0.27 ± 0.07 | -0.02 | *sink* |
| `6.1` | head | +0.18 ± 0.00 | -0.01 | *induction* |
| `mlp22` | mlp | +0.16 ± 0.00 | +0.01 | *mlp* |
| `11.4` | head | +0.13 ± 0.00 | -0.01 | *sink* |
| `5.8` | head | +0.10 ± 0.00 | -0.00 | *induction* |
| `0.1` | head | +0.09 ± 0.03 | +0.02 | **UNNAMED — candidate** |
| `9.9` | head | +0.09 ± 0.01 | +0.00 | *induction* |
| `1.4` | head | +0.09 ± 0.05 | +0.01 | *duplicate* |
| `0.8` | head | +0.08 ± 0.02 | -0.00 | **UNNAMED — candidate** |

**Candidate operators (UNNAMED, load-bearing):** —

## gpt2-large (GPT-2/absolute, 36L × 20H) — 0 unnamed load-bearing

Top components by induction ΔNLL (named in *italics*; **UNNAMED** = candidate op):

| component | kind | induction ΔNLL (μ±σ) | generic ΔNLL | status |
|---|---|---|---|---|
| `mlp0` | mlp | +13.55 ± 0.07 | +3.79 | *mlp* |
| `16.9` | head | +0.74 ± 0.04 | +0.00 | *induction* |
| `0.14` | head | +0.19 ± 0.03 | +0.02 | *duplicate* |
| `5.19` | head | +0.17 ± 0.02 | +0.00 | *duplicate* |
| `23.9` | head | +0.12 ± 0.01 | -0.00 | *sink* |
| `mlp34` | mlp | +0.11 ± 0.01 | -0.10 | *mlp* |
| `22.6` | head | +0.10 ± 0.01 | -0.00 | *sink* |
| `mlp25` | mlp | +0.10 ± 0.01 | -0.01 | *mlp* |
| `mlp35` | mlp | +0.09 ± 0.01 | -0.22 | *mlp* |
| `24.4` | head | +0.08 ± 0.00 | -0.00 | *sink* |

**Candidate operators (UNNAMED, load-bearing):** —

## gemma-2-2b (RoPE, 26L × 8H) — 6 unnamed load-bearing

Top components by induction ΔNLL (named in *italics*; **UNNAMED** = candidate op):

| component | kind | induction ΔNLL (μ±σ) | generic ΔNLL | status |
|---|---|---|---|---|
| `mlp0` | mlp | +3.65 ± 0.31 | +0.29 | *mlp* |
| `22.4` | head | +2.90 ± 0.14 | -0.02 | *induction* |
| `14.5` | head | +2.56 ± 0.10 | +0.15 | *self* |
| `17.4` | head | +2.33 ± 0.08 | +0.14 | *self* |
| `15.6` | head | +1.93 ± 0.08 | +0.20 | *self* |
| `mlp17` | mlp | +1.72 ± 0.17 | +0.00 | *mlp* |
| `16.2` | head | +1.58 ± 0.06 | +0.20 | *self* |
| `17.2` | head | +1.56 ± 0.05 | +0.13 | *prevtok* |
| `5.0` | head | +1.34 ± 0.11 | +0.14 | *prevtok* |
| `17.6` | head | +1.31 ± 0.05 | +0.10 | *prevtok* |

**Candidate operators (UNNAMED, load-bearing):** `13.7` (+0.80), `0.7` (+0.45), `5.6` (+0.43), `10.6` (+0.37), `0.6` (+0.23), `9.6` (+0.12)

## Llama-3.2-1B (RoPE, 16L × 32H) — 16 unnamed load-bearing

Top components by induction ΔNLL (named in *italics*; **UNNAMED** = candidate op):

| component | kind | induction ΔNLL (μ±σ) | generic ΔNLL | status |
|---|---|---|---|---|
| `mlp1` | mlp | +12.64 ± 0.09 | +7.35 | *mlp* |
| `mlp0` | mlp | +12.63 ± 0.03 | +4.50 | *mlp* |
| `1.8` | head | +11.41 ± 0.70 | +0.01 | *sink* |
| `1.11` | head | +8.98 ± 0.84 | -0.00 | *sink* |
| `0.31` | head | +7.26 ± 0.91 | +0.78 | **UNNAMED — candidate** |
| `1.31` | head | +5.93 ± 0.24 | +2.35 | **UNNAMED — candidate** |
| `1.29` | head | +5.57 ± 0.16 | +1.96 | **UNNAMED — candidate** |
| `0.29` | head | +2.79 ± 0.10 | +0.15 | **UNNAMED — candidate** |
| `mlp15` | mlp | +1.92 ± 0.06 | +0.81 | *mlp* |
| `0.13` | head | +1.64 ± 0.24 | +0.15 | **UNNAMED — candidate** |

**Candidate operators (UNNAMED, load-bearing):** `0.31` (+7.26), `1.31` (+5.93), `1.29` (+5.57), `0.29` (+2.79), `0.13` (+1.64), `0.28` (+1.25), `0.14` (+1.07), `0.19` (+0.55)

## Qwen2.5-1.5B (RoPE, 28L × 12H) — 2 unnamed load-bearing

Top components by induction ΔNLL (named in *italics*; **UNNAMED** = candidate op):

| component | kind | induction ΔNLL (μ±σ) | generic ΔNLL | status |
|---|---|---|---|---|
| `mlp2` | mlp | +13.75 ± 0.19 | +5.72 | *mlp* |
| `mlp1` | mlp | +13.38 ± 0.08 | +8.03 | *mlp* |
| `mlp0` | mlp | +7.71 ± 0.11 | +3.05 | *mlp* |
| `mlp26` | mlp | +1.05 ± 0.02 | +0.73 | *mlp* |
| `8.3` | head | +0.65 ± 0.01 | +0.00 | *duplicate* |
| `0.10` | head | +0.50 ± 0.09 | +0.35 | *self* |
| `2.3` | head | +0.45 ± 0.04 | -0.00 | *induction* |
| `mlp27` | mlp | +0.27 ± 0.05 | +0.59 | *mlp* |
| `13.4` | head | +0.25 ± 0.02 | +0.01 | *prevtok* |
| `18.8` | head | +0.24 ± 0.03 | +0.01 | *prevtok* |

**Candidate operators (UNNAMED, load-bearing):** `1.6` (+0.22), `1.5` (+0.11)

## How to read this

- *named* = the component is already a member of a catalogued operator class for that model (prev-token / induction / duplicate / sink / self / local, by attention-mask mass; or a GPT-2 IOI circuit head). *(Mean-ablation under-counts self-repairing classes like name-movers — see the IOI dossier — so a 'named' reading low is expected.)*
- **UNNAMED** = load-bearing on induction but not yet in any catalogued class — a candidate operator to give a [dossier](README.md). MLPs are flagged `mlp` (the COMPUTE class, see [MLP / COMPUTE](mlp_compute.md)).
- Provisional and descriptive — measurements to be checked, not settled results.

_Data: [runs/disassembly/operators/discovered_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/discovered_summary.json). Regenerate: [discovery_atlas.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/discovery_atlas.py)._