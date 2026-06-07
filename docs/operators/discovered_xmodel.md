---
title: Discovered candidates (cross-model)
---

# Discovered candidate operators — cross-model profiles

Arch-generic dossiers of the **UNNAMED** load-bearing candidates the [discovery sweep](discovered.md) surfaced in the RoPE models (which `operator_dossier.py`, GPT-2-only, could not reach). Each: which content pattern it reads, its causal ΔNLL (mean-ablation), and the channel decomposition (what addresses its key vs what it moves). Two independent harnesses agree where the **discovery ind ΔNLL** (multi-seed sweep) and the **profiled causal ind ΔNLL** (this run, fresh probes) line up. Provisional.

**23 candidates profiled** across 3 models. Sorted by profiled induction ΔNLL.

## gemma-2-2b

| head | reads | discovery ind ΔNLL | profiled causal ΔNLL (ind / gen) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |
|---|---|---|---|---|---|
| `13.7` | induction | +0.80 | +0.68 / +0.01 | 6.0 (+5%) | 8.0 (0.38) |
| `0.7` | induction | +0.45 | +0.34 / +0.06 | — (layer 0) | — |
| `10.6` | duplicate | +0.37 | +0.31 / +0.05 | 9.2 (+6%) | 8.0 (0.28) |
| `5.6` | induction | +0.43 | +0.25 / +0.01 | 1.4 (+1%) | 4.4 (0.14) |
| `9.6` | duplicate | +0.12 | +0.24 / +0.05 | 8.3 (+10%) | 8.0 (0.31) |
| `0.6` | duplicate | +0.23 | -0.12 / +0.00 | — (layer 0) | — |

## Llama-3.2-1B

| head | reads | discovery ind ΔNLL | profiled causal ΔNLL (ind / gen) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |
|---|---|---|---|---|---|
| `0.31` | induction | +7.26 | +7.99 / +0.78 | — (layer 0) | — |
| `1.31` | induction | +5.93 | +5.97 / +2.35 | 0.30 (+6%) | 0.29 (0.52) |
| `1.29` | induction | +5.57 | +5.56 / +1.96 | 0.30 (+3%) | 0.29 (0.52) |
| `0.29` | induction | +2.79 | +2.87 / +0.15 | — (layer 0) | — |
| `0.13` | duplicate | +1.64 | +1.70 / +0.15 | — (layer 0) | — |
| `0.14` | duplicate | +1.07 | +1.14 / +0.03 | — (layer 0) | — |
| `0.28` | induction | +1.25 | +1.07 / +0.05 | — (layer 0) | — |
| `0.19` | induction | +0.55 | +0.94 / +0.06 | — (layer 0) | — |
| `0.18` | duplicate | +0.45 | +0.63 / +0.18 | — (layer 0) | — |
| `0.25` | duplicate | +0.26 | +0.49 / +0.00 | — (layer 0) | — |
| `0.16` | duplicate | +0.22 | +0.45 / +0.01 | — (layer 0) | — |
| `0.22` | duplicate | +0.23 | +0.37 / +0.00 | — (layer 0) | — |
| `1.28` | duplicate | +0.30 | +0.33 / +0.06 | 0.30 (+4%) | 0.29 (0.44) |
| `0.20` | duplicate | +0.38 | +0.26 / +0.05 | — (layer 0) | — |
| `0.21` | induction | +0.21 | +0.19 / +0.00 | — (layer 0) | — |

## Qwen2.5-1.5B

| head | reads | discovery ind ΔNLL | profiled causal ΔNLL (ind / gen) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |
|---|---|---|---|---|---|
| `1.6` | duplicate | +0.22 | +0.31 / +0.06 | 0.8 (+4%) | 0.10 (0.18) |
| `1.5` | induction | +0.11 | +0.07 / +0.06 | 0.10 (+3%) | 0.11 (0.15) |

_Data: [xmodel_candidates_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/xmodel_candidates_summary.json). Regenerate (GPU): [xmodel_candidate.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/xmodel_candidate.py); re-render the page (CPU): `xmodel_candidate.py --docs-only`. The full per-op battery for these is the RoPE-dossier port (future); this is the channel + causal core._