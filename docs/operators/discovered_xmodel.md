---
title: Discovered candidates (cross-model)
---

# Discovered candidate operators — cross-model profiles

Arch-generic dossiers of the strongest **UNNAMED** load-bearing candidates the [discovery sweep](discovered.md) surfaced in the RoPE models (which `operator_dossier.py`, GPT-2-only, could not reach). Each: which content pattern it reads, its causal ΔNLL (mean-ablation), and the channel decomposition (what addresses its key vs what it moves). Provisional.

| model | head | reads | causal ΔNLL (ind / gen) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |
|---|---|---|---|---|---|
| gemma-2-2b | 13.7 | induction | +0.68 / +0.01 | 6.0 (+5%) | 8.0 (0.38) |
| Llama-3.2-1B | 0.31 | induction | +7.99 / +0.78 | — | — |
| Llama-3.2-1B | 1.31 | induction | +5.97 / +2.35 | 0.30 (+6%) | 0.29 (0.52) |
| Qwen2.5-1.5B | 1.6 | duplicate | +0.31 / +0.06 | 0.8 (+4%) | 0.10 (0.18) |

_Data: [xmodel_candidates_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/xmodel_candidates_summary.json). Regenerate: [xmodel_candidate.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/xmodel_candidate.py). The full per-op battery for these is the RoPE-dossier port (future); this is the channel + causal core._