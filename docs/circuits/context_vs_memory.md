---
title: Context vs memory
---

# Context vs memory — when an in-context fact contradicts the stored one

Synthesizes the two behaviour threads: **induction** (in-context copy) vs **factual recall** (in-weights memory). Prompt: "The capital of France is **Berlin**. The capital of France is ___" — the in-context answer (Berlin) contradicts the stored fact (Paris). **margin = logit(context) − logit(memory)** (>0 → context wins). Then the causal link: mean-ablate the model's **induction heads** (the in-context-copy mechanism) — if induction is what makes context win, ablating it swings the margin back toward memory; a random same-size head-set is the control.

| model | context-win rate (margin) | − induction heads | − random heads |
|---|---|---|---|
| gpt2 | **44%** (-0.14) | 0% (-2.61) | 19% (-1.27) |
| gpt2-medium | **81%** (+0.66) | 19% (-0.38) | 94% (+0.85) |
| gpt2-large | **56%** (+0.01) | 12% (-1.19) | 50% (-0.03) |
| gemma-2-2b | **6%** (-0.80) | 0% (-3.19) | 6% (-0.70) |
| Llama-3.2-1B | **0%** (-1.19) | 0% (-1.77) | 0% (-1.54) |
| Qwen2.5-1.5B | **0%** (-1.50) | 0% (-4.40) | 0% (-1.62) |

## How much context to override memory? (repetition sweep)

Context-win rate as the false fact is repeated **N times** in-context before the query — does *more* in-context evidence eventually flip the memory-dominant models?

| model | ×1 | ×2 | ×3 | ×5 | ×8 |
|---|---|---|---|---|---|
| gpt2 | 44% | 81% | 94% | 100% | 100% |
| gpt2-medium | 81% | 100% | 100% | 100% | 100% |
| gpt2-large | 56% | 100% | 100% | 100% | 100% |
| gemma-2-2b | 6% | 100% | 100% | 100% | 100% |
| Llama-3.2-1B | 0% | 100% | 100% | 100% | 100% |
| Qwen2.5-1.5B | 0% | 100% | 100% | 100% | 100% |

_**This reframes the memory-dominance.** At **2 repetitions every model flips to 100% context-win** — including the one-shot-memory-dominant RoPE models (Gemma 6%→100%, Llama/Qwen 0%→100%). So the RoPE "memory-dominance" is purely a **one-shot** effect: their induction needs **≥2 prior occurrences** to fire strongly enough to override memory, where GPT-2's fires on a single one. It is induction's **evidence threshold**, not a refusal to trust context — and once induction fires (≥2 reps) it overrides stored memory universally._

_**Finding — two regimes.** (1) The **GPT-2 family is context-swayable** (context-win 44–81%) and **induction is the mechanism**: ablating the induction heads collapses context-win to 0–19% (memory wins), far more than ablating a random same-size head-set (which leaves it ≈baseline or higher). So induction is what lets a fresh in-context statement override stored memory. (2) The **RoPE family is memory-dominant**: Llama and Qwen **ignore the contradicting in-context fact entirely** (0% context-win) and Gemma nearly so (6%) — on a **single** in-context statement. But the repetition sweep above shows this is a one-shot effect: at ≥2 repetitions all six flip to 100% context-win. So the difference is induction's **evidence threshold** (RoPE needs ≥2 occurrences, GPT-2 fires on 1), not a standing refusal to trust context; induction is the shared override mechanism once it fires. Provisional, ~16 capital facts, single-token answers. Data: [context_vs_memory_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/context_vs_memory_summary.json). Regenerate: [context_vs_memory.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/context_vs_memory.py). See [induction](../operators/induction.md) + [where facts live](factual_recall.md)._