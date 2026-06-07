---
title: Attention vs MLP substrate for induction
---

# Where does induction live — attention or the MLP substrate?

The [reconstruction](reconstruction.md) test kept the MLPs intact and ablated attention. This does the complement: with **all attention intact**, mean-ablate the MLP substrate and measure the induction-NLL damage (induction-NLL on repeated-random sequences), isolating the early detokenizer [MLP0](../operators/mlp_detokenizer.md). Larger ΔNLL = induction leans more on that substrate.

| model | base induction-NLL | Δ all-attention | Δ all-MLPs | Δ MLP0 only | Δ MLPs except MLP0 |
|---|---|---|---|---|---|
| gpt2 | 0.63 | +10.18 | +9.64 | +9.09 | +8.90 |
| gpt2-medium | 0.54 | +10.26 | +9.63 | +17.35 | +9.63 |
| gpt2-large | 0.50 | +9.91 | +14.49 | +11.45 | +12.06 |
| gemma-2-2b | 4.54 | +15.12 | +17.58 | +4.01 | +16.02 |
| Llama-3.2-1B | 0.74 | +14.79 | +15.20 | +12.55 | +15.16 |
| Qwen2.5-1.5B | 0.36 | +16.67 | +15.82 | +8.02 | +15.54 |

_**Findings.** (1) Induction depends **roughly equally on attention and the MLP substrate** in every model (Δ all-attention ≈ Δ all-MLPs) — it is *not* an attention-only circuit; ablating either substrate roughly equally destroys it. (2) In **GPT-2-small, MLP0 alone carries nearly the entire MLP dependence** (Δ MLP0 +9.1 ≈ Δ all-MLPs +9.6) — the [detokenizer](../operators/mlp_detokenizer.md) is *the* critical MLP for induction. (3) **Gemma is the outlier**: its induction barely needs MLP0 (Δ +4.0, vs +16.0 for the rest) — consistent with Gemma's MLP0 being a clean standalone extended-embedding (η² 0.91) that the induction computation doesn't lean on; later MLPs carry it. (4) Interaction effects recur — gpt2-medium's Δ MLP0 (+17.4) *exceeds* Δ all-MLPs (+9.6): ablating one MLP hurts more than ablating all (the later MLPs partly compensate), the same non-monotonic theme as the [redundancy](../operators/induction.md) curves._

_Δ = induction-NLL increase when that part is mean-ablated (bigger = more load-bearing for induction). Provisional, single corpus. Data: [induction_substrate_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/induction_substrate_summary.json). Regenerate: [induction_substrate.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/induction_substrate.py)._