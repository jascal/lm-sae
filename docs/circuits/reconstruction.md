---
title: Executable decompilation
---

# Executable decompilation — does the induction circuit reconstruct itself?

The catalog shows which heads are *necessary*. This tests **sufficiency**: keep ONLY the induction circuit (the induction + prev-token heads from the [cross-model dossier](operators/induction.md)), mean-ablate **every other attention head** (MLPs intact — the substrate), and measure how much induction survives.

**coverage = (NLL_all-attn-ablated − NLL_circuit-only) / (NLL_all-attn-ablated − NLL_full)** — 1 = the circuit alone fully reconstructs induction, 0 = no better than ablating all attention. A random same-size head-set is the control.

| model | circuit size / total heads | induction-NLL (full / circuit-only / all-ablated) | **circuit coverage** | random control |
|---|---|---|---|---|
| gpt2 | 8 / 144 | 0.63 / 9.07 / 10.81 | **+17%** | +1% ± 0% |
| gpt2-medium | 8 / 384 | 0.54 / 10.06 / 10.80 | **+7%** | +1% ± 0% |
| gpt2-large | 8 / 720 | 0.50 / 10.40 / 10.40 | **+0%** | +0% ± 0% |
| gemma-2-2b | 8 / 208 | 4.54 / 17.58 / 19.66 | **+14%** | -1% ± 5% |
| Llama-3.2-1B | 8 / 512 | 0.74 / 14.15 / 15.52 | **+9%** | +1% ± 2% |
| Qwen2.5-1.5B | 8 / 336 | 0.36 / 17.76 / 17.03 | **-4%** | -2% ± 4% |

_**The honest result: necessity ≠ a small sufficient circuit.** No 8-head circuit *fully* reconstructs induction in any model (best +17%, GPT-2-small). The circuit beats its random control in 4/6 models — it is the **main** contributor — but coverage is modest, and it **decays with GPT-2 scale** (small +17% → medium +7% → large +0%) and fails in Qwen (−4%): in the larger / more distributed models the top induction + prev-token heads in isolation recover essentially nothing, because induction there is spread across a supporting cast the 8-head set excludes. So the catalogued circuit is causally necessary and the dominant driver, but not an executable small-circuit decompilation on its own — consistent with the distributed / non-monotonic induction-redundancy seen in the [dossier](../operators/induction.md). Provisional, single corpus; induction-NLL on repeated-random sequences. Data: [circuit_reconstruction_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/circuit_reconstruction_summary.json). Regenerate: [circuit_reconstruction.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_reconstruction.py)._