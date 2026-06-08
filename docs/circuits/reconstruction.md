---
title: Executable decompilation
---

# Executable decompilation — does the induction circuit reconstruct itself?

The catalog shows which heads are *necessary*. This tests **sufficiency**: keep ONLY the induction circuit (the induction + prev-token heads from the [cross-model dossier](operators/induction.md)), mean-ablate **every other attention head** (MLPs intact — the substrate), and measure how much induction survives.

**coverage = (NLL_all-attn-ablated − NLL_circuit-only) / (NLL_all-attn-ablated − NLL_full)** — 1 = the circuit alone fully reconstructs induction, 0 = no better than ablating all attention. A random same-size head-set is the control.

| model | circuit size / total heads | induction-NLL (full / circuit-only / all-ablated) | **circuit coverage** (mean-abl, ±σ) | coverage (resample-abl, ±σ) | random control |
|---|---|---|---|---|---|
| gpt2 | 8 / 144 | 0.62 / 9.23 / 10.95 | **+17% ± 0%** | +31% ± 1% | +4% ± 2% |
| gpt2-medium | 8 / 384 | 0.52 / 10.24 / 10.93 | **+7% ± 0%** | +24% ± 1% | +2% ± 1% |
| gpt2-large | 8 / 720 | 0.45 / 10.50 / 10.50 | **+0% ± 0%** | +5% ± 0% | +1% ± 0% |
| gemma-2-2b | 8 / 208 | 5.22 / 17.78 / 19.76 | **+14% ± 0%** | +7% ± 0% | +2% ± 5% |
| Llama-3.2-1B | 8 / 512 | 0.73 / 14.22 / 15.69 | **+10% ± 0%** | +10% ± 0% | +1% ± 2% |
| Qwen2.5-1.5B | 8 / 336 | 0.49 / 17.56 / 17.04 | **-4% ± 1%** | +0% ± 0% | -1% ± 2% |

_Coverage is **mean ± σ over 3 probe-resample seeds** — the error bars confirm the scaling/distributedness trend is not a single-seed artifact._

## How many heads does induction need? (reconstruction curve)

Rank every head by induction-mass, keep the top-K (ablate the rest), and watch coverage grow with K — the size at which it saturates is induction's *effective* circuit size.

| model | K=4 | K=8 | K=16 | K=32 | K=64 | K=128 | K=256 |
|---|---|---|---|---|---|---|---|
| gpt2 | +6% | +8% | +13% | +21% | +23% | +97% | — |
| gpt2-medium | +2% | +3% | +6% | +9% | +15% | +23% | +23% |
| gpt2-large | +1% | +1% | +2% | +2% | +3% | +3% | +1% |
| gemma-2-2b | +1% | +17% | +20% | +30% | +34% | +13% | — |
| Llama-3.2-1B | +3% | +4% | +9% | +18% | +25% | +29% | +28% |
| Qwen2.5-1.5B | +3% | +2% | -3% | -5% | -8% | -13% | -8% |

_**No compact head-subset reconstructs induction in any model.** GPT-2-small only reaches near-full coverage at K≈128/144 (it needs nearly every head); gpt2-medium saturates at ~22% even with 256 heads; gpt2-large stays ~0% throughout; and the RoPE curves go **non-monotonic** — Gemma peaks ~32% then drops, Qwen goes **negative** (keeping more induction-mass heads *hurts* induction-NLL — the same interference / compensatory effect the [outlier digs](../operators/outlier_digs.md) traced to a synthetic-probe artifact). Induction is a property of the near-whole network, not an isolable subgraph._

## The IOI circuit (GPT-2) — is the literature's *complete* circuit sufficient?

The same test on the field's most-celebrated complete circuit (Wang et al. 2022), measured on the metric it serves — the IOI logit-difference `LD = logit(IO) − logit(S)`. Keep only the IOI circuit's **26 heads** (of 144), ablate the rest:

| circuit | LD (full / circuit-only / all-ablated) | coverage | random control |
|---|---|---|---|
| IOI (26h) | +2.67 / -0.70 / -0.01 | **-26%** | -23% ± 14% |

**Same lesson as induction, sharper:** keeping only the 26 IOI heads and mean-ablating the rest gives a **negative** logit-diff (-0.70) — the model now prefers S over IO — and is **no better than a random 26-head set**. The named circuit is not a *sufficient* isolated subgraph; it needs the rest of the network as substrate.

> **Caveat (important, read this).** This is a harsh *sufficiency-under-mean-ablation* test: mean-ablating ~120 heads pushes activations far off-distribution and severs the upstream signals the circuit reads. The original IOI result (Wang et al.) is about **necessity** + path-patching, **not** isolated mean-ablation sufficiency — so this does **not** refute it. It says the IOI computation, like induction, is not recoverable from its named heads *in isolation*; the named circuit is necessary and explanatory, but the behaviour is carried by the near-whole network. A statement about distributedness, not validity._


_**Robustness — does it survive a gentler ablation?** Mean-ablation pushes activations off-distribution, so it *understates* coverage: under **resample-ablation** (replace ablated heads with a different valid sequence's activations, on the data manifold) the GPT-2 family reconstructs more (gpt2 +17%→**+30%**, medium +7%→+24%). But **no model exceeds ~30% even under resample** — so the distributedness is real, not a mean-ablation artifact; mean-ablation just exaggerated it. The named 8-head circuit is the dominant driver, not a sufficient subgraph, under either ablation._


_**The honest result: necessity ≠ a small sufficient circuit.** No 8-head circuit *fully* reconstructs induction in any model (best +17% mean / +30% resample, GPT-2-small). The circuit beats its random control in 4/6 models — it is the **main** contributor — but coverage is modest, and it **decays with GPT-2 scale** (small +17% → medium +7% → large +0%) and fails in Qwen (−4%): in the larger / more distributed models the top induction + prev-token heads in isolation recover essentially nothing, because induction there is spread across a supporting cast the 8-head set excludes. So the catalogued circuit is causally necessary and the dominant driver, but not an executable small-circuit decompilation on its own — consistent with the distributed / non-monotonic induction-redundancy seen in the [dossier](../operators/induction.md). Provisional, single corpus; induction-NLL on repeated-random sequences. Data: [circuit_reconstruction_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/circuit_reconstruction_summary.json). Regenerate: [circuit_reconstruction.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_reconstruction.py)._