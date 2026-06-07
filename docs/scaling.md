---
title: Scaling synthesis
---

# Scaling synthesis — what tracks scale, not just architecture

The single clearest cross-cutting finding: several properties usually attributed to *architecture* (absolute-position vs RoPE) actually track **scale**. This table lines up the scale-varying quantities from across the catalog (assembled by `scaling_synthesis.py` from the committed result JSONs — no model run). Read the GPT-2 ladder (124M → 355M → 774M) top-to-bottom.

| model | params | induction key-collapse | induction redundancy | recon. coverage (mean / resample) | MLP0 token-determinism | succession MLP depth | knowledge trace peak depth |
|---|---|---|---|---|---|---|---|
| gpt2 | 124M | +39% | distributed | +17% / +30% | 0.63 | 0.09 | 0.00 |
| gpt2-medium | 355M | +8% | distributed | +7% / +24% | 0.61 | 0.00 | 0.00 |
| gpt2-large | 774M | +1% | compensatory | +0% / +5% | 0.75 | 0.20 | 0.03 |
| gemma-2-2b | 2.6B | +3% | compensatory | +14% / +7% | 0.91 | — | — |
| Llama-3.2-1B | 1.2B | +2% | distributed | +9% / +10% | 0.01 | — | 0.00 |
| Qwen2.5-1.5B | 1.5B | +0% | distributed | -4% / +0% | 0.65 | — | 0.00 |

## What the columns show

- **Induction key-collapse** — how much removing the single top prev-token writer collapses the induction head's attention. GPT-2-small **+39%** (one dominant writer) → medium +8% → large +1%: the single-writer circuit is a *small-model* trait; larger GPT-2 distribute the key like the RoPE models (~0–3%).
- **Induction redundancy** — distributed (superadditive population) in the small models, **compensatory** (non-monotonic) in gpt2-large + Gemma: the population self-interferes once big enough.
- **Reconstruction coverage** — how much the named 8-head induction circuit reconstructs in isolation. Decays with GPT-2 scale (small +17%/+30% → large +0%/+5%): bigger models spread induction across a wider supporting cast, so no compact circuit suffices ([reconstruction](circuits/reconstruction.md)).
- **MLP0 token-determinism** — the early MLP is an [extended embedding](operators/mlp_detokenizer.md); the token-determined block *widens* with GPT-2 scale (small: just L0; large: L0–L2).
- **Succession / knowledge depth** — the [succession](operators/succession.md) MLP and the [causal-trace](circuits/causal_tracing.md) knowledge site both sit deeper as the model grows (GPT-2-small ≈ L0–L2 / depth 0.1; gpt2-large ≈ L7–9 / depth 0.2 and a broad early-mid plateau).

**The thesis:** as models scale, the *same* named circuits become **more distributed** — single dominant writers give way to populations, compact circuits stop being sufficient, and the load-bearing MLP sites broaden and deepen. Absolute-vs-RoPE is a real axis (the sink, positional broadcast), but much of what looks architectural is the small models being unusually *localized*. See [Cross-model findings](FINDINGS.md).

_Assembled from the committed `runs/disassembly/**` summaries. Regenerate: [scaling_synthesis.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/scaling_synthesis.py)._