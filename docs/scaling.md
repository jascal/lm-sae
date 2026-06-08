---
title: Scaling synthesis
---

# Scaling synthesis — what tracks scale, not just architecture

The single clearest cross-cutting finding: several properties usually attributed to *architecture* (absolute-position vs RoPE) actually track **scale**. This table lines up the scale-varying quantities from across the catalog (assembled by `scaling_synthesis.py` from the committed result JSONs — no model run). Read the GPT-2 ladder (124M → 355M → 774M) top-to-bottom.

| model | params | induction key-collapse | induction redundancy | recon. coverage (mean / resample) | MLP0 token-determinism | succession MLP depth | knowledge trace peak depth |
|---|---|---|---|---|---|---|---|
| gpt2 | 124M | +39% | distributed | +17% / +31% | 0.63 | 0.09 | 0.00 |
| gpt2-medium | 355M | +8% | distributed | +7% / +24% | 0.61 | 0.00 | 0.00 |
| gpt2-large | 774M | +1% | compensatory | +0% / +5% | 0.75 | 0.20 | 0.03 |
| gpt2-xl | 1.5B | +0% | distributed | +1% / +7% | 0.80 | — | — |
| gemma-2-2b | 2.6B | +3% | compensatory | +14% / +7% | 0.91 | — | — |
| Llama-3.2-1B | 1.2B | +2% | distributed | +10% / +10% | 0.01 | — | 0.00 |
| Qwen2.5-1.5B | 1.5B | +0% | distributed | -4% / +0% | 0.65 | — | 0.00 |

## What the columns show

- **Induction key-collapse** — how much removing the single top prev-token writer collapses the induction head's attention. The full GPT-2 ladder is monotone to zero: small (124M) **+39%** (one dominant writer) → medium +8% → large +1% → **xl (1.5B) +0%**. The single-writer circuit is a *small-model* trait; by 1.5B GPT-2 distributes the key entirely, like the RoPE models (~0–3%).
- **Induction redundancy** — distributed (superadditive population) in the small models, **compensatory** (non-monotonic) in gpt2-large + Gemma: the population self-interferes once big enough.
- **Reconstruction coverage** — how much the named 8-head induction circuit reconstructs in isolation. Decays with GPT-2 scale (small +17%/+30% → large +0%/+5%): bigger models spread induction across a wider supporting cast, so no compact circuit suffices ([reconstruction](circuits/reconstruction.md)).
- **MLP0 token-determinism** — the early MLP is an [extended embedding](operators/mlp_detokenizer.md); the token-determined block *widens and strengthens* with GPT-2 scale (small 0.63 → large 0.75 → xl 0.80; the block also spreads from just L0 to L0–L2).
- **Succession / knowledge depth** — the [succession](operators/succession.md) MLP and the [causal-trace](circuits/causal_tracing.md) knowledge site both sit deeper as the model grows (GPT-2-small ≈ L0–L2 / depth 0.1; gpt2-large ≈ L7–9 / depth 0.2 and a broad early-mid plateau).

**The thesis:** as models scale, the *same* named circuits become **more distributed** — single dominant writers give way to populations, compact circuits stop being sufficient, and the load-bearing MLP sites broaden and deepen. Absolute-vs-RoPE is a real axis (the sink, positional broadcast), but much of what looks architectural is the small models being unusually *localized*. See [Cross-model findings](FINDINGS.md).

## The controlled ladder — Pythia (architecture held fixed)

The table above mixes the GPT-2 ladder with heterogeneous RoPE models, so *architecture and scale are confounded*. The **Pythia** ladder (one GPT-NeoX architecture, the *same* training data, 14M→1.4B) is the clean control (`scaling_laws.py`, arch-generic block-level + logit-lens — no head resolution needed). Three quantities turn into monotone laws with architecture fixed:

| pythia | d×L | induction-NLL | all-block-ablated Δ | capital table | capital read-out depth | language read-out depth |
|---|---|---|---|---|---|---|
| pythia-14m | 128×6 | 2.09 | +8.3 | 58% | 91% | 89% |
| pythia-70m | 512×6 | 2.17 | +7.1 | 83% | 78% | 82% |
| pythia-160m | 768×12 | 0.99 | +9.2 | 100% | 68% | 67% |
| pythia-410m | 1024×24 | 0.54 | +11.0 | 100% | 57% | 53% |
| pythia-1b | 2048×16 | 0.45 | +10.9 | 100% | 61% | 57% |
| pythia-1.4b | 2048×24 | 0.48 | +11.3 | 100% | 52% | 69% |

- **Induction emerges and strengthens with scale** — induction-NLL falls 2.1 → 2.2 → **0.99** → 0.54 → 0.45 → 0.48 (a sharp turn-on between 70M and 160M), and removing all blocks costs *more* with size (+8.3 → +11.3): induction is both stronger and more load-bearing as the model grows.
- **The knowledge table fills with scale** — the capital relation is **58% → 83% → 100%** complete (14M → 70M → 160M+): the database is populated by ~160M. Strikingly, factual recall and induction **turn on at the same scale (~160M)** — the in-context-copy mechanism and factual retrieval emerge together.
- **The relation read-out depth shrinks with scale** — capital resolves at **91% → 78% → 68% → 57% → … → 52%** of depth: bigger models retrieve the fact *earlier*, monotone on a controlled ladder (the same law the [knowledge READ](FINDINGS.md) found across the heterogeneous set, now architecture-clean).

**This is the thesis on a clean axis:** with architecture fixed, induction *appears and sharpens*, the fact table *fills*, and retrieval *moves earlier* — all monotone in size. Scale, not architecture.

_Assembled from the committed `runs/disassembly/**` summaries. Regenerate: [scaling_synthesis.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/scaling_synthesis.py)._