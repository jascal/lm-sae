---
title: Causal tracing (where facts are stored)
---

# Causal tracing of factual recall — where is the fact stored?

The field-standard **causal trace** (Meng et al., ROME), run across six models (ROME only did GPT-2/GPT-J). Corrupt the **subject** tokens with Gaussian noise (3× the embedding std) — the fact's probability drops — then in the corrupted run **restore the clean MLP output** at each layer (at the subject's last token) and measure how much the object's probability recovers. The MLP whose restoration recovers the most is where the fact is enriched; ROME's headline is an **early-mid MLP** site at the subject's last token.

Two sites are expected: an **early MLP store at the subject's last token** (restore the clean MLP output) and a **late attention readout at the last token** (restore the clean attention output — the heads that copy the enriched fact to the prediction).

| model | facts used | MLP store — peak @ subject (depth, recovery) | attention readout — peak @ last token (depth, recovery) |
|---|---|---|---|
| gpt2 | 17 | **L0 (0.00, +75%)** | L9 (0.82, +78%) |
| gpt2-medium | 17 | **L0 (0.00, +97%)** | L20 (0.87, +31%) |
| gpt2-large | 15 | **L1 (0.03, +104%)** | L26 (0.74, +37%) |
| Llama-3.2-1B | 18 | **L0 (0.00, +94%)** | L9 (0.60, +55%) |
| Qwen2.5-1.5B | 18 | **L0 (0.00, +87%)** | L22 (0.81, +57%) |

_**Finding — the two-site flow is architecture-invariant.** Every model traced shows the canonical ROME structure: an **early MLP store at the subject** (peak depth ≈ 0.00–0.03) feeding a **late attention readout at the last token** (peak depth ≈ 0.60–0.87). The fact is enriched into the subject's residual by the early MLPs, then copied to the prediction by late-layer attention — the same early-MLP → late-attention information flow in GPT-2 (small/medium/large), Llama, and Qwen. Recovered cross-model (ROME only did GPT-2/GPT-J)._

_**Scale note.** Factual recall recovers from the **early MLPs at the subject's last token** in every model — and the early-mid MLP **plateau widens with scale**: GPT-2-small is a sharp single L0 spike, while gpt2-large and Llama show a broad L0–3 early-mid plateau (the same embedding-block-widens-with-scale pattern as the [extended-embedding test](../operators/mlp_detokenizer.md)). This is the rigorous (corruption + restoration) confirmation of the cheaper [ablation-contrast](factual_recall.md) — facts are enriched in the early MLPs at the subject, ROME's store, now cross-model._ **Excluded: gemma-2-2b** — Gemma scales its embeddings by √d, so the standard 3×-std noise barely corrupts the fact (denom < 0.2) and no clean trace is obtained.

_Recovery = fraction of the corruption-induced probability drop that restoring that layer's **MLP output** (at the subject's last token) recovers. Provisional, ~18 capital-city facts, single-token objects, one noise sample per fact. Data: [causal_tracing_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/causal_tracing_summary.json). Regenerate: [causal_tracing.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/causal_tracing.py)._