---
title: Causal tracing (where facts are stored)
---

# Causal tracing of factual recall — where is the fact stored?

The field-standard **causal trace** (Meng et al., ROME), run across six models (ROME only did GPT-2/GPT-J). Corrupt the **subject** tokens with Gaussian noise (3× the embedding std) — the fact's probability drops — then in the corrupted run **restore the clean MLP output** at each layer (at the subject's last token) and measure how much the object's probability recovers. The MLP whose restoration recovers the most is where the fact is enriched; ROME's headline is an **early-mid MLP** site at the subject's last token.

| model | facts used | peak restore layer (depth, recovery) | top-3 layers (depth, recovery) |
|---|---|---|---|
| gpt2 | 17 | **L0 (0.00, +75%)** | L0 (0.00, +75%); L5 (0.45, +4%); L6 (0.55, +4%) |
| gpt2-medium | 17 | **L0 (0.00, +97%)** | L0 (0.00, +97%); L1 (0.04, +33%); L3 (0.13, +27%) |
| gpt2-large | 15 | **L1 (0.03, +104%)** | L1 (0.03, +104%); L0 (0.00, +99%); L2 (0.06, +73%) |
| Llama-3.2-1B | 18 | **L0 (0.00, +94%)** | L0 (0.00, +94%); L2 (0.13, +74%); L3 (0.20, +56%) |
| Qwen2.5-1.5B | 18 | **L0 (0.00, +87%)** | L0 (0.00, +87%); L2 (0.07, +50%); L3 (0.11, +35%) |

_**Finding.** Factual recall recovers from the **early MLPs at the subject's last token** (the ROME subject-enrichment site) in every model traced — and the early-mid **plateau widens with scale**: GPT-2-small is a sharp single L0 spike, while gpt2-large and Llama show a broad L0–3 early-mid plateau (the same embedding-block-widens-with-scale pattern as the [extended-embedding test](../operators/mlp_detokenizer.md)). This is the rigorous (corruption + restoration) confirmation of the cheaper [ablation-contrast](factual_recall.md) — facts are enriched in the early MLPs at the subject, ROME's store, now cross-model._ **Excluded: gemma-2-2b** — Gemma scales its embeddings by √d, so the standard 3×-std noise barely corrupts the fact (denom < 0.2) and no clean trace is obtained.

_Recovery = fraction of the corruption-induced probability drop that restoring that layer's **MLP output** (at the subject's last token) recovers. Provisional, ~18 capital-city facts, single-token objects, one noise sample per fact. Data: [causal_tracing_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/causal_tracing_summary.json). Regenerate: [causal_tracing.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/causal_tracing.py)._