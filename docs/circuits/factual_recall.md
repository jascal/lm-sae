---
title: Where do facts live?
---

# Where do facts live? — per-layer localization of factual recall

The catalog is about *mechanisms*; this is about *knowledge*. For a set of factual completions ("The capital of France is" → " Paris", single-token objects only), measure the object's NLL, then mean-ablate each layer's **MLP** in turn. **The confound:** raw fact-ΔNLL is dominated by the *early* MLPs — but those (MLP0 the [detokenizer](../operators/mlp_detokenizer.md)) carry **all** token processing, not facts specifically. So we control against **generic** prose-NLL ablation and report the **fact-specific excess** = (a layer's share of fact-importance) − (its share of generic-importance). That is where facts are hurt *disproportionately* — the natural-history analog of the ROME mid-layer-MLP store.

| model | #facts | raw top-MLP (depth, ΔNLL) — *detokenizer-dominated* | **fact-specific** top-MLP (depth, excess) | fact-specific peak depth |
|---|---|---|---|---|
| gpt2 | 23 | L0 (0.00, +8.4); L1 (0.09, +7.8); L2 (0.18, +6.4) | **L3 (0.27, +0.10); L9 (0.82, +0.02); L8 (0.73, +0.01)** | 0.27 |
| gpt2-medium | 23 | L0 (0.00, +14.8); L3 (0.13, +8.3); L2 (0.09, +8.0) | **L3 (0.13, +0.14); L2 (0.09, +0.13); L4 (0.17, +0.04)** | 0.13 |
| gpt2-large | 23 | L0 (0.00, +9.2); L7 (0.20, +7.9); L9 (0.26, +7.2) | **L7 (0.20, +0.04); L8 (0.23, +0.04); L9 (0.26, +0.03)** | 0.20 |
| gemma-2-2b | 24 | L3 (0.12, +0.4); L21 (0.84, +0.3); L4 (0.16, +0.2) | **L3 (0.12, +0.14); L21 (0.84, +0.11); L22 (0.88, +0.07)** | 0.12 |
| Llama-3.2-1B | 24 | L0 (0.00, +15.8); L1 (0.07, +12.1); L15 (1.00, +10.6) | **L14 (0.93, +0.03); L15 (1.00, +0.02); L11 (0.73, +0.01)** | 0.93 |
| Qwen2.5-1.5B | 24 | L1 (0.04, +13.8); L2 (0.07, +12.0); L3 (0.11, +11.7) | **L3 (0.11, +0.06); L2 (0.07, +0.05); L5 (0.19, +0.02)** | 0.11 |

_**Finding.** Raw fact-ΔNLL is dominated by the very-early detokenizer MLPs (L0–1) — they carry all token processing. Once that is controlled for, the **fact-specific** MLP importance concentrates in **early-mid layers** (depth ≈ 0.1–0.27 — gpt2 L3, gpt2-medium L3, gpt2-large L7–9, Gemma L3, Qwen L3), broadly consistent with the ROME early-mid MLP knowledge store, recovered here as natural history. The recurring outliers show again: **Gemma** adds a **late** fact site (L21–22), and **Llama** localizes facts late (L14–15, depth ≈0.93). The excess magnitudes are small — facts are also distributed — but the disproportionate fact-specific load sits early-mid._


_The **raw** column is the detokenizer confound (early MLPs hurt everything); the **fact-specific** column controls for it (fact-importance share minus generic-importance share) — that is where the *facts* live as opposed to general token processing. Proper causal localization is ROME-style subject-corruption tracing; this is a cheaper ablation-contrast proxy. Provisional, ~24 facts (capitals + a few), single-token objects. Data: [factual_recall_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/factual_recall_summary.json). Regenerate: [factual_recall.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/factual_recall.py)._