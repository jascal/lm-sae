---
title: Fact transplant (patching the MLP store)
---

# Fact transplant — does patching the MLP store rewrite the retrieved fact?

The [causal trace](causal_tracing.md) localized facts to the **early MLPs at the subject**. This is the sufficiency / "recompile" test: run "The capital of **France** is" but **patch the early-MLP output at the subject position with the same-position output from "The capital of Italy is"** — grafting Italy's subject-enrichment into France's run. If the store carries the fact, the model now predicts **Rome**, not Paris. Over ordered fact pairs: the **flip rate** (the donor's capital out-scores the original's) and the mean logit-difference shift.

| model | relation | facts | patched band | pairs | **flip rate** | mean logit-diff shift |
|---|---|---|---|---|---|---|
| gpt2 | capital | 16 | L0–2 | 64 | **100%** | +4.32 |
| gpt2 | language | 16 | L0–2 | 64 | **100%** | +2.80 |
| gpt2-medium | capital | 16 | L0–5 | 64 | **100%** | +8.76 |
| gpt2-medium | language | 16 | L0–5 | 64 | **100%** | +5.10 |
| gpt2-large | capital | 16 | L0–8 | 64 | **100%** | +8.72 |
| gpt2-large | language | 16 | L0–8 | 64 | **100%** | +4.73 |
| gemma-2-2b | capital | 16 | L0–5 | 64 | **3%** | -9.29 |
| gemma-2-2b | language | 16 | L0–5 | 64 | **11%** | -5.35 |
| Llama-3.2-1B | capital | 16 | L0–3 | 64 | **100%** | +10.23 |
| Llama-3.2-1B | language | 16 | L0–3 | 64 | **100%** | +6.54 |
| Qwen2.5-1.5B | capital | 16 | L0–6 | 64 | **100%** | +7.53 |
| Qwen2.5-1.5B | language | 16 | L0–6 | 64 | **100%** | +4.65 |

_**Finding.** Patching the early-MLP store at the subject **causally transplants the fact — a 100% flip rate in GPT-2 (all three sizes), Llama, and Qwen, for *both* relations tested** (capital and language): France's run now answers Rome / Italian, every pair. So the store carries facts **generally**, not capital-specifically — an activation-patch edit (no weight surgery), the sufficiency complement of the causal trace's necessity, and the decompile→recompile loop made concrete. **Gemma is the recurring outlier** (3% capital / 11% language flip, *negative* shift on both): patching its early-subject MLPs does NOT transplant the fact for either relation — consistent with Gemma's clean standalone MLP0 (token-determinism η² 0.91). A band-scan confirms it: **no single 25% MLP band (early, mid, or late) transplants Gemma's facts** — every band gives ~0% flip with a *negative* shift (patching only damages). So Gemma's factual storage is **distributed**, not band-localizable / editable the way the other five models' early store is. Same Gemma exceptionalism as the sink, the induction key, and redundancy._

_A high flip rate = the early store causally carries the fact. Provisional, ~16 capital facts, single-token subjects + objects, early band = first ~25% of MLPs. Data: [fact_patching_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/fact_patching_summary.json). Regenerate: [fact_patching.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/fact_patching.py). See [causal tracing](causal_tracing.md) + [DECOMPILATION.md](../DECOMPILATION.md) (the decompile→recompile loop)._