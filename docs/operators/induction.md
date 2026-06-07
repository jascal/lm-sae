# Operator `induction`

> **`op:induction`** — this is the **operator** (a head *class*: the family of heads that realize the operation). Not the [`induction` *circuit*](../circuits/induction.md), which is the *composition* (a writer-op feeding a reader-op) named after — and built around — this operator.

**content** — in-context copy: attend to the key whose predecessor token == current token, copy it

## Cross-model (catalog row) — signal/causal are mean ± σ over 3 probe-resample seeds

| model | arch | signal (±σ) | #heads | top head | depth | causal ΔNLL (±σ) |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 0.926 ± 0.006 | 22 | 5.5 | 0.45 | +0.011 ± 0.004 |
| gpt2-medium | GPT-2/absolute | 0.915 ± 0.004 | 61 | 18.5 | 0.78 | +0.003 ± 0.003 |
| gpt2-large | GPT-2/absolute | 0.969 ± 0.002 | 75 | 16.0 | 0.46 | +0.007 ± 0.000 |
| gemma-2-2b | RoPE | 0.941 ± 0.005 | 23 | 6.3 | 0.24 | -0.282 ± 0.019 |
| Llama-3.2-1B | RoPE | 0.946 ± 0.005 | 40 | 10.23 | 0.67 | +0.001 ± 0.001 |
| Qwen2.5-1.5B | RoPE | 0.994 ± 0.001 | 54 | 14.3 | 0.52 | -0.000 ± 0.002 |

## Deep dossier (GPT-2) — `operator_dossier.py --op induction`

**A · identity** (behavioural: top heads by attention mass on the induction pattern (>0.02)): heads ['5.1', '5.5', '6.9', '7.10', '7.2']. ranked: 5.1 (0.81), 5.5 (0.78), 6.9 (0.77), 7.10 (0.75), 7.2 (0.72), 5.0 (0.57)

**B · causal × tasks** (* = beyond random control): generic +0.01, induction +6.39*, copy_names +14.42*, successor +0.01, ioi +0.28*  → serves **['induction', 'copy_names', 'ioi']**

**C · channels** (reader 5.1): KEY/match top 4.11 (=prev-token head) collapse +43% (concentration 90.8×); VALUE/move top 1.10 ΔV-out 0.22 (median 0.08).

**D · composition**: IN→key 4.11(0.101), 4.7(0.095), 1.0(0.070), 3.7(0.070); OUT→value 6.7(0.052), 6.8(0.051), 6.6(0.049), 7.0(0.048).

**E · redundancy** (task `induction`): solo 5.1(+1.70), 7.2(+0.22), 6.9(+0.21), 5.5(+0.08), 7.10(-0.03); cumulative 1h +1.70 → 2h +2.25 → 3h +3.82 → 4h +5.77 → 5h +6.39 → DISTRIBUTED population (full +6.39 ≫ best single +1.70).

**F · cross-model**: gpt2 sig 0.81; gpt2-medium sig 0.97/gain +12.6; Qwen2.5-1.5B sig 1.00/gain +14.1

**G · SAE operands**: NOT RUN — needs a SAE (sae_lens / Gemma Scope); the op's feature-space read/write operands are the next layer.


_Data: `runs/disassembly/operators/dossiers/induction/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._