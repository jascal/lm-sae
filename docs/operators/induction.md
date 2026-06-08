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

## Cross-model deep dossier (arch-generic) — `operator_dossier_xmodel.py`

The deep battery's arch-generic core — behavioural head-ID + mean-ablation causal + the faithful key-only path-patch channel (the model re-applies its own RoPE) — run across **every** model, not just GPT-2. (The full A–F dossier below stays GPT-2-only: its channel/composition math is written against GPT-2's fused-QKV layout, and the named *output* ops have no published head-set off GPT-2.)

| model | top head | #heads (mass≥thr) | causal induction ΔNLL | causal generic ΔNLL | redundancy (top heads) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |
|---|---|---|---|---|---|---|---|
| gpt2-xl | 21.3 | 237 | +0.20 | -0.00 | distributed (full +0.20 ≫ best 1h +0.12) | 2.11 (+0%, conc 104×) | 2.13 (0.03) |
| gpt2 | 5.1 | 37 | +3.67 | +0.03 | distributed (full +3.67 ≫ best 1h +1.31) | 4.11 (+39%, conc 85×) | 1.10 (0.22) |
| gpt2-medium | 11.1 | 103 | +0.70 | +0.09 | distributed (full +0.70 ≫ best 1h +0.12) | 4.13 (+8%, conc 118×) | 2.14 (0.08) |
| gpt2-large | 16.0 | 145 | +0.63 | +0.01 | **compensatory** (peak +1.18@3h → full +0.63; non-monotonic) | 3.1 (+1%, conc 44×) | 3.11 (0.07) |
| gemma-2-2b | 6.3 | 51 | +0.59 | -0.31 | **compensatory** (peak +1.98@2h → full +0.59; non-monotonic) | 5.0 (+3%, conc 1672×) | 5.5 (0.07) |
| Llama-3.2-1B | 10.23 | 93 | +1.24 | +0.01 | distributed (full +1.24 ≫ best 1h +0.77) | 1.9 (+2%, conc 64×) | 1.28 (0.08) |
| Qwen2.5-1.5B | 14.3 | 98 | +2.10 | -0.00 | distributed (full +2.10 ≫ best 1h +0.51) | 2.10 (+0%, conc 17×) | 1.4 (0.07) |

_Mean-ablate the op's top behavioural heads → induction-NLL / generic-NLL damage; **redundancy** cumulative-ablates the top heads in solo-effect order (bottleneck = one head ≈ the whole op; distributed = the population far exceeds any single head; **compensatory** cases — which head triggers the recovery — are dug in [outlier mechanism digs](outlier_digs.md)); channel = remove each upstream head from the reader's key → top collapser + the value/move channel. Data: [xmodel_dossiers_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/xmodel_dossiers_summary.json). Regenerate: [operator_dossier_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_dossier_xmodel.py)._

## SAE-feature operands (section G)

What this operator reads/writes in **feature** space (monosemantic SAE latents), via the per-layer GPT-2 SAEs / Gemma Scope — see the [full SAE-operand table](sae_operands.md). _READ = dominant key-feature where the head attends (glossed by top tokens); copy-score = OV→unembed on those tokens (+ copies / − suppresses). Provisional, single corpus; for positional/addressing ops the read-feature is incidental._

| model | head | reads (SAE feature) | copy-score |
|---|---|---|---|
| gpt2 | 5.5 | `First`; `What/_What/_what`; `US` | +0.09 (copies) |
| gemma-2-2b | 6.3 | `First`; `▁gods/▁run/▁petition`; `Before` | +0.06 (copies) |

## Deep dossier (GPT-2) — `operator_dossier.py --op induction`

**A · identity** (behavioural: top heads by attention mass on the induction pattern (>0.02)): heads ['5.1', '5.5', '6.9', '7.10', '7.2']. ranked: 5.1 (0.81), 5.5 (0.78), 6.9 (0.77), 7.10 (0.75), 7.2 (0.72), 5.0 (0.57)

**B · causal × tasks** (* = beyond random control): generic +0.01, induction +6.39*, copy_names +14.42*, successor +0.01, ioi +0.28*  → serves **['induction', 'copy_names', 'ioi']**

**C · channels** (reader 5.1): KEY/match top 4.11 (=prev-token head) collapse +43% (concentration 90.8×); VALUE/move top 1.10 ΔV-out 0.22 (median 0.08).

**D · composition**: IN→key 4.11(0.101), 4.7(0.095), 1.0(0.070), 3.7(0.070); OUT→value 6.7(0.052), 6.8(0.051), 6.6(0.049), 7.0(0.048).

**E · redundancy** (task `induction`): solo 5.1(+1.70), 7.2(+0.22), 6.9(+0.21), 5.5(+0.08), 7.10(-0.03); cumulative 1h +1.70 → 2h +2.25 → 3h +3.82 → 4h +5.77 → 5h +6.39 → DISTRIBUTED population (full +6.39 ≫ best single +1.70).

**F · cross-model**: gpt2 sig 0.81; gpt2-medium sig 0.97/gain +12.6; Qwen2.5-1.5B sig 1.00/gain +14.1


_Data: `runs/disassembly/operators/dossiers/induction/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._