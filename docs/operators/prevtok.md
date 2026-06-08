# Operator `prevtok`

**positional** — previous-token head: attend to position q-1 (the induction writer / local addressing)

## Cross-model (catalog row) — signal/causal are mean ± σ over 3 probe-resample seeds

| model | arch | signal (±σ) | #heads | top head | depth | causal ΔNLL (±σ) |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 0.965 ± 0.002 | 31 | 4.11 | 0.36 | +0.010 ± 0.004 |
| gpt2-medium | GPT-2/absolute | 0.987 ± 0.000 | 53 | 5.11 | 0.22 | +0.030 ± 0.006 |
| gpt2-large | GPT-2/absolute | 0.962 ± 0.001 | 80 | 14.1 | 0.40 | +0.011 ± 0.001 |
| gemma-2-2b | RoPE | 0.879 ± 0.003 | 106 | 21.7 | 0.84 | -0.006 ± 0.039 |
| Llama-3.2-1B | RoPE | 0.685 ± 0.001 | 47 | 0.2 | 0.00 | +0.008 ± 0.002 |
| Qwen2.5-1.5B | RoPE | 0.771 ± 0.001 | 57 | 13.4 | 0.48 | +0.216 ± 0.029 |

## Cross-model deep dossier (arch-generic) — `operator_dossier_xmodel.py`

The deep battery's arch-generic core — behavioural head-ID + mean-ablation causal + the faithful key-only path-patch channel (the model re-applies its own RoPE) — run across **every** model, not just GPT-2. (The full A–F dossier below stays GPT-2-only: its channel/composition math is written against GPT-2's fused-QKV layout, and the named *output* ops have no published head-set off GPT-2.)

| model | top head | #heads (mass≥thr) | causal induction ΔNLL | causal generic ΔNLL | redundancy (top heads) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |
|---|---|---|---|---|---|---|---|
| gpt2-xl | 12.21 | 556 | +0.14 | +0.01 | distributed (full +0.14 ≫ best 1h +0.02) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| gpt2 | 4.11 | 86 | +1.14 | +0.02 | distributed (full +1.14 ≫ best 1h +0.42) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| gpt2-medium | 5.11 | 177 | +0.58 | +0.01 | distributed (full +0.58 ≫ best 1h +0.07) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| gpt2-large | 14.1 | 363 | +0.27 | +0.02 | distributed (full +0.27 ≫ best 1h +0.04) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| gemma-2-2b | 21.7 | 191 | +4.94 | +0.30 | distributed (full +4.94 ≫ best 1h +1.60) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| Llama-3.2-1B | 0.2 | 279 | +1.02 | -0.00 | distributed (full +1.02 ≫ best 1h +0.15) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| Qwen2.5-1.5B | 13.4 | 195 | +2.79 | +0.22 | distributed (full +2.79 ≫ best 1h +0.27) | — (addresses by position/key-0) | — (addresses by position/key-0) |

_Mean-ablate the op's top behavioural heads → induction-NLL / generic-NLL damage; **redundancy** cumulative-ablates the top heads in solo-effect order (bottleneck = one head ≈ the whole op; distributed = the population far exceeds any single head; **compensatory** cases — which head triggers the recovery — are dug in [outlier mechanism digs](outlier_digs.md)); channel = remove each upstream head from the reader's key → top collapser + the value/move channel. Data: [xmodel_dossiers_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/xmodel_dossiers_summary.json). Regenerate: [operator_dossier_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_dossier_xmodel.py)._

## SAE-feature operands (section G)

What this operator reads/writes in **feature** space (monosemantic SAE latents), via the per-layer GPT-2 SAEs / Gemma Scope — see the [full SAE-operand table](sae_operands.md). _READ = dominant key-feature where the head attends (glossed by top tokens); copy-score = OV→unembed on those tokens (+ copies / − suppresses). Provisional, single corpus; for positional/addressing ops the read-feature is incidental._

| model | head | reads (SAE feature) | copy-score |
|---|---|---|---|
| gpt2 | 4.11 | `US`; `cius/ius`; `I` | +0.10 (copies) |
| gemma-2-2b | 21.7 | `▁Citizen/./▁belly`; `▁the/▁a/▁to`; `First/▁first` | -0.10 (suppresses) |

## Deep dossier (GPT-2) — `operator_dossier.py --op prevtok`

**A · identity** (behavioural: top heads by attention mass on the prevtok pattern (>0.02)): heads ['4.11', '2.2', '3.2', '3.7', '2.9']. ranked: 4.11 (0.96), 2.2 (0.54), 3.2 (0.38), 3.7 (0.37), 2.9 (0.33), 1.0 (0.32)

**B · causal × tasks** (* = beyond random control): generic +0.00, induction +2.61*, copy_names +4.76, successor +0.03, ioi +0.53*  → serves **['induction', 'ioi']**

**C · channels** (reader 4.11): KEY/match top 1.10 collapse +18% (concentration 106.3×); VALUE/move top 2.9 ΔV-out 0.21 (median 0.11).

**D · composition**: IN→key 0.11(0.114), 1.8(0.105), 1.3(0.094), 1.9(0.091); OUT→value 8.3(0.064), 5.2(0.062), 5.8(0.057), 9.4(0.056).

**E · redundancy** (task `copy_names`): solo 4.11(+1.78), 3.7(+0.06), 2.9(+0.03), 2.2(-0.00), 3.2(-0.00); cumulative 1h +1.78 → 2h +2.61 → 3h +3.45 → 4h +5.11 → 5h +4.76 → DISTRIBUTED population (full +4.76 ≫ best single +1.78).


_Data: `runs/disassembly/operators/dossiers/prevtok/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._