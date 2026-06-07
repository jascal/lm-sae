# Operator `duplicate`

> **`op:duplicate`** — this is the **operator** (a head *class*: the family of heads that realize the operation). Not the [`duplicate` *circuit*](../circuits/duplicate.md), which is the *composition* (a writer-op feeding a reader-op) named after — and built around — this operator.

**content** — duplicate-token head: attend to an earlier occurrence of the same token

## Cross-model (catalog row) — signal/causal are mean ± σ over 3 probe-resample seeds

| model | arch | signal (±σ) | #heads | top head | depth | causal ΔNLL (±σ) |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 0.622 ± 0.005 | 4 | 0.5 | 0.00 | +0.114 ± 0.014 |
| gpt2-medium | GPT-2/absolute | 0.859 ± 0.003 | 6 | 7.11 | 0.30 | -0.008 ± 0.007 |
| gpt2-large | GPT-2/absolute | 0.961 ± 0.002 | 23 | 5.8 | 0.14 | +0.000 ± 0.003 |
| gemma-2-2b | RoPE | 0.856 ± 0.007 | 11 | 1.4 | 0.04 | -1.067 ± 0.027 |
| Llama-3.2-1B | RoPE | 0.735 ± 0.008 | 16 | 6.8 | 0.40 | +0.017 ± 0.005 |
| Qwen2.5-1.5B | RoPE | 0.973 ± 0.001 | 16 | 8.3 | 0.30 | +0.000 ± 0.003 |

## Cross-model deep dossier (arch-generic) — `operator_dossier_xmodel.py`

The deep battery's arch-generic core — behavioural head-ID + mean-ablation causal + the faithful key-only path-patch channel (the model re-applies its own RoPE) — run across **every** model, not just GPT-2. (The full A–F dossier below stays GPT-2-only: its channel/composition math is written against GPT-2's fused-QKV layout, and the named *output* ops have no published head-set off GPT-2.)

| model | top head | #heads (mass≥thr) | causal induction ΔNLL | causal generic ΔNLL | redundancy (top heads) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |
|---|---|---|---|---|---|---|---|
| gpt2 | 0.5 | 11 | +0.49 | +0.23 | distributed (full +0.49 ≫ best 1h +0.12) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| gpt2-medium | 7.11 | 31 | +0.15 | -0.00 | distributed (full +0.15 ≫ best 1h +0.04) | 2.13 (+13%, conc 712×) | 2.7 (0.29) |
| gpt2-large | 5.8 | 106 | +0.39 | +0.00 | distributed (full +0.39 ≫ best 1h +0.14) | 3.3 (+31%, conc 865×) | 3.17 (0.24) |
| gemma-2-2b | 1.4 | 34 | -0.28 | -0.87 | **compensatory** (peak +1.55@3h → full -0.28; self-repair) | 0.0 (+1%, conc 20×) | 0.1 (0.25) |
| Llama-3.2-1B | 0.9 | 72 | +0.94 | +0.03 | distributed (full +0.94 ≫ best 1h +0.59) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| Qwen2.5-1.5B | 8.3 | 81 | +1.37 | +0.00 | distributed (full +1.37 ≫ best 1h +0.70) | 0.4 (+0%, conc 44×) | 6.5 (0.08) |

_Mean-ablate the op's top behavioural heads → induction-NLL / generic-NLL damage; **redundancy** cumulative-ablates the top heads in solo-effect order (bottleneck = one head ≈ the whole op; distributed = the population far exceeds any single head); channel = remove each upstream head from the reader's key → top collapser + the value/move channel. Data: [xmodel_dossiers_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/xmodel_dossiers_summary.json). Regenerate: [operator_dossier_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_dossier_xmodel.py)._

## Deep dossier (GPT-2) — `operator_dossier.py --op duplicate`

**A · identity** (behavioural: top heads by attention mass on the duplicate pattern (>0.02)): heads ['0.5', '3.0', '0.1', '1.11', '0.10']. ranked: 0.5 (0.59), 3.0 (0.53), 0.1 (0.32), 1.11 (0.18), 0.10 (0.11), 1.5 (0.08)

**B · causal × tasks** (* = beyond random control): generic +0.07*, induction +0.57, copy_names +3.15*, successor -0.25, ioi +0.35*  → serves **['generic', 'copy_names', 'ioi']**

**C · channels**: reader in layer 0 — no upstream; channel skipped

**D · composition**: IN→key —; OUT→value 9.9(0.052), 7.10(0.049), 11.2(0.047), 7.2(0.047).

**E · redundancy** (task `successor`): solo 3.0(+0.00), 1.11(-0.00), 0.5(-0.01), 0.1(-0.03), 0.10(-0.06); cumulative 1h +0.00 → 2h +0.01 → 3h -0.00 → 4h -0.04 → 5h -0.25 → DISTRIBUTED population (full -0.25 ≫ best single +0.00).

**F · cross-model**: gpt2 sig 0.59; gpt2-medium sig 0.89/gain +12.6; Qwen2.5-1.5B sig 0.99/gain +14.1

**G · SAE operands**: NOT RUN — needs a SAE (sae_lens / Gemma Scope); the op's feature-space read/write operands are the next layer.


_Data: `runs/disassembly/operators/dossiers/duplicate/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._