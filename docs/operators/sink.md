# Operator `sink`

**addressing** — attention sink: park attention on key-0 (the no-op / idle register)

## Cross-model (catalog row) — signal/causal are mean ± σ over 3 probe-resample seeds

| model | arch | signal (±σ) | #heads | top head | depth | causal ΔNLL (±σ) |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 0.959 ± 0.000 | 117 | 7.2 | 0.64 | +0.024 ± 0.006 |
| gpt2-medium | GPT-2/absolute | 0.965 ± 0.000 | 335 | 9.9 | 0.39 | +0.000 ± 0.002 |
| gpt2-large | GPT-2/absolute | 0.972 ± 0.002 | 555 | 19.4 | 0.54 | +0.000 ± 0.001 |
| gemma-2-2b | RoPE | 0.059 ± 0.001 | 0 | 0.3 | 0.00 | -0.024 ± 0.043 |
| Llama-3.2-1B | RoPE | 0.997 ± 0.000 | 446 | 5.11 | 0.33 | -0.003 ± 0.002 |
| Qwen2.5-1.5B | RoPE | 0.990 ± 0.012 | 292 | 14.5 | 0.52 | -0.001 ± 0.002 |

## Cross-model deep dossier (arch-generic) — `operator_dossier_xmodel.py`

The deep battery's arch-generic core — behavioural head-ID + mean-ablation causal + the faithful key-only path-patch channel (the model re-applies its own RoPE) — run across **every** model, not just GPT-2. (The full A–F dossier below stays GPT-2-only: its channel/composition math is written against GPT-2's fused-QKV layout, and the named *output* ops have no published head-set off GPT-2.)

| model | top head | #heads (mass≥thr) | causal induction ΔNLL | causal generic ΔNLL | redundancy (top heads) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |
|---|---|---|---|---|---|---|---|
| gpt2 | 7.2 | 136 | +3.10 | +0.03 | distributed (full +3.10 ≫ best 1h +1.31) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| gpt2-medium | 9.9 | 378 | +0.41 | +0.01 | distributed (full +0.41 ≫ best 1h +0.12) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| gpt2-large | 19.4 | 687 | +0.17 | +0.00 | distributed (full +0.17 ≫ best 1h +0.04) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| gemma-2-2b | 1.0 | 67 | -0.18 | -0.49 | **compensatory** (peak +0.62@2h → full -0.18; self-repair) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| Llama-3.2-1B | 5.11 | 508 | +0.42 | -0.00 | distributed (full +0.42 ≫ best 1h +0.22) | — (addresses by position/key-0) | — (addresses by position/key-0) |
| Qwen2.5-1.5B | 14.5 | 312 | +0.19 | -0.00 | distributed (full +0.19 ≫ best 1h +0.08) | — (addresses by position/key-0) | — (addresses by position/key-0) |

_Mean-ablate the op's top behavioural heads → induction-NLL / generic-NLL damage; **redundancy** cumulative-ablates the top heads in solo-effect order (bottleneck = one head ≈ the whole op; distributed = the population far exceeds any single head); channel = remove each upstream head from the reader's key → top collapser + the value/move channel. Data: [xmodel_dossiers_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/xmodel_dossiers_summary.json). Regenerate: [operator_dossier_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_dossier_xmodel.py)._

## Deep dossier (GPT-2) — `operator_dossier.py --op sink`

**A · identity** (behavioural: top heads by attention mass on the sink pattern (>0.02)): heads ['7.2', '5.1', '6.9', '7.10', '9.9']. ranked: 7.2 (0.94), 5.1 (0.94), 6.9 (0.89), 7.10 (0.87), 9.9 (0.85), 9.6 (0.82)

**B · causal × tasks** (* = beyond random control): generic +0.01, induction +4.60*, copy_names +11.55*, successor +0.03, ioi +0.07  → serves **['induction', 'copy_names']**

**C · channels** (reader 7.2): KEY/match top 0.9 collapse +4% (concentration 261.9×); VALUE/move top 5.9 ΔV-out 0.34 (median 0.06).

**D · composition**: IN→key 4.11(0.089), 4.7(0.087), 4.6(0.073), 5.6(0.072); OUT→value 9.3(0.043), 10.9(0.042), 8.7(0.042), 11.8(0.039).

**E · redundancy** (task `generic`): solo 5.1(+0.00), 6.9(+0.00), 7.10(+0.00), 7.2(+0.00), 9.9(-0.00); cumulative 1h +0.00 → 2h +0.00 → 3h +0.01 → 4h +0.01 → 5h +0.01 → DISTRIBUTED population (full +0.01 ≫ best single +0.00).

**G · SAE operands**: NOT RUN — needs a SAE (sae_lens / Gemma Scope); the op's feature-space read/write operands are the next layer.


_Data: `runs/disassembly/operators/dossiers/sink/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._