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

## Deep dossier (GPT-2) — `operator_dossier.py --op prevtok`

**A · identity** (behavioural: top heads by attention mass on the prevtok pattern (>0.02)): heads ['4.11', '2.2', '3.2', '3.7', '2.9']. ranked: 4.11 (0.96), 2.2 (0.54), 3.2 (0.38), 3.7 (0.37), 2.9 (0.33), 1.0 (0.32)

**B · causal × tasks** (* = beyond random control): generic +0.00, induction +2.61*, copy_names +4.76, successor +0.03, ioi +0.53*  → serves **['induction', 'ioi']**

**C · channels** (reader 4.11): KEY/match top 1.10 collapse +18% (concentration 106.3×); VALUE/move top 2.9 ΔV-out 0.21 (median 0.11).

**D · composition**: IN→key 0.11(0.114), 1.8(0.105), 1.3(0.094), 1.9(0.091); OUT→value 8.3(0.064), 5.2(0.062), 5.8(0.057), 9.4(0.056).

**E · redundancy** (task `copy_names`): solo 4.11(+1.78), 3.7(+0.06), 2.9(+0.03), 2.2(-0.00), 3.2(-0.00); cumulative 1h +1.78 → 2h +2.61 → 3h +3.45 → 4h +5.11 → 5h +4.76 → DISTRIBUTED population (full +4.76 ≫ best single +1.78).

**G · SAE operands**: NOT RUN — needs a SAE (sae_lens / Gemma Scope); the op's feature-space read/write operands are the next layer.


_Data: `runs/disassembly/operators/dossiers/prevtok/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._