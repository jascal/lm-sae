# Operator `duplicate`

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

## Deep dossier (GPT-2) — `operator_dossier.py --op duplicate`

**A · identity** (behavioural: top heads by attention mass on the duplicate pattern (>0.02)): heads ['0.5', '3.0', '0.1', '1.11', '0.10']. ranked: 0.5 (0.59), 3.0 (0.53), 0.1 (0.32), 1.11 (0.18), 0.10 (0.11), 1.5 (0.08)

**B · causal × tasks** (* = beyond random control): generic +0.07*, induction +0.57, copy_names +3.15*, successor -0.25, ioi +0.35*  → serves **['generic', 'copy_names', 'ioi']**

**C · channels**: reader in layer 0 — no upstream; channel skipped

**D · composition**: IN→key —; OUT→value 9.9(0.052), 7.10(0.049), 11.2(0.047), 7.2(0.047).

**E · redundancy** (task `successor`): solo 3.0(+0.00), 1.11(-0.00), 0.5(-0.01), 0.1(-0.03), 0.10(-0.06); cumulative 1h +0.00 → 2h +0.01 → 3h -0.00 → 4h -0.04 → 5h -0.25 → DISTRIBUTED population (full -0.25 ≫ best single +0.00).

**F · cross-model**: gpt2 sig 0.59; gpt2-medium sig 0.89/gain +12.6; Qwen2.5-1.5B sig 0.99/gain +14.1

**G · SAE operands**: NOT RUN — needs a SAE (sae_lens / Gemma Scope); the op's feature-space read/write operands are the next layer.


_Data: `runs/disassembly/operators/dossiers/duplicate/` + the catalog. Regenerate: `operator_catalog_doc.py`._