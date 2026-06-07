# Operator `s_inhibition`

**output** — IOI S-inhibition: suppress the subject so the name-mover writes IO

GPT-2-only circuit op (literature DLA head-set): 7.3, 7.9, 8.6, 8.10. No published head-set in the RoPE models — not in the cross-model atlas.

## Deep dossier (GPT-2) — `operator_dossier.py --op s_inhibition`

**A · identity** (output op — heads from literature (DLA-defined; not attention-mask-readable)): heads ['7.3', '7.9', '8.6', '8.10']. ranked: 7.3, 7.9, 8.6, 8.10

**B · causal × tasks** (* = beyond random control): generic +0.01, induction -0.11, copy_names -0.13, successor -0.05, ioi +0.65*  → serves **['ioi']**

**C · channels**: output/circuit op — carried by OV→unembedding, not a key/value match (see composition out-edges).

**D · composition**: IN→key 0.11(0.060), 1.8(0.059), 5.3(0.058), 2.6(0.057); OUT→value 8.7(0.070), 9.3(0.064), 8.5(0.059), 9.10(0.044).

**E · redundancy** (task `ioi`): solo 7.9(+0.25), 8.10(+0.19), 8.6(+0.16), 7.3(+0.07); cumulative 1h +0.25 → 2h +0.43 → 3h +0.57 → 4h +0.65 → DISTRIBUTED population (full +0.65 ≫ best single +0.25).

**G · SAE operands**: NOT RUN — needs a SAE (sae_lens / Gemma Scope); the op's feature-space read/write operands are the next layer.


_Data: `runs/disassembly/operators/dossiers/s_inhibition/` + the atlas. Regenerate: `operator_catalog_doc.py`._