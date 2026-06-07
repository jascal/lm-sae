# Operator `coreference`

**content** — coreference (exploratory): pronoun -> earlier antecedent (no clean task probe here)

GPT-2-only circuit op (literature DLA head-set): 9.0. No published head-set in the RoPE models — not in the cross-model catalog.

## Deep dossier (GPT-2) — `operator_dossier.py --op coreference`

**A · identity** (circuit op — heads from literature (DLA-defined; not attention-mask-readable)): heads ['9.0']. ranked: 9.0

**B · causal × tasks** (* = beyond random control): generic +0.00, induction +0.01, copy_names +0.05, successor -0.00, ioi +0.02  → serves **none**

**C · channels**: output/circuit op — carried by OV→unembedding, not a key/value match (see composition out-edges).

**D · composition**: IN→key 5.10(0.063), 5.9(0.055), 5.11(0.054), 7.1(0.053); OUT→value 11.3(0.074), 10.11(0.053), 11.11(0.049), 10.0(0.042).

**E · redundancy** (task `generic`): solo 9.0(+0.00); cumulative 1h +0.00 → DISTRIBUTED population (full +0.00 ≫ best single +0.00).

**G · SAE operands**: NOT RUN — needs a SAE (sae_lens / Gemma Scope); the op's feature-space read/write operands are the next layer.


_Data: `runs/disassembly/operators/dossiers/coreference/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._