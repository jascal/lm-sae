# Operator `negative_mover`

**output** — copy-suppression / negative name-mover: writes against the copied token

GPT-2-only circuit op (literature DLA head-set): 10.7, 11.10. No published head-set in the RoPE models — not in the cross-model catalog.

## Deep dossier (GPT-2) — `operator_dossier.py --op negative_mover`

**A · identity** (circuit op — heads from literature (DLA-defined; not attention-mask-readable)): heads ['10.7', '11.10']. ranked: 10.7, 11.10

**B · causal × tasks** (* = beyond random control): generic +0.00, induction -0.28, copy_names -0.46, successor -0.02, ioi -0.54  → serves **none**

**C · channels**: output/circuit op — carried by OV→unembedding, not a key/value match (see composition out-edges).

**D · composition**: IN→key 3.0(0.054), 1.9(0.050), 1.6(0.050), 5.11(0.049); OUT→value 11.3(0.044), 11.10(0.044), 11.8(0.032), 11.11(0.032).

**E · redundancy** (task `ioi`): solo 11.10(-0.16), 10.7(-0.21); cumulative 1h -0.16 → 2h -0.54 → DISTRIBUTED population (full -0.54 ≫ best single -0.16).

**G · SAE operands**: NOT RUN — needs a SAE (sae_lens / Gemma Scope); the op's feature-space read/write operands are the next layer.


_Data: `runs/disassembly/operators/dossiers/negative_mover/` + the catalog. Regenerate: `operator_catalog_doc.py`._