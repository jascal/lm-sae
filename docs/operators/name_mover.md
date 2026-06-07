# Operator `name_mover`

**output** — IOI name-mover: copy the indirect-object name to the logits (output head)

GPT-2-only circuit op (literature DLA head-set): 9.6, 9.9, 10.0, 10.10. No published head-set in the RoPE models — not in the cross-model catalog.

## SAE-feature operands (GPT-2 section G)

Top head 9.6 reads SAE feature(s) `_Citizen/_citizens`, `MEN/_men/_Men`, `_it/'d/_are`; the OV copy-score on that feature's own tokens is **+0.04** (copies it). The feature-space operand basis (monosemantic features, not tokens) via the per-layer GPT-2 SAEs — see the [full SAE-operand table](sae_operands.md) for every operator. _Provisional, single corpus; for positional/addressing ops the read-feature is incidental (they attend by position, not content)._

## Deep dossier (GPT-2) — `operator_dossier.py --op name_mover`

**A · identity** (output op — heads from literature (DLA-defined; not attention-mask-readable)): heads ['9.6', '9.9', '10.0', '10.10']. ranked: 9.6, 9.9, 10.0, 10.10

**B · causal × tasks** (* = beyond random control): generic -0.00, induction +0.10, copy_names +0.54, successor +0.01, ioi +0.05  → serves **none**

**C · channels**: output/circuit op — carried by OV→unembedding, not a key/value match (see composition out-edges).

**D · composition**: IN→key 4.11(0.063), 4.7(0.063), 5.6(0.063), 3.7(0.058); OUT→value 11.3(0.064), 11.11(0.039), 11.10(0.037), 10.11(0.033).

**E · redundancy** (task `ioi`): solo 10.0(+0.11), 9.9(+0.07), 10.10(+0.01), 9.6(-0.11); cumulative 1h +0.11 → 2h +0.18 → 3h +0.19 → 4h +0.05 → BOTTLENECK (one head ≈ whole op).


_Data: `runs/disassembly/operators/dossiers/name_mover/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._