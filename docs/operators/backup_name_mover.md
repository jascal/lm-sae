# Operator `backup_name_mover`

**output** — IOI backup name-mover: the self-repair spares that wake when primaries are ablated

GPT-2 literature DLA head-set: 9.0, 9.7, 10.1, 10.2, 10.6, 11.2. No published head-set in the RoPE models — not in the cross-model catalog.

## SAE-feature operands (section G)

What this operator reads/writes in **feature** space (monosemantic SAE latents), via the per-layer GPT-2 SAEs / Gemma Scope — see the [full SAE-operand table](sae_operands.md). _READ = dominant key-feature where the head attends (glossed by top tokens); copy-score = OV→unembed on those tokens (+ copies / − suppresses). Provisional, single corpus; for positional/addressing ops the read-feature is incidental._

| model | head | reads (SAE feature) | copy-score |
|---|---|---|---|
| gpt2 | 9.0 | `_it/'d/_are`; `_Citizen/_citizens`; `US/us` | +0.03 (copies) |

## Deep dossier (GPT-2) — `operator_dossier.py --op backup_name_mover`

**A · identity** (circuit op — heads from literature (DLA-defined; not attention-mask-readable)): heads ['9.0', '9.7', '10.1', '10.2', '10.6', '11.2']. ranked: 9.0, 9.7, 10.1, 10.2, 10.6, 11.2

**B · causal × tasks** (* = beyond random control): generic -0.00, induction +0.15, copy_names +0.20, successor +0.05*, ioi +0.15  → serves **['successor']**

**C · channels**: output/circuit op — carried by OV→unembedding, not a key/value match (see composition out-edges).

**D · composition**: IN→key 5.10(0.063), 5.9(0.055), 5.11(0.054), 7.1(0.053); OUT→value 11.3(0.074), 10.11(0.053), 11.11(0.049), 10.0(0.042).

**E · redundancy** (task `ioi`): solo 10.6(+0.10), 9.7(+0.04), 10.1(+0.02), 10.2(+0.02), 9.0(+0.02), 11.2(-0.06); cumulative 1h +0.10 → 2h +0.15 → 3h +0.17 → 4h +0.20 → 5h +0.22 → 6h +0.15 → DISTRIBUTED population (full +0.15 ≫ best single +0.10).


_Data: `runs/disassembly/operators/dossiers/backup_name_mover/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._