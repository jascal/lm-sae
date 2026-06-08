# Operator `negative_mover`

**output** — copy-suppression / negative name-mover: writes against the copied token

GPT-2 literature DLA head-set: 10.7, 11.10. The RoPE head-set is now found **behaviourally** (below).

## Cross-model (found behaviourally — IOI dossier)

The literature head-set above is GPT-2. The unified [`ResidualVM`](../DECOMPILATION.md) locates this operator by the **ablation sweep** (end→IO heads whose removal *raises* the IO−S logit-diff = copy-suppression) in **every** model ([cross-model IOI dossier](../circuits/ioi_q_chain.md)) — so it is no longer GPT-2-only:

| model | heads (top) |
|---|---|
| gpt2 | `10.7`, `9.6`, `11.10`, `10.1` |
| gpt2-medium | `18.9`, `22.14`, `16.15`, `20.7` |
| gpt2-large | `32.0`, `26.0`, `29.17`, `30.0` |
| gemma-2-2b | `22.4`, `22.0`, `23.7`, `18.5` |
| Llama-3.2-1B | `12.15`, `15.12`, `12.29`, `10.25` |
| Qwen2.5-1.5B | `23.8`, `25.4`, `22.7`, `27.11` |

## SAE-feature operands (section G)

What this operator reads/writes in **feature** space (monosemantic SAE latents), via the per-layer GPT-2 SAEs / Gemma Scope — see the [full SAE-operand table](sae_operands.md). _READ = dominant key-feature where the head attends (glossed by top tokens); copy-score = OV→unembed on those tokens (+ copies / − suppresses). Provisional, single corpus; for positional/addressing ops the read-feature is incidental._

| model | head | reads (SAE feature) | copy-score |
|---|---|---|---|
| gpt2 | 10.7 | `'d/_not/_and`; `And/That/_and`; `_Citizen/_citizens` | -0.01 (≈neutral) |

## Deep dossier (GPT-2) — `operator_dossier.py --op negative_mover`

**A · identity** (circuit op — heads from literature (DLA-defined; not attention-mask-readable)): heads ['10.7', '11.10']. ranked: 10.7, 11.10

**B · causal × tasks** (* = beyond random control): generic +0.00, induction -0.28, copy_names -0.46, successor -0.02, ioi -0.54  → serves **none**

**C · channels**: output/circuit op — carried by OV→unembedding, not a key/value match (see composition out-edges).

**D · composition**: IN→key 3.0(0.054), 1.9(0.050), 1.6(0.050), 5.11(0.049); OUT→value 11.3(0.044), 11.10(0.044), 11.8(0.032), 11.11(0.032).

**E · redundancy** (task `ioi`): solo 11.10(-0.16), 10.7(-0.21); cumulative 1h -0.16 → 2h -0.54 → DISTRIBUTED population (full -0.54 ≫ best single -0.16).


_Data: `runs/disassembly/operators/dossiers/negative_mover/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._