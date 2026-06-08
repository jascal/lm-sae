# Operator `name_mover`

**output** — IOI name-mover: copy the indirect-object name to the logits (output head)

GPT-2 literature DLA head-set: 9.6, 9.9, 10.0, 10.10. The RoPE head-set is now found **behaviourally** (below).

## Cross-model (found behaviourally — IOI dossier)

The literature head-set above is GPT-2. The unified [`ResidualVM`](../DECOMPILATION.md) locates this operator by **end→indirect-object copy-attention** (heads that copy the IO name) in **every** model ([cross-model IOI dossier](../circuits/ioi_q_chain.md)) — so it is no longer GPT-2-only:

| model | heads (top) |
|---|---|
| gpt2 | `9.9`, `10.0`, `10.6`, `10.10` |
| gpt2-medium | `15.14`, `18.5`, `20.6`, `20.0` |
| gpt2-large | `22.0`, `29.0`, `23.13`, `21.8` |
| gemma-2-2b | `18.6`, `21.5`, `22.5`, `17.3` |
| Llama-3.2-1B | `12.13`, `12.2`, `11.14`, `10.13` |
| Qwen2.5-1.5B | `27.4`, `27.8`, `24.10`, `27.1` |

## SAE-feature operands (section G)

What this operator reads/writes in **feature** space (monosemantic SAE latents), via the per-layer GPT-2 SAEs / Gemma Scope — see the [full SAE-operand table](sae_operands.md). _READ = dominant key-feature where the head attends (glossed by top tokens); copy-score = OV→unembed on those tokens (+ copies / − suppresses). Provisional, single corpus; for positional/addressing ops the read-feature is incidental._

| model | head | reads (SAE feature) | copy-score |
|---|---|---|---|
| gpt2 | 9.6 | `_Citizen/_citizens`; `MEN/_men/_Men`; `_it/'d/_are` | +0.04 (copies) |

## Deep dossier (GPT-2) — `operator_dossier.py --op name_mover`

**A · identity** (output op — heads from literature (DLA-defined; not attention-mask-readable)): heads ['9.6', '9.9', '10.0', '10.10']. ranked: 9.6, 9.9, 10.0, 10.10

**B · causal × tasks** (* = beyond random control): generic -0.00, induction +0.10, copy_names +0.54, successor +0.01, ioi +0.05  → serves **none**

**C · channels**: output/circuit op — carried by OV→unembedding, not a key/value match (see composition out-edges).

**D · composition**: IN→key 4.11(0.063), 4.7(0.063), 5.6(0.063), 3.7(0.058); OUT→value 11.3(0.064), 11.11(0.039), 11.10(0.037), 10.11(0.033).

**E · redundancy** (task `ioi`): solo 10.0(+0.11), 9.9(+0.07), 10.10(+0.01), 9.6(-0.11); cumulative 1h +0.11 → 2h +0.18 → 3h +0.19 → 4h +0.05 → BOTTLENECK (one head ≈ whole op).


_Data: `runs/disassembly/operators/dossiers/name_mover/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._