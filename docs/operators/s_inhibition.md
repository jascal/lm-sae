# Operator `s_inhibition`

**output** — IOI S-inhibition: suppress the subject so the name-mover writes IO

GPT-2 literature DLA head-set: 7.3, 7.9, 8.6, 8.10. The RoPE head-set is now found **behaviourally** (below).

## Cross-model (found behaviourally — IOI dossier)

The literature head-set above is GPT-2. The unified [`ResidualVM`](../DECOMPILATION.md) locates this operator by the **ablation sweep** (the most logit-diff-load-bearing heads — the S-inhibition that lets the name-movers write IO) in **every** model ([cross-model IOI dossier](../circuits/ioi_q_chain.md)) — so it is no longer GPT-2-only:

| model | heads (top) |
|---|---|
| gpt2 | `8.10`, `8.6`, `7.9`, `5.5` |
| gpt2-medium | `19.1`, `12.3`, `13.4`, `13.13` |
| gpt2-large | `20.14`, `18.3`, `24.17`, `17.19` |
| gemma-2-2b | `23.5`, `20.6`, `16.2`, `14.3` |
| Llama-3.2-1B | `8.19`, `11.4`, `8.17`, `12.13` |
| Qwen2.5-1.5B | `24.8`, `0.6`, `11.8`, `13.4` |

## SAE-feature operands (section G)

What this operator reads/writes in **feature** space (monosemantic SAE latents), via the per-layer GPT-2 SAEs / Gemma Scope — see the [full SAE-operand table](sae_operands.md). _READ = dominant key-feature where the head attends (glossed by top tokens); copy-score = OV→unembed on those tokens (+ copies / − suppresses). Provisional, single corpus; for positional/addressing ops the read-feature is incidental._

| model | head | reads (SAE feature) | copy-score |
|---|---|---|---|
| gpt2 | 7.3 | `US/us`; `MAR/MEN/CI`; `_you/you/You` | +0.01 (≈neutral) |

## Deep dossier (GPT-2) — `operator_dossier.py --op s_inhibition`

**A · identity** (output op — heads from literature (DLA-defined; not attention-mask-readable)): heads ['7.3', '7.9', '8.6', '8.10']. ranked: 7.3, 7.9, 8.6, 8.10

**B · causal × tasks** (* = beyond random control): generic +0.01, induction -0.11, copy_names -0.13, successor -0.05, ioi +0.65*  → serves **['ioi']**

**C · channels**: output/circuit op — carried by OV→unembedding, not a key/value match (see composition out-edges).

**D · composition**: IN→key 0.11(0.060), 1.8(0.059), 5.3(0.058), 2.6(0.057); OUT→value 8.7(0.070), 9.3(0.064), 8.5(0.059), 9.10(0.044).

**E · redundancy** (task `ioi`): solo 7.9(+0.25), 8.10(+0.19), 8.6(+0.16), 7.3(+0.07); cumulative 1h +0.25 → 2h +0.43 → 3h +0.57 → 4h +0.65 → DISTRIBUTED population (full +0.65 ≫ best single +0.25).


_Data: `runs/disassembly/operators/dossiers/s_inhibition/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._