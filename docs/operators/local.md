# Operator `local`

**positional** operator (universal/addressing — measured across all models in the catalog).

## Cross-model (catalog row) — signal/causal are mean ± σ over 3 probe-resample seeds

| model | arch | signal (±σ) | #heads | top head | depth | causal ΔNLL (±σ) |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 0.336 ± 0.000 | 15 | 4.11 | 0.36 | -0.009 ± 0.006 |
| gpt2-medium | GPT-2/absolute | 0.338 ± 0.000 | 29 | 5.11 | 0.22 | +0.030 ± 0.006 |
| gpt2-large | GPT-2/absolute | 0.338 ± 0.000 | 44 | 14.1 | 0.40 | +0.012 ± 0.001 |
| gemma-2-2b | RoPE | 0.315 ± 0.000 | 49 | 0.0 | 0.00 | +0.452 ± 0.032 |
| Llama-3.2-1B | RoPE | 0.273 ± 0.000 | 8 | 0.2 | 0.00 | +0.010 ± 0.005 |
| Qwen2.5-1.5B | RoPE | 0.286 ± 0.001 | 22 | 1.4 | 0.04 | +0.216 ± 0.029 |

## SAE-feature operands (GPT-2 section G)

Top head 4.11 reads SAE feature(s) `US`, `cius/ius`, `I`; the OV copy-score on that feature's own tokens is **+0.10** (copies it). The feature-space operand basis (monosemantic features, not tokens) via the per-layer GPT-2 SAEs — see the [full SAE-operand table](sae_operands.md) for every operator. _Provisional, single corpus; for positional/addressing ops the read-feature is incidental (they attend by position, not content)._


_Data: `runs/disassembly/operators/dossiers/local/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._