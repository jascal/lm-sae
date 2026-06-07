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

## SAE-feature operands (section G)

What this operator reads/writes in **feature** space (monosemantic SAE latents), via the per-layer GPT-2 SAEs / Gemma Scope — see the [full SAE-operand table](sae_operands.md). _READ = dominant key-feature where the head attends (glossed by top tokens); copy-score = OV→unembed on those tokens (+ copies / − suppresses). Provisional, single corpus; for positional/addressing ops the read-feature is incidental._

| model | head | reads (SAE feature) | copy-score |
|---|---|---|---|
| gpt2 | 4.11 | `US`; `cius/ius`; `I` | +0.10 (copies) |
| gemma-2-2b | 0.0 | `cius/▁belly/VIR`; `UMN/GIL/▁Cai`; `▁the` | +0.02 (≈neutral) |


_Data: `runs/disassembly/operators/dossiers/local/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._