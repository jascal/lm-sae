# Operator `structural`

**structural** operator (universal/addressing — measured across all models in the catalog).

## Cross-model (catalog row) — signal/causal are mean ± σ over 3 probe-resample seeds

| model | arch | signal (±σ) | #heads | top head | depth | causal ΔNLL (±σ) |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 0.199 ± 0.001 | 2 | 3.1 | 0.27 | -0.012 ± 0.005 |
| gpt2-medium | GPT-2/absolute | 0.255 ± 0.001 | 4 | 3.1 | 0.13 | -0.032 ± 0.002 |
| gpt2-large | GPT-2/absolute | 0.391 ± 0.007 | 9 | 4.7 | 0.11 | -0.002 ± 0.004 |
| gemma-2-2b | RoPE | 0.238 ± 0.003 | 11 | 24.6 | 0.96 | -0.120 ± 0.012 |
| Llama-3.2-1B | RoPE | 0.351 ± 0.003 | 19 | 0.31 | 0.00 | +1.382 ± 0.450 |
| Qwen2.5-1.5B | RoPE | 0.172 ± 0.017 | 4 | 25.7 | 0.93 | -0.055 ± 0.012 |


_Data: `runs/disassembly/operators/dossiers/structural/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._