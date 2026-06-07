# Operator `self`

**addressing** operator (universal/addressing — measured across all models in the catalog).

## Cross-model (catalog row) — signal/causal are mean ± σ over 3 probe-resample seeds

| model | arch | signal (±σ) | #heads | top head | depth | causal ΔNLL (±σ) |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 0.839 ± 0.001 | 10 | 0.1 | 0.00 | +0.024 ± 0.010 |
| gpt2-medium | GPT-2/absolute | 0.447 ± 0.000 | 21 | 5.13 | 0.22 | +0.033 ± 0.008 |
| gpt2-large | GPT-2/absolute | 0.549 ± 0.003 | 74 | 0.14 | 0.00 | +0.012 ± 0.002 |
| gemma-2-2b | RoPE | 0.966 ± 0.001 | 154 | 25.7 | 1.00 | +0.476 ± 0.027 |
| Llama-3.2-1B | RoPE | 0.956 ± 0.001 | 60 | 15.14 | 1.00 | +0.103 ± 0.003 |
| Qwen2.5-1.5B | RoPE | 0.999 ± 0.000 | 43 | 15.7 | 0.56 | +3.733 ± 0.059 |


_Data: `runs/disassembly/operators/dossiers/self/` + the catalog. Regenerate: `operator_catalog_doc.py`._