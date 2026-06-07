# Operator `local`

**positional** operator (universal/addressing — measured across all models in the catalog).

## Cross-model (catalog row)

| model | arch | signal | #heads | top head | depth | causal ΔNLL |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 0.335 | 16 | 4.11 | 0.36 | +0.000 |
| gpt2-medium | GPT-2/absolute | 0.338 | 28 | 5.11 | 0.22 | +0.001 |
| gpt2-large | GPT-2/absolute | 0.336 | 42 | 14.1 | 0.40 | +0.016 |
| gemma-2-2b | RoPE | 0.312 | 48 | 0.0 | 0.00 | +0.505 |
| Llama-3.2-1B | RoPE | 0.273 | 8 | 0.2 | 0.00 | -0.003 |
| Qwen2.5-1.5B | RoPE | 0.285 | 22 | 1.4 | 0.04 | +0.196 |


_Data: `runs/disassembly/operators/dossiers/local/` + the catalog. Regenerate: `operator_catalog_doc.py`._