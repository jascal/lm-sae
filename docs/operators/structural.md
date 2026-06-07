# Operator `structural`

**structural** operator (universal/addressing — measured across all models in the atlas).

## Cross-model (atlas row)

| model | arch | signal | #heads | top head | depth | causal ΔNLL |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 0.192 | 2 | 3.1 | 0.27 | -0.046 |
| gpt2-medium | GPT-2/absolute | 0.247 | 4 | 3.1 | 0.13 | -0.051 |
| gpt2-large | GPT-2/absolute | 0.379 | 6 | 4.7 | 0.11 | +0.001 |
| gemma-2-2b | RoPE | 0.234 | 7 | 24.6 | 0.96 | -0.117 |
| Llama-3.2-1B | RoPE | 0.361 | 3 | 0.31 | 0.00 | +1.473 |
| Qwen2.5-1.5B | RoPE | 0.162 | 1 | 25.7 | 0.93 | -0.059 |


_Data: `runs/disassembly/operators/dossiers/structural/` + the atlas. Regenerate: `operator_catalog_doc.py`._