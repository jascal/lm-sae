# Operator `self`

**addressing** operator (universal/addressing — measured across all models in the atlas).

## Cross-model (atlas row)

| model | arch | signal | #heads | top head | depth | causal ΔNLL |
|---|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 0.831 | 10 | 0.1 | 0.00 | +0.053 |
| gpt2-medium | GPT-2/absolute | 0.446 | 21 | 5.13 | 0.22 | +0.035 |
| gpt2-large | GPT-2/absolute | 0.547 | 74 | 0.14 | 0.00 | +0.000 |
| gemma-2-2b | RoPE | 0.950 | 149 | 25.7 | 1.00 | +0.509 |
| Llama-3.2-1B | RoPE | 0.956 | 58 | 15.14 | 1.00 | +0.102 |
| Qwen2.5-1.5B | RoPE | 1.000 | 43 | 15.7 | 0.56 | +3.771 |


_Data: `runs/disassembly/operators/dossiers/self/` + the atlas. Regenerate: `operator_catalog_doc.py`._