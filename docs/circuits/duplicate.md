# Circuit `duplicate` (cross-model)

same-token reader (duplicate-token detection; IOI initiator)

**Defining edge:** `(reader-side; writer often layer-0)`

## Cross-model edge liveness (path-patch: remove the writer from the reader's key → attention collapse)

| model | reader | writer | key collapse | writer is | value mover | value ΔV-out |
|---|---|---|---|---|---|---|
| gpt2 | 0.5 | — | (skipped) | — | — | — |
| gpt2-medium | 1.11 | 0.13 | +0% | sink | 0.14 | 0.02 |
| gpt2-large | 6.17 | 1.1 | +4% | sink | 3.1 | 0.40 |
| gemma-2-2b | 1.4 | 0.2 | +3% | sink | 0.0 | 0.19 |
| Llama-3.2-1B | 1.22 | 0.17 | +9% | sink | 0.17 | 0.28 |
| Qwen2.5-1.5B | 6.6 | 0.6 | +13% | sink | 3.8 | 0.12 |

_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json). Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._