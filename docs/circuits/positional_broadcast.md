# Circuit `positional_broadcast` (cross-model)

early sink/write-hub --K--> prev-token head's key (absolute-position broadcast)

**Defining edge:** `sink-writer -> prevtok key (K)`

## Cross-model edge liveness (path-patch: remove the writer from the reader's key → attention collapse)

| model | reader | writer | key collapse | writer is | value mover | value ΔV-out |
|---|---|---|---|---|---|---|
| gpt2 | 4.11 | 1.3 | +22% | sink | 2.9 | 0.22 |
| gpt2-medium | 5.11 | 1.5 | +32% | sink | 2.14 | 0.22 |
| gpt2-large | 14.1 | 3.0 | +0% | sink | 3.0 | 0.07 |
| gemma-2-2b | 21.7 | 5.4 | +0% | sink | 0.2 | 0.05 |
| Llama-3.2-1B | 0.2 | — | (skipped) | — | — | — |
| Qwen2.5-1.5B | 13.4 | 0.0 | +0% | sink | 5.1 | 0.11 |

_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json). Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._