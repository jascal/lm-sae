# Circuit `induction` (cross-model)

> **`circuit:induction`** — this is the **circuit** (a *composition* of operators: a writer-op feeding a reader-op's K/Q/V port). Not the [`induction` *operator*](../operators/induction.md), which is the head *class* this circuit is named after (the circuit is keyed by its **reader operator**). `circuit:induction` here vs `op:induction` there.

prev-token head --K--> induction head (the in-context-copy macro)

**Defining edge:** `prevtok_head -> induction (K)`

## Cross-model edge liveness (path-patch: remove the writer from the reader's key → attention collapse)

| model | reader | writer | key collapse | writer is | value mover | value ΔV-out |
|---|---|---|---|---|---|---|
| gpt2 | 7.11 | 4.11 | +17% | prev-tok head | 1.5 | 0.26 |
| gpt2-medium | 7.2 | 2.15 | +23% | sink | 2.14 | 0.15 |
| gpt2-large | 11.19 | 3.14 | +8% | sink | 3.8 | 0.10 |
| gemma-2-2b | 4.4 | 0.0 | +18% | sink | 3.0 | 0.12 |
| Llama-3.2-1B | 2.26 | 1.20 | +70% | sink | 1.20 | 0.17 |
| Qwen2.5-1.5B | 2.3 | 1.4 | +89% | sink | 1.4 | 0.24 |

## Stage redundancy (GPT-2, `rung3_induction_chain.py`)
3-stage chain: prev-token population (17 heads) → stage-2 reader `[7, 11]` (bottleneck) → inductors. Writers are individually redundant, collectively necessary; copy-score↔induction ρ 0.2676754280202556.

_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json). Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._