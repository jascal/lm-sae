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

## Cross-model causal dossier (necessity / sufficiency / redundancy — via the ResidualVM)

The operator-dossier battery, lifted to this circuit and run on the [unified `ResidualVM`](../DECOMPILATION.md) (`find_heads` locates the heads, `ablate_heads` + `nll` measure the rest). Two next-token metrics: **induction-NLL** (in-context copy) and **generic-NLL** (general LM).

| model | reader | necessity Δind-NLL | necessity Δgen-NLL | sufficiency (keep-only, ind) | reader redundancy |
|---|---|---|---|---|---|
| gpt2 | 4.11 | +2.46 | +0.04 | +5% | distributed |
| gpt2-medium | 5.11 | +0.80 | -0.05 | +2% | distributed |
| gpt2-large | 14.1 | +0.15 | +0.01 | +0% | distributed |
| gpt2-xl | 12.21 | +0.17 | +0.00 | +0% | distributed |
| gemma-2-2b | 21.7 | +0.68 | +0.22 | -1% | bottleneck |
| Llama-3.2-1B | 0.2 | +0.28 | -0.02 | +1% | distributed |
| Qwen2.5-1.5B | 13.4 | +0.56 | +0.01 | -13% | distributed |

- **Necessity** — Δ NLL when the circuit's heads are mean-ablated (higher = more load-bearing for that behaviour). Generic-NLL necessity is small everywhere — these circuits are *task-specific*, not general-LM.
- **Sufficiency** — reconstruction coverage keeping **only** the circuit's heads (MLPs intact); a small head-set that reconstructs the behaviour is an executable decompilation. (Generic-NLL coverage is omitted as a headline — with MLPs intact a tiny head-set scores high for reasons unrelated to the circuit; induction-NLL is the meaningful attention-circuit metric. Negative = keeping so few heads is worse than the all-ablated floor, the known keep-1-is-net-negative effect.)
- **Redundancy** — reader-head solo-vs-cumulative on induction-NLL: *bottleneck* = one head carries it, *distributed* = the population shares it.

_Dossier data: [runs/disassembly/circuits/dossier_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/dossier_summary.json) ([circuit_dossier_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_dossier_xmodel.py), built on the ResidualVM)._

_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json). Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._