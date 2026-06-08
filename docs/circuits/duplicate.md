# Circuit `duplicate` (cross-model)

> **`circuit:duplicate`** — this is the **circuit** (a *composition* of operators: a writer-op feeding a reader-op's K/Q/V port). Not the [`duplicate` *operator*](../operators/duplicate.md), which is the head *class* this circuit is named after (the circuit is keyed by its **reader operator**). `circuit:duplicate` here vs `op:duplicate` there.

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

## Cross-model causal dossier (necessity / sufficiency / redundancy — via the ResidualVM)

The operator-dossier battery, lifted to this circuit and run on the [unified `ResidualVM`](../DECOMPILATION.md) (`find_heads` locates the heads, `ablate_heads` + `nll` measure the rest). Two next-token metrics: **induction-NLL** (in-context copy) and **generic-NLL** (general LM).

| model | reader | necessity Δind-NLL | necessity Δgen-NLL | sufficiency (keep-only, ind) | reader redundancy |
|---|---|---|---|---|---|
| gpt2 | 0.5 | +0.46 | +0.22 | +1% | distributed |
| gpt2-medium | 7.11 | +0.16 | -0.01 | +1% | distributed |
| gpt2-large | 5.8 | +0.41 | +0.00 | +0% | distributed |
| gpt2-xl | 4.12 | +0.35 | +0.00 | +1% | distributed |
| gemma-2-2b | 1.4 | -0.39 | -0.83 | -11% | distributed |
| Llama-3.2-1B | 0.9 | +0.97 | +0.03 | -1% | distributed |
| Qwen2.5-1.5B | 8.3 | +1.32 | -0.01 | -3% | distributed |

- **Necessity** — Δ NLL when the circuit's heads are mean-ablated (higher = more load-bearing for that behaviour). Generic-NLL necessity is small everywhere — these circuits are *task-specific*, not general-LM.
- **Sufficiency** — reconstruction coverage keeping **only** the circuit's heads (MLPs intact); a small head-set that reconstructs the behaviour is an executable decompilation. (Generic-NLL coverage is omitted as a headline — with MLPs intact a tiny head-set scores high for reasons unrelated to the circuit; induction-NLL is the meaningful attention-circuit metric. Negative = keeping so few heads is worse than the all-ablated floor, the known keep-1-is-net-negative effect.)
- **Redundancy** — reader-head solo-vs-cumulative on induction-NLL: *bottleneck* = one head carries it, *distributed* = the population shares it.

_Dossier data: [runs/disassembly/circuits/dossier_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/dossier_summary.json) ([circuit_dossier_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_dossier_xmodel.py), built on the ResidualVM)._

_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json). Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._