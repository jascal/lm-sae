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

## Cross-model causal dossier (necessity / sufficiency / redundancy — via the ResidualVM)

The operator-dossier battery, lifted to this circuit and run on the [unified `ResidualVM`](../DECOMPILATION.md) (`find_heads` locates the heads, `ablate_heads` + `nll` measure the rest). Two next-token metrics: **induction-NLL** (in-context copy) and **generic-NLL** (general LM).

| model | reader | necessity Δind-NLL | necessity Δgen-NLL | sufficiency (keep-only, ind) | reader redundancy |
|---|---|---|---|---|---|
| gpt2 | 5.1 | +5.02 | +0.06 | +15% | distributed |
| gpt2-medium | 11.1 | +2.03 | +0.07 | +5% | distributed |
| gpt2-large | 16.0 | +0.87 | +0.02 | +0% | bottleneck |
| gpt2-xl | 21.3 | +0.32 | +0.00 | +0% | distributed |
| gemma-2-2b | 6.3 | +1.64 | +0.12 | +8% | bottleneck |
| Llama-3.2-1B | 10.23 | +1.66 | -0.02 | +3% | distributed |
| Qwen2.5-1.5B | 14.3 | +3.51 | +0.01 | -2% | distributed |

- **Necessity** — Δ NLL when the circuit's heads are mean-ablated (higher = more load-bearing for that behaviour). Generic-NLL necessity is small everywhere — these circuits are *task-specific*, not general-LM.
- **Sufficiency** — reconstruction coverage keeping **only** the circuit's heads (MLPs intact); a small head-set that reconstructs the behaviour is an executable decompilation. (Generic-NLL coverage is omitted as a headline — with MLPs intact a tiny head-set scores high for reasons unrelated to the circuit; induction-NLL is the meaningful attention-circuit metric. Negative = keeping so few heads is worse than the all-ablated floor, the known keep-1-is-net-negative effect.)
- **Redundancy** — reader-head solo-vs-cumulative on induction-NLL: *bottleneck* = one head carries it, *distributed* = the population shares it.

**The induction circuit's necessity AND sufficiency both decay monotonically across the GPT-2 ladder** (gpt2, gpt2-medium, gpt2-large, gpt2-xl): necessity Δind-NLL +5.02 → +2.03 → +0.87 → +0.32; sufficiency +15% → +5% → +0% → +0%. The same scale-driven distributedness the rest of the catalog finds — the named circuit is most localized in the smallest model and dissolves into the network with scale.

_Dossier data: [runs/disassembly/circuits/dossier_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/dossier_summary.json) ([circuit_dossier_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_dossier_xmodel.py), built on the ResidualVM)._

_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json). Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._