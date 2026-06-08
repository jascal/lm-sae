# Circuit catalog — composed circuits, surveyed & collected across models

Operators are single head-classes ([`../operators/`](../operators/README.md)); **circuits** are their
*compositions* (a writer-op feeding a reader-op's K/Q/V port, chained). A **working catalog** (amateur,
exploratory, provisional — not a definitive reference) of **7 circuits** collected with the tools
here, two sources:

- **Cross-model circuit edges** — the defining composition edge of each universal-reader circuit, path-patched
  across 6 models (faithful key/value patch, arch-generic).
- **GPT-2 discovered / circuit-specific** — harvested from the committed discovery artifacts (`composition_dag`,
  `vcomposition`, `ioi_causal`, `validate_new_edges`): the IOI Q-chain, the V-composition virtual heads, and the
  **22 novel-live edges** the discovery gate found
  (of which 13 are behaviourally named). These are
  GPT-2-only (literature IOI head-sets / GPT-2 path-patch runs).

## Circuit inventory (index)

- [`induction`](induction.md) — cross-model · **also an [operator](../operators/induction.md)** (`circuit:induction` here vs `op:induction` there)
- [`positional_broadcast`](positional_broadcast.md) — cross-model
- [`duplicate`](duplicate.md) — cross-model · **also an [operator](../operators/duplicate.md)** (`circuit:duplicate` here vs `op:duplicate` there)
- [`ioi_q_chain`](ioi_q_chain.md) — GPT-2 (Q-composition chain (GPT-2-only))
- [`induction_kchain_weights`](induction_kchain_weights.md) — GPT-2 (K-composition (weight + path-patch, GPT-2))
- [`discovered_write_hub_edges`](discovered_write_hub_edges.md) — GPT-2 (DISCOVERED (novel-live composition edges))
- [`v_virtual_heads`](v_virtual_heads.md) — GPT-2 (V-composition (composed-OV 'virtual heads', GPT-2))

> **Circuit vs operator — a naming note.** A few names (duplicate, induction) appear in
> *both* this circuit catalog and the [operator catalog](../operators/README.md). A **circuit** is a *composition*
> (`circuit:induction` = prev-token → induction); the same-named **operator** is the *head class* it is named
> after and built around (`op:induction`). The coincidence is deliberate: a circuit is keyed by its **reader
> operator**. Pages cross-link to their namesake.

## Cross-model circuit-edge liveness (remove the writer from the reader's key → attention collapse %)

| circuit | defining edge | gpt2 | gpt2-medium | gpt2-large | gemma-2-2b | Llama-3.2-1B | Qwen2.5-1.5B |
|---|---|---|---|---|---|---|---|
| **induction** | prevtok_head -> induction (K) | +17% 4.11 | +23% 2.15 | +8% 3.14 | +18% 0.0 | +70% 1.20 | +89% 1.4 |
| **positional_broadcast** | sink-writer -> prevtok key (K) | +22% 1.3 | +32% 1.5 | +0% 3.0 | +0% 5.4 | (skip) | +0% 0.0 |
| **duplicate** | (reader-side; writer often layer-0) | (skip) | +0% 0.13 | +4% 1.1 | +3% 0.2 | +9% 0.17 | +13% 0.6 |

**Reading it:** the **induction** edge (prev-token → induction) is live in *every* model (and *stronger* in RoPE —
content matching lives in the key everywhere); **positional-broadcast** (sink/hub → prev-token key) is
**GPT-2-small/medium-only** (the absolute-position plumbing — RoPE reads position from the rotation, so the
prev-token key has no upstream writer to remove). Same absolute-position-family split as the operator catalog's sink.

## Discovered edges (de novo, cross-model)

Beyond the named circuits, [**discovered circuit edges**](discovered.md) runs the key-patch over the top content
readers in *every* model and keeps the edges that collapse the reader beyond a reader-matched null. It recovers
the prev-token→induction K-chain de novo in the GPT-2 family (6/2/2 live edges) and finds localized edges in
Llama (3) and Qwen (1), but **none in Gemma** (0 — its content-reader keys aren't sharply localized to one
writer; RoPE distributes the circuit). 14 live edges total.

## Executable decompilation — is the circuit *sufficient*?

Edge liveness shows the circuit's edges are **necessary**. [**Reconstruction**](reconstruction.md) tests
**sufficiency**: keep only the induction circuit's heads (induction + prev-token), mean-ablate every other
attention head (MLPs intact), and measure how much induction the circuit alone recovers — far above a random
same-size head-set. A small head-set that reconstructs most of the behaviour is an *executable* decompilation.

Each cross-model circuit page now also carries a **cross-model causal dossier** (necessity + sufficiency +
redundancy, operator-parity), generated on the [unified `ResidualVM`](../DECOMPILATION.md) debugger
(`circuit_dossier_xmodel.py`). The sharpest read: the **induction circuit's necessity *and* sufficiency both decay
monotonically across the GPT-2 ladder** (small → XL) — the named circuit is most localized in the smallest model
and dissolves into the network with scale, the same distributedness theme measured as a clean ablation battery.

## Taxonomy & gaps

- **Levels:** circuit (a DAG of operator nodes) → edge (writer-op → reader-op via a K/Q/V port) → the operator
  classes at each node ([`../operators/`](../operators/README.md)). Edges are the primitive the discovery gate scores.
- **succession / greater-than** — MLP-dominated; no clean attention-composition circuit (carried by the copy ops).
- **SSM (Mamba)** — no heads, so no composition edges; induction is present only behaviourally (`ssm_induction.py`).
- **Not yet run:** the IOI Q-chain / V-composition cross-model (no published head-sets off GPT-2); full per-edge
  path-patch of all 22 discovered edges on the
  RoPE models. The cross-model catalog covers the universal-reader edges.

## How this was made

`circuit_atlas.py` (cross-model edges + harvest) → `circuit_catalog_doc.py` (these docs). Discovery/validation:
`composition_dag.py`, `validate_new_edges.py`, `vcomposition.py`, `ioi_causal.py`, `self_repair.py`,
`rung3_induction_chain.py`. See [`../DECOMPILATION.md`](../DECOMPILATION.md).
