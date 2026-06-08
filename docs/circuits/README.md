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

## MLP nodes in the circuit DAG — the COMPUTE class, cross-model

The cross-model edges above are attention-only (head→head). But circuits also route through **MLPs**: [`mlp_circuit_xmodel.py`](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/mlp_circuit_xmodel.py) (on the ResidualVM) makes them first-class circuit **nodes** — per-layer COMPUTE importance for induction (ablate each MLP → Δinduction-NLL) + the head↔MLP composition edges that wire them in.

| model | induction head | all-MLP-ablated Δind-NLL | dominant induction-MLP(s) | detokenizer = MLP0? |
|---|---|---|---|---|
| gpt2 | `5.1` | +8.7 | L0 (+8.2), L1 (+6.3) | ✓ |
| gpt2-medium | `11.1` | +8.9 | L0 (+15.4) | ✓ |
| gpt2-large | `16.0` | +13.0 | L0 (+10.3) | ✓ |
| gemma-2-2b | `6.3` | +17.5 | L0 (+3.7), L17 (+1.6), L5 (+1.4) | ✓ |
| Llama-3.2-1B | `10.23` | +15.1 | L1 (+12.8), L0 (+12.5), L15 (+1.9) | ✗ (L1) |
| Qwen2.5-1.5B | `14.3` | +15.9 | L2 (+14.0), L1 (+13.4), L0 (+7.3) | ✗ (L2) |

- **The COMPUTE class is load-bearing for induction in *every* model** — ablating all MLPs (attention intact) costs **+8.7 to +17.5** induction-NLL, so a faithful induction circuit is **not** attention-only; the MLP nodes belong in the DAG.
- **The load-bearing MLPs are *early* everywhere** — the detokenizer / extended-embedding substrate ([MLP test](operators/mlp_detokenizer.md)) — but their **concentration tracks the family**: GPT-2 (and Gemma) pin it to a single **MLP0**, while the RoPE models **Llama (L1+L0)** and **Qwen (L2+L1+L0)** spread the substrate across the first two–three MLPs (so `detokenizer = MLP0` is GPT-2/Gemma-only; the embedding is assembled across early layers in RoPE). Same localized-in-GPT-2 / distributed-in-RoPE split the attention side shows. _(Data: [mlp_circuit_xmodel_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/mlp_circuit_xmodel_summary.json).)_


## Taxonomy & gaps

- **Levels:** circuit (a DAG of operator nodes) → edge (writer-op → reader-op via a K/Q/V port) → the operator
  classes at each node ([`../operators/`](../operators/README.md)). Edges are the primitive the discovery gate scores.
- **succession / greater-than** — MLP-dominated; no clean attention-composition circuit (carried by the copy ops).
- **SSM (Mamba)** — no heads, so no composition edges; induction is present only behaviourally (`ssm_induction.py`).
- **IOI is now cross-model** (the [`ioi_q_chain`](ioi_q_chain.md) page): the circuit's *operators* (name-movers,
  negative/copy-suppression movers, duplicate-token initiator) and its load-bearing necessity are found
  behaviourally in all 6 models via the ResidualVM — closing the old "no head-set off GPT-2" gap. The precise
  *Q-composition edge wiring* stays GPT-2-validated.
- **V-composition is now cross-model too** (the [`v_virtual_heads`](v_virtual_heads.md) page): the composed-OV
  virtual heads (an induction head's output re-read as a downstream **value**) are weight-legible in all 6 models —
  so the full Elhage **K / Q / V** composition-edge triad is now measured across models (induction K-chain, IOI
  Q-chain, V-virtual-heads). Dynamic ΔV-out confirmation stays GPT-2-validated.
- **Still GPT-2-only:** full per-edge path-patch of all
  22 discovered write-hub edges on the RoPE
  models (they are GPT-2 absolute-position plumbing — predicted absent in RoPE). The cross-model catalog covers the
  universal-reader edges, the IOI operators, and all three composition-edge types.

## How this was made

`circuit_atlas.py` (cross-model edges + harvest) → `circuit_catalog_doc.py` (these docs). Discovery/validation:
`composition_dag.py`, `validate_new_edges.py`, `vcomposition.py`, `ioi_causal.py`, `self_repair.py`,
`rung3_induction_chain.py`. See [`../DECOMPILATION.md`](../DECOMPILATION.md).
