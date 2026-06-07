---
---

*Disassembling a transformer's attention into a catalogued instruction set — `lm-sae`.*

A frozen language model computes in a basis we didn't choose. **`lm-sae`** reads that computation as a small,
reused **instruction set** — names the **operators** (single attention heads / MLPs), collects the **circuits**
(compositions of operators), validates them causally, surveys both **across model families**, and asks the harder
question: can we **decompile** it — reconstruct the *computation* faithfully, not just label the components?

This site is generated from the repo's tracked result artifacts and stays in sync with them. Source + how-to-run:
[github.com/jascal/lm-sae](https://github.com/jascal/lm-sae).

> **Mode — natural history, borrowed from biology.** A trained model is *grown*, not designed; we **catalog** the
> operators it developed, **taxonomize** them, and derive **generalizable** principles by comparing across model
> *species* (GPT-2 vs RoPE vs SSM). "Catalog," not "atlas," is deliberate — a growing, causally-tested record of
> what we've observed, not a claim of completeness. The edge over wet biology: we can **breed synthetic specimens
> and intervene at high speed** — train hosts, swap the mixer to a non-attention recurrence, and ablate /
> path-patch any component in milliseconds.

## The goal

> Exhaustively explore, taxonomize, and collect causal dossiers on **all operators and circuits** we can find,
> **across all the models we investigate** — to the limit of what the tools can find.

Two model families anchor it: the **absolute-position** GPT-2 family (small / medium / large) and the **RoPE**
family (Gemma-2-2B, Llama-3.2-1B, Qwen-2.5-1.5B), plus **SSM** (Mamba) for the no-attention control.

## The theory

| | disassembly (have) | decompilation (target) |
|---|---|---|
| unit | one head / MLP / idiom in isolation | the **composition** — which ops chain into which |
| coverage | *% of attention legible* | *% of the forward pass faithfully reconstructable* |
| validation | mean-ablation damages the metric | **recompile the program; KL ≈ host** |

The instruction set has two classes — **MOVE** = attention (a QK *addressing mode* × an OV *write op*) and
**COMPUTE** = MLP (key–value memories). An **operator** is one head/MLP class (a *family of heads*, not a single
head); a **circuit** is a composition — a writer-op feeding a reader-op's K/Q/V port, chained. **Decompilation
coverage** is `1 − KL(host ‖ recompiled[ops kept]) / KL(host ‖ all-ablated)`.

The throughline: **a model is legible in the right basis even where it is *not* legible as single SAE features.**

## The catalogs (generated, kept in sync with results)

- **[Operator catalog](operators/README.md)** — every operator *class* × every model: behavioural signal,
  membership (# heads), causal load-bearing, + deep per-op dossiers (identity · causal×tasks · K/V channels ·
  composition · redundancy · cross-model). *Induction is universal (.91–.99); the sink is common but absent in
  Gemma; RoPE leans on `self`.*
- **[Circuit catalog](circuits/README.md)** — composed circuits, cross-model edge liveness + the GPT-2 discovered circuits
  (IOI Q-chain, V-composition virtual heads, 22 novel-live edges). *The induction edge is live in all 6 models
  (stronger in RoPE); positional-broadcast is GPT-2-small/medium-only.*

## The methodology & tools

- **Disassembly** ([deep-dive](DISASSEMBLY.md)) — idiom library → opcode tables → coverage scorecard → causal
  validation → corpus robustness; 8/8 literature idioms recovered from weights, ~99% of content mass legible.
- **Decompilation** ([design + milestones](DECOMPILATION.md)) — reconstruction-coverage interpreter (M1), the
  composition-DAG extractor + path-patch gate (M2), MLP/COMPUTE nodes (M3), the forge-basis ceiling (M4), the
  cross-model ceiling (M5: the decompilable *fraction* is absolute-position-family-specific).
- **The ResidualVM debugger** — a *programmatic* discovery engine (`attribution_sweep` + `edge_probe`) that finds
  **un-named** load-bearing components and candidate circuit edges. It surfaced MLP0 (the biggest operator,
  uncatalogued) and candidate ops 7.6 / 5.9 — the catalog's growth engine.

## Cross-architecture synthesis

**Mechanisms are invariant** (idioms, induction causal in all four RoPE/abs-pos models, K/Q/V composition); **the
positional register is absolute-position-family-specific** (three independent signatures: the sink, the
positional-broadcast circuit, the decompilable fraction). One level deeper, the in-context-copy *capability*
survives even a non-attention mixer (Mamba), though the *mechanism* is unverified without head-resolution.

## Future work

MLP/COMPUTE operator family · cross-model circuit discovery · dossier the discovered candidates · per-class
SAE-feature operands · executable decompilation (recompile a validated circuit, KL ≈ host).

## Sister track

The **oracle & cov95 forge tax** investigation — what an SAE *feature basis* destroys, and what you must preserve —
is at **[Forge-tax track](FORGE_TAX_TRACK.md)**.
