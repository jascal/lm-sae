---
---

*Disassembling a transformer's attention into a catalogued instruction set — `lm-sae`.*

A frozen language model computes in a basis we didn't choose. **`lm-sae`** reads that computation as a small,
reused **instruction set** — names the **operators** (single attention heads / MLPs), collects the **circuits**
(compositions of operators), validates them causally, surveys both **across model families**, and asks the harder
question: can we **decompile** it — reconstruct the *computation* faithfully, not just label the components?

This site is generated from the repo's tracked result artifacts and stays in sync with them. Source + how-to-run:
[github.com/jascal/lm-sae](https://github.com/jascal/lm-sae). New to the vocabulary? Every technical term is
defined — with its canonical reference — in the **[Glossary](glossary.md)**.

> **Mode — natural history, borrowed from biology.** A trained model is *grown*, not designed; we **catalog** the
> operators it developed, **taxonomize** them, and derive **generalizable** principles by comparing across model
> *species* (GPT-2 vs RoPE vs SSM). "Catalog," not "atlas," is deliberate — a growing, causally-tested record of
> what we've observed, not a claim of completeness. The edge over wet biology: we can **breed synthetic specimens
> and intervene at high speed** — train hosts, swap the mixer to a non-attention recurrence, and ablate /
> path-patch any component in milliseconds.

> **Amateur, exploratory home-science.** Everything here is *a working catalog*, not *the* definitive one —
> findings are **descriptive and provisional**, measurements to be checked, not settled results. Where a mechanism
> is unknown we still list the specimen with what we measured, and say so.

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
- **[MLP / COMPUTE catalog](operators/mlp_compute.md)** — the *other* instruction class (attention MOVES, MLP
  COMPUTES), per-layer causal profile across models. *COMPUTE concentrates on an early MLP — the detokenizer — in
  5 of 6 models (catastrophic for induction); Gemma distributes it.*
- **[Per-head disassemblies](disassembly/README.md)** — the full per-head listing for each model (addressing ×
  write × content binding × operator role), with every operator-role tag **hyperlinked** to its catalog page.
- **[Discovered components](operators/discovered.md)** — the debugger run across *every* model: every head + MLP
  ranked by causal effect, flagged named-vs-**UNNAMED** (candidate new operators). *Gemma surfaced 6, Llama 16
  unnamed load-bearing candidates — leads to dossier next.*
- **[Discovered circuit edges](circuits/discovered.md)** — de-novo key-patch over the top content readers in every
  model. *Recovers prev-token→induction in the GPT-2 family (6/2/2 edges); Llama 3, Qwen 1, Gemma 0 (distributed).*

## Analyses & results (the experiments)

Beyond the catalogs, the deeper experiments — read **[Cross-model findings](FINDINGS.md)** for the narrative and
**[Scaling synthesis](scaling.md)** for the central table.

- **Mechanism depth** — **[SAE-feature operands](operators/sae_operands.md)** (what each operator reads/writes in
  *feature* space, GPT-2 + Gemma) · **[MLP extended-embedding test](operators/mlp_detokenizer.md)** (MLP0 is
  token-determined in 5/6 — Llama the outlier) · **[outlier mechanism digs](operators/outlier_digs.md)** (the
  "compensatory" suppression is a synthetic-probe artifact; high causal effect ≠ doing the named op).
- **Executable decompilation** — **[reconstruction](circuits/reconstruction.md)** (no small head-set is *sufficient*
  for induction; even IOI's 26-head circuit isn't, in isolation) · **[attention-vs-MLP substrate](circuits/induction_substrate.md)**
  (induction leans on both ~equally; MLP0 is the critical MLP).
- **Knowledge** — **[where facts live](circuits/factual_recall.md)** (ablation-contrast) and **[ROME causal
  tracing](circuits/causal_tracing.md)** (facts enriched in the early MLPs at the subject, cross-model).
- **Operator gaps** — **[succession](operators/succession.md)** (the +1 operator is 95–100% MLP-computed — data
  behind the catalog's "MLP-dominated" gap).
- **The thesis** — much of what looks *architectural* tracks **scale**: the *same* named circuits become more
  distributed as models grow ([scaling synthesis](scaling.md)).

## The methodology & tools

- **Disassembly** ([deep-dive](DISASSEMBLY.md)) — idiom library → opcode tables → coverage scorecard → causal
  validation → corpus robustness; 8/8 literature idioms recovered from weights, ~99% of content mass legible.
- **[Architecture references](architectures.md)** — the host GPT-2 block (attention = MOVE, MLP = COMPUTE) + the
  SAE, as [n-orca](https://github.com/jascal/n-orca) typed-DAG specs compiled to Mermaid.
- **Decompilation** ([design + milestones](DECOMPILATION.md)) — reconstruction-coverage interpreter (M1), the
  composition-DAG extractor + path-patch gate (M2), MLP/COMPUTE nodes (M3), the forge-basis ceiling (M4), the
  cross-model ceiling (M5: the decompilable *fraction* is absolute-position-family-specific).
- **The ResidualVM debugger** — a *programmatic* discovery engine (`attribution_sweep` + `edge_probe`) that finds
  **un-named** load-bearing components and candidate circuit edges. It surfaced MLP0 (the biggest operator,
  uncatalogued) and candidate ops 7.6 / 5.9 — the catalog's growth engine.

## Cross-architecture synthesis

See **[Cross-model findings](FINDINGS.md)** for the curated narrative. In brief: the **mechanisms are invariant**
(idioms, induction causal in all six models) and the **positional register is absolute-position-family-specific**
(the sink, the positional-broadcast circuit, the decompilable fraction) — but several things people credit to
*architecture* actually track **scale** (induction's single-prev-token-writer key is a GPT-2-*small* trait that
distributes with size; the token-determined MLP "embedding block" widens with scale). The recurring outliers are
**Gemma** (low-sink, distributed key, strongest MLP0 extended-embedding) and **Llama** (context-determined MLP0,
layer-0 induction *enablers* not inductors). Banked cautions: synthetic probes can manufacture apparent
suppression; **high causal effect ≠ doing the named operation**; present ≠ depended-on. One level deeper, the
in-context-copy *capability* survives a non-attention mixer (Mamba), though the *mechanism* is unverified.

## Future work

**Executable decompilation** (recompile a validated circuit, KL ≈ host) · circuit-edge SAE-feature content (the
operator [SAE operands](operators/sae_operands.md) exist; the per-edge version is next) · the outlier follow-ups
in [Cross-model findings](FINDINGS.md) · the pivot to **other models / more decompiler research**. *(Done: the
MLP/COMPUTE family + its extended-embedding mechanism test, cross-model circuit discovery, dossiering the
discovered candidates, the cross-model deep dossier (identity/causal/channel/redundancy) across all six models,
the outlier digs, and **per-operator SAE-feature operands on GPT-2 + Gemma** — the catalog-depth gap, now filled.)*

## Sister track

The **oracle & cov95 forge tax** investigation — what an SAE *feature basis* destroys, and what you must preserve —
is at **[Forge-tax track](FORGE_TAX_TRACK.md)**.
