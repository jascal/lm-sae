---
title: Cross-model findings
---

# Cross-model findings — what the catalog has (and hasn't) shown

A curated, **descriptive** reading of the catalog's cross-model results — natural history across six transformer
models (GPT-2 small/medium/large, Gemma-2-2B, Llama-3.2-1B, Qwen2.5-1.5B) plus a non-attention control (Mamba).
Amateur, provisional, single-corpus where noted; every claim links to the page with the data. The headline is not
"here is the mechanism" but "here is what is invariant, what scales, where the outliers are, and what we learned
not to trust."

## What looks invariant

- **Induction is universal and causally load-bearing in every model** — the universal idioms (prev-token,
  duplicate, induction) are recovered from the weights/behaviour everywhere, and mean-ablating the induction heads
  raises induction-NLL in all six ([operator catalog](operators/README.md); cross-model dossier on each op page).
- **The early MLP is largely an "extended embedding" in 5/6 models** — MLP0's output is mostly fixed by the current
  token identity (token-determinism η²: GPT-2 0.63, Gemma 0.91, Qwen 0.65), the classic detokenizer reading
  ([MLP extended-embedding test](operators/mlp_detokenizer.md)).
- **In feature space, the copy/suppress split survives** — reading operators in monosemantic SAE features (the
  [SAE-feature operands](operators/sae_operands.md), GPT-2 + Gemma): the copy ops have positive OV copy-scores on
  their read-feature's tokens and `negative_mover` is the only non-positive circuit op (copy-suppression). The
  *read*-features are cleaner on GPT-2 than Gemma (whose heads are `<bos>`/structural-heavy on this corpus) —
  provisional, single-corpus.

## What scales — within a family, not just across architectures

The sharpest lesson of the cross-model pass: several things people attribute to *architecture* (absolute-position
vs RoPE) actually track **scale**.

- **Induction's key-addressing sharpness decays with size.** GPT-2-small reads induction off **one** dominant
  prev-token writer (head 4.11, +39% key-collapse when removed); gpt2-medium +8%, gpt2-large +1% — until the
  largest GPT-2 distributes the key like the RoPE models (~0–3%). One dominant writer is a *small-model*
  phenomenon, not an absolute-position one ([induction dossier](operators/induction.md)).
- **The token-determined "embedding block" widens with scale.** In GPT-2-small only L0 is token-determined; in
  gpt2-large L0–L2 all are (~0.7) ([MLP test](operators/mlp_detokenizer.md)).
- **Induction redundancy shifts from distributed to non-monotonic with scale.** Small GPT-2 / Llama / Qwen
  induction is a distributed, superadditive population; gpt2-large and Gemma give a *non-monotonic* ablation curve
  (ablating the top heads together damages induction less than ablating a subset). **Caveat (important):** the
  digged-into mechanism is **not** negative-head self-repair — it is substantially a *synthetic repeated-random
  probe artifact* ([outlier digs](operators/outlier_digs.md)).

## Knowledge — where facts live, and moving them

The catalog is about *mechanisms*; the [knowledge axis](circuits/causal_tracing.md) is the decompiler goal ("the
model IS the database").

- **The ROME two-site flow is architecture-invariant.** [Causal tracing](circuits/causal_tracing.md) (subject
  corruption + restoration) recovers the same structure in GPT-2 ×3, Llama, Qwen: an **early MLP store at the
  subject** (peak depth ≈0) feeds a **late attention readout at the last token** (depth ≈0.6–0.9). Cross-model,
  which ROME never did.
- **The store is editable by activation patch.** [Patching](circuits/fact_patching.md) the early-MLP store at the
  subject with a different fact's activation **transplants the fact 100% of the time** (France's run answers Rome)
  in those five models — the decompile→recompile loop made concrete, sufficiency complementing necessity.

## The outliers — where the next questions are

- **Gemma-2-2B is the recurring exception across seven independent measurements**: a near-absent attention-sink
  (~4% vs 44–55%), the most *distributed* induction key, a non-monotonic (compensatory) induction-redundancy curve,
  the *strongest* MLP0 extended-embedding (η² 0.91), induction that doesn't lean on MLP0, a **late** fact site
  (vs early elsewhere), and **fact-transplant-resistant** early MLPs (3% flip vs 100%). Gemma stores and routes
  information differently enough that it falls out of nearly every cross-model regularity — the single most
  informative "third architecture" in the set.
- **Llama-3.2-1B**'s MLP0 is the lone *context*-determined early MLP (η² ≈ 0). [Dug](operators/outlier_digs.md):
  it is not intrinsic — Llama's layer-0 attention is comparable in size to the embedding **and** the most
  context-determined of any model, so MLP0 ingests a context-mixed input. Those same layer-0 heads are
  induction-**enablers**, not inductors (strong induction *causal* effect but weak induction *attention* — they set
  up the residual that a later head reads).

## Is the computation an isolable circuit? (mostly no — it's distributed)

The catalog names which heads are *necessary*. **[Executable decompilation](circuits/reconstruction.md)** tests
*sufficiency* — keep only a circuit's heads, ablate the rest, see how much behaviour survives.

- **No small head-set reconstructs induction.** The named 8-head circuit recovers at most ~17% (mean-ablation) /
  ~30% (the gentler resample-ablation) of induction; coverage **decays with GPT-2 scale** (small → large +0%) and
  the RoPE curves go non-monotonic (Qwen negative). You need *nearly every head* (GPT-2-small only hits ~full at
  K≈128/144). Robust to ablation type — so it's a real property, not an artifact.
- **Even IOI — the field's celebrated "complete" 26-head circuit — isn't sufficient in isolation** (keeping only
  it gives a *negative* logit-diff, no better than random). Caveat: a harsh mean-ablation test; this speaks to
  distributedness, not the validity of the IOI necessity/path-patching result.
- **Induction is not an attention-only circuit** — it leans roughly *equally* on attention and the
  [MLP substrate](circuits/induction_substrate.md); in GPT-2-small the early detokenizer **MLP0 alone** carries
  almost the entire MLP dependence (Gemma is the exception — its clean standalone MLP0 isn't needed by induction).

The through-line: the named circuits are causally **necessary and the dominant drivers**, but the behaviour is
carried by the near-whole network — a clean decompilation into a tiny sufficient subgraph does not exist here.
(The reconstruction-coverage numbers are **seed-stable**, ±0–1% over three probe-resample seeds, so the
scaling/distributedness trend is not a single-seed artifact.)

## Methodological cautions — banked from the digs

- **Synthetic repeated-random probes can manufacture apparent suppression.** A head that looks like it suppresses
  induction on random-token repeats can be neutral (and positive-OV) on natural text — validate on real repeats.
- **High causal effect ≠ doing the named operation.** Llama head 0.31 is the single most induction-causal head
  (+7.99 when ablated) yet does not attend induction-style — it *enables* induction downstream.
- **Magnitude ≠ dependence; present ≠ depended-on.** The attention-sink carries 44–55% of attention in three
  models yet only GPT-2 is functionally dependent on it (+42% NLL when blocked vs ~+1%).
- **Causal validation is metric-specific.** Confirm an op against the metric it serves, not generic-prose NLL.

## The positional register is absolute-position-family-specific

Three independent signatures separate the GPT-2 (learned absolute position) family from the RoPE family: the
attention-**sink**, the **positional-broadcast** circuit (early write-hubs → prev-token key, [circuit
catalog](circuits/README.md)), and the larger **decompilable fraction**. RoPE reads relative position from the
rotation, so it has no positional-broadcast plumbing to remove.

## Beyond attention

The in-context-copy *capability* survives a non-attention mixer (Mamba shows an induction-NLL gain), but the
*mechanism* is unverified without head-resolution — a documented edge of the current tools.

---

_This page is a hand-curated narrative; the numbers live in the generated [operator](operators/README.md) /
[circuit](circuits/README.md) / [MLP](operators/mlp_compute.md) catalogs, the
[extended-embedding test](operators/mlp_detokenizer.md), and the [outlier digs](operators/outlier_digs.md), each
regenerable from committed JSON. See [DISASSEMBLY.md](DISASSEMBLY.md) for the original GPT-2 method deep-dive._
