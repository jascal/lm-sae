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

- **Induction's key-addressing sharpness decays monotonically with size, to zero.** Across the full GPT-2 ladder
  (124M→1.5B) the key-collapse from removing the single top prev-token writer goes **+39% → +8% → +1% → +0%** — by
  GPT-2-XL the one-dominant-writer circuit is gone and the key is distributed like the RoPE models (~0–3%). One
  dominant writer is a *small-model* phenomenon, not an absolute-position one ([induction dossier](operators/induction.md),
  [scaling synthesis](scaling.md)).
- **The token-determined "embedding block" widens *and strengthens* with scale.** MLP0 determinism climbs
  0.63 → 0.75 → 0.80 over the GPT-2 ladder and the block spreads from L0 to L0–L2 ([MLP test](operators/mlp_detokenizer.md)).
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

### When a circuit "distributes," what is it becoming — a weighted ensemble, or heterogeneous circuits woven in?

The [cross-model dossier](circuits/induction.md) shows the induction circuit's necessity *and* sufficiency decay with
GPT-2 scale. "Distributed" could mean three different things, so we measured the full induction-head population (not
just the top heads) on two axes (`circuit_ensemble.py`, on the ResidualVM): a **functional** axis (do the heads help
the *same* token-predictions?) and a **structural** axis that is **free of the ablation-gentleness confound** — the
pairwise cosine of the heads' **OV operation-matrices** (do they do the *same thing*, weight-wise?) and the
population's **spread across depth**.

- **It is NOT a weighted ensemble of duplicates.** If the distributing circuit were many copies of one head being
  averaged, the heads' OV matrices would be aligned. They are not, in **any** model: the mean pairwise OV cosine is
  **≈0 everywhere (0.01–0.07)** and does **not** rise with scale. The induction heads are doing structurally
  *distinct* operations, not replicating one.
- **The population is spread across depth, not a replicated band** — the induction heads span **43–93% of the
  model's layers** in every model (Qwen 93%, Llama 87%, Gemma 80%), and the **functional** overlap is low too
  (cosine 0.03–0.12 — heads help largely *different* predictions). Both point to the second picture: **structurally
  heterogeneous heads at different depths with overlapping function**, the closest match to "new circuits woven in,"
  rather than one circuit cloned or one circuit cleanly decomposed.
- **Honest confound (why we don't headline a "members grow with scale" number).** The contribution-concentration
  metrics — effective-N (Hill number) and the top head's share — come out **non-monotonic** across the GPT-2 ladder
  (effective-N 2.2 → 11.3 → 2.1 → 7.4; top-share 66% → 15% → 67% → 29%), because single-head **mean-ablation removes
  less in wider models** (each head is a smaller fraction of the residual), so the per-head contribution vector gets
  noise-dominated. We therefore lead with the two **confound-free structural** facts (OV-cosine ≈0; wide layer span),
  which are weight/attention measurements, not ablation deltas.

**Verdict.** Of the user's two framings — "weighted ensemble?" vs "heterogeneous circuits with overlapping function
woven in?" — the data favors the **second** and rejects the first: as a circuit distributes it recruits
*structurally different* heads spread across depth, not duplicates of itself.

### Separable parallel circuits or one decomposition? — it splits by architecture family

"Structurally different heads woven in" is consistent with *either* several **complete parallel circuits** (each
fed by its own upstream predecessor-writer) *or* **one circuit decomposed** behind a shared front-end. The
discriminator is each induction reader's **upstream writer-dependency**: we ran the faithful key-only patch
(`circuit_writer_cluster.py`, on the `circuit_content_patch` machinery; every zero-patch sanity = 0.0) over the
*whole* induction population, then clustered the readers by *which* upstream head, removed from their key, collapses
their induction attention. Shared writers → one decomposition; distinct writers → separable circuits.

- **The GPT-2 (absolute-position) family fragments into separable sub-circuits with scale.** Across small → medium →
  large the induction readers split into **1 → 2 → 3** writer-defined clusters and their writer-profile similarity
  drops (pairwise cosine **0.58 → 0.58 → 0.21**) — GPT-2-large's readers draw on **6 distinct** top-writers, only 25%
  sharing one front-end. So the heterogeneous heads aren't one decomposed circuit; they are increasingly **separate
  circuits with different upstream wiring**, woven in as the model grows. (Not perfectly monotonic — GPT-2-XL
  partially re-concentrates, 75% on one writer / 2 clusters — so this is a trend, with n=8 readers and a coarse
  cosine-0.5 component threshold; the robust signal is GPT-2-large as the most fragmented point.)
- **The RoPE family keeps one shared writer front-end (closer to a decomposition).** Gemma / Llama / Qwen each have a
  **single** reader cluster (component count 1) with high profile-cosine (**0.68–0.76**) and one dominant
  predecessor-writer feeding most readers — Llama's layer-0 head **0.2 feeds 88%** of its induction readers, Gemma's
  **0.0** half of them. RoPE puts the predecessor signal in one early writer the whole population reads, so there the
  distribution is one circuit's labour split across readers, not parallel circuits.

**Net.** As an induction circuit "distributes," it is **not** becoming a weighted ensemble of duplicates (OV-operation
cosine ≈0). In the **absolute-position family it weaves in genuinely separable sub-circuits** (distinct upstream
writers, fragmenting with scale); in the **RoPE family it decomposes one circuit behind a shared early-writer
front-end**. The "more distributed with scale" headline is two different mechanisms underneath, split by the same
absolute-vs-RoPE line that separates the positional register everywhere else in the catalog.

### Does the population scale with INPUT size? — yes, a separate axis from model size ("same function, more inputs")

A distinct hypothesis for *why* induction distributes: not (only) more parameters, but the **same function applied
over a larger input domain** — more distinct token-types to induct over → more heads recruited, each covering a
slice. We test it by **holding the model fixed and scaling the input** (`circuit_input_scaling.py`, confound-free —
one forward pass, induction *attention mass* per head, no ablation):

- **More input token-types recruits more induction heads — in 6/7 models.** Sweeping the probe vocabulary V = 8 →
  1024 (repeat length fixed), the **scale-invariant** effective number of active induction heads (Hill number, so
  *not* just an overall-magnitude effect) **rises monotonically and then saturates**: GPT-2 12 → 20, GPT-2-medium 23
  → 52, GPT-2-large 39 → 72, GPT-2-XL 63 → 122; Llama 25 → 46, Qwen 28 → 47. The top head's share stays low and flat
  (1–8%) throughout — the extra inputs are absorbed by *recruiting more heads*, not by loading the dominant one
  harder. This is direct support for "same function block over more possible inputs": input-diversity is its own
  driver of the head count, on top of (and separable from) model size.
- **Gemma is the exception (again).** Its active induction population **saturates almost immediately** (effective-N
  28 → 21 as V grows, n_active flat ~40) and the top-share *rises* (2% → 9%) — Gemma covers a wider input domain by
  *concentrating* a fixed small head-set, not recruiting. The recurring "third architecture" falls out of this
  regularity too.
- **It's input *diversity*, not raw length.** The complementary context-length sweep (more positions, vocabulary
  fixed) runs the *other* way — n_active *drops* as the repeat lengthens — but that axis is confounded (short probes
  give few induction targets, so the per-head mass estimate is noisy and over-counts active heads), so the clean
  signal is the **vocabulary** axis: it is the breadth of the *input type-space*, not the amount of input, that
  pulls in more heads. (`runs/disassembly/circuits/input_scaling_summary.json`.)

**Synthesis across the three tests.** A distributing induction circuit is (1) **not** a weighted ensemble of
duplicates (OV-cosine ≈0), (2) made of **structurally heterogeneous heads** — separable parallel sub-circuits in the
absolute-position family, one shared-front-end decomposition in the RoPE family — and (3) its head-count is driven by
**input-domain breadth** as a first-class axis alongside model size (the same function tiled over more token-types),
GPU-cheaply confirmable on a single model. Gemma is the exception to (2)'s and (3)'s regularities, as it is to most.

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
