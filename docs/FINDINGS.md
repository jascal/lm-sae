---
title: Cross-model findings
---

# Cross-model findings — what the catalog has (and hasn't) shown

A curated, **descriptive** reading of the catalog's cross-model results — natural history across six transformer
models (GPT-2 small/medium/large, Gemma-2-2B, Llama-3.2-1B, Qwen2.5-1.5B), a non-attention control (Mamba), and a
controlled scale ladder (the GPT-NeoX **Pythia** family, 14m–1.4b — one architecture, same data, six sizes — used
for the [scaling laws](scaling.md)).
Amateur, provisional, single-corpus where noted; every claim links to the page with the data. The headline is not
"here is the mechanism" but "here is what is invariant, what scales, where the outliers are, and what we learned
not to trust."

## What looks invariant

- **Induction is universal and causally load-bearing in every model** — the universal idioms (prev-token,
  duplicate, induction) are recovered from the weights/behaviour everywhere, and mean-ablating the induction heads
  raises induction-NLL in all six ([operator catalog](operators/README.md); cross-model dossier on each op page).
- **The IOI circuit is architecture-invariant, not GPT-2-only** — found behaviourally on the ResidualVM in all six
  models, every one has **name-movers** (heads attending the end→indirect-object and copying it), **negative/copy-
  suppression movers**, and a **duplicate-token initiator**; ablating the name-movers collapses the IO−S logit-diff
  (+13% to +26%) everywhere, and the behaviour *strengthens* with GPT-2 scale (+2.88 → +3.11 → +4.09)
  ([IOI dossier](circuits/ioi_q_chain.md)). The **backup-name-mover self-repair generalises too** — in every model the
  *most ablation-load-bearing* heads are the S-inhibition type (name-movers are backed up), so the ablation and
  copy-attention rankings disagree (the Hydra effect, cross-model).
- **The early MLP is largely an "extended embedding" in 5/6 models** — MLP0's output is mostly fixed by the current
  token identity (token-determinism η²: GPT-2 0.63, Gemma 0.91, Qwen 0.65), the classic detokenizer reading
  ([MLP extended-embedding test](operators/mlp_detokenizer.md)). And the **induction circuit routes through these
  early MLPs in *every* model** — ablating all MLPs (attention intact) costs +8.7 to +17.5 induction-NLL, so a
  faithful circuit is **not** attention-only ([MLP nodes in the circuit DAG](circuits/README.md)). The substrate's
  *concentration* tracks the family, mirroring the attention side: GPT-2/Gemma pin it to a single **MLP0**, the RoPE
  models (Llama L1+L0, Qwen L2+L1+L0) spread it across the first two–three MLPs.
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

## The flagship — composition doesn't factor through SAE features (the forge tax, from the decompilation side)

The program's throughline: *a language model is legible in the right basis even where it is not legible as single SAE
features* — and the unifying claim that the **decompilation ceiling is the forge tax** (composition doesn't factor
through the features for the same reason cov95 collapses under forging). We test it directly on real LMs
(`sae_forge_tax.py`, on the ResidualVM + loaded SAEs): **force the residual through the SAE feature basis**
(decode∘encode = the forge bottleneck) at a layer, and compare the damage to the **composition** (induction-NLL —
the in-context copy that needs prev-token → induction *composition*) against the **readout** (generic next-token NLL —
what the SAE features are trained to carry), each relative to its own clean baseline.

- **The SAE feature basis taxes composition far more than readout.** In **GPT-2** the full forge (all 12 resid SAEs)
  raises induction-NLL **+1357%** but generic-NLL only **+70%** — composition is taxed ~19× the readout, and it is
  taxed more **in every one of the 12 layers** individually. The feature basis preserves what the model *reads out*
  but not what it *composes*: SAE features survive as readouts, the computation over them does not factor through
  them — the cov95 forge tax, now measured from the disassembly/reconstruction side on the host the catalog reads.
- **It holds in the RoPE outlier too, weaker.** **Gemma-2-2B**: composition +542% vs readout +399% (net +143%,
  composition-taxed-more in 5/8 layers). Weaker because Gemma's induction is already weak/distributed (base
  induction-NLL 4.87 vs GPT-2's 0.73 — the recurring Gemma exception) and its Gemma-Scope SAEs cover only 8 layers
  and reconstruct more loosely (the *readout* is heavily taxed too, +399%), narrowing the gap — but composition still
  loses more.
- **Why this is the unifying result.** The same phenomenon the [forge-tax track](FORGE_TAX_TRACK.md) measures as
  cov95 collapse (single-latent monosemanticity destroyed while mAUC/readout survives) appears here as a
  *reconstruction* cost concentrated on *composition* — connecting the two contributions (the SAE forge tax **A** and
  the disassembly/decompilation **B**) on a real LM. **Honest scope:** SAE reconstruction is lossy everywhere, so the
  signal is the *relative* tax (composition vs readout), which is robust (12/12 layers in GPT-2); the full-stack forge
  compounds reconstruction error (hence the large absolute numbers — the per-layer count is the clean comparison);
  and this is the *reconstruction/NLL* route to the tax, complementary to (and agreeing with) the cov95/mAUC route,
  not the full sae-forge `NativeModel` weight-projection ceiling test ([DECOMPILATION.md](DECOMPILATION.md) M4).

## What IS the entangled core — a compact grammar head on a content bulk

Having localized the core as a shared moderate-rank subspace ([DECOMPILATION.md](DECOMPILATION.md), `core_rank.py`),
we decompiled its **structure** (`core_basis_decompile.py` + `core_grammar.py`, GPT-2 s/m/l):

- **It is not the readout subspace, and not the named operators.** The shared union basis lies *at/below chance* in
  the unembedding's top logit directions (0.33–0.39 vs 0.36–0.40), and the catalog ops' OV-write subspaces
  (induction/prev-tok/dup/sink) sit at ~chance inside it (≈ random heads). The core is the **aggregate of every
  layer's writers**, not a few idioms and not "where logits are written."
- **Its most-shared head is a generic grammar.** Binning the shared directions by sharedness and fitting on three
  structurally distinct corpora (Shakespeare / a modern novel / Python), the **top ~16 directions are *both*
  corpus-invariant (overlap 0.44 vs chance 0.02 — 22×) *and* closed-class/grammatical (0.28 vs 0.00 random — 28× the
  base rate)**; everything deeper is *neither* (corpus-specific, content/rare-token). So there is a content-free
  grammatical scaffold — but it is a **compact head (~5–16 directions), not the whole Θ(d) core**.
- **A "simpler-than-Chomsky" grammar, and learned.** What a linear write-basis encodes is a **categorial** scaffold
  (determiner-slot, punctuation-slot, verb-slot) — a distributional POS basis, *not* recursive/hierarchical syntax
  (which, if present, lives in the *composition* of categories — the entangled bulk that pays the forge tax). And it
  emerges from data in a generic learner with no syntactic prior.
- **The big-O: Θ(model size).** Functional per-layer rank and the shared basis are both Θ(d) (a constant fraction
  ~⅓–⅖, growing with width); the grammar head is ~O(1). Low-rank simplification buys a constant factor, not a big-O
  cut — the irreducible core scales *with* the model.
- **Across the layer chain the *coupling* is area-law, but the *runnable* bond is Θ(d).** Viewed as a tensor network
  over layer-"sites" (`core_mps.py`), the cross-cut coupling spectrum has **participation ratio ~16, flat with
  depth/width** — area-law on the coupling. **But that PR is not a runnable state size:** the no-retrain TT surrogate
  (`core_tt.py`, embedding-protected running bond) shows χ≈16 *badly* degrades NLL (ΔNLL +1.4 to +3.1), the runnable
  bond is ~⅓·d, **per-layer truncation beats the running-bond TT at every χ**, and the TT *compounds* error with depth
  (χ=256 ΔNLL +0.23→+0.75→+0.98 over 12→24→36 layers — *worse* with scale, not better). So the only free CPU lever is
  core_rank's per-layer rank-⅓·d (a ~3× constant-factor FLOP saving, lossless, no retrain), **not** a χ≈16 collapse.
  The composition graph is densely coupled (adjacent > distant); the ontology of typed directions is
  grammar-at-the-rim / content-in-the-core.
- **The Θ(d) floor is a FROZEN-LINEAR artifact — with retraining it falls ~30×** (`core_distill.py`). The no-retrain
  results all freeze weights + use a fixed PCA basis, which says nothing about a *learned* representation. Training a
  per-layer rank-r update-bottleneck (init from PCA = the no-retrain floor; base model frozen; ~300 steps): a **trained
  rank-8 update (1% of d) is lossless** on GPT-2 (ΔNLL +0.03 vs the no-retrain +1.78), a ~30× rank reduction; a
  rank-256 control trains to ΔNLL≈0, ruling out a domain-adaptation confound. So the "entangled core" is **not
  irreducible** — only the frozen-linear route was blocked; detangling/compressing it is tractable *with learning*
  (the sae-forge feature-native direction). Scope: this compresses the per-layer *update*, not full internal FLOPs;
  gpt2-large needs more rank/steps (rank-8→76%, rank-64→88% in 250 steps) — the lossless rank grows modestly with
  size/budget but stays ≪ the frozen floor.
- **But "rank-8 is lossless" is metric-specific — behaviourally the forge tax is HIGH-rank** (`compose_core.py`).
  The rank-8 result above is *true-token NLL* (a retrieval-dominated loss — what `core_distill` optimised). Extracting
  the bonds and measuring *behavioural reproduction of the model* (KL to the model's own logits + top-1 agreement,
  on held-out tokens tagged retrieval-vs-composition by the actual **pylm** flat predictor) tells a sharper story.
  Under no-retrain PCA truncation, top-1 agreement with the model rises with rank for **retrieval** tokens (27%→64%
  over rank 2→64) but barely moves for **composition** (forge-tax) tokens (5%→20%, plateauing ~15-20% even at rank-64
  = 8% of d). A *random* rank-r channel scores **0% at every rank** — the update occupies a specific subspace, not any.
  KL-distilling the rank-8 bonds **to the model** (the correct extraction objective, not true-token CE — which had
  *raised* KL) halves the divergence (KL retr 2.33→1.37, comp 3.05→1.89) yet still reproduces the model's argmax on
  only **55% / 17%** of retrieval / composition tokens. A confound check (composition top-1 confidence 0.25 vs
  retrieval 0.43) is real but too small to explain the ~3× retention gap. **So the "computed not retrieved" forge tax
  is precisely the HIGH-rank component of the inter-layer update**: a rank-8 channel captures the stereotyped
  retrieval-like writes, not the composition. The extracted artifact (`pylm/compose_core_gpt2.npz`, 0.55 MB, 147 K
  params — the first time the rank-r bonds are serialised) is a *real* object, but it is the channel's low-rank bulk,
  not a sufficient composition program. This refines "the core is a tiny rank-r object" the way the recoverability
  sweep refined "compression is variance-greedy" — the optimistic form holds for one metric and breaks under the
  stricter one.
- **Made runnable (pylm).** A flat-file **grammar** idiom decompiles the scaffold ([pylm track](PYLM_TRACK.md)), but
  adds ~nothing to the *token*-level decompilable fraction (49.0→49.5%) — grammar is categorial; the n-gram modes
  already absorb it. The un-decompiled ~50% is content that is **neither n-gram nor relational fact** — the entangled
  composition, the forge tax restated. So the flat-file basis {induction, grammar, n-gram, knowledge} is *sufficient
  for half*, and the complement is the core.
- **Anatomy of the forge tax: it is COMPUTATION (not fuzzy retrieval), and mostly SYNTAX** (`forge_tax_anatomy.py`).
  `context_ceiling.py` sized the composition residual (what unbounded exact ∞-gram can't reproduce); this asks *what it
  is*. Every held-out position is assigned to the first rung of a retrieval ladder that reproduces the model's top-1:
  **flat** (pylm bounded store) → **exact-copy** (unbounded ∞-gram over the train stream) → **soft-assoc** (nearest
  neighbour by mean-pooled *input*-embedding context — a fair "is it a fuzzy associative lookup?" test, no deep
  computation) → **computed** (none). Across two models and two corpora (GPT-2 + Pythia-70m; tinyshakespeare + wikitext;
  1200 positions each): of the forge tax, extended exact-copy cracks only **6-12%**, soft-associative **1-5%**, and
  **~87-90% is genuinely computed** — robust to model *and* corpus. **The core is not a soft database**: surface
  retrieval, exact or fuzzy, barely dents it. The computed residual carries a **large syntactic share** —
  function-word + punctuation is **62-75%** (GPT-2/Shakespeare 47%+28%, Pythia 45%+37%, GPT-2/wikitext 40%+22%) — i.e.
  much of the "computation" is getting the grammatical *skeleton* right (which function word, which punctuation) under
  long-range context. But that share is **corpus-shaped**: on prose (wikitext) the *content* fraction nearly doubles
  (26%→**39%**) as verse's punctuation load drops, so "mostly syntactic" holds on verse, "syntactic-leaning, with
  substantial content" on prose. Confidence falls monotonically down the ladder (flat 0.45 → computed 0.26), so the
  computed tokens are the genuinely uncertain positions. This *names* the forge tax — **computation, syntax-heavy**,
  the quantified upstream of the recursive-syntax result below. Honest scope: the soft retriever is surface-form (input
  embeddings, ≤40 K-context DB) so "computed" = *not surface-retrievable* (a model-strength retriever would be
  circular). **Not small-model-shaped** (`forge_tax_anatomy_fieldrun.py`): running the identical ladder against
  **Qwen2.5 served by the fieldrun runtime** (per-position top-1 from `fieldrun --dump`, validated **98%
  top-1-agreement with HF transformers**, embeddings mmap'd from the bundle) on the *in-distribution* corpus
  (wikitext; Shakespeare is OOD for Qwen → an inflated, meaningless tax), across three scale points:

  | model (wikitext) | params | forge tax | computed | content / function / punct |
  |---|---|---|---|---|
  | GPT-2 | 124 M | 84% | 87% | 39 / 40 / 22 |
  | Qwen2.5-0.5B | 500 M | 62% | 89% | 39 / 38 / 23 |
  | Qwen2.5-1.5B | 1.5 B | 64% | 90% | 40 / 41 / 19 |

  The forge tax's *size* **falls then plateaus** with capability (84→62→64% — a better model n-gram-reproduces more
  of its own output) but its **makeup is scale-invariant**: computed **87→89→90%**, content/function/punct **≈40/40/20**
  — flat across 12× params and two architectures. So "the forge tax is computation (not fuzzy retrieval), syntax-leaning
  with substantial content" is a property of the trained transformer, not an artifact of small capacity.
- **The *recursive* syntax is in the composition, not the basis** (`recursive_syntax.py`). Subject–verb agreement
  across attractors (*"the key near the cabinets **is**"*) is a hierarchical dependency: the model agrees with the
  **head** ~100% across depth (gpt2 small/large, Llama), resisting the nearest noun, with the logit-diff *degrading*
  with distance (bounded forward-pass depth, TC⁰). But (a) the **flat pylm** program follows the nearest attractor
  (0% head / 100% attractor at depth ≥ 1) — it's not in the decompilation; and (b) **ablating attention collapses the
  model to the attractor** at depth ≥ 1 (depth-0 survives — the number is local), while MLP-ablation just destroys the
  readout. So **attention composition carries the head's number across the attractors.** Categorial grammar lives in
  the static basis (decompilable); recursive/hierarchical syntax lives in the composition — the entangled core the
  forge tax measures. "Simpler-than-Chomsky" in the basis, Chomskyan in the composition. **The number-mover is a
  small, distinct UNNAMED circuit** (`agreement_circuit.py`): per-head ablation localizes ~4 mid-to-late heads that
  attend verb→head (gpt2: 7.4/10.9/8.5; gpt2-large: 24.3/32.5/25.4, verb→head 0.3–0.4 vs verb→attractor ~0.05), none
  of them induction/prev-token/duplicate — a new operator class for the catalog.
- **The recursion depth limit is set by distribution, not layers** (`recursion_depth.py`). Distance (PP attractors)
  vs nesting (center-embedding) across depth 0–5, gpt2 12L/24L/36L + Llama 16L: **distance is interference-bounded**
  (≥75% through depth 5, gradual decay, never flips), **nesting breaks sooner** (center-embedding harder than
  distance), **but the nesting ceiling SHRINKS with model size** (12L→5, 16L→3, 24L→3, 36L→2) — the opposite of "more
  layers → deeper recursion." So the TC⁰ layer-bound is real *in principle* but *not* the binding constraint: all
  models have ≫ enough layers for depth ~2–3, and they fail for distributional/interference reasons (deep
  center-embedding is rare in training and unparseable for humans; bigger models commit harder to the natural local
  parse). Layers aren't the active limit; the decode loop / chain-of-thought is how a model goes deeper (TC⁰ per step,
  Turing-complete across steps).
- **Dyck bracket-matching: the bottleneck is binding-interference, not stack depth — and scale buys *hierarchy*
  preferentially** (`recursion_depth_probe.py`). A controlled probe of recursive *structure*: predicting a closing
  bracket's TYPE requires tracking the open-bracket stack. Single-forced-close design (each prompt ends where exactly
  one closer is forced, avoiding the "closing-momentum" confound of full ramps), with two families at *matched
  distance* — **DEEP** `( [ { } ] →` (deep nesting) vs **FLAT** `( [] {} () →` (flat distractor pairs). Across scale
  (HF: pythia-70m/gpt2/pythia-410m/gpt2-large; **Qwen2.5-0.5B via fieldrun**): (1) the deepest reliably-matched nesting
  **grows with model size** (6L pythia-70m → depth 0; 12L gpt2 → 7; 24–36L → the dmax-9 ceiling) — the *opposite* of
  the center-embedding result above, and consistent with it: Dyck/code nesting is *in-distribution* so scale helps,
  natural center-embedding is not so scale hurts → both say the limit is **distributional, not a hard layer bound**.
  (2) **DEEP is consistently EASIER than FLAT at matched distance, and the gap GROWS with capability** (deep−flat acc:
  pythia-70m +0.06 → gpt2 +0.20 → pythia-410m +0.24 → gpt2-large +0.45 → **Qwen-0.5B +0.54**). So the failure is not
  stack-depth overflow — the model is *good* at clean hierarchy and gets disproportionately better at it with scale;
  what it struggles with is long-range binding across same-level distractors. This refines the "bounded recursive
  evaluator" reading: it's a **hierarchy tracker whose competence scales**, bottlenecked by binding-interference.
  (Recursive *computation* — Lisp arithmetic eval — is a separate, much harder probe; base GPT-2 can't do it at all,
  so it needs Qwen-scale via fieldrun: `lisp_eval_probe.py`.)
- **Lisp evaluation: the model runs a layer-consuming recursive EVALUATOR (computation, unlike Dyck *structure*)**
  (`lisp_eval_probe.py`). Evaluating `(+ 1 (* 3 (- 5 1)))` forces bottom-up recursive descent — compute `(- 5 1)=4`,
  then `(* 3 4)=12`, then `13` — so it tests recursion that carries a *value* stack, not just bracket pointers. Lisp
  is the cleanest probe: full parenthesization ⇒ surface = parse tree (no precedence shortcuts), prefix ⇒ operator
  known before operands. Few-shot `(expr) = digit` prompt, single-digit answers (exact read), controlled nesting depth.
  - **Computation emerges with scale.** Base **GPT-2 (124 M) ≈ chance** — it cannot evaluate even `(+ 1 2)`.
    **Qwen2.5-1.5B nails depth-1 (100%)** and degrades monotonically: depth 1→6 acc **1.00 / 0.63 / 0.53 / 0.43 /
    0.37 / 0.27** (reliable to nesting depth ~3). Recursive *evaluation* is a capability that appears between 124 M and
    1.5 B — where recursive *structure* (Dyck) was already present in tiny models.
  - **Each recursive level consumes layers** (the mechanism, HF logit-lens). The layer at which the correct answer
    first becomes the logit-lens argmax rises monotonically with nesting depth: **23.8 → 25.8 → 25.9 → 26.0 → 26.6 →
    27.1** (of 28), ≈**+0.6 layers per nesting level**. Direct evidence the model evaluates bottom-up, spending depth
    of *network* per depth of *expression* — and why accuracy collapses as nesting approaches the layer budget.
  - **Notation-general (Lisp-specificity).** Prefix vs (parenthesized) infix `(a + b)` are nearly identical (depth-6
    acc 0.27 vs 0.30; same rising resolve-layer) — the recursion is about *nesting structure*, not operator position;
    Lisp just makes it maximally explicit. (Caveat: parenthesized infix doesn't test precedence-shortcutting — true
    unparenthesized-precedence infix is the open follow-up.)
  - **The through-line:** recursive STRUCTURE (Dyck matching) is a hierarchy tracker, binding-interference-limited,
    present in small models; recursive COMPUTATION (Lisp eval) is a layer-bounded evaluator, depth-degrading and
    layer-consuming, emergent with scale. Computation needs the value stack that matching doesn't — the forge tax's
    "computed, syntax-heavy" residual is, in part, exactly this bounded recursive evaluator.

## Knowledge — where facts live, and moving them

The catalog is about *mechanisms*; the [knowledge axis](circuits/causal_tracing.md) is the decompiler goal ("the
model IS the database").

- **The READ side — dump the table, and decompile where it's queried** (`relation_decompile.py`). Treating a relation
  (capital-of, language-of) as a database table: the model's table is **100% complete** for these common facts in all
  six models (every subject's object read out correctly, vs 7–11% chance), and the logit-lens locates **where in depth
  the relation resolves** (the earliest layer the object is decodable from the last-token residual). Two cross-model
  reads: (i) the read-out depth **shrinks with GPT-2 scale** — capital resolves at **76% → 60% → 40%** of depth
  (small → large) — bigger models retrieve the fact *earlier* (more post-retrieval compute); (ii) **`language` resolves
  earlier than `capital`** in almost every model (e.g. Llama 39% vs 52%, Gemma 41% vs 69%) — a more directly-bound
  attribute is queried sooner. So the read side of "the model IS the database" is concrete: the table is queryable and
  complete, and its query site is locatable and scale-dependent. (A faithful *linear* relation operator — the LRE that
  would make the table a standalone queryable view — needs the model's Jacobian, which OOMs the 2-B models on a 7.5 GB
  GPU; the robust logit-lens read is the cross-model version.)

- **The ROME two-site flow is architecture-invariant.** [Causal tracing](circuits/causal_tracing.md) (subject
  corruption + restoration) recovers the same structure in GPT-2 ×3, Llama, Qwen: an **early MLP store at the
  subject** (peak depth ≈0) feeds a **late attention readout at the last token** (depth ≈0.6–0.9). Cross-model,
  which ROME never did.
- **The store is editable by activation patch.** [Patching](circuits/fact_patching.md) the early-MLP store at the
  subject with a different fact's activation **transplants the fact 100% of the time** (France's run answers Rome)
  in those five models — the decompile→recompile loop made concrete, sufficiency complementing necessity.
- **But it's a *verified* write of the whole entity, not a single fact-row** (`fact_edit_xmodel.py`, the ROME triad
  cross-model). The early-MLP store edit is **efficacious** (band-patch flips the capital 100% in 5/6; Gemma 0%,
  resistant — its store is distributed), **generalizes** (holds under a paraphrase prompt, 100%), and is
  **position-localized** (patching a relation token instead of the subject flips nothing, 0%). The catch is
  **entity-leakage**: editing the *capital* store also flips the subject's *language* **56–100%** of the time (Llama
  the worst, a *full* entity swap at 100%; GPT-2 ~60%). So at this site the addressable unit is the **entity, not the
  fact** — you can't surgically rewrite one row without dragging the entity's other facts along. And **editability
  granularity tracks the store's concentration**: GPT-2/Llama edit at a *single* early MLP, but **Qwen needs the whole
  early band** (single-layer 0% → band 100%), exactly the model whose detokenizer substrate is spread across L0–L2
  ([MLP nodes](circuits/README.md)). The decompiler can write to the database, but the row it writes is the entity.
- **And there is no fact-addressable site *at any depth*** (`fact_site_sweep.py`, efficacy-vs-leakage per layer). We
  looked for a *later* layer where the edit stays efficacious but stops leaking — a layer where the row would be the
  *fact*. None exists: in every editable model the best fact-specificity is at **L0** (the entity store), and no deeper
  single-MLP edit keeps efficacy while shedding leakage (lowest leakage ~50%, GPT-2-large). Confirmed across six models —
  the four *editable* ones (GPT-2 small/medium/large + Llama-3.2-1B) plus two that reject the edit — and no layer in
  *any* of the editable models clears even a lenient clean-site bar (efficacy ≥50% with leakage ≤20%). **Gemma-2-2B and
  Qwen2.5-1.5B don't take the activation-patch edit at all** (efficacy 0% at
  every layer), consistent with their fact-transplant resistance — so for those the question is moot, not clean. So
  entity-addressability is **depth-invariant** — a single fact is not an independently editable row at any single-MLP
  site via activation patching. Surgical single-fact editing would need weight surgery (ROME's rank-1) or a cleaner
  basis, not a better *location* — a hard limit for the "model IS the database" framing, measured cross-model. *(Scope,
  per the necessity-vs-method discipline: this sweeps the **layer** axis only — the token-position × update-rank ×
  bystander-fact axes are not yet swept, so a clean fact-local site elsewhere in that space remains open, not excluded.)*
- **Method *or* representation? Both — and the irreducible part shrinks with scale** (`fact_rome_xmodel.py`). Is the
  entity-leakage because the activation patch is *blunt* (it swaps the whole MLP output), or because capital and
  language are genuinely *entangled*? We optimised a **targeted** ROME-style edit-value `v` (flip the capital while a
  KL term preserves the subject's essence) and traced the efficacy-vs-leakage frontier vs the blunt entity-patch
  baseline (GPT-2 ladder; the 2-B RoPE models don't fit the backprop graph on a 7.5 GB GPU — a real limit, unlike the
  activation-patch runs). A targeted edit **roughly halves** the leakage — GPT-2 62%→**38%**, GPT-2-medium 62%→**12%**,
  GPT-2-large 50%→**12%** — so *part* of the leakage is the **method** (the blunt swap drags the whole entity). But the
  leakage **floors** (12–38%) and won't go to zero however hard we preserve essence — so *part* is the
  **representation** (capital and language share the subject's write direction irreducibly). The decisive scaling
  signal: that **irreducible floor falls with size** (38% → 12%), i.e. bigger models carry **more separable** entity/
  fact representations — surgical single-fact editing is partly a method problem (a better edit helps) and gets
  *easier* with scale.

## The outliers — where the next questions are

- **Gemma-2-2B is the recurring exception across seven independent measurements**: a near-absent attention-sink
  (~4% vs 44–55%), the most *distributed* induction key, a non-monotonic (compensatory) induction-redundancy curve,
  the *strongest* MLP0 extended-embedding (η² 0.91), induction that doesn't lean on MLP0, a **late** fact site
  (vs early elsewhere), and **fact-transplant-resistant** early MLPs (3% flip vs 100%). Gemma stores and routes
  information differently enough that it falls out of nearly every cross-model regularity — the single most
  informative "third architecture" in the set.
  - **Family trait, not a Gemma-2 quirk — the controlled test** (`gemma3_anomaly_summary.json`; the ungated
    **Gemma-3-1B** vs Gemma-2-2B on the non-SAE anomalies). A second Gemma disambiguates "Gemma-2-specific" from
    "Gemma-family": **(1) the near-absent sink reproduces** — Gemma-3-1B sink signal **0.056** ≈ Gemma-2's 0.059
    (both ~0, vs 0.7–0.9 for the others), so the sink absence is a **Gemma-family** signature; **(2) the strong
    token-determined MLP0 reproduces** — Gemma-3-1B η² **0.80** (Gemma-2 0.91), both far above GPT-2's 0.63, and
    Gemma-3's determined block is even **wider** (η² stays ~0.78–0.82 through L2 where Gemma-2 has already fallen to
    0.56) — again a **family** trait. Two differences are *not* shared: in Gemma-3 the **prev-token** heads are
    causally load-bearing (ablation ΔNLL **+0.45** vs Gemma-2's −0.01) and **induction ablation bites** (+0.04 vs
    Gemma-2's −0.28 self-repair) — so Gemma-2's induction self-repair / non-load-bearing prev-token is *more*
    Gemma-2-specific. Verdict (descriptive): the **addressing/embedding anomalies (sink, MLP0) are Gemma-family**; the
    **induction-circuit redundancy differs within the family**. *(Scope: Gemma-3-4B is gated and Gemma-Scope SAEs are
    Gemma-2-only, so the two SAE-dependent anomalies — redundancy-monotonicity and store-routing — are not yet
    testable on Gemma-3; this resolves the operator/MLP0 anomalies, not all six.)*
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

**But the recruited heads do *not* cleanly tile the input by an interpretable property (a null result).** The natural
follow-up — if more inputs recruit more heads, does each head *own a slice* of the input? — we tested directly
(`circuit_domain_tiling.py`): for every induction position take the head that dominates its induction attention, then
ask whether the dominant head is predicted by the matched token's **frequency** (η² vs a label-permutation null) or
whether heads own **disjoint token sets** (Jaccard). At a 256-type vocabulary, **frequency-specialization is *not*
significant in any of 5 models** (η² ≈ its permutation null, z ≈ 0–1 — the weak gpt2 signal at a smaller vocabulary
did not replicate). The low token-set Jaccard (0.04–0.06) *looks* like a partition but **lacks a null** and is
confounded by the very distributedness above: with the top head holding only 1–8% of the induction attention, "which
head owns this position" is a **noisy** label, so no clean specialist→input-slice map exists to find. So "same
function over more inputs" **recruits** heads but they are a *redundant distributed population*, not crisp domain
specialists carving the input along token frequency — consistent with the low functional-overlap / no-duplicates
picture above. (`runs/disassembly/circuits/domain_tiling_summary.json`.)

**Synthesis across the four tests.** A distributing induction circuit is (1) **not** a weighted ensemble of
duplicates (OV-cosine ≈0), (2) made of **structurally heterogeneous heads** — separable parallel sub-circuits in the
absolute-position family, one shared-front-end decomposition in the RoPE family — (3) its head-count is driven by
**input-domain breadth** as a first-class axis alongside model size, but (4) the recruited heads do **not** crisply
specialize by an interpretable input property (token frequency): the recruitment grows a *redundant distributed*
population, not a clean input→head tiling. Gemma is the exception to (2)'s and (3)'s regularities, as it is to most.

### Does RoPE's shared early-writer front-end make induction fragile to post-training?

If RoPE hangs most of its induction population off **one** shared early predecessor-writer (Llama L0 head 0.2 feeds
88% of readers), is that a single point of failure post-training could break — or is the early node *protected*
because fine-tuning adjusts late layers more? We measured both on three base→instruct pairs of the same model
(`posttrain_drift.py`): per-layer weight drift vs depth (lazy safetensors reads) + induction survival (induction-NLL
and the prev-token writer head, base vs instruct).

- **Early layers are NOT meaningfully shielded.** The attention weight-drift is roughly flat across depth — early/late
  ratio **0.84–0.93** — and the induction writer's own layer drifts at **0.92–0.99×** the model mean. So the "early
  nodes get adjusted less" backstop is at best *marginally* true; the shared early writer is **not** protected by
  being early. Post-training reaches it.
- **Yet the circuit is resilient, not fragile — no catastrophic single-point break.** The shared writer **head
  survives** post-training where the model is lightly tuned (Qwen: ~1% drift, induction-NLL unchanged 0.35→0.35,
  writer 13.4→13.4) and even where it is heavily tuned (Llama: ~13% drift, induction-NLL degrades 0.78→1.73 but the
  writer head 0.2 **persists** — performance drops, structure holds). **Gemma reorganizes adaptively**: its writer
  *moves* (layer 21.7 → 0.0) and induction gets **better** (NLL 4.87→3.35). In none of the three did the shared
  front-end catastrophically break the population.
- **The GPT-2 comparison refutes "RoPE suffers more."** Adding two GPT-2 fine-tune pairs (gpt2 → DialoGPT-small,
  gpt2-medium → DialoGPT-medium — a real, *heavy* dialogue-domain fine-tune) lets us compare GPT-2's **distributed**
  induction (no single early node) against RoPE's **shared front-end** directly. The predecessor-writer head **survives
  in every model of both families** (gpt2-medium 5.11→5.11, like Llama 0.2→0.2 and Qwen 13.4→13.4; only Gemma
  reorganizes). GPT-2's induction actually degraded **more** in absolute terms (DialoGPT Δ **+2.57 / +0.75** vs the
  RoPE pairs' +0.96 / −0.00 / −1.52) — but it also **drifted 3–30× more** (DialoGPT attn-drift 0.34–0.38 vs RoPE
  0.01–0.14), so per-unit-drift the damage is comparable. The degradation tracks **how hard the model was fine-tuned**,
  not whether induction hangs off one shared early node — and the concentrated RoPE front-end is, if anything, *no more*
  fragile than GPT-2's distributed one. **Caveats:** DialoGPT is a *domain* fine-tune (dialogue), heavier and different
  from the RoPE *instruction*-tunes, so the two families aren't intensity-matched; n is small (2 GPT-2, 3 RoPE); and
  induction-NLL on repeated-random is mildly out-of-distribution for any post-trained model, so part of every
  degradation may be distribution shift, not circuit damage. (`runs/disassembly/posttrain_drift_summary.json`.)

- **Post-training *consolidates* the writer-dependency structure rather than breaking it.** Re-running the
  writer-dependency clustering (above) on the post-trained models — base → instruct/fine-tune, same settings, every
  zero-patch sanity 0.0 — the **shared-front-end fraction and profile-cosine *rise* in all four**: Llama 88% → **100%**
  (writer 0.2 unchanged), Qwen 75% → 88% (3.2 unchanged), Gemma 50% → 75%, and GPT-2-medium's *fragmented* **2-cluster**
  structure collapses to **1 cluster** under DialoGPT (38% → 62% shared, profile-cos 0.58 → **0.90**). So fine-tuning
  doesn't redistribute induction off the shared writer — the readers lean on it **more** afterward (plausibly because
  instruction/dialogue tuning *sharpens* attention). The distribution structure is not just robust but *reinforced*.

**Verdict.** RoPE's shared early-writer front-end is **exposed** (not depth-protected) but **robust** — under realistic
post-training the predecessor-writer head persists (or, in Gemma, reorganizes to an even-earlier one while induction
*improves*), the **direct GPT-2 comparison shows no extra fragility** from the concentration (GPT-2's distributed
induction degrades at least as much under a heavier fine-tune, writer head surviving in both), and the writer-dependency
structure is **consolidated, not broken**, by post-training. The single-point-of-failure worry isn't borne out; the
failure mode that appears is graded *performance* degradation proportional to fine-tuning intensity, not structural
breakage of the shared early node.

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

## Beyond attention — Mamba (SSM) across the themes

Mamba has no attention heads and no per-layer MLP — just a residual stream of SSM `mixer` blocks — so the
attention-based catalog (heads, K/Q/V composition, name-movers) has no analog. But the *arch-generic* themes run on
it via a Mamba-specific harness (`mamba_themes.py`, the Mamba ladder 130m/370m/790m), and the result splits cleanly
into **what is attention-specific** and **what is architecture-invariant**:

- **The copy *mechanism* is layer-distributed, not head-localized.** In-context copy works (induction-NLL 0.86 → 0.59
  → 0.53, *improving* with scale) and is causally load-bearing (ablating all SSM layers costs **+14 to +20**
  induction-NLL). But where a transformer localizes induction to a few **heads** (and GPT-2-small to *one* dominant
  prev-token writer), Mamba spreads it across **~7 layers** (effective-N 6.6–7.2 of N), no single layer carrying more
  than 23–32%. The SSM realises the same capability as a *distributed multi-layer* computation — there is no head to
  name, so the disassembly's head-circuits genuinely have no SSM counterpart.
- **But the knowledge themes are architecture-invariant.** On the *same* axes as the six transformers, Mamba behaves
  like a transformer:
  - **READ** — the relation table is **complete** (capital 100% across the ladder; language 78–100%), and the
    logit-lens read-out depth **shrinks with scale** exactly as in the transformers — capital **86% → 81% → 52%**,
    language **79% → 76% → 40%** (130m → 790m). Bigger SSMs retrieve facts earlier, the same scaling law.
  - **WRITE** — grafting a donor subject's early-layer **residual** transplants the fact **100%** of the time, and it
    is **entity-leaky** (editing the capital flips the language **91% → 100% → 100%**) — *more* leaky than the
    transformers (56–67%). So the SSM's knowledge store is the same **entity-addressable, not fact-addressable**
    residual content the transformers showed, with the same depth-invariant entanglement.

**The through-line.** What is *attention-specific* is the **mechanism's localization** — head-circuits, the
positional register, name-movers — none of which survive into the SSM (induction goes layer-distributed). What is
*architecture-invariant* is the **knowledge-storage character** — a complete, queryable table whose read-out site
shrinks with scale, stored as an editable but entity-leaky residual. "The model IS the database" is a property of the
residual-stream LM, not of attention; the *circuits* that fill the database are what the mixer choice decides.

## SAE recoverability — detection is cheap, allocation is a competition (not variance-greedy)

A cross-substrate test (econ-sae macro-regime, bio-sae ESM-2, and here on GPT-2) of *when an unsupervised SAE
recovers a known feature*. For every exact-lexical ground-truth feature on GPT-2 layer-8 residuals we measure two
cheap SAE-free predictors and two expensive measurements: **Fisher SNR** `Δμᵀ(Σ_w+λI)⁻¹Δμ` (detection theory) and
**variance-share** `p(1−p)‖Δμ‖²/trΣ` (rate–distortion) against a linear-probe AUC (presence) and the **SAELens
24 576-feature** dictionary's best-latent recovery AUC (allocation). Reproduce: `scripts/recoverability_theory.py`
(summary in `runs/substrate/recoverability_theory_summary.json`); synthesis in the workspace `SUPERVISION_DEPENDENCE.md`.

- **Presence ≠ allocation, on a real LM.** 6 / 28 features are *present yet dropped* — a probe reads them at
  AUC ≥ 0.85 but the SAE recovers them at < 0.85. They are **the entire lexical tier**: `is_capitalized` (probe
  **1.00**, SAE 0.72), `has_leading_space` (1.00 / 0.72), `len2` (0.99 / 0.63). Diffuse token *properties* are
  maximally detectable yet poorly recovered; sharp one-token detectors (the `token` tier) recover at cov95 **89.5%**.
- **Presence is Fisher.** Partial `fisher→probe | var_share` **+0.64** — detection theory predicts readability, the
  same as on econ-sae and ESM-2.
- **Allocation is *not* variance-share.** Partial `var_share→SAE | fisher` is **−0.35** (negative); `fisher→SAE`
  **+0.51**. The diffuse lexical features carry *higher* variance-share than the rare sharp tokens yet recover worse,
  because an over-complete production SAE **splits / absorbs** a common diffuse property across many latents — so no
  single latent cleanly encodes it. This is the same sign as ESM-2 (−0.26 to −0.30): the rate-distortion predictor
  does not survive cross-substrate.

**The through-line.** Rate-distortion governs *reconstruction* (variance captured); SAE interpretability needs
*monosemantic allocation* (one latent per feature). They diverge for **rare** meaning (no co-firing mass for a latent)
and **diffuse** meaning (split across latents) — exactly the dropped set. The robust law is two-axis: *detection is
cheap and near-universal; unsupervised recovery is a competition for latents won by distinctiveness and statistical
mass, not by variance-share.* "Compression is variance-greedy" holds only where Fisher is held roughly constant.

---

_This page is a hand-curated narrative; the numbers live in the generated [operator](operators/README.md) /
[circuit](circuits/README.md) / [MLP](operators/mlp_compute.md) catalogs, the
[extended-embedding test](operators/mlp_detokenizer.md), and the [outlier digs](operators/outlier_digs.md), each
regenerable from committed JSON. See [DISASSEMBLY.md](DISASSEMBLY.md) for the original GPT-2 method deep-dive._
