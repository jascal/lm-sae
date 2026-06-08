# From op-catalog to decompilation — a research design

[`DISASSEMBLY.md`](DISASSEMBLY.md) reads a transformer's attention as a **first-order instruction set**:
per-head QK/OV opcodes, named idioms, a coverage scorecard ("% of attention *legible*"), causal validation,
and cross-model replication. This document scopes the next phase: turning that disassembly into an
**executable decompilation** — reconstructing the *computation*, not just labeling components — and argues
the obstacle is the *same* entanglement the forge tax measures.

This is a design doc (a target + a metric + milestones), not a results writeup. It is the flagship that
ideas (i) cross-model breadth and (ii) multilingual invariance de-risk: you want the disassembler proven on
≥3 architectures before you trust a single decompilation-coverage number.

> **The decompilation taken literally — see the [pylm track](PYLM_TRACK.md):** a whole small LLM reimplemented as a
> *small pure-Python program + flat-file knowledge store* (no neural-net code or concepts), validated against the
> corpus and the model. It makes the decompilable fraction a **running artifact** (~50% of GPT-2 in 49 lines; 56% of
> Pythia-14m; shrinking with scale) — the disassembly→decompilation arc reaching its endpoint, and the forge-tax
> ceiling visible as the half it can't reach.

## The gap: disassembly ≠ decompilation

| | disassembly (have) | decompilation (want) |
|---|---|---|
| unit | one head / one idiom in isolation | the **composition** — which ops chain into which |
| coverage metric | *% of attention legible* (named or weight-binding) | *% of the forward pass faithfully reconstructable* |
| MLP | first-order read→write token catalog | MLP neurons as first-class ops (key–value memories) |
| validation | mean-ablation damages the metric (necessary) | **recompile the extracted program; KL ≈ host** (sufficient) |
| output | a human-readable listing | a runnable reduced model + a symbolic trace |

The disassembly already took the first two rungs of composition: the **induction macro** read as
`OV_A ∘ QK_B` (prev-token head feeds the induction head; weight-composed diagonal vs behavior ρ≈0.78,
path-patch-gated), and the **IOI chain** `duplicate → S-inhibition → name-mover` read as a product of
Q-composition scores. Decompilation is the *general* algorithm those two are instances of.

## The load-bearing idea: reconstruction coverage

Replace "% legible" with a metric that has teeth — **does the extracted op-graph, compiled back into a
runnable model, reproduce the host's next-token distribution?**

```
reconstruction_coverage(budget B) =
    1 − KL( host ‖ recompiled[ ops selected under budget B ] ) / KL( host ‖ mean-ablated-everything )
```

- **Recompile = forge.** sae-forge's `NativeModel.from_projected_weights` already takes a basis + projected
  weights and produces a runnable model (`forward_mode=native_in_basis`). A decompilation is a *structured*
  forge: keep exactly the ops the disassembly named (heads' QK/OV in the operand basis + catalogued MLP
  neurons), mean-ablate the rest, run, measure KL. This is the "recompile" half of larql's
  decompile→query→**recompile** loop, used here as the faithfulness oracle.
- **Sweep the budget** (how many ops kept) → a coverage *curve*, the decompilation analog of the
  coverage scorecard's single number. The area under it is "how much of the model the catalog explains,
  *executably*."
- **Per-task traces.** For a templated task (IOI, induction, greater-than), emit the **symbolic trace**:
  the ordered DAG of ops that fire to produce the output, each edge path-patch-validated. A correct trace
  + a faithful recompile = a decompilation of that behavior.

> **First results are live — and humbling.** The reconstruction-coverage idea is now run across six models for the
> induction circuit and (GPT-2) the IOI circuit: **[Executable decompilation](circuits/reconstruction.md)** (keep
> the circuit's heads, ablate the rest, measure induction-NLL / IOI logit-diff recovery + the budget *curve*) and
> the **[attention-vs-MLP substrate](circuits/induction_substrate.md)** split. The honest headline: **no small
> head-set is a *sufficient* decompilation** — induction recovers ≤30% from its named 8 heads (even under the
> gentler resample-ablation), needs ~all heads to fully reconstruct, and leans on attention and the MLP substrate
> roughly equally; even IOI's celebrated 26-head circuit is not sufficient in isolation. The named circuits are
> necessary and the dominant drivers, but the behaviour is distributed across the near-whole network. So a faithful
> decompilation here is *not* a tiny op-graph — the budget curve, not a single small circuit, is the real object.

## Four work-items

### 1. The composition DAG extractor
Generalize rung-2/3 into the standard algorithm: from the weights, score every directed edge
`OV_g → QK_h` (head g's write feeds head h's query/key/value) as a composition bilinear in the operand
basis; threshold + path-patch-validate to keep live edges. Output: the model's circuit DAG (heads + MLP
nodes, typed edges Q/K/V-composition). The induction macro and IOI chain become the first two validated
sub-DAGs; the extractor produces them automatically and finds new ones.

### 2. The MLP gap
The catalog is attention-centric; MLPs carry real computation (greater-than is MLP-dominated — the OV probe
sees only its attention-side shadow). Treat MLP neurons as ops: in-direction (gate/up read) and
out-direction (down write) in the operand basis = a key→value memory; cluster into named MLP idioms
(detokenization, entity attributes, successor arithmetic). Add them as nodes in the DAG (1) and as keepable
ops in the recompile (the metric).

### 3. The operand-basis ceiling
First-order disassembly uses centroid / SAE operands that are imperfect; composition only reads cleanly in a
basis where the ops *compose* cleanly. This is where decompilation meets the rest of the program: the
two-basis forge's **U_C (composition subspace)** is exactly "the basis in which circuits survive." (The
specific *writer-output* U_C construction was retracted — see DISASSEMBLY.md — but the *target* stands: find
the basis that makes reconstruction_coverage high at low budget.) `polygram` models the operand-dictionary
geometry; the decompiler consumes it.

### 4. Executable verification & emission
Wire (1)+(2) into the recompile metric (the load-bearing idea), and emit the DAG in a checkable form:
`n-orca` can express the extracted circuit as a typed neural DAG; `larql` can hold the decompiled model as a
queryable index ("the model IS the database"). A decompilation is *accepted* only if recompile-KL clears a
threshold — no unverified symbolic story counts.

## The unifying claim (why this is one program, not a side quest)

> Composition is not cleanly feature-readable for the **same reason** cov95 collapses under forging:
> the composition is entangled (the entanglement tower's irreducible high-χ core).

So **"how much can we decompile" and "how big is the un-forgeable core" are the same quantity measured from
two directions** — reconstruction_coverage from the disassembly side, `1 − cov95-tax` / the tower's
convergent core from the forge side. The falsifiable prediction:

> The reconstruction-coverage curve **plateaus** at the same fraction the entanglement tower identifies as
> the irreducible entangled core (~the low-χ substrate), and **no op-budget closes the gap** — because the
> residual *is* the entangled composition, which by the tower's no-go is not decompilable into first-order
> ops. Decompilation has a hard ceiling, and that ceiling is the forge tax.

If true, this is the result: a single number — measured independently as (a) the decompilable fraction of
the forward pass and (b) the forgeable/low-χ fraction — that bounds how interpretable a transformer can be
made *without* retraining. If the curve instead reaches ~1.0, the entanglement story is wrong and the model
is fully decompilable (also a publishable result).

> **Flagship result on real LMs — composition doesn't factor through the SAE features (`sae_forge_tax.py`).** The
> sharp prediction of the unifying claim, tested directly: force the residual through the SAE feature basis
> (decode∘encode, the forge bottleneck) and compare the damage to the **composition** (induction-NLL) vs the
> **readout** (generic next-token NLL). The feature basis taxes **composition far more than readout** — GPT-2: the
> full forge raises induction-NLL **+1357%** but generic-NLL only **+70%** (composition taxed ~19×, *and* taxed more
> in **all 12 layers** individually); Gemma-2-2B: +542% vs +399% (net +143%, 5/8 layers — weaker, as its induction is
> already distributed and its Gemma-Scope SAEs reconstruct loosely). This is the cov95 forge tax measured from the
> *disassembly/reconstruction* side: SAE features survive as readouts, the computation **over** them does not factor
> **through** them — the two contributions (forge tax / disassembly) meeting on one host. (Honest scope: SAE
> reconstruction is lossy everywhere, so the *relative* composition-vs-readout tax is the signal, robust at 12/12
> layers in GPT-2; this is the reconstruction/NLL route, complementary to the cov95/mAUC route, **not** the full
> `NativeModel` weight-projection ceiling test of M4. `runs/disassembly/sae_forge_tax_summary.json`.)

### What form is the core? (not an SAE, not a dense slab — a program)
"SAE vs dense" is a false dichotomy for the entangled remainder. The forge tax says the core has **no sparse
feature (SAE) basis** in which composition is monosemantic — that rules out the SAE form. But the χ-ladder
showed composition *is* factorable in **weight/attention-composition coordinates** (induction = `OV_A∘QK_B`;
IOI = a head chain), and the instruction-tensor result found the program is **low-rank (~5–11 templates vs
~132 random) — in operand-pair space, not residual-direction space** (which is exactly why the residual-
direction search, joint-`U_C`, failed). So the core's natural form is a **program**: a compact DAG of reused
weight-space bilinear templates over *now-explicit* operands. Subtracting the clean low-χ substrate is then a
**change of coordinates, not a sparsification** — it makes the operands explicit so the remainder reads as
"bilinears with clean arguments," but it does **not** shrink the dense magnitude (the tower's ~24% irreducible
variance floor; the retrain no-go). Consequence for this program: the lever that "simplifies" the core is the
**composition-DAG extractor in operand coordinates** (work-item 1+2), *not* a better SAE — and the
reconstruction-coverage curve should plateau at the entangled-core fraction precisely because first-order
*feature* ops can't express the composition, while *template* ops over operands can. (Open at scale: whether
the template count stays small and whether the irreducible floor itself grows with capability — the tower
convergence was measured on small hosts.)

> **Core-rank result — the core is heavy-tailed low-rank, and CPU-simplifiable without retraining (`core_rank.py`).**
> Measuring the empirical rank of each layer's **residual update** (its contribution to the residual) over 6 models:
> the spectrum is **heavy-tailed** — a tiny dominant head (participation ratio **5–33**, ~1–2% of d) but a real tail,
> so the *functional* rank (energy-95 / accuracy-preserving) is **~⅓ of d** (hundreds of directions: rank-95
> 250→398→476 across the GPT-2 ladder), and the dominant-direction count **grows with scale** (PR 5→13→24) —
> consistent with the entangled core growing with capability, and with the forge tax's "no tiny basis." So the core
> decompiles into a **moderate-rank** structure, not ~5 templates. **But it is substantially CPU-simplifiable for
> free:** a *no-retrain* low-rank projection of every layer's update onto its top-r PCA subspace is **~lossless at 70%
> rank** in GPT-2 (≈ +0.05 NLL at 40%), and **Gemma-2 is lossless at ~20%** (its update is the lowest functional rank
> — ~80% of write-dims droppable), while **Llama/Qwen** are the least compressible (need ~70%+; higher functional
> rank). A concrete efficiency lever — rank-r per-layer factorisation, no retraining — with the compressible fraction
> **architecture-dependent** and shrinking with scale. (Caveat: this is the per-layer *write-subspace* rank, not the
> full input→output map; data-dependent PCA. `runs/disassembly/core_rank_summary.json`.)
>
> **And the layers SHARE that subspace.** The cross-layer sharing measure: the *union* of all layers' top-64 write-
> subspaces has effective rank **278 (GPT-2) / 517 (large) / 658 (Gemma) / 570 (Llama)** — far below the no-sharing
> bound (nL·64, capped at d): the whole model's per-layer writes collectively span only **~⅓–⅖ of d**, and layer pairs
> are **~2× more aligned than chance** (overlap 0.05–0.16 vs 0.03–0.08 random). So the entangled core is a **shared
> moderate-rank subspace**, not nL distinct ones — the strongest decompile/simplify lever yet: one shared
> ~300-direction basis captures most layers' composition (decompile *its* structure; or a single global low-rank
> projection serves every layer on CPU). Sharing is partial (overlap ≪ 1) and grows with scale — compact, but not a
> few templates.
>
> **…but a *single* global basis is not a free lunch (honest refinement).** Forcing *every* layer through the one
> shared union-basis at low rank is **much worse** than per-layer PCA (GPT-2 at 10% rank: **+4.48** global vs +0.47
> per-layer ΔNLL; Gemma +8.44 vs +0.38), only catching up at ~40–70% rank (Gemma's global basis is even slightly
> *better* than per-layer at 70%, −0.39 — its sharing is strongest). So the partial overlap does **not** let you
> collapse the core to one tiny global basis: the efficient simplification is **per-layer** low-rank (lossless at
> ~70% GPT-2 / ~20% Gemma), with a moderate-rank *global* basis a second-best. The core is a *shared-but-partial*
> moderate-rank subspace — compact, reused across layers, but with real per-layer structure: "not a few templates,
> not a dense slab," made quantitative.

### What IS the shared subspace? — decompile its *structure* (`core_basis_decompile.py`)

Knowing the core is a shared ~300-direction subspace ("how big") invites the deeper question: **is that basis
interpretable** ("what is it")? Three tests on the union basis `Ug` (top-K, K = its effective rank), GPT-2 s/m/l:

| test | what it asks | result (gpt2 / medium / large) | reading |
|---|---|---|---|
| **(A) token/logit** | does `Ug` lie in `W_U`'s top-K logit subspace? | **0.33 / 0.39 / 0.39** vs chance **0.36 / 0.40 / 0.40** | **at/below chance** — the core is *not* the readout subspace; it is upstream compute |
| **(B) operators** | is `Ug` built from the catalog ops' OV-write subspaces? | induction/prev-tok/dup/sink **0.39–0.47**, random heads **0.39–0.46**, chance 0.36–0.40 | named ≈ random — the core is the **aggregate of all writers**, not a few named idioms |
| **(C) logit-lens** | do `Ug`'s directions decode to coherent tokens? | peak-z **5.2 / 5.9 / 5.1** vs random **4.6 / 4.8 / 4.5** | **readable** — top dirs = function-words, punctuation, a verb axis, a register axis |

So the entangled core's shared basis is **logit-lens-readable per direction, but is neither the readout subspace
nor the named-operator basis** — it is a broad write-basis whose *individual* directions decode to grammatical /
lexical classes while *collectively* it is the sum of every layer's writers. (`runs/disassembly/core_basis_summary.json`.)

### Is the core a *generic grammar*? — the compact head is, the bulk is content (`core_grammar.py`)

The logit-lens (C) hint — directions decode to grammatical *classes*, not content — is the Chomsky hypothesis: **is
the core a content-free grammatical scaffold?** A grammar core would be both *generic* (corpus-invariant) and
*grammatical* (closed-class). Fitting `Ug` on three structurally distinct corpora (Shakespeare drama / a modern novel
/ Python source) and binning the shared directions by sharedness rank (GPT-2):

| sharedness rank | cross-corpus overlap vs chance | closed-class fraction vs random | verdict |
|---|---|---|---|
| **top-16** (most shared) | **0.44 vs 0.02** (22× chance) | **0.28 vs 0.00** (28× base) | **generic AND grammatical** — the grammar head |
| mid 16–64 | 0.23 vs 0.06 (4×) | 0.03 | fading |
| deep 64–K | 0.35 vs 0.28 (~1×) | 0.00 | content (corpus-specific) |
| content tail | 0.43 vs 0.36 (~1×) | 0.00 | content / rare-token (`inventoryQuantity`, `_Lear`) |

**The dissociation is clean: the top ~16 most-shared directions are *both* corpus-invariant (22× chance) *and*
closed-class/grammatical (28× the base rate); everything deeper is neither.** So there **is** a generic grammar — but
it is a **compact head of ~5–16 directions**, not the whole Θ(d) core. This is the "simpler-than-Chomsky" version: a
finite, content-free **categorial** scaffold (determiner-slot, punctuation-slot, verb-slot, pronoun-slot) — a learned
distributional POS basis, **not** the recursive/hierarchical syntax of generative UG (which, if present, lives in the
*composition* of these categories across positions — i.e. in the entangled bulk that pays the forge tax, not in the
static write-basis). And it is **learned, not innate** — a generic transformer grows the scaffold from data with no
syntactic prior. **Cross-architecture (RoPE):** Llama-3.2-1B reproduces the dissociation — top-16 directions
**0.41 vs 0.01 chance (41×) corpus-invariant** and **0.30 vs 0.01 (30×) closed-class** (with a clean reflexive-pronoun
direction `_himself _ourselves _herself _themselves` and a punctuation direction visible through the multilingual-byte
noise), everything deeper neither. The grammar head is **architecture-general**, not a GPT-2 quirk.
(`runs/disassembly/core_grammar_summary.json`.)

### Big-O of the core: Θ(model size), not a few templates

Putting the rank results together answers "how many templates / what scaling": the functional per-layer rank is
**Θ(d)** (rank-95 ≈ ⅓–⅖·d at *every* scale: 250/768, 398/1024, 476/1280, 509/2304 — a constant *fraction*, growing
with width), the shared union basis is **Θ(d)** (K ≈ ⅓–⅖·d), so the full per-layer composition is **~Θ(nL·d)
directions ≈ Θ(nL·d²) parameters — a constant fraction of the model** (the forge-tax floor). The "few templates"
(the dominant-variance head, PR 5–33) is only the *tip* and grows slowly (~O(nL)); the **grammar head is smaller
still (~O(1), ~16 directions)**. Net: low-rank truncation buys a **constant factor** (~⅓–⅔), **not** a big-O
reduction; and the irreducible core scales **with the model**, consistent with "the entangled core grows with
capability." The pylm sister track makes the consequence runnable — a flat-file **grammar** idiom decompiles the
compact scaffold (see [pylm track](PYLM_TRACK.md)), but adds little to the *token*-level decompilable fraction
(grammar is categorial; the n-gram modes already absorb it), and the un-decompiled ~50% is exactly the content that
is **neither n-gram nor relational fact** — the entangled composition, the forge tax restated.

### The core as a TENSOR NETWORK — MPS bond-dimension levels + the composition graph (`core_mps.py`)

The per-layer write rank is Θ(d) (above), but that is the *single-cut* rank. Treating the sequence of per-layer
updates (Δ₁ … Δ_nL) as a state over layer-"sites" and measuring the **bond dimension χ** across each layer cut (the
effective rank of the cross-correlation between layers ≤ L and > L — the q-orca / entanglement-tower view) gives a
very different number, and the cleanest CPU-simplification result yet:

| GPT-2 | layers | mean χ | χ range across cuts | max-bond mid-stack | χ / max-bond (mid) |
|---|---|---|---|---|---|
| gpt2 | 12 | **15.9** | 15–16 | 144 | 0.11 |
| gpt2-medium | 24 | **15.5** | 14–16 | 288 | 0.056 |
| gpt2-large | 36 | **16.4** | 14–18 | 432 | 0.037 |

**The bond dimension's *participation ratio* is ~16 and FLAT — independent of depth and width.** Adding layers to
either side of a cut does *not* raise the dominant cross-cut coupling: an **area-law** signature on the *coupling
spectrum*. **Honest correction (see the TT surrogate below):** χ≈16 here is the **participation ratio** of the
cross-cut correlation spectrum — a *coupling-concentration* measure dominated by a few high-variance directions — **not**
the rank you need to *reconstruct* the computation (exactly the core_rank gap: participation ratio ≈ 16, but rank-95 ≈
250). So "the layers communicate through a 16-dim channel" is true of the dominant *coupling*, **false** as a runnable
state dimension. The honest statement: the cross-layer coupling is heavy-tailed-low-rank (dominant head O(1), flat with
depth), the *runnable* bond is Θ(d) (next), and the simplification lever stays core_rank's per-layer low-rank.

#### …and the TT surrogate confirms it — the runnable bond is Θ(d), not 16 (`core_tt.py`)

Turning the bond into a **runnable, no-retrain surrogate** settles the CPU question. The residual carries two things —
the token EMBEDDING (full-rank, injected once) and the inter-layer COMPUTATION — so the faithful tensor-train protects
`resid_0` in full and forces only the *carried computation* through a χ-dim bond:
`resid_{L+1} ← resid_0 + B_L Bᵀ_L (resid_{L+1} − resid_0)`. ΔNLL vs the full model (generic next-token):

| ΔNLL | χ=16 | χ=64 | χ=256 | reading |
|---|---|---|---|---|
| **TT** (emb-protected bond) gpt2 / med / large | +1.37 / +2.94 / +3.11 | +0.77 / +1.94 / +2.41 | **+0.23 / +0.75 / +0.98** | the χ≈16 bond is **unusable**; runnable bond ~⅓·d |
| **per-layer** (core_rank) | +1.15 / +2.09 / +2.71 | +0.54 / +1.14 / +1.34 | **+0.10 / +0.27 / +0.36** | **better at every χ** — the real lever |
| **resid** (full-residual control) | +1.44 / +2.90 / +3.14 | ≈ TT | ≈ TT | emb-protection barely helps |

Three honest findings: (1) **at the area-law bond χ≈16, NLL is badly degraded** (+1.4 to +3.1) — the participation
ratio is not a runnable state size; (2) the **runnable no-retrain bond is ~⅓·d**, and **per-layer truncation beats the
running-bond TT at every χ** (the TT compounds error across cuts, so it gets *worse* with depth: χ=256 ΔNLL
+0.23→+0.75→+0.98 as layers go 12→24→36 — the *opposite* of "bigger = more compressible"); (3) protecting the
embedding barely helps (TT ≈ resid control), so the un-compressible content is the *computation*, not the tokens. **So
the only free CPU lever is core_rank's per-layer rank-⅓·d — a constant-factor (~3×) FLOP saving, lossless, no retrain —
NOT a χ≈16 tensor-train collapse.** The hard floor is the Θ(d) core (the forge tax): the irreducible composition is a
constant *fraction* of the model, so no fixed-size surrogate beats the model's own size without retraining. (A
*retrained* TT — learn the cores instead of PCA-projecting activations — could reach a lower bond; that is the
sae-forge feature-native direction, outside the no-retrain frame. `runs/disassembly/core_tt_summary.json`.)

#### …but with RETRAINING the floor falls ~30× — the core is *not* irreducible (`core_distill.py`)

The no-retrain results above all freeze the weights and use a *linear, fixed (PCA) basis* — so the Θ(d) floor they
hit says nothing about whether a *learned* low-rank representation exists. This builds one: a rank-r bottleneck on each
layer's update, **initialised from PCA** (step 0 = the no-retrain floor) and then **trained** (base model frozen, only
the bottleneck factors, ~300 steps). The floor collapses:

| GPT-2 rank (per-layer update) | no-retrain ΔNLL | **trained ΔNLL** |
|---|---|---|
| 8 (1% of d) | +1.78 | **+0.03** (≈ lossless) |
| 16 (2%) | +1.28 | −0.10 |
| 32 (4%) | +0.97 | −0.18 |
| 256 (33%, adaptation control) | +0.24 | −0.00 |

**A *trained* rank-8 update (1% of d) is lossless, vs the no-retrain floor of ~⅓·d (256) — a ~30× rank reduction.**
The rank-256 control trains to ΔNLL ≈ 0 (no meaningful domain-adaptation bonus), so the comparison to the full model
is fair: trained rank-8 genuinely matches it. **So the "Θ(d) entangled core" was an artifact of *freezing + linear
PCA*, not a property of the function** — exactly the right caution: the no-retrain experiments measure the wrong thing
for "is the core compressible." Scale nuance: gpt2-large needs more rank/steps (rank-8 recovers 76%, rank-64 88% in
250 steps; rank-512 *overfit* — an optimization artifact of a huge bottleneck under-trained), so the lossless rank
grows modestly with size and training budget — but it stays far below the no-retrain floor. **Honest scope:** this
compresses the per-layer *update* (the residual write), not the full internal attention+MLP FLOPs, and uses a small
eval (20 chunks) + cheap training; a rigorous version (more steps, general eval, factoring the output projection for
real FLOP savings) is the natural follow-up. But the qualitative claim is settled: **detangling/compressing the core
is tractable with learning — "irreducible" is falsified; only the frozen-linear route was blocked.**
(`runs/disassembly/core_distill_summary.json`.)

#### …but compression is legibility-NEUTRAL — small ≠ legible for free (`compress_legibility.py`)

The feature-native endgame (train an N× smaller model → extract clean features / knowledge / circuits → the small
Python program → runtime explainability) hinges on whether compression *aligns* with interpretability. It does not, for
NLL-only compression: logit-lens-scoring the rank-32 directions the *trained* bottleneck writes into vs the untrained
PCA directions, both interpretability proxies are **unchanged** — peak-z 5.00 → 4.96, closed-class fraction 0.10 →
0.09 (marginally *lower*, within noise). So distillation buys the **size** win (#142) but **not** a feature-cleanliness
win: compression and per-feature monosemanticity are **decoupled** under an NLL objective. **Consequence for the
pipeline:** to get *both* small *and* legible you must train with an explicit **legibility / sparsity term** (the
sae-forge feature-native objective) — pure NLL-distillation packs the same entanglement into fewer dimensions. The
enabling facts are all in hand (smaller = more decompilable; compression is tractable; feature/knowledge/circuit
extractors + pylm + runtime idiom-traces exist), but the missing ingredient is the legibility objective, not more
compression. (`runs/disassembly/compress_legibility_summary.json`.)

#### …and a naive legibility term can't fix it on a tight bottleneck — the small / legible / complete trade-off triangle (`feature_native_distill.py`)

Adding the obvious legibility objective — an **L1 sparsity penalty on the bottleneck codes** (the sae-style term) —
does *not* rescue legibility on a compressive rank-32 bottleneck: sweeping λ ∈ {0, 0.01, 0.05, 0.2}, peak-z stays ~4.95
and the code-active fraction stays ~0.50 (L1 *doesn't sparsify*), while NLL only rises (+0.19 at λ=0.2). The reason is
structural: a *tight* (rank < d) bottleneck has **no room** to be sparse — every direction is genuinely used, so L1
shrinks them all uniformly instead of zeroing any. **Legibility needs *overcompleteness*** — an SAE has *more* features
than dimensions and *then* sparsifies, which is the room to specialize — and overcompleteness is the **opposite of
compression**. So the user's "train smaller *and* extract clean features" runs into the forge tax as a **trade-off
triangle — small / legible / complete, pick ~two**:

| corner | small | legible | complete | what it is |
|---|---|---|---|---|
| **pylm** | ✓ (49 LOC + flat) | ✓ (every token = a named idiom) | ✗ (~half) | the decompiled program |
| **low-rank distill (#142)** | ✓ (~30×) | ✗ (legibility-neutral) | ✓ (lossless) | compression |
| **SAE / feature-native** | ✗ (overcomplete) | ✓ (sparse monosemantic) | ✓ | the legible basis |

The forge tax *is* this triangle: monosemanticity (overcomplete + sparse) and compactness (tight low-rank) pull in
opposite directions, and the full-fidelity composition resists both unless you pay one of them. The whole pipeline
(train smaller → extract features/knowledge/circuits → the small Python program → runtime explainability) is buildable,
but it cannot be small *and* legible *and* complete at once — you pick the corner per use-case (pylm for an auditable
runtime; distillation for a cheap one; an overcomplete feature-native model for an interpretable one).
(`runs/disassembly/feature_native_distill_summary.json`.)

**The composition graph** (mean-squared canonical correlation between layer-pair write coords) is densely coupled —
every pair far above chance (0.34–0.56 vs 0.009) — with **adjacent-layer coupling > distant** (0.49–0.53 vs 0.34–0.37)
and the strongest edges clustered at the **output-assembly end** (late-layer pairs) plus the embedding edge `0→1`. A
banded DAG over layers: local-dominant but globally coupled, condensing through the low bond.

**Ontology** (each layer's top write directions typed grammar / content by logit-lens): grammar concentrates at the
**boundaries** — the embedding end (detokenise into grammatical classes) and, most strongly, the **output end** (write
the grammatical next-token classes) — and is sparse in the **middle** (content / compute). The U-shape is cleanest in
gpt2-small (`4 5 1 1 2 2 1 2 3 7 5 8`); in larger models the output-end concentration dominates and the embedding-end
grammar is partly masked by rare-/byte-token directions. So the **typed graph**: grammatical scaffold at the rim,
entangled content composition in the core — the ontology of "what the core is made of," per layer.
(`runs/disassembly/core_mps_summary.json`.)

### Where the *recursive* syntax lives — in the composition, not the basis (`recursive_syntax.py`)

The static basis holds only the **categorial** grammar (POS-class directions). The open question: is the genuinely
Chomskyan part — hierarchy, long-range dependency, recursion — in that basis, or in the **composition**? The cleanest
test is subject–verb agreement across intervening attractors (Linzen et al.): *"The key near the cabinets **is/are**"*
— the verb must agree with the head (`key`, singular), **not** the nearest noun (`cabinets`, plural). A flat local
program follows the nearest noun; true hierarchical syntax tracks the head across depth. Three read-only measurements:

| attractor depth | MODEL (full) | MODEL, attention-ablated | FLAT pylm (head / attractor) |
|---|---|---|---|
| 0 (local) | **100%** (Δ+5.0) | 98% (Δ+2.1) — number is on the last token | 53% / 47% (no attractor yet) |
| 1 | **100%** (Δ+2.7) | **0%** (Δ−2.2) — follows the attractor | **0% / 100%** |
| 2 | **97%** (Δ+2.4) | 1% (Δ−2.2) | 0% / 100% |
| 3 | **100%** (Δ+2.0) | 0% (Δ−2.3) | 0% / 100% |

*(GPT-2 small; logit-diff Δ = lp(correct) − lp(wrong).)* Three findings, all confirming the hypothesis:
- **The model does true hierarchical agreement** — ~100% across depth, resisting the attractor; and the logit-diff
  **degrades with distance** (+5.0 → +2.0), the bounded-depth signature of a single forward pass (TC⁰: recursion isn't
  free — it weakens with depth, and would need the decode loop / scratchpad to go deeper).
- **The flat decompilation cannot** — pylm (n-gram + induction + categorial grammar) is **0% head / 100% attractor**
  at depth ≥ 1: it always follows the nearest noun. The hierarchical dependency is **not** in the flat program (nor in
  the categorial grammar head it decompiles). *(Caveat: pylm surfaces a verb token in only n≈6–12 of the depth ≥ 1
  stimuli — out-of-distribution sentences — but of those it is 100% attractor-driven.)*
- **Ablating attention collapses it to the attractor** — depth-0 agreement survives (the head's number is on the last
  token itself), but at depth ≥ 1 the model flips to following the nearest noun (0%, logit-diff negative). So
  **attention carries the head's number across the attractors** — the dependency lives in the attention **composition**,
  not the static grammar head. **Cross-architecture:** Llama-3.2-1B (RoPE) replicates it crisply (full 100/100/100/94%
  across depth; attention-ablated 95% at depth 0 → **3–4%** at depth ≥ 1); gpt2-large likewise (full ~91–100%,
  attention-ablated → chance). MLP-ablation destroys the verb readout entirely (chance) in all three — the clean
  dissociation is attention.

This closes the grammar loop and ties it to the forge tax: **the categorial grammar is in the static write-basis
(decompilable, #133); the recursive/hierarchical syntax is in the composition (the entangled core the forge tax
measures).** "Simpler-than-Chomsky" in the basis, Chomskyan in the composition — demonstrated, not assumed.
(`runs/disassembly/recursive_syntax_summary.json`.)

**Which heads carry it — the number-mover circuit (`agreement_circuit.py`).** Per-head ablation over the depth-≥1
stimuli localizes a **small, distinct circuit** (GPT-2, base agreement logit-diff +2.55): the load-bearing heads are
**7.4** (ablation drop +0.85), **10.9** (+0.51), **8.5** (+0.50), **2.10** (+0.50), concentrated in **mid-to-late
layers**. The late movers show the clean signature — they attend **from the verb to the HEAD noun**, not the nearest
attractor (**10.9: verb→head 0.31 vs verb→attractor 0.07**; 10.5: 0.35 vs 0.06; 11.10: 0.29 vs 0.09) — literally
*moving the head's number to the verb slot*. There is also an early **attractor-tracker** (0.3 attends 97% to the last
token; 2.10 leans to the attractor) — the local pathway the agreement heads override. Crucially, **none of the
number-movers are induction / prev-token / duplicate heads** (all `UNNAMED`): the agreement circuit is a **distinct
operator class**, a new catalog entry — the composition that does the recursive work, made of heads the copy-idiom
catalog doesn't name. **Cross-scale:** gpt2-large reproduces it — the movers are again UNNAMED mid-to-late heads
(24.3 with verb→head 0.42 vs 0.03; 32.5, 25.4, 15.10, 14.0 at the same ~60–90% relative depth).
(`runs/disassembly/agreement_circuit_summary.json`.)

### The recursion depth limit — distance is interference-bounded, nesting breaks sooner, but the ceiling does *not* grow with layers (`recursion_depth.py`)

Is recursion depth-limited by the number of layers? Theory says a single forward pass is a fixed-depth circuit (TC⁰),
so **genuine nesting** (each embedded clause resolved before the outer one — stack-like) is layer-bounded, while mere
**distance** (the head noun is far away) is **not** — attention jumps directly across any distance in one layer.
Testing both — PP-distance (*"the key near the dogs near the tables is"*) vs center-embedding (*"the key that the dogs
that the cat sees chase is"*), scoring the outer verb's agreement with the head across depth 0–5:

| nesting ceiling (acc ≥ 75%) | gpt2 12L | Llama 16L | gpt2-medium 24L | gpt2-large 36L |
|---|---|---|---|---|
| **distance** (PP modifiers) | 5+ | 5+ | 5+ | 5+ (gradual, never crosses 0) |
| **nesting** (center-embedding) | **5** | **3** | **3** | **2** (logit-diff crosses 0 at depth 5) |

- **Distance is interference-bounded, not layer-bounded** — accuracy stays ≥ 75% through depth 5 in every model, the
  logit-diff decays gradually (+5 → +1) and never flips: more attractors just add noise to a one-hop lookup.
- **Nesting breaks sooner** — center-embedding degrades faster (logit-diff halves by depth 2; flips negative for
  gpt2-large), consistent with genuine recursion being harder than distance.
- **…but the nesting ceiling does *not* grow with layers — it *shrinks* with model size** (12L → 5, 16L → 3, 24L → 3,
  36L → 2). The *opposite* of "more layers → deeper recursion."

**Honest conclusion: the TC⁰ layer-bound is real in principle but is *not* the binding constraint here.** Every model
has 12–36 layers — far more than the depth ~2–3 where center-embedding breaks — so they fail for **distributional /
interference** reasons, not layer exhaustion: deep center-embedding is vanishingly rare in training (and unparseable
for humans past ~2), each level adds more opposite-number signal (nouns *and* inner verbs), and a bigger/better model
commits *harder* to the dominant local-natural parse → it breaks *earlier*. So the answer to "is recursion
layer-limited?" is: **yes in principle (one pass is TC⁰), but empirically these models cap at center-embedding depth
~2–3 — far below any layer limit — and layer count is not the active constraint; training distribution + interference
are.** The genuine layer-bound would bind only for *in-distribution* recursion pushed past the data's depth, which
natural language essentially never does (the human depth-2 center-embedding limit is itself this phenomenon); to go
deeper a model uses the **decode loop / chain-of-thought** (TC⁰ per step, Turing-complete across steps), trading
within-pass depth for sequence length. *(Caveat: nesting adds more+stronger attractors per level than distance, so
"nesting harder" is partly more interference, not purely stack depth; the robust, size-monotone result is the ceiling
shrinking with capability.)* (`runs/disassembly/recursion_depth_summary.json`.)

### Does the decode loop buy back the depth? — chain-of-thought, partially (`cot_depth.py`)

Theory's escape hatch: a single pass is TC⁰, but the **decode loop** is Turing-complete *across* steps, so chain-of-
thought should trade within-pass depth for sequence length and lift the ceiling. Testing it on the canonical
bounded-recursion task — **Dyck bracket matching** at nesting depth d (balanced vs one mismatched closer, chance 50%) —
with an instruction-tuned model that can reason (Qwen2.5-1.5B-Instruct), **direct answer** vs **CoT**:

| depth | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| **direct** | 85% | 65% | 65% | 65% | 55% | **50%** (chance) |
| **CoT** | 50% | **85%** | **74%** | 50% | **71%** | 43% |

- **Direct collapses cleanly to chance by depth 12** — the single-forward-pass recursion limit, demonstrated on a clean
  recursion task (no distributional confound this time: brackets are uniform across depth).
- **CoT beats direct at moderate depth (4/6/10)** — the loop *does* buy back depth, as the theory predicts.
- **But it is noisy and reliability-bounded at 1.5B**: CoT trails direct at depths 2/8/12, and its *coverage* drops with
  depth (a parseable "Answer:" emerges in only 14/20 cases at depth 12 — the small model rambles past the token budget
  without concluding). So the escape hatch is real but its quality is gated by the model's own reasoning reliability;
  a larger / reasoning-tuned model would show it cleaner (as the CoT literature does).

Net: the within-pass depth limit is real and clean (direct → chance), and the decode loop is a genuine but
reliability-bounded escape — **depth is traded for tokens, only as well as the model can actually reason step by step.**
(`runs/disassembly/cot_depth_summary.json`.)

## Execution model: an interpreter over the op-graph ("ResidualVM")

The recompile-KL harness is most useful not as a one-shot metric but as a **steppable interpreter** over the
extracted op-graph: run the named ops on the residual bus, in selectable *fidelity modes*, and watch
reconstruction-KL — which makes decompilation **debuggable**, not just measurable. (Execution-model framing
contributed by Grok; integrated + corrected here.)

**The honest mechanics — dataflow per pass, "VM" only at the loop.** It is tempting to cast this as a von
Neumann machine (fetch–decode–execute over a stored program). That is the wrong abstraction for a *single
forward pass*: there is no program counter, no instruction fetched per cycle, no data-dependent control flow,
and the weights are never modified by the data path. One pass is a **fixed-depth dataflow circuit** (closer to
an ASIC / systolic array; Merrill's TC⁰) — the "ISA" (the op-catalog) is **hardwired and applied in
parallel**, a *description* of fixed functional units, not a runtime dispatch. The stored-program /
von-Neumann character appears **only at the autoregressive loop**: the residual stream + KV-cache as a
read/write tape, the decode step as the clock, chain-of-thought as working memory (see the
`llm-as-accreting-vm` framing). So the interpreter executes a **fixed circuit per token** and is a **VM at the
generation level** — not a stored-program CPU per layer.

Formal grounding (why the DAG/loop split is load-bearing, not decorative): the object is a **clocked
sequential machine** = combinational core (the fixed DAG) + state register (KV-cache + the growing token
sequence) + clock (the decode step). The **DAG alone is weak** — one bounded-depth pass sits in **TC⁰**
(Merrill–Sabharwal); the **loop supplies the power** — transformer + decoding + an unbounded scratchpad is
Turing-complete (Pérez et al., CoT-expressivity). Two precisions: (a) the recursion goes through a **discrete
token bottleneck** — high-dim state is sampled to a *token* and re-embedded, so depth-per-step is bounded but
steps are unbounded (hard problems buy back missing within-pass depth with *longer* CoT); (b) the program is
**fixed, not self-modifying** — the same DAG every step, only the data changes (microcode/ASIC-like; Turing
power lives entirely in the outer tape+clock). The decompilation payoff: **the loop is a clean recurrence —
the hard part is reading the DAG**, and `reconstruction_coverage` is exactly the measure of how much of that
fixed high-dim DAG reduces to a compact symbolic program over explicit operands.

What the frame *does* buy, mapped to checked-in results:

| VM concept | what it actually is here | grounded in |
|---|---|---|
| memory hierarchy (registers/L1 vs main memory) | entanglement-tower levels: low-χ monosemantic core (addressable, cov95-high) vs high-χ entangled remainder | `mps_tower_*`, serve-tower cov95-saturation-vs-capability-cliff |
| fidelity modes: full / preserve-hybrid / decompiled | exact host / verbatim-pin ~6–12% of atoms + forge the rest / run only the recompiled op-graph | `preserve_hybrid_tiny.py`, sae-forge `NativeModel` |
| ISA vs model-specific microcode | the op-catalog (idioms) is invariant across architectures **and** languages; the **sink/plumbing policy is model-specific** | the 4-model + multilingual results |
| associative memory bank | MLP neurons as content-addressable key→value stores | work-item 2 (the MLP gap) |
| the recompiler / JIT | sae-forge projects the kept ops into a runnable module | reconstruction-coverage metric above |

**What it adds beyond the metric (the real new lever): an interactive debugger.** Step layer-by-layer;
breakpoint when a named idiom fires; inspect the low-χ "registers" (SAE latents); **ablate / preserve / swap a
single op and watch reconstruction-KL move live**. That turns reconstruction-coverage from a number into a
tool for *localizing* where decompilation fails — i.e. it operationalizes milestones 1+4.

**The hard constraint the frame must not hide.** The entangled core is **preserve-or-pay**, not "approximate
main memory": by the tower's no-go, fidelity modes move you *along* the interpretability↔capability frontier,
never off it. An interpreter that "approximates the core cheaply" is just choosing a point on that frontier
(and paying the capability cliff) — the ceiling is the forge tax, restated.

Demarcation: adopt the execution layer, the memory-hierarchy mapping, the fidelity modes, and the ISA-vs-
microcode framing; **drop the von-Neumann mechanics** (no per-layer instruction fetch) in favor of
dataflow-circuit-per-pass + VM-at-the-loop. `COMPOSE` / `PRESERVE` / `TOWER_TRUNCATE` are not opcodes the model
runs — they are *extracted descriptions* (the DAG) or *execution modes* (knobs on the interpreter).

## Milestones (each a PR, gated on the prior)

1. **Recompile-KL harness, built as the interpreter** (§Execution model) — **DONE (v1, GPT-2): `scripts/disassembly/residual_vm.py`** (see First result). v1 recompiles by *keeping ops at full fidelity and mean-ablating the complement*; v2 = feature-basis recompilation via sae-forge `NativeModel` (the ceiling test, milestone 4). **Now also a reusable `ResidualVM` class** in that file — the consolidated, arch-generic intervention layer the ~20 disassembly scripts each re-implemented (load + corpus-mean + `ablate_heads` / `ablate_mlps` / `patch_mlp` / `patch_attn` / `trace` / `attribution` as **composable context managers**); `python residual_vm.py --demo` reproduces the induction reconstruction coverage through it. (The biggest tooling gap identified after the cross-model program: a single steppable debugger so an experiment is ~10 lines and the load/hook/merge bugs live in one place.) **Feature-level interventions are now in too** — `vm.set_feature(layer, feat, target)` / `ablate_feature` edit individual **SAE latents** (GPT-2 jbloom / Gemma Scope) directly in the residual; `python residual_vm.py --feat-demo` shows that ablating the subject's dominant feature (a France/geography latent promoting "Alps") at L5 of " France" drops `logp(" Paris")` by ~9 nats — feature-native surgery, the rung from head/MLP-level interventions toward editing the *catalogued features* themselves.
2. **Composition-DAG extractor** — weight-space edge scorer + path-patch gate; auto-recover induction + IOI;
   report new sub-DAGs. (Generalizes `path_patch_induction.py` / `composition_graph.py`.) **DONE (GPT-2):
   `scripts/disassembly/composition_dag.py`** — static composition predicts dynamic writer specificity
   (ρ=+0.37); the canonical induction K-chain and IOI Q-chain are auto-recovered AND live, imposters/random
   rejected (0% FP); 22 new live edges surfaced (see Composition-DAG section).
3. **MLP ops** — neuron key→value catalog + named MLP idioms; add to the DAG + the recompile. **DONE (GPT-2):
   `scripts/disassembly/mlp_ops.py`** — the recompile now charges for MLPs (M1/bridge kept heads only); MLPs are
   load-bearing, concentrated in L0 (the detokenizer); head↔MLP composition edges are weight-legible (see MLP
   ops section).
4. **The ceiling test** — reconstruction-coverage plateau vs the tower's entangled core, same host; the
   unifying claim stands or falls. **First result DONE (v2, tiny GPT): `scripts/cov95_forge_tax/ceiling_test.py`** — content/factorability axes decouple; the capability plateau is GPU-scale-gated (see Ceiling test section).
5. **Cross-model** — repeat the ceiling on Gemma-2 / Llama-3 / Qwen-2.5 (idea i) to test whether the
   decompilable fraction is architecture-invariant like the mechanisms are. **DONE (op-selection ceiling, 5
   models incl. the gpt2-medium control): `scripts/gemma/cross_model_ceiling.py`** — the named circuit beats
   random everywhere (mechanisms invariant) but the decompilable *fraction* is **not** invariant; the gpt2-medium
   control **disentangles** it — the high fraction tracks the **absolute-position family, not scale** (see
   Cross-model ceiling section). The *forge-basis* ceiling stays SAE/GPU-gated for non-GPT-2.
6. **Executable decompilation & knowledge, across six models** — **DONE.** The reconstruction-coverage idea (M1)
   run as a *sufficiency* test over GPT-2 ×3 + Gemma/Llama/Qwen: **no small head-set is sufficient** for induction
   (≤30% even under resample-ablation; seed-stable ±0–1%) and even IOI's 26-head circuit isn't, in isolation
   ([reconstruction](circuits/reconstruction.md), [substrate](circuits/induction_substrate.md)). Knowledge ported
   cross-model: ROME causal tracing recovers an architecture-invariant early-MLP-store → late-attention-readout
   flow, and the store is **editable by activation patch** (100% fact-transplant in 5/6, generalizing across
   relations; Gemma's storage is distributed) ([tracing](circuits/causal_tracing.md),
   [transplant](circuits/fact_patching.md)). **Nuance vs M5:** the *forge-basis decompilable fraction* (M5) is
   absolute-position-**family**-specific, but the *head-ablation reconstruction coverage / circuit distributedness*
   (M6) tracks **scale** — different metrics, both true ([scaling synthesis](scaling.md)): small models are
   unusually localized, so a faithful decompilation is the **budget curve**, not a tiny op-graph.

## First result (milestone 1, GPT-2)

`residual_vm.py` on GPT-2 (Shakespeare; floor = KL(host ‖ all-144-heads-mean-ablated) = 1.92): keep a head-set
at full fidelity, mean-ablate the complement, sweep the budget B by marginal ablation importance vs a random
control + the named induction circuit.

- **Attention is distributed / redundant.** The coverage curve rises *gradually* — **128 of 144 heads are
  needed for 90% coverage**; no small set reconstructs the forward pass. The single most-important head in
  isolation is even net-negative (keep 1, ablate 143 → slightly below the all-ablated floor): heads interact.
  (Same program-wide redundancy seen in the circuit work.)
- **But the named catalog is coverage-efficient.** Top-B beats random-B at every budget except B=1 (largest
  gap mid-range, Δ≈+0.31 at B=24–32), and the **5-head induction circuit (prev-token 4.11 + induction
  5.0/5.5/6.9/7.11) reconstructs +0.164 coverage vs +0.032 for a random 5-head set — ~5×**. The disassembly's
  named ops are disproportionately load-bearing (importance ranks: 7.11 #2, 4.11 #8). The op-catalog buys real
  reconstruction.
- **Scope (honest).** v1 keeps the kept heads at *full fidelity* (exact weights), so coverage → 1 as B → all
  *by construction* — it measures **which/how-many ops matter** (op-selection coverage), **not** the
  entangled-core ceiling. The plateau-below-1 prediction (forge tax as decompilation ceiling) needs the
  **feature-basis recompilation** (sae-forge `NativeModel`, milestone 4), where kept ops must be expressed in a
  clean basis and composition bottlenecks. Milestone 1 delivers the interpreter + metric + the op-selection
  result; the ceiling test is the next build. `runs/disassembly/residual_vm_gpt2_summary.json`.

## Composition-DAG extractor (milestone 2) — first result (GPT-2)

`composition_dag.py` unifies the two precursors into one extractor that reads the **call graph**, not one
idiom. `composition_graph.py` gave the static adjacency (Elhage Q/K-composition on raw weights, mean-write
removed) but only validated the single prev-token→induction K-edge; `path_patch_induction.py` gave the dynamic
gate but measured an *induction-specific* collapse, so it could only confirm induction. M2 generalizes both:
score the full K/Q composition DAG over all causal head pairs, then gate the strongest edges with an
**idiom-agnostic** dynamic metric — the mean total-variation change in the reader's attention pattern when the
writer is removed from that port (**ΔTV**), defined for *any* reader. GPT-2, Shakespeare, weights + two forward
passes; 170 gated edges (105 K / 65 Q); attention recompute is exact (max|Δ| = 9.85e-7 vs the model).

The one methodological move that makes this work: raw ΔTV **grows with reader depth/magnitude**, so a global
random null is confounded (early-layer write-hubs dominate). The fix is a **reader-matched null** — for each
reader, compare the real writer against *random causal writers into the same reader head* (the path-patching
null that isolates writer *specificity* from reader depth). An edge is "live" if its ΔTV beats its reader's
matched 2σ null; **specificity = ΔTV − matched null**.

- **Static composition predicts dynamic liveness across the whole graph.** The headline (cleaner, depth-
  unconfounded) metric is **Spearman(static, reader-matched specificity) = +0.37**; the raw
  Spearman(static, ΔTV) = +0.52 is higher but inflated by the depth co-scaling both quantities share, so the
  specificity correlation is the one to trust. Either way the weight-space score is a graph-wide predictor of
  which writes actually shape which reads — the broad version of `path_patch_induction`'s induction-only ρ,
  not a single-idiom result.
- **The induction K-chain is auto-recovered AND live.** Static prev-token→induction K-composition 0.069 vs
  causal baseline 0.042 (1.6×) vs random 0.039. Dynamically, the **canonical prev-token head 4.11 → inductors
  is 4/5 live** (4.11→5.5/5.0/7.11/6.9 clear their matched nulls; 4.11→5.1 marginal), the strong-edge median
  Δinduction is +0.015, and the **top edge collapses 56% of induction attention** under key-path patching (the
  original strong readout, retained for induction edges).
- **The IOI Q-chain is auto-recovered.** Static S-inhibition→name-mover Q-composition 0.065 vs causal 0.042
  (1.5×); the recovered chain is duplicate-token (3.0) → S-inhibition (8.10/8.3/10.0) → name-mover
  (9.9/10.0/11.2). Dynamically **5/11 S-inhib→name-mover Q-edges are live** — the real S-inhibition heads
  8.10/8.3 reshape the name-movers' query attention; the spurious 6.7 (a Q-composition false-positive) does
  **not** (negative specificity).
- **The gate is selective — it rejects imposters.** Across the named cross-product, **43% of edges are live vs
  0% random false-positives**. The selectivity is the point: the *non-canonical* high-prev-token writers
  (2.2/3.2/3.7, which have high prev-token attention but aren't the prev-token head) → inductors are mostly
  **dead**, and the spurious S-inhib head is dead. The extractor keeps the real sub-circuit and discards the
  cross-product noise.
- **22 new live edges surfaced** above their reader-matched 2σ nulls, not in induction/IOI — dominated by
  early-layer **write-hubs** feeding many readers (consistent with positional / duplicate-token hubs). The
  highest-specificity candidates:

  | port | edge | ΔTV | reader null | specificity |
  |---|---|---|---|---|
  | K | 0.9→2.9 | 0.444 | 0.035 | **+0.410** |
  | K | 0.9→2.5 | 0.449 | 0.044 | **+0.404** |
  | K | 1.8→10.9 | 0.293 | 0.007 | **+0.287** |
  | K | 1.8→9.3 | 0.333 | 0.088 | **+0.245** |
  | Q | 0.9→1.3 | 0.259 | 0.018 | **+0.240** |
  | K | 1.8→3.2 | 0.362 | 0.133 | **+0.230** |
  | Q | 1.10→2.11 | 0.270 | 0.055 | **+0.215** |
  | K | 1.8→10.5 | 0.198 | 0.021 | **+0.177** |

  Two heads recur as hubs: **0.9** (a layer-0 writer dominating several layer-1/2 readers) and **1.8** (a
  layer-1 writer reaching *long-range* into late-layer keys, 9.3/10.5/10.9). These are *candidate* sub-DAGs,
  **not** validated circuits — behavioral labeling (and targeted single-edge path-patching, as for induction)
  is the obvious follow-up; the full ranked list is in the summary JSON's `novel_live_edges`.

**Feeds the recompile harness (milestones 1/4) — now tested, see the next section.** The live edges this
extractor confirms are exactly the *keepable ops* the ResidualVM reconstruction-coverage interpreter
(`residual_vm.py`) should retain: M1 selects heads by marginal ablation importance, but the DAG supplies the
**structured op-set** (which writer→reader wires carry the computation), so a DAG-guided keep-set is the
natural upgrade to M1's flat head-budget — and the recompile-KL then verifies the extracted sub-DAG *executes*.
The M1↔M2 bridge below confirms this. Completing the DAG with V-composition + MLP nodes (milestone 3) is what
makes that op-set whole.

**Scope (honest).** (a) The dynamic gate runs on natural text, so the IOI Q-edges are confirmed by generic
attention reshaping (ΔTV), not the IOI-task logit-difference — that task-specific causal validation already
lives in `ioi_causal.py`; M2's contribution is the *unified static→dynamic* recovery. (b) ΔTV measures whether
an edge *reshapes the reader's attention pattern* (Q/K composition); V-composition (writing values without
moving attention) needs a different readout, not done here. (c) MLP nodes are absent — they are milestone 3.
**Compute:** GPT-2, CPU-feasible — weights + two forward passes over the corpus (one for behavioural labels,
one for the path-patch gate); the full run is **~70 s wall-clock and ~5 GB RAM on CPU** (no GPU), so it scales
to any HF model the box can hold a forward pass of. `runs/disassembly/composition_dag_summary.json` (re-run the
script to regenerate the figure).

### Validating the new write-hub edges (follow-up to M2)

`validate_new_edges.py` takes the 22 new live edges to the next rung: a **targeted single-edge path-patch with
a behavioural readout** (the induction-style strong test, generalized). For each edge A→B it surgically removes
A's output from B's port, recomputes B's attention, and measures the collapse of B's *named* components —
{prev-token, duplicate, induction, sink} — against a **reader-matched random-writer null**. A pattern collapse
beyond the null *names* the edge's function; ΔTV with no named collapse is real-but-unlabeled shaping.

- **The write-hubs are early SINK heads broadcasting a positional signal.** All three hubs (0.11, 0.9, 1.8) and
  the minor ones (1.3, 1.7, 1.9) self-label **sink** (their own attention parks on position-0). **13/22 edges
  resolve to a named function — 9 prev-token, 3 sink, 1 duplicate.** Removing these hubs from a downstream key
  collapses that head's **prev-token attention**: e.g. `0.9→2.5` −56%, `1.8→9.3 / 1.8→10.9` −45%, and — the
  headline — `1.3→4.11` −11% and `1.8→4.11` −8% into the *canonical prev-token head itself*.

  | edge | writer→reader | pattern | rel-collapse (beats null) |
  |---|---|---|---|
  | K | 0.9→2.5 | prev-token | −56% |
  | K | 1.8→9.3 | prev-token | −45% |
  | K | 1.8→10.9 | prev-token | −45% |
  | Q | 0.9→1.3 | sink | −42% |
  | Q | 0.9→1.10 | prev-token | −23% |
  | Q | 0.9→1.2 | prev-token | −23% |
  | K | 1.8→11.8 | prev-token | −14% |
  | K | **1.3→4.11** | **prev-token** | **−11%** |
  | K | 0.11→1.1 | prev-token | −10% |
  | K | **1.8→4.11** | **prev-token** | **−8%** |
- **The prev-token mechanism is not self-contained in 4.11.** It *reads a positional signal piped in from early
  sink heads*; remove that input and 4.11's previous-token addressing degrades. This edge-resolves the
  disassembly's **position/structure register** and reframes the **sink**: a sink head is a no-op in *where it
  reads* (parks on pos-0) but its **OV-write is a load-bearing positional broadcast** — attention-pattern and
  write-content are decoupled. *Hypothesis* (consistent with the sink-ablation result — GPT-2 is the only
  family member that depends on its sink, position-independently = the absolute-positional-embedding signature):
  these hubs propagate GPT-2's learned absolute positions; the exact encoding pathway is left open.
- **Honest scope.** (a) The reader-matched null *includes other early/sink heads* that also carry positional
  signal, so the test is **conservative** — several of the 9 "unlabeled" edges still collapse prev-token (e.g.
  `0.9→2.9` −55%) but don't beat that positional null (those readers' prev-token mass is fragile to *any* key
  perturbation), so they are real shaping not attributable to one writer. (b) The readout is attention-pattern
  collapse (like M2's ΔTV), evidence of a positional-broadcast role, not a task-level loss metric.

  *Next* (sharpen the mechanism): (a) **task-level readout** — re-do the edge patch as a forward-pass
  intervention and read next-token KL / induction-NLL, turning "reshapes the pattern" into "changes the
  output"; (b) **the positional-embedding test** — re-run the patch with `wpe` zeroed/randomized: if these
  hubs propagate GPT-2's *absolute* positions, their prev-token collapse should vanish without `wpe`; (c)
  **cross-model prediction** — RoPE models (Gemma-2 / Llama-3 / Qwen-2.5) do **not** depend on their sink
  (sink-ablation result), so they should show **no** such absolute-positional-broadcast circuit — the
  prev-token signal is carried by RoPE in the QK directly, not piped from a sink head. A clean falsifier of the
  "GPT-2 absolute-positions" hypothesis. `runs/disassembly/validate_new_edges_summary.json` (~40 s on CPU;
  re-run to regenerate the figure).

### Cross-model: the positional broadcast is GPT-2-specific (forward-pointer (c), done)

`scripts/gemma/cross_model_positional.py` runs the cross-model test. A *key-only* causal path-patch that
respects each model's RoPE is non-trivial to do faithfully across architectures, so the test goes to the
**representational** question the broadcast hypothesis really turns on: **is the prev-token head's position
carried in the key *content* (absolute, must be written into the residual) or in the QK *rotation* (RoPE,
applied at attention time)?** For each model it finds the top prev-token head, captures its **pre-rotation**
keys over the corpus, and decomposes the key variance into the fraction explained by absolute **position** vs by
**token** identity. The prediction: GPT-2 position-dominated (its keys must encode *where* — exactly what the
sink heads broadcast), the RoPE models token-dominated (keys encode *what*; position is the rotation).

*Method:* a between-group variance decomposition on the head's key vectors. `position_fraction` = the
between-position-index variance (variance of the mean key at each absolute position) over the total key-
covariance trace; `token_fraction` = the same over token identity (mean key per frequent token). Both are
fractions of the same total, so `pos/tok` is a clean within-head ratio comparable across architectures.

| model | pos. encoding | prev-token head | key var: **position** | key var: token | pos/tok |
|---|---|---|---|---|---|
| **GPT-2** | absolute (`wpe`) | 4.11 | **59%** | 18% | **3.3** |
| Gemma-2-2B | RoPE | 21.7 | 7% | 23% | 0.30 |
| Llama-3.2-1B | RoPE | 0.2 | 2% | 64% | 0.04 |
| Qwen-2.5-1.5B | RoPE | 13.4 | 9% | 25% | 0.35 |

**CONFIRMED — a clean 1-vs-3 split (~10×).** Only GPT-2's prev-token key is position-dominated; every RoPE
model's is token-dominated. This is the cross-model *explanation* that ties the positional thread together: GPT-2
encodes absolute position as key content → its prev-token head must *read that content* → it depends on the early
sink heads that **broadcast** it (the `validate_new_edges` collapse) → and GPT-2 is the only family member that
**depends on its sink** (the sink-ablation result, position-independently = the absolute-positions signature). The
RoPE models need none of this — position rides in the rotation — so they have no sink dependence and no
positional-broadcast circuit. The figure (`cross_model_positional.png`, regenerable) shows the bars flip:
position > token only for GPT-2. Note **Llama-3.2-1B's prev-token head is in *layer 0*** (head 0.2) and is the
most token-pure of all (pos/tok 0.04) — it reads the raw token embedding directly and leans entirely on RoPE for
position, the cleanest case of the RoPE pattern.

**Scope.** This is the *representational* confirmation (the key **is** position-encoded only in GPT-2),
corroborating the *causal* GPT-2 result; a faithful key-only causal path-patch across RoPE models
(forward-pointer (a)) is the heavier next step — **now done** (next subsection). *Also next:* re-run on an
oracle-supervised host (#19/#20) — does training a more legible model shift the prev-token key's
position-vs-token content, i.e. does supervision touch the positional machinery or only the feature substrate?
~33 s for all four models. `runs/gemma/cross_model_positional_summary.json`.

### Cross-model: the *causal* key-only path-patch (the heavier confirmation)

`key_patch_cross_model.py` does the causal complement #26 deferred. The representational test said GPT-2's
prev-token key carries position-content and RoPE's doesn't; the causal test asks: **does removing an upstream
head's *key content* collapse the prev-token attention?** It's a forward-pass intervention so each model applies
its **own** RoPE — for the top prev-token head B, replace B's key input with `norm(resid − A_out)` (q untouched →
key-only) for each upstream head A, and re-read B's attention. RoPE rotates the patched key, so the relative
match is preserved; only the key *content* changes. The intervention is exact: the **zero-patch sanity is
0.00e+00 on every model** (q/v untouched).

| model | pos. enc. | prev-token head | top key-patch collapser | prev-token collapse | robust z |
|---|---|---|---|---|---|
| **GPT-2** | absolute | 4.11 | **sink head 1.3** | **−22%** | **118.8** |
| Gemma-2-2B | RoPE | 21.7 | 5.4 (non-sink) | −0% | 4.8 |
| Qwen-2.5-1.5B | RoPE | 13.4 | 0.0 (non-sink) | −0% | 1.8 |

**CONFIRMED causally — GPT-2's prev-token attention is carried by KEY CONTENT; the RoPE models' by the rotation.**
Removing the **sink head 1.3** from GPT-2's prev-token key collapses its prev-token attention **−22%**, a huge
standout (z 118.8 vs a median 0.1% over the other upstream heads) — and the collapser *is a sink head*, the
positional broadcaster from #25. In the RoPE models, **no** upstream head's key-content removal collapses
prev-token (max −0%): their rotation, untouched by the content patch, still aligns q−1. This is the decisive
causal pairing for #26's representational split, and the third+fourth causal/representational signatures of
GPT-2's learned absolute positions (with the sink-dependence and the cross-model ceiling). For scale: the −22%
comes from removing **one** upstream head's key-content with the rest of the model fully intact — vs a **0.1%
median** over the other ~47 upstream heads (the figure `key_patch_cross_model.png` is the lone red GPT-2 bar
beside the flat RoPE bars). The zero-patch sanity (0.0) is the numerical-fidelity check: `norm(resid − A_out)`
reproduces the clean key exactly when A_out is forced to zero.

**Scope.** Llama-3.2's prev-token head is in *layer 0* (no upstream heads to patch), so the RoPE side rests on
Gemma + Qwen — a real limitation for any model whose prev-token head sits at the very bottom (the patch needs an
*upstream* writer). The patch is the *direct* A→B key path (`resid − A_out` at B's layer). *Next:* the same
key-only / value-only forward patch generalizes to **any** circuit — induction heads, the validated V-edges
(#28) — to probe content-vs-rotation dependence beyond prev-token; and re-running it on an oracle-supervised host
tests whether supervision (#19/#20) shifts the key-content dependence. `key_patch_cross_model.py`,
`runs/gemma/key_patch_cross_model_summary.json` (~85 s, 4 models).

### Generalizing the patch across circuits & channels — match (key) vs move (value)

`circuit_content_patch.py` runs the same faithful key-only patch on three circuits — **prev-token** (positional:
attend to q−1), **induction** (content: attend to the key whose *predecessor token* == the current token),
**duplicate** (content: attend to an earlier occurrence of the *same* token). The sharp question: is the
key-content-vs-rotation split (#26/#31) about the *architecture*, or about the *addressing type*? RoPE's rotation
supplies *position*, never token *content* — so a content match must live in the key content in **every**
architecture, while only positional matching can move to the rotation.

| circuit | type | GPT-2 | Gemma-2-2B | Llama-3.2-1B | Qwen-2.5-1.5B |
|---|---|---|---|---|---|
| **prev-token** | positional | **+22% ✓** | +0% | skip (L0) | +0% |
| **induction** | content | **+17% ✓** | **+18% ✓** | **+70% ✓** | **+89% ✓** |
| duplicate | content | skip (L0) | +3% | +9% | +13% ✓ |

(✓ = the top upstream head's key-content removal collapses that circuit ≥10% and ≥3× the upstream-head median;
every zero-patch sanity is 0.0.)

**GENERALIZED — key-content dependence is about the ADDRESSING TYPE, not the architecture.** The *positional*
circuit (prev-token) is key-content-dependent **only in GPT-2** (the RoPE models read position from the rotation,
−0%), but the *content* circuit (induction) is key-content-dependent in **every** model (+17…+89%): removing the
upstream predecessor-writer from the induction head's key collapses induction everywhere — the rotation cannot
supply the predecessor *token*, so it must be in the key content. So #26/#31's GPT-2-vs-RoPE split is specifically
the **positional register**; the **content instruction set** (token-identity matching) is universal — exactly
consistent with the mechanism-invariance results (induction is causally load-bearing in all four). A nuance: in
GPT-2 the induction collapser *is* its prev-token head (4.11 — the canonical K-composition writer), but in the
RoPE models it is an **early** head (their dominant prev-token head is *late* — e.g. Gemma's is layer 21,
downstream of its layer-4 induction head — so induction is fed by a separate early predecessor-writer); the
universal fact (induction is key-content-fed) holds, the specific writer differs.

**The move (value) channel — universal even where the key isn't.** The same forward patch runs **value-only**:
feed `norm(resid − A_out)` to the reader's *value* and measure the change in its OUTPUT (ΔV-out, the #28 readout).
RoPE rotates Q/K but **never** the value — so what each circuit *moves* should be content-dependent in *every*
architecture, even for the positional circuit whose *key* is rotation-only in RoPE.

| circuit | type | GPT-2 | Gemma | Llama | Qwen |
|---|---|---|---|---|---|
| prev-token | positional | 0.22 ✓ | 0.05 | skip | 0.11 ✓ |
| induction | content | 0.26 ✓ | 0.12 ✓ | 0.17 ✓ | 0.24 ✓ |
| duplicate | content | skip | 0.19 ✓ | 0.28 ✓ | 0.12 ✓ |

(top value-patch ΔV-out; ✓ = >0.05.) **The move channel is universal** — value-content dependence everywhere,
*including* prev-token (GPT-2 0.22, Qwen 0.11) whose KEY collapses only in GPT-2. So the architecture-specific
positional register is confined to the **key/match (score) channel**; what heads *move* is content in every model
because the value is never rotated. The value channel is also markedly more **distributed** than the key channel
— the top value-mover is only ~2–3× the upstream median (vs ~10–100× for keys), so no single head dominates what
a circuit moves (the redundancy theme — cf the V-edges adding nothing to the recompile keep-set, #30).

**NET (the addressing register, decomposed):** only **positional matching** is architecture-specific (GPT-2 puts
it in the key content via the sink broadcast; RoPE in the rotation); **content matching** (induction/duplicate
keys) and **all moving** (every circuit's value) are universal — consistent with mechanism-invariance. **Scope.**
Induction is the clean content circuit; duplicate's readers are early/layer-0 (GPT-2 skips, noisier elsewhere);
prev-token skips for Llama (layer-0 reader). Every zero-patch sanity is 0.0. `circuit_content_patch.py`,
`runs/gemma/circuit_content_patch_summary.json` (~4 min, 4 models × 3 circuits × 2 channels).

## Circuit-structured keep-set selection (M1↔M2 bridge) — first result (GPT-2)

`dag_recompile.py` closes the loop between the two milestones: it feeds the M2-extracted live sub-DAG into M1's
reconstruction-coverage harness (same mean-ablation metric, `coverage = 1 − KL(host‖keep)/KL(host‖all-ablated)`)
and asks whether the **weight-cheap** DAG (weights + 2 forward passes) picks the keep-set as well as M1's
**expensive** marginal-ablation importance ranking (one forward pass per head). GPT-2, Shakespeare, floor KL
1.82; ~95 s on CPU (most of it the 144-head importance ranking the DAG sidesteps).

- **The path-patch-confirmed circuit IS the recompile keep-set — and it beats the greedy importance set at equal
  size.** The 12-head induction+IOI live sub-DAG reconstructs **+0.333** coverage vs **+0.230** for the 12
  individually-most-important heads (greedy top-B) and **+0.038** for random-12 — i.e. **145% of the
  greedy-optimal** at equal budget, +0.295 over random. It *beats* greedy importance because marginal ablation
  is myopic (ranks heads by their *individual* effect) while the DAG selects a *coordinated interacting circuit*
  — exactly the regime M1 flagged ("heads interact; the single most-important head in isolation is net-negative").
  So the auto-extracted circuit is a coverage-efficient keep-set **without** the per-head ablation sweep.
- **But raw connectivity is NOT a generic output-importance proxy** — the honest dissociation. Ranking *all*
  heads by DAG-connectivity (summed incident live-edge specificity) only weakly tracks marginal importance:
  Spearman = +0.24 over all 144 heads but **+0.02 among the heads the DAG actually gated**. And the larger
  keep-set that adds the new write-hubs (`dag_all_live`, 30 heads) reaches only **67%** of the greedy-optimal
  coverage (+0.361 vs +0.541). Reason: **ΔTV measures attention-*reshaping*, not output-*importance*** — the new
  early-layer write-hubs (0.9→2.x, 1.8→{9.3,10.5,10.9}) strongly shape downstream attention yet are
  output-redundant (mean-ablating them barely moves next-token KL). *Mechanistic hypothesis:* this is the
  program-wide redundancy seen throughout the stack (substrate/core redundancy in the tower; the redundant
  prev-token *population* feeding a bottleneck inductor in rung-3) — an early write-hub broadcasts positional /
  duplicate-token signal along *many parallel paths*, so removing any one (or even the hub's whole output, mean-
  ablated) leaves the downstream readers able to recover it elsewhere; high attention-influence, low *marginal*
  output-importance.

  As an *ordering*, DAG-connectivity still beats random at every budget (it front-loads the circuit heads) but
  lags the marginal-importance ordering at small budgets — consistent with "connectivity finds the circuit, not
  the importance rank":

  | budget B | top-importance | DAG-connectivity | random |
  |---|---|---|---|
  | 4 | +0.153 | +0.054 | +0.025 |
  | 12 | +0.230 | +0.186 | +0.082 |
  | 24 | +0.391 | +0.416 | +0.170 |
  | 48 | +0.610 | +0.636 | +0.408 |
  | 64 | +0.715 | +0.690 | +0.426 |

**Takeaway.** The bridge confirms the program's central use of the extractor: a *path-patch-confirmed* sub-DAG
is the structured op-set the recompiler should keep (rivals/beats greedy importance, ≫ random, no ablation
sweep) — but the gate's ΔTV is a *circuit-liveness* signal, not a drop-in importance score, so the keep-set
must come from the confirmed circuit, not from thresholding raw connectivity. This is the M1→M4 hand-off: the
DAG keep-set is what the feature-basis recompilation (milestone 4) should express in a clean basis.

*Next:* (a) feed *validated* new edges (single-edge path-patch of the write-hub candidates from M2) into this
keep-set so the recompiler grows beyond the two textbook circuits; (b) make the gate jointly attention-liveness
*and* output-importance (e.g. an attribution / logit-effect term alongside ΔTV) so connectivity becomes a true
keep-set score; (c) re-run on an oracle-supervised host (#19/#20) — a more legible host should yield a cleaner
live DAG and a higher-coverage keep-set. The script takes `--dag-summary` so any of these DAGs (other corpora,
supervised models) drops straight in. `runs/disassembly/dag_recompile_summary.json` (re-run to regenerate the
figure).

### Adding the value pathway (V-edges) to the bridge

The bridge now also consumes the V-composition DAG (`vcomposition.py`, `--v-summary`). The K/Q circuit covers
attention *routing*; do the composed-OV **virtual heads** (the layer-6 value-readers, induction content re-read
as a value) carry reconstruction the K/Q circuit missed? We add the **22 live V-edge heads** to the 12-head K/Q
circuit and measure the incremental coverage against the same number of random and importance-optimal additions:

| keep-set | coverage | Δ vs K/Q |
|---|---|---|
| K/Q circuit (12 heads) | +0.333 | — |
| + 22 **V-edge** heads | +0.338 | **+0.004** |
| + 22 random heads | +0.412 | +0.079 |
| + 22 top-importance heads | +0.532 | +0.198 |

**No — the value-pathway heads are output-redundant.** Adding the 22 V-edge virtual heads lifts coverage by only
**+0.004**, *less than adding 22 random heads* (+0.079), and their marginal-importance ranks are mid-pack
(23–134 of 144). So **ΔV-out, like ΔTV, is an edge-COUPLING score, not head output-importance**: the V-edges are
a real composed-OV coupling (#28 — removing the writer changes the reader's *output*), but those virtual heads
are not load-bearing for the model's *next-token* output (keeping them barely helps; the reader is driven by the
writer yet itself contributes little to the logits). This **unifies the bridge's lesson across all three Elhage
ports** — K/Q (ΔTV) *and* V (ΔV-out) composition-coupling both fail to predict output-importance; the recompile
keep-set is the **path-patch-confirmed named circuit**, full stop, not any coupling score. (Consistent with the
program-wide redundancy theme: composition is distributed; a few named heads are load-bearing, the rest is
redundant coupling.) `runs/disassembly/dag_recompile_summary.json`, `value_pathway` field.

## MLP ops in the DAG + the recompile (milestone 3) — first result (GPT-2)

`mlp_ops.py` adds the **COMPUTE** instruction class. M1 and the bridge kept/ablated attention heads only — MLPs
ran at full fidelity, so the coverage metric never charged for them — yet MLPs carry real computation
(greater-than is MLP-dominated; `mlp_catalog.py` read the neuron key→value vocabulary). M3 extends the
mean-ablation harness to keep/ablate **MLP layers** as well as heads (floor = all heads *and* all MLPs ablated),
adds **head↔MLP composition edges** to the DAG, and names the load-bearing MLPs. GPT-2, Shakespeare, floor KL
3.39.

- **MLPs are load-bearing — and the load is concentrated in L0.** Removing *all* MLPs (heads intact) collapses
  coverage to **−0.019** (≈ floor): attention alone cannot reconstruct the forward pass. The single most
  important op in the whole recompile is the **layer-0 MLP** (marginal importance **+0.772**; next are L11 +0.08,
  L1 +0.07) — GPT-2's **detokenizer**, the same `e_ext = e + MLP0(·)` enrichment the QK/copy disassembly already
  reads through. Its top neurons read sentence-boundary punctuation (`. ; ?`) and write structural/line-start
  tokens (newline, `I`, `First`) — detokenization / boundary formatting.
- **A few MLPs reconstruct most (attention intact).** Sweeping the MLP budget with all heads kept, the
  top-importance MLPs dominate a random-MLP control at every budget (L0 first):

  | MLPs kept | top-importance | random |
  |---|---|---|
  | 1 | +0.118 | −0.002 |
  | 2 | +0.257 | −0.004 |
  | 4 | +0.436 | +0.313 |
  | 8 | +0.819 | +0.595 |

  A **combined sparse op-set** of just the 12 M2 circuit heads + the top-4 MLP layers reaches **+0.286** coverage
  — a tiny MOVE+COMPUTE program (16 ops of 156).
- **The DAG gains MLP nodes.** Head→MLP read edges (`‖OV_a · W_in^L‖`, mean-write removed) and MLP→head write
  edges (`‖W_out^L · W_{Q/K}^B‖`) are weight-legible; top edges e.g. `2.1→L2`, `11.0→L11` (read) and `L1→2.2`,
  `L0→4.11` (write). So the call graph now has typed head↔MLP edges, not just head↔head.

**Scope (honest).** (a) Mean-ablating *all* MLPs is severe (L0 dominates), so "attention-only / MLP-only" are
**necessity** statements, **not** a clean attention-vs-MLP credit split — attention's reconstruction value is
the MLP-intact bridge (#23), where circuit heads reconstruct with MLPs on. (b) Static head↔MLP composition does
**not** rank MLP recompile-importance (Spearman = **−0.43**): the most important MLPs are *early* (L0) and have
the *fewest* incoming head→MLP edges (a depth confound), so the DAG edges give **structure**, importance comes
from the recompile — the same lesson as the bridge's "ΔTV ≠ KL-importance." (c) The full per-neuron catalog +
low-rank/named-idiom analysis lives in `mlp_catalog.py`; M3 reuses its read→write naming for the load-bearing
layers only. (d) MLP→MLP dynamic gating remains future work (V-composition is now done — next section).
`runs/disassembly/mlp_ops_summary.json` (re-run to regenerate the figure).

## V-composition — the value pathway (completes the DAG edge types) — first result (GPT-2)

M2 scored K- and Q-composition and gated them with **ΔTV** — the change in the reader's attention *pattern*. By
construction ΔTV cannot see **V-composition** (Elhage's third edge type): head A's output feeding head B's
**value** changes *what B moves*, not *where B attends*. V-composition is how heads chain OV circuits ("virtual
heads" / composed copies), so the DAG was incomplete. `vcomposition.py` adds it with the matching readout —
static `comp_V(A→B) = ‖OV_A·W_V^B‖/(‖OV_A‖‖W_V^B‖)` (mean-write removed) + a dynamic **ΔV-out**: remove A from
B's *value*, recompute B's **output contribution** (attention fixed), measure the relative change in B's residual
write; reader-matched null. GPT-2, faithful patches (recompute vs model 9.85e-7).

- **A clean K/V double dissociation — V is a separable pathway, not a relabelled K-edge.** The strongest V-edges
  change B's **output** (median ΔV-out **0.214**) but barely its **attention** (ΔTV **0.020**); the strongest
  K-edges are the mirror (ΔTV **0.142**, ΔV-out 0.067). The scatter is L-shaped: V-edges top-left
  (output-moving), K-edges bottom-right (attention-shaping), null in between (figure
  `runs/disassembly/vcomposition.png`). This is exactly the pathway M2's ΔTV readout was blind to. (ΔV-out is
  normalized to B's *own* output norm, so the median 0.214 means removing one writer shifts ~21% of B's residual
  write — a large single-edge effect for the value pathway, ~3× the K-edges' incidental 0.067.)
- **Static V-composition predicts dynamic ΔV-out (ρ +0.36)** — the value pathway is weight-legible like K/Q.
- **The top V-edges are composed-OV "virtual heads," and they're interpretable** — induction heads (layer 5) feed
  *layer-6 values*: the induction-moved content is re-read as a value by a layer-6 head and moved onward (a 2-hop
  OV circuit). For `5.9→6.7`, ΔV-out 1.32 means removing 5.9 changes 6.7's output by >100% of its norm — 5.9
  *dominates* 6.7's value.

  | V-edge (A→B) | static V-comp | ΔV-out | ΔTV | reading |
  |---|---|---|---|---|
  | `5.9→6.7` | 0.070 | **1.32** | 0.043 | induction 5.9 → 6.7's value (composed OV) |
  | `5.5→6.7` | 0.074 | 0.87 | 0.032 | induction 5.5 → 6.7's value |
  | `5.5→6.6` | 0.072 | 0.72 | 0.046 | induction 5.5 → 6.6's value |
  | `5.9→6.0` | 0.077 | 0.60 | 0.033 | induction 5.9 → 6.0's value |
  | `5.9→7.3` | 0.076 | 0.54 | 0.039 | induction 5.9 → 7.3's value |
  | `3.0→4.3` | 0.081 | 0.44 | 0.026 | early duplicate 3.0 → 4.3's value |
- **V-composition is weaker/secondary to attention routing** — mean V/K composition 0.80, and K has the stronger
  top edges (0.114 vs 0.081), consistent with Elhage's finding that GPT-2 composition is mostly Q/K. The value
  pathway is real but the minority edge type.

**Scope (honest).** (a) The readout is the change in B's *direct output* (value patch, attention fixed) — the
value analog of ΔTV, not a task-level loss metric. (b) The `spearman_staticK_vs_dTV` reported here (−0.17) is
**range-restricted** (computed over only the top-static-K slice, a narrow static range) and is *not* the
canonical K static→dynamic agreement — that is M2's +0.37 over a broad top+random edge set; here the K-edges are
only the dissociation control. (c) Q-value (Q-composition's value analog) and MLP→head V-edges are natural
extensions. The DAG now carries **K, Q and V** head-edge types. `runs/disassembly/vcomposition_summary.json`
(~75 s on CPU; re-run to regenerate the figure).

## Ceiling test (milestone 4) — first result (v2, tiny GPT)

`ceiling_test.py` recompiles the tiny GPT by forcing computation **through the SAE feature basis** (sae-forge
`NativeModel`, `native_in_basis`), sweeping SAE width 1–8×, measuring three things at once: forged-model
output faithfulness (KL vs host; unigram floor 2.17), feature-content retention (mAUC), and monosemantic
factorization (cov95).

| width | forged-model KL | cov95 host→forged | mAUC host→forged |
|---|---|---|---|
| 1× | 43.3 | 0.60→0.00 | 0.92→0.67 |
| 2× | 5.11 | 0.64→0.00 | 0.92→0.78 |
| 4× | 5.08 | 0.64→0.12 | 0.93→0.84 |
| 8× | 5.08 | 0.60→0.16 | 0.92→0.85 |

The single-ceiling prediction is **refined into two axes — and the capability axis is scale-confounded**:
- **Feature content reconstructs; monosemantic factorization does not.** mAUC retention ~86% (rises with
  width) vs cov95 retention ~11% (collapses) — the forge tax in the recompilation frame: the basis carries
  *what* the model represents, not a *monosemantic factorization* of it.
- **The forged model's output is globally broken at every width** (KL 5–43 ≫ unigram 2.17 → negative capability
  coverage; 1× catastrophic, 2–8× plateau at KL≈5.08). This is the known tiny-whole-model-forge artifact, **not**
  the entangled-core ceiling — so this substrate *cannot* isolate the capability plateau-vs-core (the doc's
  central prediction). Settling that needs a **high-quality GPU-scale forge** (SAELens + polygram compression)
  the 8 GB box can't run.

So milestone 4 **builds the harness and settles the content/factorability axis** (decoupled, robust across
widths) and **identifies the capability-ceiling test as GPU-scale-gated** — an honest partial result: the
unifying "one ceiling" claim is wrong as stated (≥2 axes), and the clean capability test is deferred to better
forge hardware. (The tower's ~24% irreducible core is the target a GPU-scale capability curve would be
compared against.) `ceiling_test.py`, `runs/cov95_forge_tax/ceiling_test_summary.json`.

## Cross-model ceiling (milestone 5) — first result (5 models, abs-pos vs RoPE)

The forge-basis ceiling (M4) needs per-layer SAEs + sae-forge forging, which Llama-3.2-1B / Qwen-2.5-1.5B don't
have and which is globally broken at whole-model scale — so the cross-model ceiling goes to the **op-selection**
metric, which is arch-generic: `cross_model_ceiling.py` runs M1's reconstruction-coverage harness
(`coverage = 1 − KL(host‖keep)/KL(host‖all-ablated)`) on five models — two **absolute-position** (GPT-2 144h,
gpt2-medium 384h) and three **RoPE** (Gemma-2-2B, Qwen-2.5-1.5B, Llama-3.2-1B) — and compares the **named
induction circuit** (prev-token + induction heads, found behaviorally) to an **equal-size random** keep-set. The
gpt2-medium control is the key: it is *larger* than two of the RoPE models, so it separates **scale** from
**architecture-family**. The question: is the *decompilable fraction* architecture-invariant like the mechanisms
are?

| model | pos. enc. | heads | induction circuit | random | lift | ratio |
|---|---|---|---|---|---|---|
| **GPT-2** | absolute | 144 | +0.202 (**20% of pass**) | +0.049 | **+0.153** | **4.1×** |
| **gpt2-medium** | absolute | 384 | +0.197 (**20%**) | +0.012 | **+0.185** | **16.9×** |
| Gemma-2-2B | RoPE | 208 | +0.031 (3%) | +0.022 | +0.009 | 1.4× |
| Qwen-2.5-1.5B | RoPE | 336 | +0.090 (9%) | +0.031 | +0.060 | 2.9× |
| Llama-3.2-1B | RoPE | 512 | +0.034 (3%) | −0.000 | +0.034 | n/a |

**The decompilable fraction is *not* architecture-invariant — and the gpt2-medium control disentangles *why*:
it's the absolute-position FAMILY, not scale.** The named circuit beats random in **every** model (lift > 0 →
the op-catalog is real and the *mechanisms* are invariant). But its coverage-share splits cleanly by
position-encoding: the **absolute-position** GPT-2 family reconstructs **~20%** of the pass (4–17× random)
*regardless of size*, while **every RoPE** model sits at **3–9%** (1.4–2.9×). The control is decisive —
**gpt2-medium has 384 heads, more than Qwen (336) and Gemma (208), yet keeps the full 20% / 16.9× fraction** the
larger RoPE models lack. So the earlier GPT-2-smallest-and-only-absolute confound resolves in favour of
**family**: GPT-2's absolute positions **concentrate** the induction circuit into a few load-bearing heads; RoPE
**distributes** it across many (more redundant), so a fixed named circuit explains proportionally less. The
random-budget curves echo it: the absolute-pos curves *rise* with heads kept, the RoPE curves stay
flat/declining (random head-sets add ~nothing). This is the **same GPT-2-family-is-special pattern** as the sink
(only GPT-2 depends on it) and the positional-broadcast circuit (only GPT-2 has it) — three independent
signatures of GPT-2's learned absolute positions.

**Scope (honest).** (a) **Op-selection** ceiling (heads at full fidelity, complement mean-ablated); the
**forge-basis** ceiling (M4) stays SAE/GPU-gated for non-GPT-2 — when per-layer SAEs land for the RoPE models it
can be run apples-to-apples. (b) **gpt2-large (720 heads) was tested and excluded** — its 5-head circuit
collapses to ~0% because the behavioral induction-head identification fails at that head count on the small eval
(it picks a last-layer "inductor"); the disentanglement rests on gpt2-medium, where the circuit is clean. (c)
The circuit is a fixed ~5 heads (behavioral); the size-controlled lift makes the conclusion robust to the exact
heads (`--n-prev/--n-ind` expose it for sensitivity). `cross_model_ceiling.py`,
`runs/gemma/cross_model_ceiling_summary.json` (~3 min, 5 models).

## Reachability — host-width × oracle-supervision (first result)

The ceiling test asks what's *achievable*; this asks what's *reachable by training*. The entanglement-tower
retrain no-go ("you can't train interpretability in without losing capability") used a *reconstruction*
bottleneck — the mAUC axis that already survives forging, not cov95. `host_width_sweep.py` retries on the
right axis: train tiny GPTs from scratch across host widths, with/without an **auxiliary oracle-feature-
recovery loss** (a linear head from the residual must predict the exact per-token oracle labels), and measure
native cov95 + capability.

| host width | params | LM-loss unsup→sup | cov95 unsup→sup | Δcov95 |
|---|---|---|---|---|
| 32 | 1.7M | 6.54→6.42 | 0.48→0.62 | +0.14 |
| 64 | 3.4M | 6.21→6.16 | 0.62→0.79 | +0.17 |
| 128 | 7.2M | 5.99→5.99 | 0.69→0.76 | +0.07 |
| 256 † | 16.0M | 6.04→6.05 | 0.45→0.69 | +0.24 |

† w256 is **undertrained** (16M params on 107k tokens; its unsup LM loss 6.04 is *worse* than w128's 5.99) —
its low unsupervised cov95 is a compute-budget artifact, not scarcity counter-evidence.

- **Reachability — CONFIRMED.** Oracle-supervision lifts native cov95 at *every* width (+0.07…+0.24, mean
  **+0.155**) at **zero/negative capability cost** (mean **−0.037 nats** — it slightly *helps* LM loss). So
  interpretable, equally-capable solutions are not only existent (superposition is linear compression —
  decompressing preserves the function) but **reachable by training pressure**, with the manufactured-oracle
  substrates serving as the *training signal*, not just the grader. This is the constructive counter to the
  tower retrain no-go: the no-go used the wrong (reconstruction) axis; supervising the oracle-feature axis
  lifts monosemanticity for free.
- **Scarcity — partially supported.** Unsupervised cov95 *rises* with host width in the well-trained regime
  (0.48→0.62→0.69 for 32→64→128), consistent with superposition being capacity-driven; it drops at 256 (0.45),
  but that host is **undertrained** (16M params on 107k tokens — its LM loss is also worse than w128), a budget
  artifact, not counter-evidence. So the forge tax is *partly* a capacity-scarcity artifact — relieved by width
  up to the training budget, and relieved more cheaply by supervision.

**Feeds milestone 1:** supervision yields cleaner, more monosemantic low-χ residuals — which are exactly the
"registers" the ResidualVM interpreter reads, so a supervised host should decompile further (higher
reconstruction-coverage at lower op-budget) than an unsupervised one. That's the direct hand-off to the
recompile harness.

### What kind of pressure lifts cov95? (aux-mode comparison)
Is the lift real monosemanticity or just linear recoverability? `monosemantic_aux.py` compares aux modes at a
fixed well-trained width (128), training from scratch, **over 3 seeds** (per-seed model init + batch order +
eval-SAE init):

| aux mode | LM-loss | cov95 (mean ± std, 3 seeds) |
|---|---|---|
| none | 6.02 | 0.68 ± 0.04 |
| linear (recoverability) | 6.01 | **0.76 ± 0.00** |
| decorr (orthogonal read-directions) | 6.00 | 0.76 ± 0.00 |
| dedicated (one-neuron-per-feature) | 6.00 | 0.69 ± 0.00 |
| sparsedict (full in-loop aligned SAE) | 6.16 | **0.48 ± 0.07** |

Paired across seeds: **`linear > none` in 3/3 (+0.080 ± 0.043); `sparsedict < none` in 3/3 (−0.195 ± 0.086);
`linear > sparsedict` in 3/3 (+0.276)** — every signal survives the noise.

**The simple linear-recoverability proxy is robustly the *best*** — and *every* "more direct" monosemanticity
objective fails to beat it: orthogonalizing the probe's read-directions is **inert** (decorr ≡ linear, identical
every seed), dedicated raw neurons sit *below* `none`'s mean, and the **full sparse-dictionary-in-the-loop** (a
jointly-trained TopK SAE on the residual, reconstruction + sparsity, first F latents aligned to the oracle) is
robustly the **worst** — cov95 0.48 (below `none` in 3/3) *and* the only one with a real capability cost
(+0.16 nats). The heavier the direct pressure, the worse it gets. So the cov95 lift comes from making features
**linearly prominent** (so the downstream SAE can isolate them), **not** from forcing axis-alignment /
sparse-coding in the host residual: the SAE does the factorization; an in-loop dictionary's reconstruction
pressure distorts the representation in a way that doesn't transfer to the fresh eval SAE. This answers "is it
just recoverability?" — yes, and recoverability *is* the effective lever for SAE-measured cov95. **Bonus:**
supervision also makes cov95 **variance-free** (linear pins exactly 22/29 features over threshold every seed,
vs `none`'s noisy 0.62–0.72) — it lifts cov95 *and* makes it reproducible. `monosemantic_aux.py`,
`runs/cov95_forge_tax/monosemantic_aux_summary.json`.

### Is the lift real monosemanticity, or the eval-SAE finding what we planted? (non-SAE cross-check)
cov95 fits a TopK SAE — so does the linear aux just make features recoverable in a way a linear-ish SAE
prefers (circular)? `legibility_crosscheck.py` scores the none→linear lift in **three bases** (same
symmetric-AUC scorer, only one involving an SAE), width 128, 3 seeds:

| basis | none → linear | Δ (mean ± std) | up in |
|---|---|---|---|
| sae (fitted TopK dictionary) | 0.68 → 0.76 | +0.08 ± 0.04 | 3/3 |
| neuron (raw residual dims, no fit) | 0.68 → 0.76 | +0.08 ± 0.07 | 2/3 |
| pca (rotated basis, fit *without* labels) | 0.36 → 0.69 | **+0.33 ± 0.09** | 3/3 |

**Corroborated — the lift is genuine monosemanticity, not circular.** Both SAE-free metrics rise too: features
become single-detector-isolable by *raw neurons* (+0.08) and, most strongly, by *PCA components* (+0.33). So
supervision doesn't merely make features recoverable in a way the eval-SAE prefers — it makes them
axis-isolable in the natural and rotated bases as well. (For the unsupervised model, neuron-cov95 *equals*
sae-cov95 exactly — the fitted SAE adds nothing over raw neurons here, so it can't be inflating the result.)
**Mechanism:** the biggest lift is in PCA — unsupervised pca-cov95 is only 0.36 (the oracle features sit *off*
the high-variance axes), so supervision **pushes the features into the residual's principal (high-variance)
directions**, where *any* axis-aligned probe (neuron, PCA, or SAE) isolates them. That's a basis-independent
signature of real monosemanticity, and it explains *how* recoverability becomes monosemanticity: prominence =
high-variance placement. `legibility_crosscheck.py`, `runs/cov95_forge_tax/legibility_crosscheck_summary.json`.

Caveats + scope: the aux loss pressures **linear recoverability** of the oracle (which, per the aux-mode
comparison above, is the *effective* lever — direct decorr/dedicated/sparsedict objectives don't beat it, and
the in-loop sparse dict actively hurts). The scarcity trend is only cleanly visible
in the **well-trained regime** (≤w128 here); confirming it at w256+ needs more tokens/steps (compute scaling),
not more width. Tiny hosts, short training. **Status of the open levers:** the *training-pressure* axis is
settled (decorr/dedicated/sparse-dict all tested, none beats linear-recoverability; multi-seed-confirmed) and
the *measurement* worry is resolved (non-SAE bases — raw-neuron + PCA — corroborate the lift, see cross-check
above → genuine monosemanticity). **Remaining follow-ups:** (1) multi-seed +
adequately-trained wide hosts to clean the scarcity curve; (3) richer oracles (spaCy POS/NER) + curriculum
annealing of the aux weight; (4) polygram geometry penalties. But the direction is clear and the cost is
~zero, so the reachability lever is real. `host_width_sweep.py`,
`runs/cov95_forge_tax/host_width_sweep_summary.json`.

### Does feature legibility buy circuit legibility? (oracle-supervised DAG)

The reachability lever makes the *feature substrate* more legible (cov95 up). `oracle_supervised_dag.py` asks the
sharp follow-up: does it also make the *composition* more legible — or are knowledge (features) and computation
(the DAG) the **separate axes** the forge tax says they are? Train tiny GPTs unsupervised vs oracle-supervised
(the `linear` lever), 3 seeds, and on each model measure feature legibility (cov95) alongside circuit legibility
(static→dynamic composition ρ; prev-token→induction recovery) and the prev-token head's key position-vs-token
content (the #26 probe), paired by seed.

| metric | none → linear | seeds + |
|---|---|---|
| **cov95** (feature) | 0.690 → 0.736 (Δ **+0.046**) | 2/3 |
| induction-recovery (circuit) | 1.18 → 1.35 (Δ **+0.16**) | 3/3 |
| static→dynamic ρ (circuit) | 0.44 → 0.49 (Δ +0.04 ± **0.42**) | noise-dominated |
| prev-token key **position** fraction | 0.33 → 0.13 (Δ **−0.20**) | 0/3 (drops) |

**PARTIAL / SUBSTRATE-DOMINATED — largely separate axes.** Supervision robustly reshapes *what the residual
represents*: cov95 lifts, and the prev-token head's key shifts **token-ward** every seed (Δpos −0.20) — the
feature-recovery loss injects token/lexical content into the residual, which the key inherits, making it *less*
positional. Its effect on the *composition* is at most marginal: induction-recoverability lifts a small but
consistent +0.16 (3/3), while the broad static→dynamic agreement is noise-dominated (±0.42) on a 16-head host.
So the lever acts mainly on the **feature substrate**, with only a small spillover to circuit recoverability —
consistent with the program's knowledge-≠-computation thesis (you cannot supervise circuits into existence with
a feature loss). **Scope (honest):** the tiny 4-layer host is underpowered for circuit metrics — its induction
is marginal (1.1–1.4× baseline) and its prev-token key is token-dominated even unsupervised (unlike GPT-2's 4.11
at 59%), so the static→dynamic ρ is high-variance; a definitive circuit-legibility test needs a host with real
circuits (GPT-2 scale), which the reachability lever can't retrain on this budget. `oracle_supervised_dag.py`,
`runs/cov95_forge_tax/oracle_supervised_dag_summary.json`.

## Instruction reuse vs specialization — is the op-catalog an ISA? (the "LLM-as-VM" test)

The disassembly reads the named ops as a reusable *instruction set*. But are the same instructions recruited
across *different tasks* (genuine reuse → one ISA), or does each task have dedicated heads (specialization)? We've
shown the idioms are invariant across languages and architectures, never across *tasks within a model* —
`instruction_reuse.py` is that test. It builds the **head-class × task causal matrix**: mean-ablate each named
op-class and measure the damage to three distinct programs — **generic LM** (held-out next-token NLL),
**induction** (NLL on the 2nd copy of a repeated random sequence), **IOI** (logit(IO)−logit(S)). A class
"serves" a task if ablating it damages it beyond a random-head control.

The matrix is run with **five** tasks — generic, induction-copy, **copy-names** (a 2nd copy task, different
content), **successor** (consecutive-number runs → increment), IOI:

| op-class | generic | induction | copy-names | successor | IOI | serves |
|---|---|---|---|---|---|---|
| prev-token | +0.01 | **+0.65 ✓** | **+1.69 ✓** | +0.45 | +0.16 | copy family |
| **induction** | +0.03 | **+6.53 ✓** | **+8.03 ✓** | **+5.46 ✓** | **+0.26 ✓** | **ALL 4 in-context** |
| duplicate-token | +0.05 | **+0.73 ✓** | **+7.46 ✓** | **+22.7 ✓** | −0.05 | copy + successor |
| name-mover | −0.00 | +0.10 | +0.50 | +0.42 | +0.06 | (self-repair) |
| S-inhibition | +0.00 | −0.10 | −0.13 | −0.19 | **+0.58 ✓** | IOI |
| negative-mover | +0.00 | −0.27 | −0.46 | −0.32 | −0.60 | (writes *against* IO) |

(values = ablation damage, + hurts the task; ✓ = beyond the random-head control.)

**A reused low-level core + task-specific output heads.** Thickening the matrix from 3 to 5 tasks *flips* the
read toward reuse:
1. **None of the named ops are load-bearing for *generic* LM** (every generic cell ≈ 0) — the catalog is
   in-context-task instructions **recruited on demand**, not an always-on ISA.
2. **The low-level copy/addressing ops are genuinely *reused* across the copy-family:** `induction` is
   load-bearing for **all four** in-context tasks; prev-token + duplicate serve the copy+successor family. So the
   shared substrate ops transfer.
3. **Only the *output* head is task-specific:** S-inhibition serves IOI alone.
4. **A bonus:** `successor` is carried by the *copy* ops (induction +5.46, duplicate +22.7), not a dedicated
   successor head — **the model increments partly by in-context copying**.

So the named ops *are* a **reusable instruction set for the shared substrate, composed into task-specific
circuits at the output** — closer to the VM "shared ISA + per-task programs" than the 3-task version suggested,
but with the architecture-specific caveat that none are always-on. **Caveat:** name-movers read ~0 only because
mean-ablation triggers the **IOI self-repair** (quantified in `self_repair.py`) — not genuine unimportance.
`instruction_reuse.py`, `runs/disassembly/instruction_reuse_summary.json`.

**Self-repair, made causal (`self_repair.py`).** The name-mover caveat above is quantified: ablating the
*primary* name-movers (9.6/9.9/10.0) drops the IOI logit-diff by **−0.002** (the circuit looks robust), but
ablating primaries **and** backups (9.0/9.7/10.x/11.2) together drops it **−1.04** (+2.57 → +1.53). The backups
are causally **idle when primaries are present** (Δ +0.26 from removing them) and **carry the logit-diff once
primaries are gone** (Δ +1.04 — **4× larger**). So the op is load-bearing, but a redundant backup pathway masks
it under single-class ablation — a clean instance of the program-wide redundancy (the named circuit ships *hot
spares*), and exactly why mean-ablation under-counts name-movers. `runs/disassembly/self_repair_summary.json`.

## Does the in-context-copy op survive a non-attention mixer? — SSM vs attention sweep (data, not a verdict)

`instruction_reuse.py` found **induction** (in-context copy) to be the one genuinely *reused* low-level op across
tasks. The disassembly reads that op as **attention** (a QK predecessor-match feeding an OV copy). Does anything
like it appear when the sequence-mixer is **not** attention at all? **Mamba** is a pure state-space model — no
heads, no QK match; mixing is a learned linear recurrence (a scan). `ssm_induction.py` runs this as an
**exhaustive sweep, reporting raw measurements rather than a clean conclusion**: two families across three sizes
each (Mamba 130m/370m/790m; GPT-2 small/medium/large), each measured for (a) **induction GAIN** = (1st-copy NLL −
2nd-copy NLL) on repeated random sequences, over **3 seeds × 3 context lengths**; and (b) a per-layer
**localization** profile = mean-ablate each layer's mixer (Mamba `mixer.out_proj` / GPT-2 `attn.c_proj`), record
the induction-NLL increase.

| model | mixer | L | hidden | induction gain (μ±σ) | gain @ len 12/24/48 | top layer (depth, ΔNLL, conc) |
|---|---|---|---|---|---|---|
| gpt2 | attention | 12 | 768 | **+12.54**±0.20 | +12.8 / +12.5 / +12.3 | L5 (d0.45, +0.90, c0.27) |
| gpt2-medium | attention | 24 | 1024 | **+12.45**±0.21 | +12.7 / +12.5 / +12.2 | L3 (d0.13, +0.52, c0.21) |
| gpt2-large | attention | 36 | 1280 | **+12.38**±0.18 | +12.6 / +12.4 / +12.2 | L0 (d0.0, +15.85, **c0.92 — artifact**) |
| mamba-130m | **SSM** | 24 | 768 | **+12.10**±0.35 | +12.4 / +12.2 / +11.7 | L0 (d0.0, +8.73, c0.47) |
| mamba-370m | **SSM** | 48 | 1024 | **+12.33**±0.27 | +12.5 / +12.4 / +12.1 | L0 (d0.0, +4.70, c0.38) |
| mamba-790m | **SSM** | 48 | 1536 | **+12.28**±0.30 | +12.6 / +12.4 / +11.9 | L1 (d0.02, +4.43, c0.32) |

**What the data shows (descriptive):**
1. **The in-context-copy GAIN is present and large in both families, in a narrow band — +12.10..+12.54 — across
   6 models, 3 sizes/family, 3 seeds, 3 lengths.** The behaviour that the disassembly attributes to attention is
   produced just as strongly by a model with *no attention*. The gain shrinks slightly with context length
   (12 > 24 > 48) in **every** model — a consistent secondary trend.
2. **The localization differs by family — but the headline number is confounded by coarse ablation.** Mamba
   concentrates the effect on an **early** layer (L0/L1, large ΔNLL +4.4..+8.7) **plus a mid-deep secondary peak**
   (L17/L34/L23, visible in `runs/gemma/ssm_induction.png`); GPT-2 small/medium **distribute** it across early-mid
   layers with small per-layer magnitudes (concentration 0.27/0.21). **gpt2-large's L0 spike (+15.85, conc 0.92)
   is an artifact**: mean-ablating L0 in a 36-layer model destroys the detokenizer, not "induction" — so the
   auto-summary's "attention concentration 0.47" is dragged up by that one outlier (gpt2/medium are 0.21–0.27).
   Within Mamba, concentration *falls* with size (0.47 → 0.38 → 0.32).

**What is NOT concluded (the held-off pattern).** This is evidence the in-context-copy **capability** is
architecture-invariant — it does **not** show the **mechanism** is the same. (1) the GAIN is a behavioural NLL
effect; (2) whole-layer mean-ablation removes everything a layer does, an over-estimate of "induction
localization"; (3) the two families' baselines/magnitudes are not directly comparable — compare *within* family
across size, not across families by magnitude; (4) single seed for the localization profile; (5) the SSM has no
heads, so there is **no head-level resolution** — we cannot point to a "Mamba induction head," only a layer. So:
the copy op survives the loss of attention *behaviourally*, and even survives the loss of *heads as a unit of
analysis*; whether the SSM implements it by anything resembling a QK-match→OV-copy is **unverified here** and
would need an SSM-native circuit probe (the recurrence's state-update, not a head). This complements the
cross-model ceiling result (mechanisms invariant, positional register absolute-position-family-specific): here
the **mixer class itself** changes and the copy *capability* persists. `ssm_induction.py`,
`runs/gemma/ssm_induction_summary.json` (~SSM models run on the sequential Mamba kernel — slow but exact).

## Deep per-operator dossier — the full battery on ONE instruction (induction, first)

The disassembly studied each named op *piecemeal*: induction lived in `composition_dag`/`ssm_induction`,
prev-token in `cross_model_positional`/`key_patch_cross_model`, name-movers in `self_repair`, the key/value
channel in `circuit_content_patch`, reuse-across-tasks in `instruction_reuse`. `operator_dossier.py` inverts
that: pick **one** operator and run **every** measurement on it in a single report — `--op induction` (default),
extensible to `prevtok|duplicate|name_mover|s_inhibition`. Six sections (A identity · B causal×tasks · C K/V
channels · D composition in/out · E redundancy curve · F cross-model). First dossier — **induction on GPT-2**:

- **A · IDENTITY (behavioural, not hardcoded).** Top heads by attention mass on the induction pattern:
  **5.1 (.81), 5.5 (.78), 6.9 (.77), 7.10 (.75), 7.2 (.73)** (then 5.0 .57). The literature set 5.0/5.1/5.5/6.9 is
  recovered, but behavioural ranking puts **7.10/7.2 ahead of the literature's 7.11** on this probe — a head-set
  the downstream sections then inherit (see the B caveat).
- **B · CAUSAL × TASKS.** Ablating those 5 heads: generic **+0.01**, induction **+6.39\***, copy-names **+14.42\***,
  successor **+0.01**, IOI **+0.28\*** (\* = beyond the random-head bar). Serves induction-copy + copy-names + IOI;
  **not** generic. **Honest discrepancy:** `instruction_reuse.py` found induction served *successor* (+5.46) with
  the **literature** set (incl. 7.11/5.0); the behavioural top-5 here does **not** (+0.01). So *which heads you call
  "induction" changes the causal story* — the successor-by-copying claim is sensitive to the 7.11-vs-7.10 head
  choice. A real caveat the piecemeal scripts hid by each fixing its own set.
- **C · CHANNELS (match vs move, in one view).** Reader 5.1 (zero-patch sanity 0.0): **KEY/match** collapses
  **+43%** when the prev-token head **4.11** is removed from its key (median +0%, **90.8× concentration** — sharp,
  sparse, one writer); **VALUE/move** is **distributed** (top mover 1.10, ΔV-out 0.22 vs 0.08 median). The K/V
  dissociation `circuit_content_patch` found across circuits, here localized to the one op.
- **D · COMPOSITION (local call-graph).** IN→key dominated by **4.11** (.101) — the same K-composition edge C
  confirms causally — then 4.7/1.0/3.7; OUT→value feeds **layer-6/7** heads (6.7/6.8/6.6/7.0), the composed-OV
  "virtual heads" `vcomposition.py` named (induction-moved content re-read as a value downstream).
- **E · REDUNDANCY (the ablation curve).** Primary task induction (baseline +0.674). Per-head **solo**: 5.1 +1.70,
  7.2 +0.22, 6.9 +0.21, 5.5 +0.08, 7.10 −0.03. **Cumulative**: 1h +1.70 → 2h +2.25 → 3h +3.82 → 4h +5.77 → 5h
  **+6.39**. The full-set effect (+6.39) **vastly exceeds** the best single head (+1.70) **and the sum of solos
  (~2.18)** → **strongly superadditive**: the induction heads are a **jointly-necessary / synergistic population**,
  not a single bottleneck and not independent — removing them together hurts far more than their individual
  removals predict. (This is the per-op generalization of `self_repair`'s redundancy contrast and the
  progressive-ablation "redundancy depth" probe, folded into the dossier.)
- **F · CROSS-MODEL.** Induction behavioural signal + gain elsewhere: gpt2 .81; **gpt2-medium .97 / gain +12.60**;
  **Qwen-2.5-1.5B (RoPE) 1.00 / gain +14.07**. Strong everywhere, including RoPE — consistent with induction being
  the content-universal op (the cross-model ceiling + SSM results).

**Net (descriptive):** consolidating the scattered measurements onto one operator reproduces each prior thread
*and* surfaces what the piecemeal layout obscured — the head-set sensitivity of the causal profile (B) and the
**superadditivity** of the redundancy (E). SAE-feature operands (what induction reads/writes in feature space —
the "subword name-completion" finding) are the one battery section **not** run here (needs a SAE); flagged as the
next layer. `operator_dossier.py --op induction`, `runs/disassembly/operators/dossiers/induction/gpt2.json`. The
harness runs the same six sections for the other operators.

## The operator CATALOG — every operator class, surveyed across every model

The dossier goes deep on one op; the **catalog** goes wide and turns the survey into a browsable artifact
(`docs/operators/`, tree under `runs/disassembly/operators/`). Two instruments:
**`operator_atlas.py`** (the cross-model matrix) + **`operator_dossier.py`** (the deep per-op pages), stitched into
docs by **`operator_catalog_doc.py`**. The key framing the survey forced: **each "operator" is a CLASS — a family
of heads — not a single head.** The catalog reports per-(class, model): behavioural **signal** (max head mass on the
op's pattern), **membership** (# heads carrying it), top head + depth, and a uniform **causal ΔNLL** (mean-ablate
top-3, generic prose).

**The catalog matrix — 7 universal/addressing operator classes × 6 models** (GPT-2 small/medium/large + Gemma-2-2B /
Llama-3.2-1B / Qwen-2.5-1.5B). Signal (membership in parens for GPT-2):

| class | kind | gpt2 | -medium | -large | gemma-2 | llama-3.2 | qwen-2.5 |
|---|---|---|---|---|---|---|---|
| prevtok | positional | .96 (31h) | .99 | .96 | .84 | .68 | .77 |
| induction | content | .92 (22h) | .91 | .96 | .94 | .94 | **.99** |
| duplicate | content | .62 (4h) | .86 | .96 | .85 | .74 | .97 |
| **sink** | addressing | .95 (117h) | .96 | .96 | **.07 (0h)** | 1.00 | 1.00 |
| self | addressing | .83 (10h) | .45 | .55 | .95 | .96 | **1.00** |
| local | positional | .34 (16h) | .34 | .34 | .31 | .27 | .28 |
| structural | structural | .19 (2h) | .25 | .38 | .23 | .36 | .16 |

**What the survey shows (descriptive):**
1. **Induction is the most universal class** — signal .91–.99 in *all six* (the content-copy op the SSM port +
   cross-model ceiling already flagged as architecture-invariant). prev-token and duplicate are present
   everywhere too (prev-token weaker in RoPE, .68–.84).
2. **The sink is the one class that splits by architecture — and it splits 5-vs-1, not GPT-2-vs-rest:** present
   in the GPT-2 family (.95–.96, 117–553 heads) AND Llama/Qwen (1.00, 292–472 heads), but **absent in Gemma-2
   (.07, 0 heads)**. So "attention sink" is *common but not universal*; Gemma trained without one. **And it is
   not load-bearing on prose anywhere** (causal ΔNLL ≈ 0 every model) — *present ≠ depended-on* (the
   `sink_ablation` magnitude-vs-dependence point, now in the matrix; GPT-2's deeper dependence needs the
   blocked-attention probe, not mean-ablation).
3. **RoPE models lean on `self` where GPT-2 leans on the sink** — self signal 1.00/.96/.95 (RoPE) vs .45–.83
   (GPT-2), and self is the **only universally load-bearing class on prose** (causal ΔNLL Qwen **+3.77**, Gemma
   +0.51) — the diagonal/current-token read is a RoPE workhorse.
4. **`structural` is sparse (1–7 heads) but occasionally load-bearing** — Llama's lone structural head ΔNLL
   **+1.47**.

**The deep dossiers — 9 GPT-2 operator classes** (`docs/operators/<op>.md`): the 7 idiom/addressing classes
behaviourally found + the IOI **circuit** classes (name-mover, **backup name-mover**, **negative/copy-suppression**,
S-inhibition, coreference), which are **GPT-2-only** (literature DLA head-sets, no published set in the RoPE
models) so they sit outside the cross-model matrix. Highlights the catalog surfaced: the **sink "class" is largely
content heads in their idle state** (its top-mass heads are 5.1/6.9/7.10 — the induction heads — so ablating
"sink" hurts induction +4.6, not generic); **copy-suppression ablation *raises* IOI** (−0.54, it writes against
IO); **backup name-movers serve successor**; and the **head-set sensitivity** the piecemeal scripts hid (the
behavioural induction top-5 omits 7.11, so it does *not* damage successor where `instruction_reuse`'s literature
set did +5.46).

**Taxonomy (answering "N operators or N classes?"): classes.** (1) *class* = the operation (these rows);
(2) *instance* = an individual head (the `disassemble_gpt2.py` per-head listing; each dossier §A lists members);
(3) *variant* = intra-class structure (induction writer-branching; token vs subword-name-completion inductors;
sink-as-idle-state). **Documented gaps:** succession/greater-than is **MLP-dominated** (no clean attention head —
carried by the copy ops); **SSM/Mamba has no heads** (induction present only *behaviourally*, `ssm_induction.py`).
**Not yet run:** the SAE-feature operands per class (needs a SAE); the deep dossier §C/§D/§E ported to the RoPE
models (the atlas covers them at class level; full-depth cross-model is the next layer). `operator_atlas.py`,
`operator_dossier.py`, `operator_catalog_doc.py`; `docs/operators/`; `runs/disassembly/operators/atlas_summary.json`.

## The CIRCUIT catalog — composed circuits, surveyed & collected across models

Operators are single head-classes; **circuits** are their *compositions* (a writer-op feeding a reader-op's K/Q/V
port, chained). The same cataloging machinery applied one level up: `circuit_atlas.py` (cross-model edge liveness +
harvest of the GPT-2 discovery artifacts) + `circuit_catalog_doc.py` → `docs/circuits/`. The primitive is the
**edge** (writer-op → reader-op via a port); the discovery gate (`composition_dag` ΔTV path-patch vs a
reader-matched null) is what *collects* them. **7 circuits catalogued** from two sources.

**Cross-model circuit-edge liveness** (remove the writer from the reader's key → % attention collapse, 6 models):

| circuit | defining edge | gpt2 | -medium | -large | gemma-2 | llama-3.2 | qwen-2.5 |
|---|---|---|---|---|---|---|---|
| **induction** | prev-tok head --K--> induction | +17% | +23% | +8% | +18% | **+70%** | **+89%** |
| **positional_broadcast** | sink/hub --K--> prev-tok key | +22% | +32% | +0% | +0% | (skip) | +0% |
| duplicate | same-token reader (writer often L0) | (skip) | +0% | +4% | +3% | +9% | +13% |

1. **The induction edge (prev-token → induction) is live in *every* model — and *stronger* in RoPE** (Llama +70%,
   Qwen +89% vs GPT-2 +8–23%): content matching lives in the key everywhere, so removing the predecessor-writer
   collapses the induction reader's attention universally. (Cross-model **mechanism**-invariance, at the *edge*
   level now, not just the node.)
2. **positional_broadcast is GPT-2-small/medium-only** (+22%/+32%, ≈0 in gpt2-large/Gemma/Qwen, skip Llama) — the
   same absolute-position signature as the operator catalog's sink: RoPE reads position from the rotation, so the
   prev-token key has *no upstream writer to remove*. The decompilable plumbing circuit is family-specific even
   though the content circuit is universal.

**GPT-2 discovered / circuit-specific** (harvested from the committed discovery runs, no recompute — GPT-2-only):
the **IOI Q-chain** duplicate→S-inhibition→name-mover (5 Q-edges live; causal z from `ioi_causal`: negative/
copy-suppression **z=62** writes against IO, duplicate z=6.0; **self-repair**: −primaries ΔLD −0.002 but −both
−1.04, backups are hot spares); the **V-composition "virtual heads"** (induction 5.9 → layer-6 value 6.7,
ΔV-out 1.32 — composed-OV, changes *what* is moved); the induction **K-chain weights** (canonical writer 4.11,
4/5 edges live); and the **22 NOVEL-LIVE discovered edges** (13 behaviourally named — mostly early sink/write-hub →
prev-token-key broadcasters: the collection-goal output, `validate_new_edges`).

**Taxonomy & gaps.** Levels: circuit (a DAG of operator nodes) → edge (writer→reader via a port) → operator class
at each node. **Gaps:** succession/greater-than is **MLP-dominated** (no clean attention-composition circuit);
**SSM/Mamba has no heads** (no composition edges; induction only behavioural). **Not yet run:** the IOI Q-chain /
V-composition *cross-model* (no published head-sets off GPT-2); per-edge path-patch of all 22 discovered edges on
the RoPE models. `circuit_atlas.py`, `circuit_catalog_doc.py`; `docs/circuits/`;
`runs/disassembly/circuits/atlas_summary.json`.

## ResidualVM debugger — a programmatic discovery engine (the catalog's growth tool)

The catalogs above are *surveys of what we named*; the debugger is the **tool that finds what we have not**. It is
**programmatically steppable** (not a human REPL — it returns structured data; a UI could sit on top): it
instruments the forward pass as a steppable interpreter and exposes `trace` (per-head/MLP residual writes),
`intervene` (mean-ablate any heads/MLPs → next-token KL), `logit_lens_step` (per-layer KL-to-final: *where* the
answer is decided), **`attribution_sweep`** (ablate **every** head + **every** MLP one-at-a-time, rank by causal
effect on a behaviour, and **flag the strong components the catalog has NOT named** = candidate new operators), and
**`edge_probe`** (path-patch each upstream component out of a reader → candidate circuit edges). `main()` *drives*
it; it is the engine for being exhaustive, not a place to hand-poke.

Run on GPT-2 across three behaviours (induction / IOI / generic LM), it immediately produced **new** catalog work:
- **MLP0 had the largest single-component causal effect across all three behaviours** (induction ΔNLL **+7.8**, IOI **+7.5**,
  generic +1.8) — far above any attention head. The discovery engine's first finding is that **the biggest
  operator in the model is an MLP we have not catalogued** (the L0 detokenizer): a direct pointer to the
  MLP/COMPUTE gap (the attention catalog is only half the instruction set).
- **Candidate UNNAMED operators**, by behaviour: induction → **7.6** (+0.36, *more* load-bearing than the
  prev-token head 4.11) and 2.11/8.1/8.5; IOI → **5.9** (+1.00), 8.3, 0.10, 1.3; generic → none above +0.05 (it is
  distributed + MLP-carried — corroborates the operator catalog's "no named attention op serves generic LM").
- **A discovered circuit**: edge-probing the discovered op **7.6** shows it is fed by the induction heads (5.1
  ΔNLL +1.26, 6.9) + the prev-token head (4.11 +0.40) + the candidate writers **2.11 / 5.9** — i.e. 7.6 is a
  *downstream induction-consumer* (a late induction-output mover) the literature head-set omits. A new circuit
  node + its in-edges, found mechanically.
- **logit-lens step**: induction next-token KL-to-final falls 5.8 → 0.3 by **layer L6** (decided exactly where the
  induction heads fire), 0.1 by L9.

So the debugger closes the loop the goal asks for: it *generates* candidate operators (7.6, 5.9, 8.3, MLP0) and
candidate circuit edges (2.11→7.6, 5.9→7.6) that are not yet in the catalog — the next things to dossier.
`residual_vm_debugger.py`, `runs/disassembly/residual_vm_debugger_summary.json` (GPT-2; intervention harness is
arch-generic, the QK edge-probe is GPT-2-specific here).

## The other instruction class — MLP / COMPUTE, across models

Attention **MOVES** operands; the MLP **COMPUTES** on them — and a salient discovery-sweep result was that
**MLP0 had the largest single-component causal effect measured**. The operator catalog was attention-only; `mlp_atlas.py` adds
the COMPUTE class, surveyed across architectures: per (model, layer) it mean-ablates the whole MLP block and reads
the causal ΔNLL on generic prose + induction. (Mamba/SSM is excluded — no separate MLP block, the COMPUTE analog
of "no attention heads".)

| model | L | all-MLP ablated ΔNLL (generic) | top generic MLP (depth, Δ) | top induction MLP (depth, Δ) |
|---|---|---|---|---|
| gpt2 | 12 | +2.09 | L0 (d0.0, +1.70) | L0 (d0.0, **+11.72**) |
| gpt2-medium | 24 | +2.69 | L0 (d0.0, +7.32) | L0 (d0.0, **+20.94**) |
| gpt2-large | 36 | +5.28 | L0 (d0.0, +3.67) | L0 (d0.0, **+13.57**) |
| gemma-2-2b | 26 | **+10.74** | L25 (d1.0, +0.84) | L0 (d0.0, +4.25) |
| llama-3.2-1b | 16 | +4.18 | L1 (d0.07, +7.35) | L1 (d0.07, **+12.65**) |
| qwen-2.5-1.5b | 28 | +4.29 | L1 (d0.04, +7.64) | L2 (d0.07, **+13.91**) |

**What the survey shows (descriptive):**
1. **COMPUTE concentrates on an *early* MLP — the detokenizer — in 5 of 6 models** (GPT-2 family L0, Llama L1,
   Qwen L1–L2): the single biggest COMPUTE op sits at depth ~0, and its ablation is **catastrophic for induction**
   (+11.7…+20.9 ΔNLL — far above any single attention head, confirming the discovery engine's MLP0 finding).
2. **Gemma-2 is the exception — distributed COMPUTE:** no single MLP dominates (top generic only +0.84), yet the
   *whole-stack* ablation is the largest of all (+10.74). Its COMPUTE is spread across layers, not concentrated in
   a detokenizer. (A second cross-architecture split, alongside Gemma's missing sink in the operator catalog.)
3. **The whole-MLP-stack ablation is large in every model** — COMPUTE is load-bearing *everywhere*, unlike any
   single attention-op class (no named attention op damages generic LM; the MLP stack always does).

**GPT-2 deep characterization (harvested):** the COMPUTE vocabulary **is low-rank** (`mlp_catalog.py`: transform
participation 22 vs 1666 random — a small reused set of read→write templates, heavier-tailed than attention's ~5);
**MLPs carry the reconstruction coverage** (`mlp_ops.py`: MLP-only coverage +0.46 vs attention-only ≈0 — they
interact); head↔MLP composition edges exist in weight space. Per-**neuron** read→write idioms are GPT-2-only (the
cheap token-unembedding basis); the cross-model rows are per-**layer** causal profiles. `mlp_atlas.py`,
`runs/disassembly/operators/mlp_compute_summary.json`, [`docs/operators/mlp_compute.md`](operators/mlp_compute.md).

## Boundaries / risks

- **Recompile faithfulness is OOD-sensitive** (partial reconstructions are low-norm inputs to `lm_head`);
  use norm-preserving mean-ablation + a random-op-budget control (the lesson from the entanglement-tower
  ablations) so the curve measures structure, not norm artifacts.
- **Path-patching cost** scales with edges; the weight-space edge scorer must prune before patching.
- **The operand basis is the bottleneck** (the P2 lesson recurs): a wrong basis caps coverage regardless of
  the algorithm. Expect to co-design the basis (U_C / polygram) with the extractor.
- **Single-token-prediction framing** — a faithful decompilation of next-token logits is not a proof of the
  model's *mechanism* off-distribution; scope claims to the measured distribution.

## Dependencies (cross-repo)

`sae-forge` (recompile/verify — the metric), `polygram` (operand-dictionary geometry — the basis), `n-orca`
(emit the circuit DAG as a typed graph), `larql` (decompile→query→recompile as a queryable index). The
disassembly toolkit here (`scripts/disassembly/`, `scripts/gemma/`) is the front-end that produces the ops.
