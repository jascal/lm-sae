# The oracle & forge-tax track (sister investigation)

`lm-sae` runs two complementary investigations on a frozen text LLM. The **disassembly →
decompilation** program — reading attention as an instruction set and cataloguing its operators and
circuits — is the headline and lives in [`../README.md`](../README.md) +
[`DECOMPILATION.md`](DECOMPILATION.md) + [`DISASSEMBLY.md`](DISASSEMBLY.md). **This** document holds the
*other* track: the **exact-lexical oracle** and the **cov95 forge tax** — the question of whether an SAE
*feature basis* can carry the model's computation, graded against a manufactured answer key (the same
instrument `bio-sae`/`econ-sae` run against Pfam / a stock-flow economy).

The two tracks share a thesis: **a language model is legible in the right basis even where it is *not*
legible as single SAE features.** The disassembly track shows *where* the computation lives (attention
ops/circuits); this track measures *what the SAE forge destroys* (monosemanticity / cov95) and *what you
must preserve rather than re-learn*.

## Results at a glance

| # | finding | key number | §|
|---|---------|-----------|--|
| 1 | The exact-lexical oracle is real; sharp/diffuse split reproduces `bio`'s shape | host cov95 0.64, token tier 0.89, lexical 0.11 | 1 |
| 2 | **The cov95 forge tax replicates on a language model** | cov95 0.65 → **0.12**; mAUC robust (0.93→0.85) | 2 |
| 2 | The tax is **emergent**, not over-completeness-driven | forged cov95 stays collapsed at *every* width incl. 1× | 2 |
| 2 | **Preserve-verbatim** is the lever (not concentrate / not retrain) | K≈32–64 of 512 atoms recovers host cov95 | 2 |
| 2 | Relations are **compiled**, not composed; label-free preserve-selection is **FALSIFIED** | relational-bigram single-cov95 = 1.0 | 2 |
| 3 | A model decomposes into a low-χ interpretable **core** + high-χ capable **tail**… | cov95 saturates at 3 levels; capability all in the tail | 3 |
| 3 | …and you **cannot train the entanglement away** (but supervision lifts native cov95 for free) | every recon-retrain raises the core; oracle-aux raises cov95 +0.155 at ~0 cost | 3 |
| 4 | The two-basis writer-output `U_C` circuit-preservation claim was **RETRACTED** | writer-OV ≈ random-OV at matched compression (0/6) | 4 |

## 1. The substrate & the oracle

`bio-sae`'s recipe, retargeted: *ESM-2 + Pfam-from-a-DB* → *GPT-2 + lexical-labels-from-a-rule*.
Per token we compute **deterministic** labels (no tagger, no noise), tiered sharp→diffuse like
Pfam/GO:

- **token** — "this token == ` the` / ` of` / …" (one-token detectors; the sharp tier)
- **lexical** — capitalization / punctuation / digit / length buckets
- **struct** — word-boundary / newline

`cov95` asks: does a single SAE latent detect each known feature at AUC ≥ 0.95?

| metric (GPT-2 resid) | self-trained SAE | **SAELens** dictionary |
|---|---|---|
| host cov95 (all) | 0.607 | **0.643** |
| token tier | 0.89 | 0.89 (mAUC 0.98) |
| lexical tier | 0.00 | **0.11** (mAUC 0.79) |
| host mAUC | 0.874 | 0.918 |

A real dictionary partly recovers the lexical tier (0.00 → 0.11) — so that tier is genuinely
*diffuse*, not a training artifact. Scripts: `common/build_lm_bundle.py`,
`common/forge_cov_mechanism.py`, `substrate/sae_lens_eval.py`. For the *forge* loop (below) a
CPU-feasible **tiny GPT-2** (`n_embd=128`, 4 layers, 7.2M params) is trained from scratch
(`substrate/train_tiny_gpt.py`), with the SAE on its final layer so the forged residual is
directly decodable.

## 2. The cov95 forge tax on a language model

**"Forging"** (program-specific term) = re-expressing a trained model's weights so its residual
stream is written in a fixed SAE feature basis, producing a runnable model whose computation
happens *in feature coordinates* (via sae-forge's `native_in_basis`). It asks: can the SAE basis
carry the model's actual computation, not just label its activations?

Forging an SAE basis into the model **preserves mAUC but collapses cov95** —
monosemanticity, not accuracy, is what the forge taxes. On the tiny GPT
(`cov95_forge_tax/whole_loop_tiny.py`):

| | cov95 (all) | token tier | mAUC |
|---|---|---|---|
| host | 0.654 | 0.94 | 0.930 |
| **forged** | **0.115** | **0.18** | 0.849 |

Canonical signature: **mAUC robust (91% retained), cov95 collapses (~18% retained), sharp
one-token detectors hit hardest.**

**The tax is emergent, not over-completeness-driven** (`width_sweep_tiny.py`). Sweeping SAE
width 1×–16× over-complete, forged cov95 stays collapsed at **every** width — including 1×
(no over-completeness at all). So over-completeness is exonerated; the tax is a property of
the *deep-transformer forward pass*. This is `bio`'s "emergent" regime, **not** `econ`'s
rank/over-completeness regime — and real LLMs are deep transformers, so the LM target sits in
the emergent regime, where the lever is **preserve-verbatim**, not concentrate.

**Preserve-verbatim recovers the tax, constructively** (`common/preserve_hybrid_tiny.py`):
keep the top-K oracle-reading atoms verbatim + forge the rest.

| K verbatim (of 512) | 0 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| combined cov95 | 0.12 | 0.46 | **0.62** | **0.65** | 0.65 |
| token tier | 0.18 | 0.71 | 0.94 | 0.94 | 0.94 |

Preserving **K≈32–64 atoms (6–12% of the basis)** fully recovers host cov95 — the LM analog
of `bio`'s P1 knee.

Two sharpening negatives:
- **Label-free selection is FALSIFIED** (`residual_selector_tiny.py`). The hoped-for
  shortcut — "preserve the atoms fine-tuning can't recover" — is *anti*-informative (overlap
  0.00 with the oracle's top-64). Reconstruction-hardness is orthogonal to detector-value, so
  preserve-selection genuinely needs a **value** signal (labels / downstream importance).
- **Relations are compiled, not composed** (`pair_cov95_tiny.py`). Bilinear *pair* detectors
  read no relational signal a single latent can't: even strict relational bigrams have
  single-latent cov95 = 1.0. The model **compiles** frequent inference into dedicated unary
  features (a JIT memoizing hot paths); only novel/un-compiled composition stays high-χ in the
  entangled core. So χ tracks **compilation/novelty, not logical arity** — a *static* oracle
  can't probe the core's inference.

## 3. The entanglement tower (M0…Mn)

"Harvest the cleanest features, subtract, repeat" → an additive tower `X ≈ M0 + M1 + … + core`
where χ (monosemanticity vs the oracle) falls and variance tapers across levels
(`entanglement_tower/mps_tower_tiny.py`). Three predictions hold: a real entanglement
**taper** (χ 0.99→0.77), a graceful **dial** (cov95 saturates at 3 levels while fidelity keeps
climbing), and **convergence** to an irreducible ~24% entangled core.

The retrain experiments prove there is **no shortcut**:
- complement-routing retrain **backfires** — re-entangles the core (0.24 → 0.75)
  (`mps_tower_retrain_tiny.py`);
- geometry-forcing retrain is better but a **no-go** — it still can't drive the core below the
  original, because *training toward capability inherently increases entanglement*
  (`mps_tower_geoforce_tiny.py`).

…but those retrains used a **reconstruction** bottleneck — the mAUC axis that already survives. On the *right*
axis, the no-go lifts: training from scratch with an **auxiliary oracle-feature-recovery loss** raises native
cov95 at **every** host width (+0.07…+0.24, mean +0.155) at **zero/negative capability cost** — interpretable,
equally-capable solutions are *reachable* via supervision, with the substrates as training signal
(`cov95_forge_tax/host_width_sweep.py`, `monosemantic_aux.py`, `legibility_crosscheck.py`,
`oracle_supervised_dag.py`; see [`DECOMPILATION.md`](DECOMPILATION.md)).

Serving the tower (`serve_tower_tiny.py`) sharpens the frontier: the **interpretability** dial
works (cov95 saturates at ~4 levels) but the **capability** dial is a cliff — the low-χ levels
are predictively *inert*; by `lm_head` linearity the entangled **core alone** predicts as well
as the full model. Clean features = the **substrate** (*what* the model reads); the core = the
**composition** (*how* it predicts). Capability is irreducibly entangled — so the right
response is to *decompose and choose a truncation*, not to train the entanglement away.

## 4. The two-basis forge — and a retraction

The forge tax motivated a **two-basis forge**: `U_A` (assertion → preserves cov95) + `U_C`
(composition → meant to preserve circuits). A specific `U_C` construction — the orthonormalised
union of circuit *writer heads'* OV-output rowspace ("writer-output `U_C`") — was tested here
and **RETRACTED**.

The original metric `excess = induction_kl − complement_kl` is **gameable**: a basis can lower
"excess" by *damaging the complement*, not by preserving the circuit. Compression-controlled
re-validation (`two_basis_forge/forge_compression_controlled.py`) showed writer-OV ≈ random-OV
at matched complement-KL and never below the recon-only baseline; the broadened re-run across
layers × seeds (`forge_revalidate_broad.py`) confirmed **0/6 writer-wins**. The claim is retired
in the `sae-forge` docs + a runtime warning. Kept as an honest negative: the preserve-verbatim
lever (§2) stands; the writer-output circuit-preservation shortcut does not. (`U_A` assertion-preserve
is a separate, surviving result — it replicates the §2 preserve lever through the production pipeline.)

## Scripts

`scripts/common/` (oracle + cov95 instrument), `scripts/substrate/` (tiny GPT + SAELens eval),
`scripts/cov95_forge_tax/` (§2 + reachability), `scripts/entanglement_tower/` (§3),
`scripts/two_basis_forge/` (§4). See [`../scripts/README.md`](../scripts/README.md) for the per-script guide.
