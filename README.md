# lm-sae — the language-model ground-truth oracle substrate

The `*-sae` substrates (bio-sae, econ-sae, sm-sae) manufacture a **known feature
factorization** so an SAE/forge can be *graded* against it. The program's terminal
target is real LLMs — but a real LLM has **no oracle** (its "features" are whatever
the SAE finds, with no answer key). `lm-sae` fills that gap: a **frozen text LLM +
an exact, externally-computed feature oracle**, so the cov95 / forge-tax / preserve
instrument runs on the actual target.

## Recipe A (this MVP) — probed frozen LLM, exact-lexical oracle

The bio-sae recipe retargeted: *ESM-2 + Pfam-from-a-DB* → *GPT-2 + lexical-labels-
from-a-rule*. Per GPT-2 token we compute **deterministic** labels (no tagger, no
noise), tiered sharp→diffuse like bio's Pfam/GO:

- **token** — "this token == ' the' / ' of' / …" (one-token detectors; the sharp tier)
- **lexical** — capitalization / punctuation / digit / length buckets (sharp-ish)
- **struct** — word-boundary / newline (medium)

`cov95` then asks: does a single SAE latent on GPT-2's residual stream detect each
known feature at AUC ≥ 0.95?

## Status: MVP works; first numbers on a real LM

```
scripts/build_lm_bundle.py     # GPT-2 (cached) → layer-6 acts + exact-lexical Y → data/lm_bundle_gpt2.npz
scripts/forge_cov_mechanism.py # train TopK SAE → per-tier cov95/mAUC + N1 (rank/LN/TopK)
# run with bio-sae's venv (shared for the MVP): /home/allans/code/bio-sae/.venv/bin/python
```

First result (gpt2 layer 6, 16k tokens, self-trained TopK SAE w2048/k32):

| metric | value |
|---|---|
| host cov95 (all) | 0.607 |
| **token tier** (one-token detectors) | **0.89** |
| lexical tier | 0.00 |
| host mAUC | 0.874 |
| N1-rank | **sensitive** (rank-128/768 → 0.32; full → 0.61) |
| N1-LayerNorm | exonerated (0.607 ≈ host) |
| N1-TopK | exonerated (flat across k) |

The oracle is real and the sharp/diffuse split reproduces bio's shape.

### With a real SAELens dictionary (`scripts/sae_lens_eval.py`)

Swapping the self-trained SAE for a published **SAELens** SAE
(`jbloom/GPT2-Small-SAEs-Reformatted`, `blocks.8.hook_resid_pre`, 24576 feats,
layer 8) resolves the main caveat:

| metric | self-trained | **SAELens** |
|---|---|---|
| host cov95 (all) | 0.607 | **0.643** |
| token tier | 0.89 | 0.89 (mAUC 0.98) |
| lexical tier | **0.00** | **0.11** (mAUC 0.79) |
| host mAUC | 0.874 | 0.918 |

The real dictionary partly recovers the lexical tier (0.00 → 0.11) — so that tier is
genuinely *diffuse*, not just a training artifact. N1-rank stays **rank-sensitive**
with the real dictionary (token tier 71% of host at 17% rank-fraction, vs bio's Pfam
96% at 40%) — GPT-2's lexical/token features are high-dimensional, not low-rank
concentrated. (N1-LN drops 0.64→0.46, likely the raw-trained SAE's input-scale
sensitivity — a caveat, not a clean exoneration.)

### The forge (GPT2Adapter): machinery verified; faithful forge is GPU-scale

- The GPT-2 forge **runs end-to-end** (forge → next-token logits; `native_in_basis`).
- **Raw slices of the 24k SAELens SAE are numerically DEGENERATE** — meanKL(host‖forged)
  = 58 (N=256), **17245** (N=768), 15 (N=1536): garbage, non-monotonic. You cannot
  naively forge a 32×-over-complete basis (`scripts/forge_gpt2.py`, a negative
  control). This is the GPT-2 over-completeness wall.
- The repo's **polygram path works**: slice 64 → polygram-compress to **11 kept** →
  forge a 124M-param GPT-2 → faithfulness **KL 21.1** (`runs/forge_example_summary.json`).
  Sane KL — but 11 features is a heavily-compressed *smoke*, not a faithful forge.
- **Next (GPU-scale):** a faithful forge (hundreds–thousands of features + polygram
  tuning) and the **forged-cov95 tax** — hook the forged layer-8 residual, decode,
  re-score the lexical oracle. That's the lm-sae "whole loop" for text (the bio
  whole-loop analog); it needs a GPU, not this CPU MVP.

## Whole loop on a tiny trainable GPT (CPU) — the forge tax replicates on an LM

The GPT-2 + 24k-SAELens forge is GPU-scale (over-completeness wall). A **tiny
GPT-2-config model trained from scratch** (`n_embd=128, 4 layers, 7.2M params`, the
CPU-feasible nanochat stand-in; reuses the existing `GPT2Adapter`) makes the whole
loop tractable: train → SAE → **forge** → forged-cov95. SAE on the **final** layer so
the forged residual is directly decodable (`scripts/train_tiny_gpt.py`,
`scripts/whole_loop_tiny.py`).

| | cov95 (all) | token tier | lexical | mAUC |
|---|---|---|---|---|
| host | 0.654 | 0.94 | 0.11 | 0.930 |
| **forged** | **0.115** | **0.18** | 0.00 | 0.849 |
| **tax** | **0.65 → 0.12** | 0.94 → 0.18 | — | 0.93 → 0.85 |

**The cov95 forge tax replicates on a language model**, with the canonical
signature: **mAUC robust (91% retained), cov95 collapses (~18% retained), sharp
one-token detectors hit hardest.**

### N1-width on the LM (`scripts/width_sweep_tiny.py`): the tax is EMERGENT, not over-completeness-driven

Sweeping SAE width 1×–16× over-complete settles which regime the LM is in:

| over-complete | 1× | 2× | 4× | 8× | 16× |
|---|---|---|---|---|---|
| host cov95 | 0.615 | 0.615 | 0.654 | 0.692 | 0.692 |
| **forged cov95** | **0.00** | 0.04 | 0.12 | 0.04 | 0.15 |
| mAUC retained | 0.67 | 0.88 | 0.91 | 0.93 | 0.94 |

**Forged cov95 stays collapsed at every width — including 1× (no over-completeness
at all, where it's 0.00).** So over-completeness is **exonerated** for cov95; if
anything it mildly *helps* (via redundancy) and clearly helps mAUC retention (rises
0.67→0.94). ⇒ **the LM's cov95 tax is EMERGENT — bio's regime, not econ's.**

This **corrects the earlier guess** that a trainable host → econ's *concentrate*
regime. The regime is set by **host architecture, not trainability**: a *deep
transformer forward* (bio's ESM-2, this tiny GPT) → **emergent** tax → **preserve**
lever; econ's *shallow dense fc1/fc2 bridge* → rank/over-completeness tax →
concentrate. **Real LLMs are deep transformers, so the LM target is the
emergent/preserve regime** — preserve-verbatim is the lever, not concentrate.

### P1 on the LM (`scripts/preserve_hybrid_tiny.py`): preserve-verbatim recovers the tax

The width sweep said the LM is emergent → the lever is **preserve-verbatim**. This
confirms it constructively: keep the top-K oracle-reading SAE atoms **verbatim**
(host readout) + the rest forged, sweep K.

| K verbatim (of 512) | 0 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| combined cov95 | 0.12 | 0.31 | 0.46 | **0.62** | **0.65** | 0.65 |
| token tier | 0.18 | 0.47 | 0.71 | 0.94 | 0.94 | 0.94 |

Preserving **K≈32–64 atoms (6–12% of the basis) fully recovers host cov95** (0.12 →
0.65); the sharp one-token detectors snap back 0.18 → 0.94. **The lm-sae analog of
bio's P1 knee** (K≈160/1024 = 16%) — preserve-verbatim is the lever for the LM's
emergent cov95 tax, by construction, at small K.

### Label-free residual selector (`scripts/residual_selector_tiny.py`): FALSIFIED

The algorithm hoped the preserve set could be picked **label-free** from the
*post-training residual* — "the atoms fine-tuning can't recover are the ones to
preserve." It doesn't work. Comparing selectors by their preserve-hybrid cov95
recovery (real projection tax; diffuse = projection forge):

| selector | reaches host cov95 (0.68) at K≈ | overlap w/ oracle top-64 |
|---|---|---|
| **oracle** (host strength, labels) | **64** (12% of atoms) | 1.00 |
| random / norm / frag_proj (static P2 signals) | ~256–512 (50–100%) | 0.11–0.16 |
| **frag_train** (post-training residual) | **512** (100% — i.e. useless) | **0.00** |

The post-training residual is the **worst** selector — *anti*-informative (overlap
0.00). Reconstruction-residual ranks atoms by how hard they are to *reproduce*,
which is orthogonal-to-anti-correlated with how *valuable* they are as detectors
(the sharp detectors are high-variance, **easy** to reconstruct; the residual-hard
atoms are noise). ⇒ **preserve-selection genuinely needs a *value* signal (labels /
downstream importance), not reconstruction fidelity** — sharpening P2. The "what
training can't recover" shortcut is dead.

*(Secondary, caveated: fine-tuning the tiny forged model over-recovered cov95
0.16 → 0.60 — a same-data / tiny-model artifact where distillation nearly copies the
host; it does NOT generalize to a frozen, held-out LLM, where bio showed distillation
leaves cov95 floored. So preserve remains the lever for the real target.)*

### The M0…Mn entanglement tower (`scripts/mps_tower_tiny.py`) — the reverse algorithm

Phase 1 of the "harvest the cleanest features first, subtract, repeat" idea: build an
**additive tower** `X ≈ M0 + M1 + … + Mn` where each level is the next entanglement
band (χ-meter = monosemanticity against the oracle — *labels all the way*). On the
tiny LM (fixed-model v0), all three predictions hold:

| level | M0 | M1 | M2 | M3 | M4 | M5 | M6 | M7 |
|---|---|---|---|---|---|---|---|---|
| monosemanticity (χ) | 0.99 | 0.98 | 0.96 | 0.91 | 0.87 | 0.81 | 0.79 | 0.77 |
| variance captured | 0.36 | 0.14 | 0.07 | 0.09 | 0.06 | 0.03 | 0.01 | 0.01 |

- **Taper:** later levels are monotonically *more entangled* (χ ↑) and capture *less*
  variance — a real entanglement spectrum.
- **Dial (graceful truncation = MPS truncation):** keeping `M0..Mk`, **cov95
  (interpretability) saturates at 3 levels (0.46→0.65, token tier 0.94)** while
  **fidelity (capability) keeps climbing (0.36→0.76).** The low-χ core holds *all* the
  interpretability; the high-χ tail adds capability at **zero interpretability cost.**
  Truncation level is the capability↔interpretability dial, made smooth.
- **Convergence:** residual variance flattens 0.64 → … → **0.24** — "until stable"
  reaches a fixed point: an irreducible ~24% entangled core.

So most of the (tiny) LM's monosemantic content lives in a small low-χ core, with a
bounded entangled tail — the optimistic, falsifiable claim, here confirmed. (v0 is
fixed-model; the from-scratch *retrain-between-rounds* loop is the next step, and should
push the residual core lower.)

### Retrain-between-rounds (`scripts/mps_tower_retrain_tiny.py`) — the complement-routing version BACKFIRES

Phase 1b: after harvesting a level, freeze that subspace and fine-tune the model with
gradients routed only through the *complement* (re-express computation in the freed
capacity), then re-harvest. Then a fixed v0 decomposition of the adapted model:

```
entangled core:  original model 0.24  ->  adapted model 0.75   (3x WORSE)
```

It makes the model **less** forgeable. Diagnosis: freezing the clean directions and
forcing gradients into the complement teaches the model to predict from the *leftover
(entangled) directions* — i.e. it learns to **entangle**. **Freeing capacity by removing
forgeable features does not pressure toward forgeability; it pressures toward using
whatever's left, which is the messy subspace.**

So this disambiguates the algorithm: **"retrain until stable" needs explicit
*forgeability pressure* (geometry-forcing — train *through* the basis, the
`forge_aware_train_tiny.py` lever that *halved* the tax), not just freed capacity.** The
right Phase 1b is **harvest + geometry-forcing retrain**; harvest + complement-routing
re-entangles. (The fixed-model v0 tower — taper, dial, convergence — still stands; this
is about how to *improve* it.)

## Honest caveats (this is an MVP)

1. **Self-trained SAE, not SAELens.** `sae_lens` isn't installed here, so the SAE is
   self-trained (600 steps, 16k tokens) — token-identity-dominated (lexical tier =
   0). Swap in a **SAELens** production dictionary (the real plan) and the lexical
   tier should partly recover. This makes the **N1-rank result preliminary**: the
   host-side rank probe largely reflects which token atoms sit in the top-r by norm,
   an SAE-internal property, not GPT-2's intrinsic structure.
2. **Host-side probe, not the forge.** The *actual* forge tax (forged-vs-host cov95)
   and the **preserve hybrid (P1)** need the sae-forge `GPT2Adapter` — the immediate
   next step. This MVP is bio-sae's N1 *core* (host-side), nothing more.
3. **Partial oracle.** Lexical primitives cover a slice of GPT-2's features; cov95
   here measures "do known lexical primitives survive," not total interpretability.

## Next

- Swap the self-trained SAE for a **SAELens** GPT-2 resid SAE (sae-forge already
  ingests the format).
- Wire the **sae-forge `GPT2Adapter`** (`native_in_basis`) → forged cov95 tax + N1 on
  the forged path + **P1 preserve hybrid** → does the frozen-LLM target sit in the
  *preserve* regime?
- Add the **spaCy** syntactic/semantic tiers (POS/NER/dep) for a richer oracle.
- **Recipe B** (planted synthetic corpus + from-scratch nanochat LM) for a *perfect*
  oracle and the *concentrate* / trainable-host arm.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
