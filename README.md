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
one-token detectors hit hardest.** This is the trainable-host arm (econ's
*concentrate* regime); the SAE is 4× over-complete, so whether this tax is emergent
or over-completeness-driven is the next probe (sweep `--width` → N1-width on the LM).

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
