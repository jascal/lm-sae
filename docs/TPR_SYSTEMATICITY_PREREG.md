# Is GPT-2 small's binding systematic? — pre-registration (v2)

**Status: written 2026-10-03, committed and pushed before any code for this study was written or run.** This is the
follow-up to [`TPR_VS_SAE.md`](TPR_VS_SAE.md). That study found:
- **P2 fails:** a supervised TPR barely decodes withheld (filler, role) pairs.
- **P4 fails as pre-registered:** its interaction statistic did not measure conjunction.
- A post-hoc split-half test suggested conjunctive SAE latents.

This document re-tests both questions under one pre-registration:
- **(a)** P2 under the conditions of McCoy, Soulos, Linzen & Smolensky (2026, arXiv:2608.29530): an L2,1-regularised
  TPR, a nonlinear 6-layer unpacking decoder, and mid-layer period encodings.
- **(b)** The split-half conjunction test, with its definition, threshold and nulls fixed here.

Verdicts will be reported exactly against these rules. Anything not fixed here is **post hoc**: it is labelled as
such and never repairs a failed prediction.

## 1. Models, stimuli, sites

**Models.**
- `gpt2` (GPT-2 small, 12 layers, absolute positions). **Primary**: the question in the title is about this model.
- `Qwen/Qwen2.5-0.5B` (24 layers, RoPE). **Secondary**.
- Both are run frozen, in float32, from the Hugging Face cache.

**Fresh stimuli.** Same templates as v1, new data:
- **LIST:** `Here is a list of words: w1, …, wn.` with n ∈ {3,4,5} and distinct nouns.
  - Nouns come from a **new** candidate list disjoint from v1's `NOUNS`, defined in the script before any run.
  - The vocabulary is the first 80 candidates that are single-token (with a leading space) in **both** tokenizers.
  - 24,000 contexts.
- **SVO:** `The S V the O.` S ≠ O. Occupations and verbs are v1's lists, filtered to single-token in both tokenizers,
  capped at 40 and 16. All S-V-O combinations are shuffled and 16,000 taken.
- **Seeds:** stimulus seed 11, withheld-pair seed 111, split seed 12, fit seeds {0, 1, 2}, SAE seed 0, split-half
  seed 7. v1 used 0, 100 and 1.

**Site.** The residual stream at the final period, `hidden_states[ℓ+1]`.
- Layers at 25 / 50 / 75 % depth: GPT-2 {3, 6, 9}; Qwen {6, 12, 18}.
- **The 50 % layer is primary for every verdict.** The others are reported descriptively.
- Encodings are z-scored per dimension with statistics from the dictionary-training split.

**Roles.** LIST: bidirectional `(i, n−1−i)`, 12 roles. SVO: subject, verb, object.

**Withheld pairs** are drawn before any split or fit:
- **LIST:** 10 % of (noun, role) pairs.
- **SVO:** 25 % of (occupation, subject|object) pairs. This is raised from v1's 10 % so that strata with two or more
  withheld pairs have cases.
- Contexts containing any withheld pair are removed from TPR / atomic / SAE training. Test contexts are stratified by
  their number of withheld pairs `n_w ∈ {0, 1, 2, ≥3}`.

**Splits** (by context, split seed 12):
- 33 % decoder-training (real encodings only; withheld pairs allowed);
- 42 % dictionary-training (withheld contexts removed), of which 10 % is held out as **validation** for λ;
- 25 % test.

## 2. (a) P2 under the paper's conditions

**Unpacking decoder.** This follows the paper's period-unpacking model: a decoder-only Transformer with 6 layers,
hidden size 1,024, 16 heads, FF 4,096 and dropout 0.1.
- A linear layer maps the z-scored encoding to one prefix position. The decoder then autoregressively emits the
  filler sequence (LIST: the nouns then EOS; SVO: S, V, O then EOS) over a closed vocabulary of fillers + EOS.
- Training: real encodings of the decoder split only; AdamW, lr 3e-4, batch 256, 40 epochs, teacher forcing.
- Evaluation: greedy decoding. **Accuracy is exact match of the whole sequence.**
- One decoder per (model, family, layer). It is never trained on reconstructions.

**TPR.** `x̂ = W(Σ f_{σ(s)} ⊗ r_s) + b` with d_F = 32 and d_R = number of roles, trained by MSE on the dictionary
split. Adam, lr 3e-3, 3,000 full-batch steps.
- **L2,1 regulariser** (our operationalisation of the paper's App. L): `λ (Σ_a ‖f_a‖₂ + Σ_s ‖r_s‖₂)`, a group-lasso
  over the rows of the filler and role embedding matrices.
- λ ∈ {0, 1e-4, 1e-3, 1e-2}. It is **selected by exact-match unpacking accuracy on the seen-pair validation split**:
  no withheld pair is involved in selection.
- Three fit seeds. The verdict uses the **mean** over seeds; the range is reported.

**Comparators.**
- `atomic`: one atom per seen (filler, role) pair, zero for unseen pairs.
- `real`: the decoder on real test encodings, the ceiling.

**Strong baseline** (the paper's §8): `1/n_w!` per context. Known pairs are placed correctly; the `n_w` novel fillers
are placed at random.

**P2-strict (pass/fail, per family, GPT-2 primary layer).** TPR exact match on contexts with `n_w ≥ 1` is within
**0.15** of TPR exact match on `n_w = 0`.

**P2-paper (pass/fail, per family, GPT-2 primary layer).** For at least one stratum `n_w = m ≥ 2` with ≥ 30 test
contexts, TPR exact match exceeds `1/m!` (one-sided exact binomial, p < 0.01, using the seed-mean accuracy rounded
to counts).

**Verdict on "GPT-2 small's binding is systematic" (this site, this fit family):**
- **established** iff P2-strict passes for **both** LIST and SVO;
- **refuted** iff P2-paper fails for **both**;
- otherwise **partial / open**.

Qwen results are reported under the same rules as secondary evidence and do not change the GPT-2 verdict.

## 3. (b) The split-half conjunction test

**SAEs.** TopK, k = 32, widths {1,024, 8,192}. Trained on dictionary-split encodings at the primary layer; Adam,
lr 1e-3, batch 512, 4,000 steps, decoder columns unit-normed. They are analysed on dictionary + test contexts.

**Eligible latent.**
- Active (> 0) on ≥ 50 contexts.
- At least 10 (filler, role) cells with ≥ 3 contexts in **each** half.

**Halves.** Contexts are split uniformly at random, seed 7.

**Conjunctive (definition).** Compute per-cell mean activations on half A and half B.
- Fit a weighted additive model `grand + row_f + col_r` to A's cell means (weights = A counts; 50 backfitting
  iterations).
- Score both A's raw cell means and A's additive fit as predictors of B's cell means, by B-count-weighted squared
  error over the shared eligible cells.
- The latent is **conjunctive** iff the raw-cell error is strictly lower.

**Nulls and controls**, computed on the same contexts, halves and eligibility rules:
1. **Permutation null.** Each latent's activations are permuted across contexts.
2. **Planted-additive control (specificity).** For each eligible latent, fit
   `a_ctx ≈ c + Σ_{pairs p in ctx} (row_{f_p} + col_{r_p})` by least squares on all contexts. Synthesise
   `a′ = fitted + ε`, with `ε ~ N(0, σ²)` and `σ²` the residual variance. This is a latent that is additive by
   construction, with matched noise.
3. **Diagnostic only, not a gate.** The same planted latent passed through ReLU at the threshold that matches the
   real latent's activation frequency. This measures how much a sparse nonlinearity alone looks conjunctive.

**P4′ (pass/fail, per model × family, primary layer).** All three must hold, for **both** widths:
1. the conjunctive fraction among eligible latents is ≥ 0.50;
2. it exceeds the permutation-null fraction by ≥ 0.30;
3. the planted-additive control's conjunctive fraction is ≤ 0.10. If (3) fails, the test is non-specific and P4′ is
   **void**, not passed.

**Verdict on "conjunctive SAE latents" (GPT-2 primary):**
- **established** iff P4′ passes for both LIST and SVO;
- **refuted** iff (3) holds and the observed fraction is ≤ null + 0.10 for both;
- otherwise **open**, which includes void.

## 4. What is not claimed

- Any verdict is about this site, these templates and these fits. "Refuted" means the paper's systematicity test fails
  here, not that GPT-2 lacks systematic binding everywhere.
- Conjunctive latents, even if established, are not shown here to cause the forge tax.

## 5. Artifacts

- Script: `scripts/tpr/systematicity_v2.py`.
- Summary: `runs/tpr/systematicity_v2_summary.json`.
- Results: appended to this file under **Results**, below a line marking the end of the pre-registration. Nothing
  above that line changes after the first run.

---
*End of pre-registration.*
