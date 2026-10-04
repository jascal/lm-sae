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

## Results

### gpt2 (layers [3, 6, 9], primary 6)

**LIST**: test strata n_w = 0/1/2/≥3: [3933, 1758, 279, 30].

| layer | arm | n_w = 0 | n_w = 1 | n_w = 2 | n_w ≥ 3 |
|---|---|---|---|---|---|
| 3 | real | 0.551 (n=3933) | 0.439 (n=1758) | 0.308 (n=279) | 0.300 (n=30) |
| 3 | tpr (λ=0.0001, 3-seed mean) | 0.338 (n=3933) | 0.133 (n=1758) | 0.048 (n=279) | 0.000 (n=30) |
| 3 | atomic | 0.580 (n=3933) | 0.003 (n=1758) | 0.000 (n=279) | 0.000 (n=30) |
| 6* | real | 0.613 (n=3933) | 0.531 (n=1758) | 0.419 (n=279) | 0.367 (n=30) |
| 6* | tpr (λ=0, 3-seed mean) | 0.403 (n=3933) | 0.157 (n=1758) | 0.054 (n=279) | 0.000 (n=30) |
| 6* | atomic | 0.643 (n=3933) | 0.002 (n=1758) | 0.000 (n=279) | 0.000 (n=30) |
| 9 | real | 0.637 (n=3933) | 0.532 (n=1758) | 0.444 (n=279) | 0.333 (n=30) |
| 9 | tpr (λ=0, 3-seed mean) | 0.364 (n=3933) | 0.120 (n=1758) | 0.033 (n=279) | 0.000 (n=30) |
| 9 | atomic | 0.656 (n=3933) | 0.000 (n=1758) | 0.000 (n=279) | 0.000 (n=30) |

- **P2-strict** (primary layer): n_w=0 0.403, n_w≥1 0.140 → **FAIL**
- **P2-paper**: n_w=2: 15/279 vs 1/2! = 0.500, p = 1; n_w=3: 0/30 vs 1/3! = 0.167, p = 1 → **FAIL**
- TPR seed range by stratum: {'0': [0.40427154302597046, 0.40401729941368103, 0.40071192383766174], '1': [0.15813423693180084, 0.15813423693180084, 0.1535836160182953], '2': [0.05376344174146652, 0.05376344174146652, 0.05376344174146652], '3': [0.0, 0.0, 0.0]}; λ selection (seen-pair validation): {'0.0': 0.4387291967868805, '0.0001': 0.42813917994499207, '0.001': 0.40544629096984863, '0.01': 0.2950075566768646}

| SAE width | eligible | conjunctive | permutation null | planted additive | ReLU-additive (diag.) |
|---|---:|---:|---:|---:|---:|
| 1024 | 887 | 0.897 | 0.000 | 0.000 | 0.000 |
| 8192 | 1397 | 0.849 | 0.000 | 0.000 | 0.001 |

- **P4′**: specific=True, passes=True, refuting=False → **PASS**

**SVO**: test strata n_w = 0/1/2/≥3: [2264, 1477, 259, 0].

| layer | arm | n_w = 0 | n_w = 1 | n_w = 2 | n_w ≥ 3 |
|---|---|---|---|---|---|
| 3 | real | 0.998 (n=2264) | 1.000 (n=1477) | 1.000 (n=259) | — |
| 3 | tpr (λ=0, 3-seed mean) | 0.998 (n=2264) | 0.177 (n=1477) | 0.030 (n=259) | — |
| 3 | atomic | 0.998 (n=2264) | 0.000 (n=1477) | 0.000 (n=259) | — |
| 6* | real | 0.994 (n=2264) | 0.993 (n=1477) | 1.000 (n=259) | — |
| 6* | tpr (λ=0, 3-seed mean) | 0.996 (n=2264) | 0.197 (n=1477) | 0.046 (n=259) | — |
| 6* | atomic | 0.996 (n=2264) | 0.002 (n=1477) | 0.000 (n=259) | — |
| 9 | real | 0.974 (n=2264) | 0.983 (n=1477) | 0.988 (n=259) | — |
| 9 | tpr (λ=0, 3-seed mean) | 0.984 (n=2264) | 0.237 (n=1477) | 0.067 (n=259) | — |
| 9 | atomic | 0.984 (n=2264) | 0.003 (n=1477) | 0.000 (n=259) | — |

- **P2-strict** (primary layer): n_w=0 0.996, n_w≥1 0.175 → **FAIL**
- **P2-paper**: n_w=2: 12/259 vs 1/2! = 0.500, p = 1 → **FAIL**
- TPR seed range by stratum: {'0': [0.9960247278213501, 0.9960247278213501, 0.9960247278213501], '1': [0.13540960848331451, 0.13540960848331451, 0.32092079520225525], '2': [0.019305018708109856, 0.019305018708109856, 0.10038609802722931]}; λ selection (seen-pair validation): {'0.0': 0.9972972869873047, '0.0001': 0.9972972869873047, '0.001': 0.9972972869873047, '0.01': 0.9621621370315552}

| SAE width | eligible | conjunctive | permutation null | planted additive | ReLU-additive (diag.) |
|---|---:|---:|---:|---:|---:|
| 1024 | 764 | 0.669 | 0.001 | 0.000 | 0.021 |
| 8192 | 1038 | 0.732 | 0.006 | 0.000 | 0.026 |

- **P4′**: specific=True, passes=True, refuting=False → **PASS**

### Qwen/Qwen2.5-0.5B (layers [6, 12, 18], primary 12)

**LIST**: test strata n_w = 0/1/2/≥3: [3933, 1758, 279, 30].

| layer | arm | n_w = 0 | n_w = 1 | n_w = 2 | n_w ≥ 3 |
|---|---|---|---|---|---|
| 6 | real | 0.562 (n=3933) | 0.458 (n=1758) | 0.380 (n=279) | 0.233 (n=30) |
| 6 | tpr (λ=0.0001, 3-seed mean) | 0.332 (n=3933) | 0.115 (n=1758) | 0.019 (n=279) | 0.000 (n=30) |
| 6 | atomic | 0.595 (n=3933) | 0.001 (n=1758) | 0.000 (n=279) | 0.000 (n=30) |
| 12* | real | 0.638 (n=3933) | 0.530 (n=1758) | 0.462 (n=279) | 0.367 (n=30) |
| 12* | tpr (λ=0.0001, 3-seed mean) | 0.419 (n=3933) | 0.191 (n=1758) | 0.050 (n=279) | 0.000 (n=30) |
| 12* | atomic | 0.682 (n=3933) | 0.000 (n=1758) | 0.000 (n=279) | 0.000 (n=30) |
| 18 | real | 0.645 (n=3933) | 0.548 (n=1758) | 0.462 (n=279) | 0.367 (n=30) |
| 18 | tpr (λ=0, 3-seed mean) | 0.430 (n=3933) | 0.169 (n=1758) | 0.047 (n=279) | 0.011 (n=30) |
| 18 | atomic | 0.705 (n=3933) | 0.003 (n=1758) | 0.000 (n=279) | 0.000 (n=30) |

- **P2-strict** (primary layer): n_w=0 0.419, n_w≥1 0.169 → **FAIL**
- **P2-paper**: n_w=2: 14/279 vs 1/2! = 0.500, p = 1; n_w=3: 0/30 vs 1/3! = 0.167, p = 1 → **FAIL**
- TPR seed range by stratum: {'0': [0.4202898442745209, 0.4182557761669159, 0.4172387421131134], '1': [0.19567690789699554, 0.19340158998966217, 0.18430034816265106], '2': [0.05017921328544617, 0.05734767019748688, 0.04301075264811516], '3': [0.0, 0.0, 0.0]}; λ selection (seen-pair validation): {'0.0': 0.4447806477546692, '0.0001': 0.44780635833740234, '0.001': 0.4387291967868805, '0.01': 0.31013616919517517}

| SAE width | eligible | conjunctive | permutation null | planted additive | ReLU-additive (diag.) |
|---|---:|---:|---:|---:|---:|
| 1024 | 890 | 0.791 | 0.000 | 0.000 | 0.000 |
| 8192 | 1453 | 0.685 | 0.000 | 0.000 | 0.001 |

- **P4′**: specific=True, passes=True, refuting=False → **PASS**

**SVO**: test strata n_w = 0/1/2/≥3: [2264, 1477, 259, 0].

| layer | arm | n_w = 0 | n_w = 1 | n_w = 2 | n_w ≥ 3 |
|---|---|---|---|---|---|
| 6 | real | 0.988 (n=2264) | 0.993 (n=1477) | 0.985 (n=259) | — |
| 6 | tpr (λ=0, 3-seed mean) | 0.994 (n=2264) | 0.185 (n=1477) | 0.013 (n=259) | — |
| 6 | atomic | 0.994 (n=2264) | 0.005 (n=1477) | 0.000 (n=259) | — |
| 12* | real | 0.991 (n=2264) | 0.990 (n=1477) | 0.996 (n=259) | — |
| 12* | tpr (λ=0, 3-seed mean) | 0.997 (n=2264) | 0.186 (n=1477) | 0.026 (n=259) | — |
| 12* | atomic | 0.997 (n=2264) | 0.003 (n=1477) | 0.000 (n=259) | — |
| 18 | real | 0.984 (n=2264) | 0.984 (n=1477) | 0.996 (n=259) | — |
| 18 | tpr (λ=0, 3-seed mean) | 0.998 (n=2264) | 0.272 (n=1477) | 0.046 (n=259) | — |
| 18 | atomic | 0.998 (n=2264) | 0.001 (n=1477) | 0.000 (n=259) | — |

- **P2-strict** (primary layer): n_w=0 0.997, n_w≥1 0.162 → **FAIL**
- **P2-paper**: n_w=2: 7/259 vs 1/2! = 0.500, p = 1 → **FAIL**
- TPR seed range by stratum: {'0': [0.9973497986793518, 0.9973497986793518, 0.9973497986793518], '1': [0.15098172426223755, 0.15098172426223755, 0.25592416524887085], '2': [0.0154440151527524, 0.0154440151527524, 0.04633204638957977]}; λ selection (seen-pair validation): {'0.0': 0.9945945739746094, '0.0001': 0.9945945739746094, '0.001': 0.9945945739746094, '0.01': 0.9891892075538635}

| SAE width | eligible | conjunctive | permutation null | planted additive | ReLU-additive (diag.) |
|---|---:|---:|---:|---:|---:|
| 1024 | 946 | 0.794 | 0.006 | 0.004 | 0.022 |
| 8192 | 1312 | 0.766 | 0.006 | 0.002 | 0.030 |

- **P4′**: specific=True, passes=True, refuting=False → **PASS**

### Verdicts (GPT-2 small, primary layer, by the pre-registered rules)

- Systematic binding: **refuted**.
- Conjunctive SAE latents: **established**.


## Reading (interpretation; not part of the pre-registered verdicts)

- **The information is present; the TPR composition is not.**
  - On contexts with withheld pairs, the unpacking decoder fed **real** encodings stays near its seen-pair level:
    GPT-2 L6 LIST 0.531 / 0.419 at n_w = 1 / 2 vs 0.613 at n_w = 0; SVO ≈ 1.0 throughout.
  - The L2,1 TPR, fit only on seen pairs, collapses there (LIST 0.157 / 0.054; SVO 0.197 / 0.046). At n_w = 2 it is
    an order of magnitude *below* the paper's 1/2! baseline.
  - So GPT-2 does encode the unseen (filler, role) pairs. But their encoding is **not** the bilinear composition
    `W(f ⊗ r)` of the filler and role vectors that the seen pairs determine. In the paper's terms, the
    representational test of systematic binding fails at this site, and the same holds for Qwen2.5-0.5B.
- **λ selection did not rescue P2.** The seen-pair validation chose λ ∈ {0, 1e-4} everywhere; larger λ only hurt.
  This was the pre-registered criterion. Choosing λ by withheld-pair accuracy would be post hoc and is not reported.
- **Conjunctive latents are robust.** Every model × family × width passes P4′ with the specificity control at ≈ 0.
  The ReLU-thresholded additive diagnostic stays ≤ 0.03, so sparse nonlinearity alone does not produce the effect.
  This replaces v1's failed P4 with a pre-registered pass.
- **Consistency.** Both findings fit one picture: GPT-2 small binds fillers to roles **conjunctively**, with
  pair-specific structure that SAEs mirror as conjunctive latents. That structure is not a systematic tensor-product
  code that generalizes to unseen pairs. Whether this causes the forge tax remains `open` (§4).

**Implementation notes.**
- The summaries are written per model, as `runs/tpr/systematicity_v2_{gpt2,Qwen2.5-0.5B}_summary.json`, rather than
  the single file named in §5.
- `--smoke` runs (stimulus seed 999, tiny sizes) were used only to find bugs, before the script was committed
  (`752ccab`). This pre-registration was pushed earlier (`615bc13`).
- Results render from the summaries with `scripts/tpr/report_systematicity_v2.py`.

