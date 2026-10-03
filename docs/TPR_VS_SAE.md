# Binding vs atoms: a supervised TPR dictionary against flat SAEs on GPT-2 (`empirical`)

McCoy, Soulos, Linzen & Smolensky (2026, arXiv:2608.29530) report that LLM hidden states are closely approximated by
**linearly-transformed tensor product representations** (TPRs): `x ≈ W(Σᵢ fᵢ ⊗ rᵢ) + b`, with fillers `fᵢ` (tokens)
bound to roles `rᵢ` (structural positions). They argue against the atom picture behind SAEs:
- **2a, bag of atoms:** `x ≈ Σ e(filler)` cannot encode order.
- **2b, one atom per (filler, role) pair:** can encode order, but cannot generalize to unseen pairs.

This matters for lm-sae because it predicts the forge-tax pathologies in `THEORY_PROBLEM.md`. If representations bind
fillers to roles multiplicatively, a flat dictionary has to spend one latent per conjunction. Its width then grows
with `|fillers|×|roles|`, and its per-token active count stays high.

This note pre-registers a direct test on GPT-2 small, then reports it.

## Pre-registration (written 2026-10-03, before any run)

**Stimuli.**
- **LIST:** `Here is a list of words: w1, w2, …, wn.` with n ∈ {3,4,5}. The wᵢ are distinct single-token nouns.
- **SVO:** `The S V the O.` S and O are distinct single-token occupation nouns; V is a single-token past-tense verb.

**Site.** GPT-2 small residual stream (`hidden_states[ℓ+1]`, the output of block ℓ) at the final period, for
ℓ ∈ {3, 6, 9, 11}. The paper's period-encoding site. Activations are z-scored per dimension using statistics from the
dictionary-training split. Every dictionary is fit in that space.

**Roles.**
- LIST: bidirectional `(i, n−1−i)`, giving 12 roles.
- SVO: subject, verb, object.

**Dictionaries.**

| name | form | supervised? |
|---|---|---|
| `tpr` | `W(Σ f_{σ(s)} ⊗ r_s) + b`, d_F = 32, d_R = #roles | yes (role-filler labels) |
| `atomic` | `Σ e(filler, role) + b`: one atom per seen pair (2b) | yes |
| `additive` | `Σ e_F(filler) + Σ e_R(role) + b`: atoms for fillers and roles, no binding | yes |
| `bag` | `Σ e_F(filler) + b` (2a) | yes |
| `sae-m` | TopK SAE (k = 32), widths m ∈ {param-matched to `tpr`, 1024, 8192} | no |

**Withheld pairs** (the paper's §8 test). 10% of LIST (filler, role) pairs and 10% of SVO (noun, subject|object)
pairs are drawn at random before any fit. Every context containing one is removed from dictionary training. Test
contexts are stratified by the number of withheld pairs they contain: 0, 1, ≥2.

**Metrics.** All on held-out test contexts.
1. **Structure readout.** This is the paper's "period-unpacking" check, made linear. Per-role softmax readouts, plus a
   length head for LIST, are trained on *real* encodings from a separate readout split. Each dictionary's
   reconstruction is then fed to them. The score is exact-match of the whole structure. The reference is the
   readout's accuracy on real encodings.
2. **FVU** in z-space.
3. **Splice KL.** Write the reconstruction back at the period, at layer ℓ. Score
   `1 − KL(orig ‖ spliced) / KL(orig ‖ mean-spliced)` on the next-token distribution at the period. Pre-stated caveat:
   later tokens can read the list tokens directly, so this metric is weakly structure-dependent.

**Predictions (if the paper's claim transfers to GPT-2 small).**
- **P1.** `tpr` beats `additive` and `bag` on structure readout by ≥ 20 points at some layer.
- **P2.** On contexts with ≥ 1 withheld pair, `tpr` structure readout stays within 15 points of its 0-withheld score,
  while `atomic` falls by ≥ 30 points (it has no atom for the unseen pair).
- **P3.** The **parameter-matched** SAE scores below `tpr` on structure readout. The SAE needs a much larger width to
  reach `tpr`'s structure readout, if it reaches it at all.
- **P4.** Among active latents of the widest in-domain SAE, the majority are not additive in (filler, role). The
  latent's mean activation over the (filler, role) grid is poorly fit by a row-plus-column model (interaction
  R² share > 0.5); that is, they are conjunctive.

**Kill / reinterpret.**
- P1 fails at every layer: the paper's representational claim does not transfer to GPT-2 small at this site.
- P2 fails because `tpr` also collapses on withheld pairs: there is no systematic binding, and pairs are atoms.
- P3 or P4 fail: SAEs are not paying a conjunction tax here, and this experiment does not explain the forge tax.

**Scope.** A single model (GPT-2 small), two templated stimulus families, supervised role schemes. Results are
`empirical` for this site only. The *how* question (which heads write the roles) is out of scope.

**Amendment (after a 3k-context smoke test, before the full run).** On real LIST encodings the linear readout reached
only 1.1% exact match at that size, so exact match alone would be uninformative. A secondary metric is added:
**per-role filler accuracy** (mean over present roles), reported alongside exact match. No prediction or threshold
was changed.

## Results

Run 2026-10-03: GPT-2 small on an RTX 5050, ~25 min. Artifacts:
- `runs/tpr/tpr_vs_sae_summary.json` (all layers and dictionaries);
- `runs/tpr/latent_null_summary.json` (post-hoc control).

LIST has 100 nouns × 12 roles, 120 withheld pairs; split: readout 7,920 / dictionary 6,622 / test 6,000. SVO has 56
fillers × 3 roles, 8 withheld pairs; split: 5,280 / 5,456 / 4,000.

### Layer 6 (LIST peaks here on real encodings)

ro = exact-match structure readout; ra = per-role filler accuracy (secondary); KL rec. = splice KL recovered.

| dictionary | params | LIST ro | LIST ra | LIST FVU | SVO ro | SVO ra | SVO FVU |
|---|---:|---:|---:|---:|---:|---:|---:|
| real encodings | — | 0.351 | 0.765 | — | 1.000 | 1.000 | — |
| `tpr` | 299k / 76k | 0.151 | 0.601 | 0.210 | 0.827 | 0.940 | 0.148 |
| `atomic` | 830k / 68k | 0.242 | 0.697 | 0.410 | 0.810 | 0.934 | 0.141 |
| `additive` | 87k / 46k | 0.011 | 0.257 | 0.558 | 0.301 | 0.657 | 0.192 |
| `bag` | 78k / 44k | 0.009 | 0.258 | 0.575 | 0.301 | 0.657 | 0.192 |
| `sae` param-matched (195 / 50) | 300k / 78k | 0.109 | 0.558 | 0.066 | 0.406 | 0.788 | 0.117 |
| `sae-1024` | 1.57M | 0.241 | 0.689 | 0.045 | 0.864 | 0.953 | 0.034 |
| `sae-8192` | 12.6M | 0.227 | 0.682 | 0.048 | 0.849 | 0.948 | 0.032 |

Params are given as LIST / SVO where they differ. The layer pattern is the same at 3, 9 and 11.
- On LIST, real-encoding exact match is 0.18 / 0.35 / 0.27 / 0.18 at layers 3 / 6 / 9 / 11. A linear readout cannot
  read whole 3–5-word lists from the period.
- On SVO, real encodings read out perfectly at every layer.
- Splice KL recovered is 0.80–0.99 at layers 3–9 for every dictionary, and near 0 for all of them at layer 11. As
  pre-stated, it does not discriminate.

### Withheld pairs (per-role accuracy by number of withheld pairs in the context)

| | LIST L6: 0 / 1 / ≥2 | SVO L6: 0 / 1 / ≥2 |
|---|---|---|
| real | 0.782 / 0.740 / 0.701 | 1.000 / 1.000 / 1.000 |
| `tpr` | 0.641 / 0.539 / 0.465 | 1.000 / 0.699 / 0.426 |
| `atomic` | 0.773 / 0.584 / 0.416 | 1.000 / 0.667 / 0.352 |
| `sae-1024` | 0.711 / 0.656 / 0.602 | 1.000 / 0.765 / 0.519 |

For the withheld pair *itself*, the accuracy implied from these rows (seen pairs at their 0-withheld rate):
- `atomic`: ≈0% by construction;
- `tpr`: ≈10% (SVO) and ≈23% (LIST); chance is 1/56 and 1/100.

The real encodings decode those pairs fine. The SAE's better figures are not comparable: it reconstructs from the
activation itself, which contains the pair, while `tpr` and `atomic` see only the symbolic labels.

### Latent structure (P4)

The pre-registered statistic was the interaction share of cell means over the (filler, role) grid. It **failed its
post-hoc noise check**. Shuffling each latent's activations across contexts gives a *higher* median interaction share
on LIST (0.91) than the real latents (0.74–0.76). On a sparse grid, noise alone looks conjunctive, so the
pre-registered LIST "pass" (0.67–0.93 conjunctive) is an artifact.

On SVO the pre-registered statistic gives only 0.13–0.18 conjunctive, below the 0.5 threshold. On LIST its
apparent pass does not survive the null. **Pre-registered P4 therefore fails.**

A different test, a **split-half** comparison, was written *after* that failure. It is **not** a result: it is an
`open` follow-up, and its threshold and its definition of "conjunctive" were chosen after seeing the data. It needs
its own pre-registration before it can support anything. It compares half A's raw cell means with half A's additive
fit at predicting half B's cell means (layer 6, `runs/tpr/latent_null_summary.json`):

| SAE | LIST conjunctive (held out) | LIST shuffled | SVO conjunctive (held out) | SVO shuffled |
|---|---:|---:|---:|---:|
| 1024 | 0.886 | 0.000 | 0.676 | 0.000 |
| 8192 | 0.858 | 0.000 | 0.762 | 0.001 |

## Verdicts against the pre-registration

- **P1 — passes on SVO, fails on LIST exact match.**
  - SVO: `tpr` beats `bag`/`additive` by +51–53 points at every layer.
  - LIST exact match: the best gap is +15.4 points (L9), below the 20-point threshold, under a readout ceiling of
    0.18–0.35.
  - On the secondary per-role metric the LIST gap is +22–38 points.
  - Binding is necessary; additive atoms for fillers and roles are no better than a bag.
- **P2 — FAILS.**
  - `tpr` loses 30–57 points of exact match on contexts with withheld pairs.
  - It decodes an unseen pair only ≈10–23% of the time: above `atomic`'s 0% and chance, but far from systematic.
  - The paper's strong generalization does **not** reproduce for GPT-2 small at this site with this fit. The paper's
    L2,1 regularizer, which it reports helps generalization, was not pre-registered or used.
- **P3 — mostly holds.**
  - The parameter-matched SAE is below `tpr` on SVO at every layer (0.32–0.45 vs 0.81–0.83 exact match), and on
    LIST at L6 and L9.
  - It is roughly tied at L3 (0.067 vs 0.064) and above at L11 (0.085 vs 0.071).
  - The SAE reaches or exceeds `tpr` from width 1024 (≈5× / 20× the parameters).
  - `atomic`, at 2.8× `tpr`'s parameters on LIST, is the best symbolic dictionary in-distribution.
- **P4 — FAILS.**
  - The pre-registered interaction-share statistic is below threshold on SVO (0.13–0.18).
  - It scores *higher* under the permutation null than on real latents on LIST (0.91 vs 0.74–0.76), so it does not
    measure conjunction there.
  - The later split-half test (LIST 0.86–0.89, SVO 0.68–0.76, shuffled ≈0) is an `open` follow-up, not a repair of
    P4. It does not change this verdict.

## Reading for lm-sae

1. **The binding half of the paper's claim transfers; the systematicity half does not, here.**
   - A bilinear, role-bound dictionary is far more parameter-efficient than a flat SAE at carrying the *structure*:
     SVO at 76k parameters reaches 0.83 exact, where an SAE needs ~1.6M.
   - The nulls show binding is required.
   - But the fitted TPR barely generalizes to unseen pairs. GPT-2 small's period encodings behave more like
     "role-bound, partly pair-specific" than like a clean systematic TPR.
2. **A forge-tax mechanism is a hypothesis, not a result here (open).**
   - The evidence actually obtained is P3: the SAE matches the 76k-parameter TPR's structure readout only at ~20× the
     parameters.
   - The pre-registered test of *why* (P4: conjunctive latents, option 2b) **failed**.
   - The post-hoc split-half numbers suggest conjunctive latents, but they are an un-preregistered follow-up.
   - Two templated families on one model would not establish the forge-tax link even if P4 had passed.
3. **FVU is not the right score.** SAEs reconstruct far more variance (FVU 0.03–0.07 vs 0.15–0.3) while carrying
   *less* structure at matched size. A dictionary compared on FVU alone would get the ranking backwards.

## Next (not run; each would be a new pre-registration)

- Pre-register the split-half conjunction test (definition of "conjunctive", threshold, null, layers) and run it on
  fresh stimuli/seeds.
- Add the paper's L2,1 regularizer and a nonlinear unpacking decoder, then re-test P2.
- Factored SAE: an unsupervised bilinear dictionary (learned role × filler codes), compared with TopK at equal
  parameters on natural text, measuring the cov95 forge tax directly.
- The same test on a RoPE model (Qwen2.5-0.5B), for the decompilable-fraction family split.
