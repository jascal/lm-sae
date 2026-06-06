# Disassembling GPT-2 attention as an instruction set

This documents the `lm-sae` disassembly thread: a pipeline that reads GPT-2's attention as a small,
reused **instruction set**, quantifies how completely a catalog of named operators explains the model's
attention, localizes the gaps, **causally validates** the named operators, and checks that the claims are
not artifacts of the test corpus.

The unifying thesis: GPT-2's computation is **legible in the right basis** — QK/OV in *feature/operand*
coordinates and a catalog of weight-grounded **idioms** — even though it is *not* legible as single
residual SAE features (the cov95 tax). Most of attention is positional/structural plumbing; the
content-carrying minority is largely named, and the named operators are causally load-bearing and
corpus-robust.

Caveat on novelty: the published literature has **no** von-Neumann analogy and **no** complete attention
op-catalog — only a piecemeal list of head types (induction, prev-token, IOI name-movers/S-inhibition,
copy-suppression, successor, greater-than). The closest formal framings are Elhage et al.'s QK/OV +
residual-bus picture, Weiss/Lindner's RASP/Tracr (an op set for what transformers *can* compute), and
Merrill's TC⁰ bound. This thread **assembles, quantifies, causally tests, and corpus-checks** such a
catalog; it does not reproduce an existing one.

## Pipeline

| stage | script | PR | what it produces |
|------|--------|----|------------------|
| Idiom library | `idiom_library_v2.py` | #2/#3 | 8 literature-validated idioms + coreference + the composed IOI chain |
| Coverage scorecard | `coverage_scorecard.py` | #2 | "% of attention the catalog explains" + the dark-head work-list |
| SAE-feature operands | `sae_opcode_table.py` | #2 | richer operand basis; resolves dark heads token-identity misses |
| Causal validation | `causal_validation.py`, `ioi_causal.py` | #4 | mean-ablation: are the named heads *load-bearing*? |
| Corpus robustness | `corpus_robustness.py` | #5 | which claims are corpus-invariant vs corpus-conditioned |

Run order (GPT-2, CPU): idiom library → opcode tables → scorecard (it consumes the idiom/opcode summaries)
→ causal validation → corpus robustness. SAEs for the SAE-operand table download per layer on demand.

## The idiom catalog (weight-grounded, literature-validated)

`idiom_library_v2.py` recovers, from the **weights** (one forward pass only for the behavioral signatures),
**8/8** idioms with published GPT-2-small head sets, plus 2 exploratory and a composed circuit:

| idiom | recovered heads | how read |
|------|-----------------|----------|
| prev-token | 4.11 | Δ=1 attention |
| induction | 5.0/5.5/6.9/7.11 | token-after-prev-occurrence attention |
| duplicate-token | 0.1/0.5/1.5/3.0 | same-token-earlier attention |
| copy / name-mover | 9.6/9.9/10.0/10.10 | OV→unembed diagonal dominance (MLP0-extended, NAME operands, late band) |
| backup name-mover | 10.2/10.6 | name-copy late, minus primaries |
| negative name-mover | 10.7/11.10 | most-negative name-copy, late |
| copy-suppression | 10.7 | most-negative OV→unembed diagonal |
| S-inhibition | 7.3/8.6/8.10 (canonical) | Q-composition into name-movers |
| coreference *(exploratory)* | — | pronoun → earlier number/gender-compatible pronoun |
| succession / greater-than *(exploratory)* | — | OV off-by-one / boost-greater over ordinals/numbers |

The composed **IOI chain** `3.0 → 8.10 → 9.6/10.0/9.9` (duplicate → S-inhibition → name-mover) is read
straight from the weights as a product of Q-composition scores.

## Coverage: what fraction of attention is explained?

`coverage_scorecard.py` priority-buckets every head's attention mass and credits the long-range (content)
mass via three channels: a validated idiom, a token-operand legible binding, or an SAE-feature legible
binding. The split is corpus-conditioned, so we report **both** a verse and a prose baseline:

| bucket | Shakespeare (verse) | WikiText (prose) |
|--------|---------------------|------------------|
| sink (no-op) | 45.5% | 45.3% |
| self | 8.4% | 8.8% |
| prev | 8.5% | 8.8% |
| structural | 11.1% | 4.8% |
| local | 12.7% | 15.5% |
| **long-range (content)** | **13.7%** | **16.9%** |

On prose (the general-text baseline), of the 16.9% content mass: **22% named by a validated idiom,
72% token-operand legible, ~5% only-SAE-legible, ~2% genuinely dark.** The lone persistently-dark head is
**1.2** (anti-legible, diffuse). The headline: most of GPT-2's attention is **plumbing** (sink ~45%); the
content-carrying minority is largely legible, and a growing share is *named*.

SAE-feature operands (`sae_opcode_table.py`) resolve dark heads token-identity can't — e.g. head **9.8**
(`function-words → proper-name fragments`) — and surfaced the coreference idiom (`9.0: _their → _they`).
They give **interpretable** content opcodes, not a higher legible count.

## Causal validation: are the named heads load-bearing?

Mean-ablate a named idiom's heads; measure the damage to its **own** metric vs the complement vs
layer-matched random heads.

- **Induction-NLL** (`causal_validation.py`): ablating induction heads raises induction-predictable NLL by
  **+0.256 = 36% of baseline, z=8.6**; prev-token z=2.5 (composition). Negative controls (copy-suppression,
  succession) show no induction-specific damage.
- **IOI logit-diff** (`ioi_causal.py`, synthetic data → corpus-independent): **negative name-movers
  10.7/11.10 → ΔLD +2.09, z=15–62** (they write *against* the IO); **S-inhibition (canonical 7.3/8.6/8.10)
  → ΔLD −2.30, z=−3.5**.

Two transferable lessons:
1. **Causal validation is metric-specific.** A behavioral *name* is necessary but not sufficient — ablate
   against the metric the idiom serves. The induction and IOI circuits **share upstream subject-detection**
   and diverge only at the task-specific output heads, so the dissociation is partial.
2. **Causal validation audits the catalog.** It caught that the idiom library mis-named S-inhibition (its
   weak Q-composition score surfaced 10.0/6.7; the canonical 7.3/8.6/8.10 are the causal ones), and that
   name-movers are backup-redundant (fragile under single-set ablation).

## Corpus robustness: invariant vs conditioned

`corpus_robustness.py` re-runs the corpus-dependent measurements on Shakespeare (verse) and WikiText-2
(prose) with a **shared** operand set (tokens frequent in both):

- **Corpus-INVARIANT** (mean Spearman 0.84): head identities — prev-token **0.99**, duplicate 0.85,
  induction 0.73 — and per-head **QK opcode legibility 0.81** (38 shared operands). The literature heads
  replicate across verse and prose → weight-grounded, not corpus artifacts. (The IOI causal results are
  corpus-independent by construction — synthetic data.)
- **Corpus-CONDITIONED**: the coverage percentages. Verse inflates `structural` ~2× (11.1% → 5.0%) and
  deflates `content` (13.8% → 16.8% on the robustness run; agrees with the scorecard table above to ~0.2pp).
  The qualitative conclusions hold; the magnitudes are corpus-specific.

Method note: the opcode-legibility cross-corpus estimate is operand-count-sensitive (~9 shared operands →
noisy 0.43–0.75; 38 → stable 0.81); it needs ~20k tokens/corpus for enough shared high-frequency tokens.

## Boundaries (honest)

- Coverage **magnitudes** are corpus-conditioned (use the prose baseline for general text).
- Causal claims are **metric-specific** — confirmed on the metric each idiom serves, not universally.
- `coreference` overlaps `duplicate_token` (the dup mechanism applied to pronouns), and its SAE *weight*
  binding (9.0) diverges from its raw *attention* signal — weight-binding ≠ attention.
- Single base model (GPT-2-small); SAE-operand table covers layers 1/4/9; greater-than is MLP-dominated
  (the OV probe sees only the attention-side shadow).

## Cross-model: Gemma-2-2B (parity)

The whole framework **ports to a recent RoPE / GQA / RMSNorm model** (`scripts/gemma/`), at matched detail.
Architecture handling: GQA (query head `h` → kv head `h//(H/n_kv)`); content opcode
`M_h = W_Q^h⊤ W_K^{kv} / √query_pre_attn_scalar` (=√256); RMSNorm gain-fold (`1+weight`, no mean-subtraction);
and the **unrotated content-QK** (R₀) reading, which separates RoPE's positional axis from the content binding.
Operands at the SAE layer = Gemma Scope (`gemma-scope-2b-pt-res`, JumpReLU, width-16k) decoder directions; at
every layer = a universal per-layer **token-centroid** basis (the parity move — the same kind of basis GPT-2's
listing uses, computed low-rank so all 208 heads stay cheap).

| axis | GPT-2-small | Gemma-2-2B | invariant? |
|------|-------------|------------|------------|
| plumbing fraction (same Shakespeare corpus) | 86.7% | 87.7% | **yes (~87%)** |
| attention-sink | **45.6%** | **3.9%** | **no — sink is GPT-2-family-specific** |
| induction causal (mean-ablation, induction-NLL) | z = 8.6 | z = 8.3 | **yes (mechanism is causal in both)** |
| QK content-opcode legibility (SAE-feature coords) | most heads z>2 | 7/8 at L12; peaks mid-network | **yes (legible in the right basis)** |
| OV write (copy vs transform) | mostly transform | 52 copy / 156 transform | **yes** |

`disasm_portable.py` (behavioral + coverage on any HF model), `gemma_opcode_table.py` (QK opcode table with
Gemma Scope operands), `gemma_causal.py` (induction-NLL ablation), `gemma_layer_sweep.py` (legibility across
depth), and `disassemble_gemma.py` (the unified per-head listing at GPT-2 parity: all-layer QK token-bind + OV
WRITE + GeGLU MLP catalog + SAE-layer feature opcode). The one residual non-parity is intrinsic: GPT-2's named
*circuit* roles (IOI name-movers, S-inhibition) come from a published head-set with no Gemma equivalent, so
Gemma carries the behavioral idiom tags + causal flags rather than named-circuit tags. Conclusion: **mechanisms
and legibility are architecture-invariant; the *composition of the plumbing* (the attention-sink) is
architecture-specific.**

## Downstream hook (and a retraction)

The causally-validated, corpus-robust writer heads (induction 5.x/6.9/7.11, prev-token 4.11) were proposed as an
evidence-backed preserve-set for **writer-output `U_C`** in `sae-forge` (the two-basis forge). That circuit-
preservation claim was **RETRACTED**: the `excess = induction_kl − complement_kl` metric is gameable (a basis can
lower it by damaging the complement), and compression-controlled re-validation found writer-OV ≈ random-OV at
matched complement-KL (0/6 writer-wins across layers × seeds; `scripts/two_basis_forge/forge_revalidate_broad.py`).
The disassembly itself stands; the shortcut from "these are the writer heads" to "forging their OV-output subspace
preserves the circuit" does not.
