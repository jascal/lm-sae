# Paper kit — a short NEMI-workshop paper from this repo

This document is a **self-contained kit**: an agent with LaTeX, matplotlib, and web access
should be able to write a short (4–6 page) workshop paper from it without re-deriving anything.
Every quantitative claim points to the exact `runs/*_summary.json` it comes from (all tracked),
so figures and tables can be regenerated. Numbers below are the committed results.

The repo holds **two related contributions** that can be one paper or two:
- **(A) The cov95 forge tax + preserve-verbatim lever**, generalized from `bio-sae` to a
  language model (§Results A).
- **(B) Disassembling attention as a causally-validated, cross-model instruction set**
  (§Results B). For a NEMI submission, **(B) is the stronger standalone paper** (novel,
  self-contained, two models, causal); (A) is a strong "results" or appendix thread. Pick the
  framing in the abstract; both share §Methods and §Reproducibility.

---

## Suggested title / abstract (B-framing)

**Title:** *Attention is a small instruction set: a causally-validated, cross-model op-catalog
of what sparse autoencoders miss.*

**Abstract (draft).** Sparse autoencoders (SAEs) recover a model's *features* but not its
*composition*. We make this concrete by **re-expressing a model's weights so that its residual
stream is written in a fixed SAE feature basis** — yielding a runnable model that computes in
feature coordinates (we call this *forging* the model). Forging preserves predictive accuracy
(mean best-latent detection AUC) yet collapses single-latent monosemanticity (the fraction of
known features each detectable by one latent at AUC≥0.95): SAE features survive as readouts but
the computation over them does not factor through them. We argue the composition is nonetheless
legible in a *different* basis. Reading attention as QK/OV bilinears over an operand basis, we
assemble a catalog of weight-grounded **idioms** (induction, previous-token, duplicate,
name-mover families, copy-suppression, S-inhibition), quantify the fraction of attention it
explains, and show the named operators are **causally load-bearing** (mean-ablation;
induction-NLL z=8.6) and **corpus-robust** (head identities ρ≈0.84 across verse and prose). We
then port the entire reading to **Gemma-2-2B** (RoPE/GQA/RMSNorm) and find the mechanisms and
their legibility are **architecture-invariant**, while one piece of the plumbing — the
attention-sink, 46% of GPT-2's attention mass — is **GPT-2-family-specific** (≈4% in Gemma).
Most of attention is positional plumbing; the content-carrying minority is largely named,
causal, and shared across models.

> **Writing note — define the coined terms, do not assume them.** "Forge / forging / forge tax",
> "cov95", "the oracle", "operand basis", "preserve-verbatim", "U_A / U_C", and "χ" are
> **vocabulary internal to this research program**, not standard ML terms a NEMI reader will
> know. The abstract above introduces *forging* on first use; the paper must do the same for
> every coined term (or substitute a standard phrasing). The **Terminology** section below gives
> a one-line external-facing gloss for each — use it, and prefer the standard term where one
> exists (e.g. "monosemanticity recovery" over the bare metric name "cov95").

---

## Terminology (project-coined — define on first use, prefer standard phrasing)

These terms are internal to this research program. The right-hand column is the external-facing
gloss to use in the paper; **none of them should appear unexplained**.

| internal term | what to write for an external reader |
|---------------|--------------------------------------|
| **forge / forging** | re-express (project) a trained model's weights so its residual stream is written in a fixed SAE feature basis, producing a runnable model whose computation happens in feature coordinates. (Mechanism: project the host's read/write weights onto the SAE decoder directions; run with `forward_mode=native_in_basis`.) Not a standard term — define it the first time. |
| **forge tax** | the degradation caused by forging — specifically, the collapse of single-latent monosemanticity (cov95) even though predictive accuracy (mAUC) is largely preserved. Call it "the cost of forcing computation through the SAE basis," then optionally name it. |
| **cov95** | monosemanticity-recovery score = fraction of *known* (oracle) features for which **some single SAE latent** is a detector at AUC ≥ 0.95. Write "single-latent monosemanticity recovery (cov95)". |
| **mAUC** | mean over known features of the **best** single-latent detection AUC. Write "mean best-latent detection AUC (mAUC)". Standard-ish, but still gloss it. |
| **the oracle** | the exact, externally-computed ground-truth feature labels per token (token identity / lexical class / structure). Write "ground-truth feature labels" — the novelty is that they are *exact and external*, unlike a real LLM where features have no answer key. |
| **operand basis** | the set of directions (token unembeddings / per-layer token centroids, or SAE decoder rows) in which QK/OV bilinears are read; the "operands" the attention opcode binds. Gloss as "a basis of interpretable directions used as the coordinates for the QK/OV bilinear." |
| **idiom / op-catalog** | a named, weight-grounded attention behavior (induction, prev-token, …) and the catalog of them. Tie to the literature head-type names; "idiom" is our umbrella term. |
| **preserve-verbatim / preserve set / P1** | keep a chosen subset of SAE atoms as **exact host readouts** (not forged) while forging the rest. Write "verbatim-preserving a subset of features." |
| **U_A / U_C** | the two subspaces of the "two-basis forge": U_A preserves single-feature *assertions* (recovers cov95), U_C is meant to preserve *composition*/circuits. Only needed for the retraction (§6); define if used. |
| **χ (chi)** | a monosemanticity score of an entanglement "band" against the oracle (high χ = clean/monosemantic, low χ = entangled). Only needed for the entanglement-tower thread; gloss as "monosemanticity of a residual subspace." |
| **substrate** | within the program, a dataset/model with a *manufactured* known feature factorization (bio/econ/sm); for this paper just say "a model with ground-truth feature labels." |

For the **B-framing paper** (the op-catalog), only *forge / forge tax / cov95 / mAUC / operand
basis / idiom* are load-bearing; U_A·U_C and χ can be omitted entirely.

---

## Contributions (citable claims)

1. **The cov95 forge tax replicates on a language model**, with the canonical signature: mAUC
   robust, cov95 collapses, sharp one-token detectors hit hardest. *(A)*
2. **The tax is emergent, not over-completeness-driven** — collapsed at every SAE width
   including 1× — placing real LLMs in the "preserve-verbatim," not "concentrate," regime. *(A)*
3. **Preserve-verbatim is the lever**: keeping ~6–12% of atoms verbatim recovers host cov95;
   the selection provably needs a *value* signal (label-free selection is anti-informative). *(A)*
4. **A weight-grounded, causally-validated, corpus-robust op-catalog** for GPT-2 attention,
   with a coverage scorecard ("% of attention explained"). *(B)*
5. **The catalog ports to Gemma-2-2B at matched detail**; mechanisms/legibility are
   architecture-invariant, the attention-sink is architecture-specific. *(B)*
6. **An honest retraction**: a specific two-basis "writer-output `U_C`" circuit-preservation
   claim is falsified under compression control — a methodological caution about gameable
   circuit metrics. *(both)*

---

## Background & related work

The literature has a **piecemeal list of head types** but, to our knowledge, **no complete
attention op-catalog, coverage metric, or cross-model parity check** of the kind assembled here.
Anchor citations the writer should fetch and cite:

- **QK/OV + residual-stream framing**: Elhage et al., *A Mathematical Framework for Transformer
  Circuits* (Anthropic, 2021). Our opcode `B_h[X,Y] = d_X^⊤ M_h d_Y` is their QK circuit in an
  operand basis.
- **Induction heads**: Olsson et al., *In-context Learning and Induction Heads* (2022).
- **IOI circuit** (name-movers, S-inhibition, duplicate-token, backup/negative name-movers):
  Wang et al., *Interpretability in the Wild* (2023).
- **Copy-suppression**: McDougall et al. (2023).
- **Successor heads / greater-than**: Gould et al. (2023); Hanna et al. (2023).
- **Attention-sink**: Xiao et al., *Efficient Streaming LMs with Attention Sinks* (2023) — our
  cross-model result (sink is GPT-2-family-specific) is a direct, citable contrast.
- **SAEs / monosemanticity**: Bricken et al. (2023); Cunningham et al. (2023). **Gemma Scope**:
  Lieberum et al. (2024) — the JumpReLU SAE suite we use as the Gemma operand basis.
- **What transformers can compute (op-set framings)**: Weiss et al. RASP (2021); Lindner et al.
  Tracr (2023); Merrill & Sabharwal TC⁰ bounds. These bound *capability*; we *measure usage*.

Differentiator sentence: *prior work names individual head types; we assemble them into a single
catalog, measure the fraction of attention they explain, causally test each, check corpus
robustness, and show the reading transfers to a second architecture.*

---

## Methods (formulas + setup)

### Models & data
- **GPT-2-small** (124M; 12L×12H, d=768, tied embeddings, learned-positional, LayerNorm).
- **Tiny GPT-2** trained from scratch (`n_embd=128`, 4L, 7.2M) for the CPU forge loop.
- **Gemma-2-2B** (26L×8H, d=2304, head_dim=256, GQA n_kv=4, RoPE, RMSNorm, attn_logit_softcap=50).
- **SAEs**: self-trained TopK; published **SAELens** GPT-2 resid SAEs
  (`jbloom/GPT2-Small-SAEs-Reformatted`); **Gemma Scope** (`gemma-scope-2b-pt-res`, JumpReLU,
  width-16k) for Gemma operands.
- **Corpora**: tinyshakespeare (verse) and WikiText-2 (prose); the cross-model coverage
  comparison uses the **same** corpus (Shakespeare) for both models.

### The oracle and cov95 / mAUC
Per token, deterministic labels in tiers (token-identity / lexical / structural). For a known
feature, **cov95** = fraction of features for which *some single SAE latent* attains detection
AUC ≥ 0.95; **mAUC** = mean best-latent AUC. cov95 measures monosemanticity (one latent ⇒ one
known feature); mAUC measures accuracy. *(`common/build_lm_bundle.py`, `forge_cov_mechanism.py`)*

### The forge & the tax
Project the host's weights into the SAE basis (saeforge `SubspaceProjector` →
`NativeModel.from_projected_weights`, `forward_mode=native_in_basis`) to get a runnable
forged model; re-score the oracle on the forged residual. The **tax** = host cov95 − forged
cov95. **Preserve-verbatim (P1)**: keep the top-K oracle-reading atoms as exact host readouts,
forge the rest, sweep K. *(`cov95_forge_tax/`, `common/preserve_hybrid_tiny.py`)*

### Attention opcodes (the core of B)
For head `h`, operand directions `d_X` (token unembeddings / per-layer token centroids, or SAE
decoder rows), folded by the pre-attention norm gain and unit-normalized:
- **QK opcode** `B_h[X,Y] = d_X^⊤ M_h d_Y`, with `M_h = W_Q^{h⊤} W_K^h / √head_dim` (GPT-2) or
  `W_Q^{h⊤} W_K^{kv(h)} / √query_pre_attn_scalar` (Gemma, GQA). Top off-diagonal entry = the
  head's content binding `X→Y`.
- **OV write channel** `V_h[X,Y] = d_X^⊤ (W_O^h W_V^{h/kv}) d_Y`; diagonal dominance
  (`mean diag − mean off-diag`) classifies **copy** vs **transform**.
- **Legibility** = Spearman ρ between `B_h` and empirical attention over operand pairs, z-scored
  against an operand-label permutation null (z>2 = legible).
- **Addressing buckets** (Gemma): per-head attention mass split into self / sink / prev /
  structural / local / long-range → dominant mode.
- **Idioms**: behavioral signatures (prev-token = Δ=1; duplicate = same-token-earlier; induction
  = token-after-prev-occurrence), z>1.5 across heads; name-mover families from OV→unembed
  diagonal dominance; IOI chain from Q-composition products.

### Gemma architecture handling (the port)
GQA expansion; RMSNorm gain-fold (`1+weight`, no mean-subtraction); and reading the **unrotated**
content-QK (R₀) so RoPE's positional modulation is a separate axis from the content binding.
The MLP catalog uses the GeGLU gate_proj (read) / down_proj (write) projected onto operands.
*(`gemma/gemma_opcode_table.py`, `disassemble_gemma.py`)*

### Causal validation
**Mean-ablate** an idiom's heads (replace their output by its corpus mean) and measure damage to
*its own* metric vs the complement vs layer-matched random heads. Metrics: **induction-NLL**
(NLL on induction-predictable positions) and, on synthetic IOI templates (corpus-independent),
**logit-difference** (IO − S). *(`causal_validation.py`, `ioi_causal.py`)*

---

## Results A — the forge tax (numbers + sources)

| claim | numbers | source JSON |
|------|---------|-------------|
| oracle is real (host) | cov95 0.643 (SAELens), token 0.89, lexical 0.11, mAUC 0.918 | `sae_lens_eval_summary.json`, `cov_mechanism_summary.json` |
| **tax replicates** (tiny LM) | cov95 0.654→**0.115**; token 0.94→0.18; mAUC 0.930→0.849 | `whole_loop_tiny_summary.json` |
| **tax is emergent** | forged cov95 collapsed at every width (0.00 at 1×); mAUC retained 0.67→0.94 | `width_sweep_tiny_summary.json`, `wl_w*_summary.json` |
| **preserve-verbatim** | cov95 K=0→0.12, K=32→0.62, K=64→0.65; token 0.18→0.94 | `preserve_hybrid_tiny_summary.json` |
| label-free selection FALSIFIED | oracle reaches host at K≈64; post-training residual overlap 0.00 | `residual_selector_tiny_summary.json` |
| relations compiled | relational-bigram single-cov95 = 1.0 (no pair signal) | `pair_cov95_tiny_summary.json` |
| entanglement tower | χ 0.99→0.77; cov95 saturates at 3 levels; core converges to ~0.24 | `mps_tower_tiny_summary.json` |
| retrain no-go | complement-routing core 0.24→0.75; geometry-forcing 0.24→0.51 | `mps_tower_{retrain,geoforce}_tiny_summary.json` |
| serve: capability all in core | low-χ levels predictively inert; core alone ≈ full model | `serve_tower_tiny_summary.json` |

**Figure A1** (forge tax bar): host vs forged {cov95, token-tier, mAUC} from
`whole_loop_tiny_summary.json`. **Figure A2** (width sweep line): forged cov95 & mAUC-retained
vs over-completeness from `width_sweep_tiny_summary.json`. **Figure A3** (preserve knee): cov95 &
token-tier vs K from `preserve_hybrid_tiny_summary.json`.

## Results B — the op-catalog (numbers + sources)

| claim | numbers | source JSON |
|------|---------|-------------|
| idioms recovered from weights | **8/8** literature idioms + IOI chain | `idiom_library_v2_summary.json` |
| coverage (Shakespeare) | plumbing 86.3%, sink 45.5%, content 13.7%; of content ~99% legible, ~2% dark (head 1.2) | `coverage_scorecard_summary.json` |
| coverage (prose) | plumbing 83.1%, content 16.9%, ~22% named, 98.9% legible | `coverage_scorecard_wikitext_summary.json` |
| causal: induction | induction-NLL +0.256 = 36% of baseline, **z=8.6**; prev-token z=2.5 | `causal_validation_summary.json` |
| causal: IOI | baseline logit-diff 2.75 (acc 0.988, n=160); negative name-movers 10.7/11.10 **z=62**; induction z=−3.7 | `ioi_causal_summary.json` |
| corpus robustness | prev-token ρ=0.99, dup 0.84, induction 0.75 (mean ≈0.84); coverage %s corpus-conditioned | `corpus_robustness_summary.json` |

**Figure B1** (coverage stacked bar, verse vs prose): the 6 attention buckets + the content
split (named / token-legible / sae-only / dark) from the two `coverage_scorecard_*` JSONs.
**Figure B2** (causal): ΔNLL (induction set vs complement vs random) from
`causal_validation_summary.json`; ΔlogitDiff per idiom from `ioi_causal_summary.json`.
**Table B1** (idiom catalog): idiom → recovered heads → reading, from
`idiom_library_v2_summary.json` (and `docs/disassembly.md`).

## Results — cross-model parity (the headline of B)

| axis | GPT-2-small | Gemma-2-2B | source |
|------|-------------|------------|--------|
| plumbing fraction (Shakespeare) | 86.7% | 87.7% | `disasm_portable_summary.json`, `disasm_portable_gemma2_summary.json` |
| **attention-sink** | **45.6%** | **3.9%** | same |
| self / local / prev plumbing | 8 / 12 / 9% | 31 / 21 / 18% | same |
| induction causal (induction-NLL) | z=8.6 | z=8.3 | `causal_validation_summary.json`, `gemma_causal_summary.json` |
| QK content-opcode legibility | most heads z>2 | 7/8 at L12; peaks mid-network | `gemma_opcode_table_summary.json`, `gemma_layer_sweep_summary.json` |
| OV write copy/transform | mostly transform | 52 copy / 156 transform | `gemma2_disassembly.json` (write_hist) |

**Figure C1** (cross-model plumbing composition): grouped bars of the 6 buckets for both models
from the two `disasm_portable_*` JSONs — visually carries "sink is GPT-2-specific." **Figure C2**
(legibility vs depth): mean z and n-legible per layer from `gemma_layer_sweep_summary.json`.
The full per-head listings (`gpt2_disassembly.txt`, `gemma2_disassembly.txt`) are the qualitative
appendix; regenerate via `disassemble_{gpt2,gemma}.py` (git-ignored, not tracked).

## Limitations (state these explicitly)

1. **Partial oracle** — covers a slice of features; cov95 ≠ total interpretability.
2. **Small hosts** — GPT-2-small / 7.2M tiny GPT / Gemma-2-2B, not frontier scale; the tiny GPT
   compiles relations aggressively; full-GPT-2 forge over a 24k SAE hits an over-completeness
   wall (negative control).
3. **First-order disassembly** — single-component instructions + induction; superposition and the
   imperfect operand basis cap fidelity; MLP catalog is a weight-only qualitative read.
4. **Coverage magnitudes corpus-conditioned**; **causal claims metric-specific**.
5. **Retracted sub-result** (writer-output `U_C`) — included as a methodological caution, not a
   positive result.

## Reproducibility

- **Environment**: python 3.12; `sae-forge==0.14.0` (PyPI), `torch==2.11.0` (cu128 for the RTX
  5050 / Blackwell GPU used for Gemma), `transformers==5.10`, `numpy==1.26.4`. `requirements.txt`.
- **Run order**: see [`../scripts/README.md`](../scripts/README.md). Disassembly (CPU): idiom
  library → opcode tables → scorecard → causal → corpus robustness → `disassemble_gpt2.py`.
  Gemma (GPU): `disasm_portable.py` → `gemma_opcode_table.py` → `gemma_causal.py` →
  `gemma_layer_sweep.py` → `disassemble_gemma.py`.
- **Data behind every figure**: the tracked `runs/*_summary.json` listed above
  ([`../runs/README.md`](../runs/README.md)). Large listings are git-ignored and regenerated.

## Claims ledger (one line per quantitative claim → file)

```
host cov95 0.643 / token 0.89 / lexical 0.11        runs/sae_lens_eval_summary.json
forge tax cov95 0.654->0.115 (mAUC 0.93->0.85)      runs/whole_loop_tiny_summary.json
emergent: forged cov95 collapsed at 1x..16x         runs/width_sweep_tiny_summary.json
preserve knee K~32-64 recovers cov95                runs/preserve_hybrid_tiny_summary.json
label-free selection overlap 0.00 (falsified)       runs/residual_selector_tiny_summary.json
relational-bigram single-cov95 = 1.0                runs/pair_cov95_tiny_summary.json
tower chi 0.99->0.77; core ~0.24                     runs/mps_tower_tiny_summary.json
retrain no-go (0.24->0.75 / 0.24->0.51)             runs/mps_tower_{retrain,geoforce}_tiny_summary.json
8/8 idioms from weights                              runs/idiom_library_v2_summary.json
coverage: sink 45.5%, content 13.7%, ~2% dark       runs/coverage_scorecard_summary.json (+ _wikitext)
causal induction-NLL z=8.6                           runs/causal_validation_summary.json
IOI neg name-movers z=62                             runs/ioi_causal_summary.json
corpus-robust head identities (prev rho 0.99)       runs/corpus_robustness_summary.json
GPT-2 sink 45.6% vs Gemma 3.9% (same corpus)        runs/disasm_portable{,_gemma2}_summary.json
Gemma induction causal z=8.3                         runs/gemma_causal_summary.json
Gemma 7/8 QK opcodes legible at L12                  runs/gemma_opcode_table_summary.json
Gemma OV write 52 copy / 156 transform              runs/gemma2_disassembly.json (write_hist)
writer-output U_C RETRACTED (0/6)                    runs/forge_revalidate_broad_summary.json
```
