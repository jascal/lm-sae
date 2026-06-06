# lm-sae — a language-model ground-truth oracle, and a disassembler for what the SAE misses

`lm-sae` is one instrument in a broader research program whose other substrates
**manufacture** a known feature factorization — a dataset or model whose ground-truth features
are computed externally — so that SAE/forge recovery can be *graded* against an answer key. The
program's terminal target, though, is real LLMs — and a real LLM has **no oracle**: its
"features" are whatever the SAE finds, with nothing to check them against.

`lm-sae` fills that gap with two complementary instruments on a **frozen text LLM**:

1. **An exact, externally-computed feature oracle** — deterministic per-token labels (token
   identity, lexical class, structure) — so the cov95 / forge-tax / preserve instrument that
   `bio-sae` runs against Pfam can run against an *actual* language model.
2. **A disassembler** — because the headline finding is that an SAE *misses* most of the
   model's computation. The disassembler reads the model's attention as a small, reused
   **instruction set** in the right (QK/OV-in-operand) basis, scores how much of it a named
   op-catalog explains, **causally validates** the named operators, and shows the same
   reading **ports to a recent model** (Gemma-2-2B), at matched detail.

The throughline: **a language model is legible in the right basis even where it is *not*
legible as single SAE features.** Most of attention is positional plumbing; the
content-carrying minority is largely named, causally load-bearing, and corpus-robust — and
the part the SAE forge destroys (monosemanticity / cov95) is exactly the part you must
*preserve* rather than re-learn.

> **Status.** A CPU/single-GPU research MVP. GPT-2-small and a from-scratch tiny GPT for the
> CPU loops; Gemma-2-2B (bf16, one RTX 5050) for the cross-model port. Every table below is
> backed by a tracked `runs/*_summary.json` (see [`runs/README.md`](runs/README.md)).

---

## Results at a glance

| # | finding | key number | where |
|---|---------|-----------|-------|
| 1 | The exact-lexical oracle is real; sharp/diffuse split reproduces `bio`'s shape | host cov95 0.64, token tier 0.89, lexical 0.11 | §1 |
| 2 | **The cov95 forge tax replicates on a language model** | cov95 0.65 → **0.12**; mAUC robust (0.93→0.85) | §2 |
| 2 | The tax is **emergent**, not over-completeness-driven | forged cov95 stays collapsed at *every* width incl. 1× | §2 |
| 2 | **Preserve-verbatim** is the lever (not concentrate / not retrain) | K≈32–64 of 512 atoms recovers host cov95 | §2 |
| 2 | Relations are **compiled**, not composed; label-free preserve-selection is **FALSIFIED** | relational-bigram single-cov95 = 1.0 | §2 |
| 3 | A model decomposes into a low-χ interpretable **core** + high-χ capable **tail**… | cov95 saturates at 3 levels; capability all in the tail | §3 |
| 3 | …and you **cannot train the entanglement away** | every retrain raises the entangled core | §3 |
| 4 | GPT-2 attention is a reused **op-catalog**; 8/8 literature idioms recovered from weights | ~99% of content mass legible, ~2% dark | §4 |
| 4 | The named heads are **causally load-bearing** and **corpus-robust** | induction-NLL z=8.6; head identities ρ≈0.84 across corpora | §4 |
| 5 | The disassembler **ports whole to Gemma-2-2B** (RoPE/GQA/RMSNorm), at GPT-2 parity | induction-NLL z=8.3; 7/8 QK opcodes legible | §5 |
| 5 | Plumbing fraction is **model-invariant** (~87%); its *composition* is not | attention-sink 46% (GPT-2) vs 4% (Gemma) | §5 |
| 6 | The two-basis writer-output `U_C` circuit-preservation claim was **RETRACTED** | writer-OV ≈ random-OV at matched compression (0/6) | §6 |

---

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

Serving the tower (`serve_tower_tiny.py`) sharpens the frontier: the **interpretability** dial
works (cov95 saturates at ~4 levels) but the **capability** dial is a cliff — the low-χ levels
are predictively *inert*; by `lm_head` linearity the entangled **core alone** predicts as well
as the full model. Clean features = the **substrate** (*what* the model reads); the core = the
**composition** (*how* it predicts). Capability is irreducibly entangled — so the right
response is to *decompose and choose a truncation*, not to train the entanglement away.

## 4. Disassembly — GPT-2 attention as an instruction set

If the SAE misses the composition, read the composition directly. Full write-up:
[`docs/DISASSEMBLY.md`](docs/DISASSEMBLY.md). Pipeline (CPU): idiom library → opcode tables →
coverage scorecard → causal validation → corpus robustness.

- **Idiom catalog** (`disassembly/idiom_library_v2.py`): **8/8** literature idioms recovered
  from the *weights* (prev-token, induction, duplicate, copy/name-mover, backup & negative
  name-mover, copy-suppression, S-inhibition), plus the composed IOI chain read straight from
  Q-composition scores.
- **Coverage** (`coverage_scorecard.py`): most of attention is **plumbing** (attention-sink
  ~45% on GPT-2); of the content-carrying minority (~14% Shakespeare / ~17% prose), **~99% is
  legible** (named idiom / token-operand / SAE-feature binding) and only **~2% is genuinely
  dark** (the lone persistently-dark head is 1.2).
- **Causal validation**: ablating induction heads raises induction-NLL by **+0.256 = 36% of
  baseline, z=8.6** (`causal_validation.py`); on synthetic IOI, negative name-movers
  10.7/11.10 move the logit-diff with **z up to 62** (`ioi_causal.py`). Causal validation is
  **metric-specific** and **audits the catalog** (it caught a mis-named S-inhibition set).
- **Corpus robustness** (`corpus_robustness.py`): head identities are **corpus-invariant**
  (prev-token ρ=0.99, mean ρ≈0.84 across Shakespeare/WikiText); the coverage *percentages* are
  corpus-conditioned (report the prose baseline for general text).

## 5. Cross-model — the same disassembler on Gemma-2-2B

The whole framework **ports to a recent RoPE / GQA / RMSNorm model** (`gemma/`), at GPT-2
parity. The architecture handling: GQA (query head `h` → kv head `h//(H/n_kv)`), content
opcode `M_h = W_Q^h⊤ W_K^{kv} / √query_pre_attn_scalar`, RMSNorm gain-fold (`1+weight`, no
mean-sub), and reading the **unrotated** content-QK (R₀) so RoPE's positional axis is
separated from the content binding.

- **Behavioral + coverage** (`disasm_portable.py`, same Shakespeare corpus as GPT-2):

  | | GPT-2-small | Gemma-2-2B |
  |---|---|---|
  | plumbing fraction | 86.7% | 87.7% |
  | attention-sink | **45.6%** | **3.9%** |
  | content (long-range) | 13.3% | 12.3% |

  The **plumbing fraction is model-invariant (~87%)**, but its *composition* is not: the heavy
  attention-sink is **GPT-2-family-specific**, not universal — Gemma plumbs via self/local/prev
  instead. A clean cross-model dissociation.
- **Causal** (`gemma_causal.py`): ablating Gemma's induction heads raises induction-NLL at
  **z=8.3** — replicating GPT-2's z=8.6. The induction mechanism is causal in both.
- **Content opcodes** (`gemma_opcode_table.py`): **7/8** QK content bindings legible (z>2) at
  layer 12 in Gemma Scope feature coords; legibility **peaks mid-network**
  (`gemma_layer_sweep.py`).
- **Full disassembly at parity** (`disassemble_gemma.py`): all 208 heads get an addressing
  bucket + behavioral idiom tags + a **QK token-operand binding** + an **OV copy/transform
  WRITE** class (WRITE histogram: 156 transform / 52 copy), plus a per-layer **GeGLU MLP
  catalog** (read-tokens → write-tokens) and, at the SAE layer, the feature-native QK/OV
  opcode — the same fields GPT-2's listing carries.

**Why the two listings used to differ (now resolved).** GPT-2's listing decoded a QK/OV
binding for *every* head at *every* layer because its operand basis — token unembeddings /
per-layer token centroids — is **universal and cheap** (768-dim, tied embeddings, no RoPE).
Gemma's first port only decoded opcodes at the single Gemma-Scope SAE layer, because the
feature-operand basis is a *per-layer* SAE and the weight-space extraction is layer-specific
under RoPE/GQA/RMSNorm. Parity was reached by giving Gemma the **same universal token-centroid
operand basis at every layer** (computed the low-rank way so all 208 heads stay cheap) and
adding the GeGLU MLP catalog; the per-layer Gemma-Scope opcode remains as the richer
*feature-native* extra at the SAE layer (the analog of GPT-2's separate `sae_opcode_table`).
The one residual non-parity is intrinsic: GPT-2's named *circuit* roles (IOI name-movers,
S-inhibition) come from a published head-set that has no Gemma equivalent, so Gemma carries
the behavioral idiom tags + causal flags rather than named-circuit tags.

## 6. The two-basis forge — and a retraction

The forge tax motivated a **two-basis forge**: `U_A` (assertion → preserves cov95) + `U_C`
(composition → meant to preserve circuits). A specific `U_C` construction — the orthonormalised
union of circuit *writer heads'* OV-output rowspace ("writer-output `U_C`") — was tested here
and **RETRACTED**.

The original metric `excess = induction_kl − complement_kl` is **gameable**: a basis can lower
"excess" by *damaging the complement*, not by preserving the circuit. Compression-controlled
re-validation (`two_basis_forge/forge_compression_controlled.py`) showed writer-OV ≈ random-OV
at matched complement-KL and never below the recon-only baseline; the broadened re-run across
layers × seeds (`forge_revalidate_broad.py`) confirmed **0/6 writer-wins**. The claim is retired
in the `sae-forge` docs + a runtime warning. This section is kept as an honest negative: the
preserve-verbatim lever (§2) stands; the writer-output circuit-preservation shortcut does not.

---

## Repository map

```
lm-sae/
├── README.md            ← you are here (theory + results guide)
├── requirements.txt     ← standalone deps (sae-forge from PyPI, torch, transformers)
├── docs/
│   ├── DISASSEMBLY.md   ← the disassembly thread (GPT-2 + Gemma cross-model), in depth
│   ├── PAPER.md         ← self-contained kit for a short NEMI-workshop paper
│   └── listings/        ← committed full per-head disassembly listings (GPT-2 + Gemma)
├── scripts/             ← grouped by research thread — see scripts/README.md
│   ├── common/          ← shared substrate + the core cov95/forge instrument
│   ├── substrate/       ← the models under test (tiny GPT, SAELens eval)
│   ├── cov95_forge_tax/ ← §2
│   ├── entanglement_tower/ ← §3
│   ├── disassembly/     ← §4 (GPT-2 op-catalog)
│   ├── two_basis_forge/ ← §6
│   └── gemma/           ← §5 (cross-model port)
└── runs/                ← result artifacts; *_summary.json tracked — see runs/README.md
```

## How to run

Standalone — its own venv, `sae-forge` from PyPI, no bio-sae path:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
# GPU (RTX 50-series / Blackwell, sm_120): install the cu128 torch wheel
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# the cov95 instrument on GPT-2
.venv/bin/python scripts/common/build_lm_bundle.py
.venv/bin/python scripts/common/forge_cov_mechanism.py

# the whole forge loop on the tiny GPT (CPU)
.venv/bin/python scripts/substrate/train_tiny_gpt.py
.venv/bin/python scripts/cov95_forge_tax/whole_loop_tiny.py

# the GPT-2 disassembly pipeline (CPU)
.venv/bin/python scripts/disassembly/idiom_library_v2.py
.venv/bin/python scripts/disassembly/coverage_scorecard.py --corpus wikitext
.venv/bin/python scripts/disassembly/causal_validation.py
.venv/bin/python scripts/disassembly/disassemble_gpt2.py

# the cross-model port (needs the GPU)
.venv/bin/python scripts/gemma/disasm_portable.py --model google/gemma-2-2b
.venv/bin/python scripts/gemma/gemma_causal.py
.venv/bin/python scripts/gemma/disassemble_gemma.py
```

See [`scripts/README.md`](scripts/README.md) for the full per-group guide and run order.

## Honest caveats

1. **Partial oracle.** The lexical primitives cover a *slice* of an LLM's features; cov95 here
   measures "do known lexical primitives survive," not total interpretability.
2. **Small hosts.** GPT-2-small, a 7.2M tiny GPT, and Gemma-2-2B — not frontier scale. The tiny
   GPT compiles aggressively (a caveat on §2's "compiled relations"), and the forge over a
   24k-feature SAELens SAE hits an over-completeness wall on full GPT-2 (`forge_gpt2.py`, a
   negative control) — the faithful large-scale forge needs the polygram whole-loop.
3. **Coverage magnitudes are corpus-conditioned** (use the prose baseline for general text);
   **causal claims are metric-specific** (confirmed on the metric each idiom serves).
4. **First-order disassembly.** Single-component instructions + the induction idiom;
   superposition and the imperfect centroid/feature operand basis cap fidelity. The MLP catalog
   is a weight-only qualitative read.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
