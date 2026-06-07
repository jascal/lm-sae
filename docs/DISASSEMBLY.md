# Disassembling attention as an instruction set

This documents the `lm-sae` disassembly thread: a pipeline that reads a transformer's attention as a
small, reused **instruction set**, quantifies how completely a catalog of named operators explains the
model's attention, localizes the gaps, **causally validates** the named operators, checks that the claims
are not artifacts of the test corpus, and **shows the same reading ports across architectures**.

The method is presented on **GPT-2-small** as the primary worked example (§Pipeline–§Corpus robustness),
then ported whole to **Gemma-2-2B** (RoPE/GQA/RMSNorm) in §Cross-model — where the headline is that the
mechanisms and their legibility are *architecture-invariant*, while one piece of the plumbing (the
attention-sink) is *architecture-specific*.

The unifying thesis: the model's computation is **legible in the right basis** — QK/OV in *feature/operand*
coordinates and a catalog of weight-grounded **idioms** — even though it is *not* legible as single
residual SAE features (the cov95 tax). Most of attention is positional/structural plumbing; the
content-carrying minority is largely named, and the named operators are causally load-bearing and
corpus-robust.

Caveat on novelty: the published literature has **no** von-Neumann analogy and **no** complete attention
op-catalog — only a piecemeal list of head types (induction, prev-token, IOI name-movers/S-inhibition,
copy-suppression, successor, greater-than). The closest formal framings are Elhage et al.'s QK/OV +
residual-bus picture, Weiss/Lindner's RASP/Tracr (an op set for what transformers *can* compute), and
Merrill's TC⁰ bound. This thread **assembles, quantifies, causally tests, corpus-checks, and
cross-model-replicates** such a catalog; it does not reproduce an existing one.

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

The **Gemma-2-2B** port (`scripts/gemma/`, needs a GPU) mirrors this pipeline with an arch-agnostic core:
`disasm_portable.py` (behavioral idioms + coverage on any HF model) → `gemma_opcode_table.py` (QK opcodes
in Gemma Scope feature coords) → `gemma_causal.py` (induction-NLL ablation) → `gemma_layer_sweep.py`
(legibility across depth) → `disassemble_gemma.py` (the unified per-head listing, at GPT-2 parity). See
§Cross-model.

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
- Four base models (GPT-2-small, Gemma-2-2B, Llama-3.2-1B, Qwen2.5-1.5B); the GPT-2 SAE-operand table covers
  layers 1/4/9; greater-than is MLP-dominated (the OV probe sees only the attention-side shadow). Named
  *circuit* roles (IOI name-movers, S-inhibition) are GPT-2-only (no published IOI head-set for the others) —
  the non-GPT-2 models carry the universal idioms + behavioral tags + causal flags. The feature-native SAE
  opcode is Gemma-only (Gemma Scope); Llama/Qwen use the universal token-operand bind.

## Cross-model: Gemma-2, Llama-3, Qwen-2.5

The whole framework **ports across the RoPE / GQA / RMSNorm / gated-MLP family** (`scripts/gemma/`, GPU). The
weight-space disassembler is **arch-generic**: the per-architecture constants — RMSNorm gain offset (Gemma's
zero-centered `1+weight` vs plain `weight` for Llama/Qwen), QK scale (`query_pre_attn_scalar` vs `√head_dim`),
and whether a per-layer feature SAE exists — live in `arch_config.py`, so one `--model` flag runs **Gemma-2-2B,
Llama-3.2-1B, and Qwen2.5-1.5B** (GPT-2 keeps its own `disassemble_gpt2.py`). **Shared handling**: GQA (query
head `h` → kv head `h//(H/n_kv)`); content opcode `M_h = W_Q^h⊤ W_K^{kv} / scale`; the **unrotated content-QK**
(R₀) reading that separates RoPE's positional axis from the content binding; a universal per-layer
**token-centroid** operand basis at every layer (low-rank, so all heads stay cheap), plus the feature-native
Gemma-Scope opcode at the SAE layer where a SAE exists (Gemma only; Llama/Qwen skip it). The Gemma deep-dive
below is the worked example; the four-model synthesis follows.

### Idioms & coverage (the portable layer)
`disasm_portable.py` recovers the universal idioms from Gemma's weights/behavior: **prev-token** (0.0, 20.1,
21.6, 21.7, …), **duplicate** (1.4, 3.2, …), **induction** (4.4, 6.1, 6.2, …). On the *same Shakespeare
corpus* used for GPT-2, the attention budget is: self 31% · sink **3.9%** · prev 17% · structural 14% ·
local 21% · long-range (content) 12% → **plumbing 87.7%** (vs GPT-2's 86.7%). The plumbing *fraction* matches
GPT-2; its *composition* does not — Gemma has almost no attention-sink and plumbs via self/local/prev instead.

### QK content opcodes, across depth
`gemma_opcode_table.py`: at layer 12, **7/8 heads have a behaviorally-legible content binding (z>2)** in
Gemma Scope feature coords (e.g. pronoun→verb, title-numeral completion). `gemma_layer_sweep.py` shows
legibility **peaks mid-network** — L6 **7/8** (mean z≈3.8), L12 5/8, weak early/late (L0 0/8) — i.e. content-
addressing concentrates in the induction/composition band, with early layers positional and late layers
output-formatting. The richest single layer is **L6**: head 6.0 binds verb→verb (tense/voice) at z=16.7,
6.4 does title-numeral completion at z=8.0, and the causal induction heads 6.2/6.3 bind number→noun and
OV-write proper-noun completions — consistent with induction-copying repeated wikitext entities.

### Causal validation
`gemma_causal.py` mean-ablates the recovered idiom heads (induction-NLL baseline 4.47): **induction is
load-bearing, z=8.3** (heads 4.4/6.2/6.3/22.2/22.3/22.4) — replicating GPT-2's z=8.6 — and **prev-token is
load-bearing, z=7.0** (0.0/20.1/21.6/21.7). Both are induction-specific (the complement barely moves). The
induction mechanism is causal in both architectures.

### The unified listing, at GPT-2 parity
`disassemble_gemma.py` emits, for **all 208 heads**: an addressing bucket + behavioral idiom tags + `*CAUSAL*`
flag + a **QK token-operand binding** + an **OV copy/transform WRITE** class (histogram **156 transform / 52
copy**), plus a per-layer **GeGLU MLP catalog** (read-tokens → write-tokens) and, at the SAE layer, the
feature-native QK/OV opcode — the same fields GPT-2's listing carries. The one residual non-parity is
intrinsic: GPT-2's named *circuit* roles (IOI name-movers, S-inhibition) come from a published head-set with
no Gemma equivalent, so Gemma carries behavioral idiom tags + causal flags rather than named-circuit tags.

### Cross-model synthesis (four models, four families)

All on the **same Shakespeare corpus** for apples-to-apples (`disasm_portable.py`); induction causality from
each model's `*_causal_summary.json`:

| axis | GPT-2-small | Gemma-2-2B | Llama-3.2-1B | Qwen2.5-1.5B | invariant? |
|------|-------------|------------|--------------|--------------|------------|
| plumbing fraction | 86.7% | 87.7% | 89.4% | 86.6% | **yes (~87%)** |
| **attention-sink** | 45.6% | **3.9%** | 55.0% | 44.4% | **no — high (44–55%) in 3/4; Gemma the low outlier** |
| universal idioms (prev/dup/induction) recovered | yes | yes | yes | yes | **yes** |
| induction causal (mean-ablation, induction-NLL z) | 8.6 | 8.3 | 27.3 | 14.9 | **yes — load-bearing in all 4** |

**Conclusion: the mechanisms (idioms) and their causal load-bearing-ness are architecture-invariant across
four models spanning four families, and the plumbing *fraction* is invariant too (~87%) — but its
*composition* is not. The attention-sink is high (44–55%) in GPT-2, Llama, and Qwen, while Gemma-2 is a
striking low-sink (~4%) outlier.** This **corrects** the earlier two-model reading that the sink was
"GPT-2-family-specific": with four models the sink is near-universal and *Gemma* is the exception — a clean
illustration of why a third/fourth architecture is worth testing. (The SAE-feature opcode legibility and the
copy/transform WRITE split are reported per-model in the Gemma deep-dive; legibility needs a per-layer SAE
[Gemma only], and the copy/transform split is threshold-defined, so neither is a cross-model invariant.)

### Is the sink load-bearing? Ablation — magnitude ≠ dependence
`sink_ablation.py` blocks attention to key-position-0 at every layer (rewrites the 4D causal mask; query 0
keeps self-attention) and measures next-token NLL on the same short-context corpus — removing the *option* to
sink and forcing each head to redistribute onto content. The sink-fraction drops to 0 under the hook
(intervention check passed for all four).

| model | sink mass | baseline NLL | sink-blocked NLL | ΔNLL | ΔNLL % |
|---|---|---|---|---|---|
| **GPT-2** | 45.6% | 5.00 | 7.08 | **+2.09** | **+42%** |
| Gemma-2-2B | 2.1% | 7.42 | 7.60 | +0.18 | +2% |
| Llama-3.2-1B | 55.0% | 4.13 | 4.17 | +0.04 | +1% |
| Qwen2.5-1.5B | 44.5% | 3.98 | 4.03 | +0.05 | +1% |

**Sink magnitude does not predict sink dependence.** Only **GPT-2** is functionally dependent on its sink
(+42% NLL); **Llama and Qwen sink even harder (55%, 44%) yet shrug off its removal (+1%)** — their large sink
is a genuinely redistributable no-op — and Gemma (low sink) is likewise unaffected (+2%). This refutes both
the naive "the sink is a universal load-bearing stabilizer" reading *and* the guess that sink magnitude
tracks dependence: the only outlier on *dependence* is GPT-2.

**Position-resolved (ΔNLL by query position)** sharpens it. All four peak at the earliest positions (little
context to redistribute onto) and decay, but two signatures separate GPT-2 from the RoPE models:

| | ΔNLL @ p1 | early (p1–8) | late (p32+) |
|---|---|---|---|
| **GPT-2** | +9.25 | +5.23 | **+1.57** |
| Gemma-2 | −0.81 | +0.51 | +0.15 |
| Llama-3.2 | +1.30 | +0.34 | **+0.00** |
| Qwen-2.5 | +0.79 | +0.36 | **+0.01** |

(1) GPT-2's early spike is ~7× the others' (p1 +9.25 vs ≤+1.3) — even at position 1 the RoPE models cope via
self/local attention, GPT-2 cannot. (2) The decisive one: **GPT-2 keeps a persistent ~+1.5-nat floor at
positions 32+** (where dozens of content tokens are available to redistribute onto), while all three RoPE
models fall to **~0**. So GPT-2 reads its sink for prediction *at every position*, not just when context is
short; the RoPE models don't depend on it at all once any context exists.

**Leading hypothesis (untested):** GPT-2's uniqueness tracks its **learned absolute positional embeddings** —
position 0 is a genuine absolute-position anchor heads rely on, so blocking it disrupts GPT-2's positional
computation; the other three use **RoPE** (relative), so key-0 is not a positional anchor and is freely
redistributable. (Alternative: a GPT-2-specific massive-activation / register read at pos-0.) **Caveats:** this
is *short context* (ctx 96, all keys present) — a different regime from the StreamingLLM result, where the
sink is essential for *long-context KV-cache eviction*, which this does not probe; and the absolute baseline
NLLs are not cross-comparable (tokenizer / no-BOS-per-chunk / Gemma's logit-softcap), so only the within-model
Δ is the signal. `sink_ablation.py`, `runs/gemma/sink_ablation_*_summary.json`.

### Multilingual: the ops are language-universal
`multilingual_ops.py` runs the behavioral disassembly on the **same domain (Wikipedia) in six languages
across four scripts** — en/fr/de (Latin), zh (CJK), ru (Cyrillic), ar (Arabic) — on the two multilingual
models.

**Mechanism heads are language-invariant.** Per-head idiom-score vectors correlate near-perfectly across
language pairs: **prev-token Spearman +0.98**, **induction +0.88 (Gemma) / +0.83 (Qwen)**, duplicate
+0.83 / +0.77 — and the *same* top induction heads run in every language: Gemma **{4.4, 6.2, 6.3, 22.2/3/4}**
(its causally-validated induction set) and Qwen **{2.3, 14.0, 14.3, 19.3}**, whether the input is English,
Chinese, Russian, or Arabic.

**The attention budget barely shifts with script** (stronger invariance than expected). Gemma-2-2B, per
language:

| lang | sink | self | prev | structural | local | content |
|---|---|---|---|---|---|---|
| en | 2% | 32% | 18% | 7% | 26% | 15% |
| zh | 2% | 34% | 19% | 10% | 22% | 13% |
| ru | 2% | 32% | 18% | 9% | 25% | 15% |
| ar | 2% | 32% | 19% | 7% | 25% | 15% |

(fr/de track en; Qwen likewise holds sink 47–51% across all six.) The only systematic script effect is small:
the **structural** fraction dips for CJK/Arabic (Qwen 3% for zh/ar vs 6–7% Latin; Gemma 7% for ar) —
consistent with fewer whitespace/newline tokens in those scripts.

⇒ **the attention instruction set is language-universal**: the same idiom heads fire in the same proportions
regardless of language; **language lives at the *operand* (token-identity) level**, not in which heads run or
how attention is budgeted. Combined with the cross-architecture result, the ops are invariant across **both
architecture and language** — what varies is the operands (and, across families, the sink).
`multilingual_ops.py`, `runs/gemma/multilingual_ops_{gemma2,qwen25_15b}_summary.json`.

### The full listings
The complete per-head listings are committed as reference artifacts (regenerate with the disassemblers):

- [`listings/gpt2_disassembly.txt`](listings/gpt2_disassembly.txt) — all 144 GPT-2 heads + MLP (Shakespeare).
- [`listings/gemma2_disassembly.txt`](listings/gemma2_disassembly.txt) — all 208 Gemma heads + MLP, SAE layer 12 (WikiText).
- [`listings/gemma2_disassembly_L6.txt`](listings/gemma2_disassembly_L6.txt) — the peak-legibility Gemma layer-6 decode.
- [`listings/llama32_1b_disassembly.txt`](listings/llama32_1b_disassembly.txt) — all 512 Llama-3.2-1B heads + MLP (token-operand basis).
- [`listings/qwen25_15b_disassembly.txt`](listings/qwen25_15b_disassembly.txt) — all 336 Qwen2.5-1.5B heads + MLP (token-operand basis).

(`scripts/disassembly/disassemble_gpt2.py` → `runs/disassembly/`; `scripts/gemma/disassemble_gemma.py --model …`
→ `runs/gemma/` for Gemma/Llama/Qwen. The `runs/` copies + per-head `.json` are git-ignored and regenerated on
demand. Llama-3.2-1B was run via the ungated `unsloth/Llama-3.2-1B` mirror — identical weights to the gated
`meta-llama/Llama-3.2-1B`, which the code defaults to once you have access.)

## Downstream hook (and a retraction)

The causally-validated, corpus-robust writer heads (induction 5.x/6.9/7.11, prev-token 4.11) were proposed as an
evidence-backed preserve-set for **writer-output `U_C`** in `sae-forge` (the two-basis forge). That circuit-
preservation claim was **RETRACTED**: the `excess = induction_kl − complement_kl` metric is gameable (a basis can
lower it by damaging the complement), and compression-controlled re-validation found writer-OV ≈ random-OV at
matched complement-KL (0/6 writer-wins across layers × seeds; `scripts/two_basis_forge/forge_revalidate_broad.py`).
The disassembly itself stands; the shortcut from "these are the writer heads" to "forging their OV-output subspace
preserves the circuit" does not.
