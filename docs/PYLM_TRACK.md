---
title: pylm track — decompiling an LLM to pure Python
---

# The pylm track — can a whole small LLM be decompiled to a *small pure-Python program* + flat files?

*A sister effort to the [forge-tax track](FORGE_TAX_TRACK.md) and the [disassembly→decompilation](DECOMPILATION.md)
program. Where the catalog *names* the operators and the flagship measures the *forge tax* as a fraction, this track
takes the decompilation literally: **reimplement the model's behaviour as a small Python program over flat-file
knowledge — with no neural-net code or concepts — and measure how much of the model it reproduces.** Code lives in
[`pylm/`](https://github.com/jascal/lm-sae/tree/main/pylm).*

## The goal and the hard constraint

> Fully decompile and reimplement a whole small LLM in Python **without using any neural-net code or concepts**,
> validate the output against a corpus, bias toward **small actual code**, but allow **flat files for the knowledge
> store**.

The constraint is on the **decompiled artifact**: the thing that *runs as the reimplemented LM* must contain no
matrix, no attention, no layer, no ML library — only the catalogued idioms as plain Python over flat data. It is
*not* a constraint on the **decompiler**: the one-time process that *extracts* the flat files is allowed to read the
model (the MI tools run forwards / read weights), exactly as a disassembler reads a binary to emit source. The model
is the **subject** of decompilation, not a runtime dependency of pylm.

### Dependencies — the split that proves the constraint

| file | role | imports |
|---|---|---|
| **`pylm/lm.py`** | **the decompiled LM (the artifact that runs)** | **`json`, `pathlib` only — pure stdlib, ZERO ML** |
| **`pylm/grammar.py`** | **the GRAMMAR idiom (runs)** | **pure stdlib** (the `closed_ids()` builder takes a tokenizer; used only by the decompiler) |
| `pylm/build.py` | decompiler (corpus route) | `transformers` *tokenizer* (a flat-file BPE, not a net) |
| `pylm/capture.py` | decompiler (model route, via MI) | `torch` + `transformers` — *reads the model* to extract flat files |
| `pylm/validate.py` | validator | `torch` only behind `--no-model` — the ground-truth ceiling, **not part of pylm** |
| **`pylm/numpy_lm.py`** | **the composition kernel (Tier B — runs)** | **`numpy` only — CPU matmul, no torch/GPU** |
| **`pylm/explain.py`** | **unified 'explain this prediction' (runs)** | **`numpy` only** (+ tokenizer for human-readable I/O) |
| `pylm/export_weights.py` | one-time weight export (build) | `torch` + `transformers` — *reads the model* to dump flat `.npz` weights |

So pylm itself runs with **no neural network and no ML dependency** — the goal's constraint, satisfied and auditable.

### Runtime tiers — retrieval (stdlib) + composition (numpy), never torch

The decompilation has a hard fidelity ceiling at ~half (the *forge tax*: the composition is **proven dense genuine
computation**, not a flat lookup, so no bigger table crosses it — see `DECOMPILATION.md`). But the composition is a
**TC⁰ circuit**, so it runs on a CPU as plain numpy matmuls. That gives three buildable tiers, and the headline is *no
deep-learning framework at runtime*:

| tier | fidelity | runtime packages | data |
|---|---|---|---|
| **A · retrieval** (`lm.py`) | ~50% token / ~83% acc (GPT-2) | **Python stdlib only** | flat JSON store (~1.6 MB) + tokenizer |
| **B · + composition** (`numpy_lm.py`) | → the full model | **+ `numpy`** (CPU matmul) | + flat weight arrays (`.npz`, Θ model size) |
| **C · + router** (`numpy_lm.py --route-frac`) | full, cheaper | + `numpy` | + a small router head; computes ~60% of MLP/token |

Building the stores/weights is the only torch step (`capture.py`, `export_weights.py`) — run once to *extract*; *running*
pylm needs only stdlib (A) or stdlib + numpy (B/C). Scaling sharpens the split: the decompilable (flat) fraction **falls**
with model size (56%→45%), so bigger models lean more on the numpy compute kernel and the weight arrays — the *computed*
part is exactly what doesn't keep pace. And the composition's per-token cost is bounded and **pre-computable from the
architecture** (it touches only ~60% of the MLP, k/m≈0.4, budget B≈1.4·k), so the Tier-C router budget is known a priori
(`router_kernel.py`).

**Validated.** The Tier-B kernel (`numpy_lm.py`, ~60 lines of numpy) reproduces GPT-2 **exactly — 100% top-1 agreement
with torch** over a 498 MB flat `.npz` (fp32), with `torch` never imported at runtime; 50.7% next-token top-1 on held-out
tiny-Shakespeare (= GPT-2's own number). The Tier-C routing (`--route-frac 0.6`, compute only the top-60% active MLP
neurons/token) costs ~3.5 pp (47.2%) for ~40% less MLP compute. So the whole runtime is **flat dict lookups (stdlib) +
numpy matmuls (CPU) + flat weight arrays** — the model on a laptop, no framework.

**Weight precision (smaller flat files).** `export_weights.py --dtype {float32,float16,int8}` controls the on-disk
store; `numpy_lm.py` upcasts on load so the kernel is identical. For GPT-2: fp32 497 MB (100% torch-agreement, 51.3%
acc), fp16 248 MB (100%, 51.3%), int8 124 MB (98.0%, 51.0%) — per-output-column symmetric int8 with an fp16 scale, so
¼ the bytes at ~2 pp agreement cost. (RAM is always fp32 — the kernel dequantises on load — so precision buys *download
and disk*, not memory.)

**Beyond GPT-2 — the RoPE family** (`export_weights_rope.py` + `numpy_rope.py`, ~80 lines of numpy). The modern
laptop-grade architectures (Llama-3.2, Qwen2.5) need RMSNorm + rotary position embedding + grouped-query attention +
SwiGLU instead of GPT-2's LayerNorm/learned-position/GELU — a different interpreter over the *same* flat-weights-plus-
numpy story. Validated on **Qwen2.5-0.5B**: fp32 (1976 MB) is **exact — 100% top-1 agreement with torch**, logit
max-abs-diff 1e-4; int8 (495 MB) holds 92.6% (the large tied-vocab embedding, doing double duty as the unembed, is the
quant-sensitive part). The kernel auto-handles q/k/v bias (Qwen) and tied vs. untied embeddings. So a capable modern
0.5B model runs in pure numpy on a CPU — ~2 GB resident (fp32 in RAM), which fits an 8 GB laptop comfortably; a 1B
(Llama-3.2-1B) is ~5 GB resident, the practical ceiling before the kernel would need to keep weights int8 *in RAM* and
dequantise per-matmul.

### Explain this prediction — the two halves fused (`explain.py`)

For any context, `explain.py` prints the prediction with **both** of its readings (numpy-only, no torch):

- **RETRIEVAL** — which symbolic idiom `lm.py` fired (induction-*N* / n-gram backoff / knowledge lookup / grammar
  skeleton), its prediction, its evidence, and whether it **agrees with the model**.
- **COMPOSITION** — the live circuits read straight off the real forward pass at the predicting position: attention
  heads **named** by their `idiom_library` signature (previous-token / duplicate-token / induction; attention-sink heads
  are collapsed to a NO-OP count), plus the top-activating MLP features, each **named by the vocabulary it promotes**
  (its write weight projected to the unembed — the neuron's direct-logit "feature"). So both halves are named, not bare
  indices.

The agree/differ flag makes the **forge tax legible per token**. Three regimes show up immediately:
- *retrieval agrees & a copy circuit is live* — on a repeated phrase the symbolic idiom is `induction-3` **and** GPT-2's
  real induction heads (L5.H1, L7.H2, L10.H6, …) all attend to the copied token: the flat half and the computed half are
  doing the same thing.
- *retrieval differs & the MLP carries it* — "…the city of" → the model says **Paris** while the n-gram store says
  "the"; no copy/induction head predicts Paris — it comes from the MLP feature stack, and the named features make the
  mechanism legible: the live neurons promote `{towns, town, suburb}`, `{China, Japan, asia}`, `{TX, shire, Janeiro}` —
  GPT-2's geography/place features firing for "city of". The token is in the computed half, and you can see *which*
  features pay the tax.
- *attention idle* — most heads sit on the sink (NO-OP), surfaced as a count so the few content-carrying circuits stand
  out. This is the explain surface the eventual API will serve.

`explain.py --sequence` runs this over a whole passage and aggregates it into a **per-text forge-tax breakdown**: what
fraction of tokens the flat store reproduces vs the dense composition carries, bucketed by provenance (induction /
n-gram-grammar / knowledge / composition-carried), plus the most-used live circuits and named features across the
passage. On a short repeated-then-factual passage, ~28% is flat-store-reproducible, ~17% in-context induction, and ~69%
composition-carried — with the L0 duplicate-token heads and L5 induction heads as the most-used circuits. That is the
decomposition the API returns for an arbitrary text: every token attributed to a half, and the carrying circuits named.

## The program (small) and the store (flat data)

**The program** — `PyLM.predict` in `lm.py`, **~62 lines** of plain Python (+ a 44-line `grammar.py`). The reused
instructions:
- **induction (in-context copy)** — the keystone idiom (an inline-cache macro): the longest local context (≥ 2
  tokens) that has recurred earlier *in this sequence* → predict the token that followed it. Pure list scan.
- **n-gram backoff** — 4-gram → trigram → bigram → unigram successor lookup. Pure dict lookup.
- **grammar (closed-class skeleton)** — the model's content-free grammatical scaffold (see below): collapse every
  content token to a single `OPEN` symbol, keep function-words/punctuation as themselves, look up the grammatical
  **skeleton → next-token** table. Fires below the lexical n-gram (where it is too sparse), above the unigram floor.

**The store** — a flat JSON of successor tables (~1–1.7 MB). Built two ways:
- **corpus route** (`build.py`): n-gram counts fit from the corpus — a *surrogate* for the model.
- **model route** (`capture.py`, the MI capture): run the model over the corpus (the model is the ultimate MI probe)
  and distil **its own** next-token predictions into the same flat tables — so the store holds *what the model would
  say* after each local context. This is decompiling the **model**, not the corpus.

## Results (GPT-2, tiny-Shakespeare, 4000 held-out positions)

| decompiler | program | pylm corpus top-1 | model corpus top-1 | **pylm↔model agreement (decompilable fraction)** |
|---|---|---|---|---|
| corpus-fit (`build.py`) | 49 LOC + ~1.7 MB | 31.9% | 34.6% | **35%** |
| **model-capture (`capture.py`)** | 49 LOC + ~1.6 MB | 29.0% | 34.6% | **49%** |

- **Capturing from the model raises the decompilable fraction 35% → 49%.** A 49-line pure-Python program + a flat
  store distilled from the model reproduces GPT-2's *exact* next token **49% of the time** — with no neural network in
  the running artifact. The model-capture trades a little raw corpus accuracy (it mimics the model, errors and all)
  for far higher *fidelity to the model* — the right objective for decompiling the model rather than fitting the text.
- **Induction carries genuine weight, as the catalog predicts.** The longest in-context match (`induction-3`) is the
  most accurate instruction at **68%**; single-token "induction" is ~chance noise (excluded); the n-gram store carries
  the statistical bulk.

## The decompilable fraction shrinks with scale (the thesis, made literal)

Running the decompile→validate loop across a controlled ladder (Pythia 14m→1.4b — one GPT-NeoX architecture, same
data, six sizes; `ladder.py`) shows the **decompilable fraction falls monotonically with model size**:

| pythia | model corpus top-1 | **decompilable fraction (pylm↔model)** | pylm reproduces (of model acc) |
|---|---|---|---|
| 14m | 29.4% | **56.3%** | **95%** |
| 70m | 33.4% | 52.3% | 92% |
| 160m | 38.8% | 50.2% | 82% |
| 410m | 44.3% | 45.8% | 75% |
| 1b | 47.7% | 46.1% | 70% |
| 1.4b | 49.4% | **44.9%** | **69%** |
| *gpt2 (124M, ref)* | 37.2% | 49.7% | 83% |

- **The smallest LLM is 56% reproducible by a 49-line pure-Python program; the largest, 45%.** A *tiny* model
  (Pythia-14m) is **95%** as accurate as itself when reimplemented in pure Python — it is almost entirely n-gram +
  induction. As the model grows, the fraction a small symbolic program reproduces **drops** (56 → 45%), and the
  fraction of its *accuracy* that decompiles drops faster (95 → 69%). GPT-2 (124M) lands at 49.7%, exactly where its
  size predicts — the curve is architecture-general.
- **This is the central thesis as a running artifact:** the part of an LLM that *is* a small program over flat
  knowledge **shrinks as the model scales**, and the irreducible remainder — the entangled core — **grows with
  capability.** It is the *third* independent measurement of that same growth: (a) the [SAE-feature composition
  tax](DECOMPILATION.md), (b) [read-out depth / fact-separability](scaling.md), and now (c) the pure-Python
  decompilable fraction. Three routes, one core.

**Modern models are *less* decompilable than older models of the same size — it falls with *capability/era*, not just
parameter count.** Taking pylm off the GPT-2/Pythia era to recent models (`runs/pylm/modern_decompile_summary.json`):

| model (era) | size | model corpus top-1 | **decompilable fraction** | pylm reproduces (of acc) |
|---|---|---|---|---|
| Pythia-1.4b (2023) | 1.4 B | 49.4% | 44.9% | 69% |
| **Qwen2.5-1.5B** (2024) | 1.5 B | 40.2% | **34.6%** | 58% |
| **Llama-3.2-3B** (2024) | 3 B | 51.0% | **31.3%** | 45% |

A modern 1.5 B (Qwen2.5, 34.6%) is **~10 points less** flat-reproducible than an *older* 1.4 B (Pythia, 44.9%), and
Llama-3.2-3B is the lowest measured (31.3%). Modern training (far more data, better methods) pushes *more* of the
model's behaviour into the entangled composition — so the same "core grows with capability" trend holds across **era**,
not only size. (Measured on tiny-Shakespeare like the rest; the relative modern-vs-old gap at matched size is the
signal.)

## The ceiling — why it stops short of 100% (the forge tax, made runnable)

pylm reproduces ~half the model and no more, and it *can't* reach 100%: the un-reproduced fraction is the entangled
**composition** the [flagship](DECOMPILATION.md) shows does not factor through any clean basis — the model's accuracy
lives in full-context composition that a flat local-context lookup + an in-context-copy macro cannot hold (distilling
the model into local n-gram tables provably discards it). pylm turns "the decompilable fraction" from a number into a
**running artifact**: ~half of a real LLM *is* a small symbolic program over flat knowledge; the rest is the
irreducible core the forge tax measures.

## Sequence-level validation — what free generation does (and why teacher-forced is the metric)

Teacher-forced top-1 agreement (above) is the decompilable fraction. A harsher, sequence-level check
(`rollout.py`): roll out generation from held-out seeds with both pylm and the model.

- **Free greedy generation diverges almost immediately** — fidelity horizon **~1 token** (median 0) before pylm's
  greedy rollout first differs from the model's. This is *not* a pylm failure: greedy rollout is trajectory-chaotic
  (one different token forks the path), so *any* two distinct models diverge within a token or two. It just means
  free-rollout match is the wrong yardstick.
- **Teacher-forced on the model's own generated path, agreement is 59–63% and *rises* to 74–84%** by token 20 —
  because the model's greedy generation becomes repetitive and pylm's **induction macro nails the loop** (the model
  loops under greedy; pylm reproduces *why*). Consistent with, and slightly above, the held-out teacher-forced fraction.
- **pylm's greedy generation vs the actual corpus: ~4%** — greedy text (model *or* pylm) doesn't reproduce natural
  text; this measures greedy-vs-natural, not decompilation fidelity.

Net: the right "how much of the model" metric is **teacher-forced top-1 agreement** (49–55%); pylm is a faithful
*next-token* decompiler, not a trajectory-matcher (no symbolic model is, under greedy).

## The flat-file knowledge store — "the model IS the database", literally

Beyond statistics, pylm carries a **flat fact table** (`knowledge.py`): relations read out of the model
(`The capital of {S} is →` argmax over the *object set* — the constrained readout `relation_decompile` uses, so the
fact is read out even when a high-frequency token outranks it in full vocab) into a flat JSON `{capital: {France:
Paris, …}}`. `PyLM` (given `--knowledge`) does a pure-Python **relational lookup** that fires *before* induction/
n-gram — answering factual prompts the n-gram never saw, and *generalising across phrasings* (it's the relation
operator, not a surface n-gram):

```
'The capital of France is' → 'Paris'   [knowledge:capital]    (n-gram alone: 'a')
'The capital of Japan is'  → 'Tokyo'   [knowledge:capital]    (n-gram alone: 'the')
'The language of Italy is' → 'Italian' [knowledge:language]   (n-gram alone: 'the')
```

The facts are read out of **gpt2-large at 100%** correct — but **gpt2-small's are mostly wrong** (France→"the"):
factual recall *emerges with scale* (~160M; see the [scaling laws](scaling.md)), so the database a model carries is a
function of its size. pylm faithfully decompiles whichever it is — including the small model's ignorance.

## Levers that plateau — the decompilable fraction is a real ceiling, not a tuning artifact

Two pushes to raise it both **failed**, which is the point (the program kills its own ideas): (1) **deeper memorised
context** — adding a 5-gram store left GPT-2's decompilable fraction flat (49.0% → 48.6%; held-out 5-grams are too
sparse); (2) **store-first arbitration** — trusting the model-captured n-gram over induction *lowered* agreement
(49.7% → 48.4%): induction-first is right because the model genuinely *does* induction (the in-context copy matches it
better than the corpus-modal). So ~50% (GPT-2) is a **genuine ceiling** — the remaining half is real composition /
generalisation a flat store + an in-context-copy macro cannot hold, the entangled core the forge tax measures.

## The context ceiling — longer context can't crack the core (the ∞-gram, `context_ceiling.py`)

The sharpest version of "more flat data won't help": push the n-gram store to **unbounded context** — the ∞-gram /
longest-suffix predictor over the whole training stream — and measure the decompilable fraction (∞-gram↔model top-1)
as the allowed context length K grows. On GPT-2 / tiny-Shakespeare:

| context cap K | 1 | 2 | 3 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| ∞-gram ↔ model | 26.5% | 32.3% | 32.5% | 31.7% | 31.7% | 31.9% | **31.9%** |

**It saturates at a trigram and never moves** — the mean matched suffix tops out at **3.4** because held-out exact
matches longer than ~3 tokens *do not exist* in a 300 K-token store. So an *unbounded*-context corpus lookup ≈ a
trigram: the **memorization ceiling is 31.9%**, and the **composition residual — the 68% not crackable by any amount of
flat context — is genuine composition, not missing memory.** (This is the rigorous form of the 5-gram-plateau lever.)
pylm reaches ~49% only by adding **in-context induction** (copying from the *prompt itself*, not the corpus store) —
the cheaply-recoverable part of the composition — leaving ~50% as the irreducible core the forge tax measures. Net for
the resource question: you **cannot** crack more of the core by storing longer context (the data is too sparse for it
to matter); the un-decompiled half is composition. *(Caveat: the trigram saturation is partly a 300 K-token store-size
artifact — a far larger corpus would surface some longer matches — but the gain is bounded by, and well below, the
composition residual.)*

## Showcase — the whole small LLM (Pythia-14m), decompiled and generating

The goal's literal headline. Pythia-14m (6 layers, d128) is the smallest real LLM here and the most-fully decompiled:
a **49-line pure-Python program + a 1.5 MB flat store reproduces 95% of its next-token accuracy** (pylm 26.5% vs the
model's 27.8%) and **55% of its exact top-1 tokens** — with no neural net running. It generates (pure-Python
`predict`, the tokenizer for I/O only):

```
seed:  "The meaning of life is"
pylm (greedy):  " a good man,\nI am not sure,\nI am not sure, …"        (induction-3 loop, 28/40 tokens)
pylm (temp 0.8):" not the\n\nI am a man,\n\nI am not sure, I will not\nbeautiful, and I have to the other way …"
```

Rough — it is a 14M-parameter model — but it is a *whole* small LLM running as a tiny symbolic program over flat
files, every token attributable to a named idiom. (`pylm/store_pythia14m.json`,
`runs/pylm/validate_pythia14m_summary.json`.)

## The flat-file GRAMMAR idiom — "if there is a grammar, it goes in flat files too"

The core-structure analysis ([`core_grammar.py`](DECOMPILATION.md)) found the entangled core's *most-shared*
directions form a compact, corpus-invariant, **closed-class scaffold** — a generic grammar (top-16 directions: 22×
chance cross-corpus overlap, 28× the closed-class base rate). A grammar decompiles to a flat file the same way a fact
table does: `grammar.py` collapses every content token to one `OPEN` symbol, keeps function-words/punctuation as
themselves, and stores the grammatical **skeleton → next-token** table (model-captured, 12,272 entries / 487
closed-class ids for GPT-2). `PyLM` consults it *below* the lexical n-gram, *above* the unigram floor — a content-free
generalisation that fires where the lexical table is too sparse.

| GPT-2, 4000 held-out positions | decompilable fraction | what fires when the bigram misses |
|---|---|---|
| without grammar (skel stripped) | 49.0% | `unigram` 2% @ 3% |
| **with grammar (flat skeleton table)** | **49.5%** | `grammar-3` 2% @ 2% · `grammar-2` 0% |

**The grammar slots in exactly where the unigram fallback used to fire, and barely beats it — a confirmed plateau
(+0.5pp).** This is the *right* null result, and it says something precise: **grammar predicts the next *category*
(slot), but the decompilable fraction is a next-*token* metric.** Which exact token fills a grammatical slot is
already absorbed into the n-gram modes (the modal successor after `the` is itself grammatical), so the grammar is
**token-redundant with the n-gram store** — real in the *geometry* (the corpus-invariant closed-class core head),
nearly invisible in token accuracy. It joins the 5-gram and store-first levers as a confirmed ceiling: you cannot
tabulate your way past the composition.

### What the decomposition says about "the content is all flat-file-able"

Tease the grammar out and the decompilable content does split into the flat idioms — `induction-3` (11% @ 68%),
`quad`/`trigram`/`bigram` (n-gram, flat), and the `knowledge` relation table (flat) on factual corpora. **But that
is the *optimistic limit*, not the whole model.** Summed, every flat idiom + the induction macro reproduces **49.5%**
of GPT-2; the complementary **~50.5% is irreducible** — and the per-instruction accuracies show why it is *not* just
content we have yet to tabulate: the bigram fires on 29% of tokens but is right only 19% of the time, the trigram
23% — positions where the local table **tries and is wrong** because the model's actual token depends on longer-range
composition. So the full decompilation is **{induction macro} + {grammar, n-gram, knowledge flat files} ≈ half the
model; the entangled composition ≈ the other half** — the three flat buckets are the complete flat-file-representable
basis, sufficient for half, and the complement is the forge tax made runnable.
(`pylm/store_grammar.json`, `runs/pylm/validate_grammar_summary.json` vs `validate_nogrammar_summary.json`.)

### The grammar, made readable (`grammar_rules.py`)

The flat store keeps the grammar as token ids; `grammar_rules.py` decodes it into a human-readable inventory — the
**closed-class lexicon by grammatical category** (the scaffold the idiom operates over) plus example skeleton rules:

| category | words (GPT-2 single-token closed class) |
|---|---|
| determiners (21) | a an any each either every her his its my neither no our some that the their these this those your |
| prepositions (22) | about above after at below between by down for from in into of off on out over through to under up with |
| conjunctions (18) | although and as because but if nor or since so than then unless when where whereas while yet |
| pronouns (21) | he hers him i it me mine ours she theirs them they us we what which who whom whose you yours |
| auxiliaries (24) | am are be been being can could did do does had has have having is may might must shall should was were will would |
| particles (14) | also hence here however just no not now only there therefore thus too very |
| punctuation (18) | `. , ; : ! ? ' " ( ) [ ] { } - — … /` |

**What is universal vs model-specific.** The *lexicon and categories* are language-universal (these are English's
closed class) and the *scaffold* is corpus-invariant + cross-architecture (the core-structure result: the grammar head
overlaps 22× chance across corpora and reproduces on Llama). The *skeleton → next-token transition table* (`skel`,
12 K rules), by contrast, is **model/tokenizer-captured** — the categories stay, the specific transitions are read out
of a given model. So the grammar is published as: a universal categorial inventory + a model-specific transition table,
both regenerable from `pylm/store_grammar.json` → `runs/pylm/grammar_rules_summary.json`.

## The auditable artifact — per-token explanation at BOTH levels (`runtime_explain.py`)

The "auditable corner" of the small/legible/complete triangle, made concrete: an explainer that attributes every
predicted token at *both* the **symbolic** level (which pylm idiom fired — induction / n-gram / grammar / knowledge)
*and* the **model-circuit** level (which mechanism carries it, with attention evidence), and *verifies* the symbolic
idiom against the real circuit. For an induction prediction it locates the model's induction head (behaviourally) and
reports its attention from the query to the **copy-source** position; content predictions are attributed to the
distributed MLP/composition bulk, knowledge to the readout. A demo trace:

```
[16] model ' on'  | pylm ' on'  [induction-2] ✓  ← induction head 5.5 (attn 0.37 → pos 3 'on')
[17] model ' the' | pylm ' the' [induction-3] ✓  ← induction head 5.5 (attn 0.47 → pos 4 'the')
[11] model ' other' | pylm ' world' [trigram]  ✗  ← MLP/content (distributed)
```

Over 299 held-out tokens (GPT-2): the attribution mix is `induction-3` 15% @ 80% agreement (clean head evidence),
`induction-2` 10% @ 39%, and the n-gram/content idioms (`bigram`/`trigram`/`quad`) 74% @ ~50–58% (the distributed
bulk). The symbolic **induction idiom is confirmed in the model** — the located head 5.5 attends a mean **0.26** to the
copy-source over the 76 induction tokens. So every prediction is **attributable end-to-end: program idiom ↔ model
circuit ↔ evidence** — a runtime-explainable artifact that fuses the decompiled program (pylm), the operator/circuit
catalog (induction located + verified), and the live model (ResidualVM attention probing). This is the *legible* corner
made runnable; it reproduces the decompilable ~half and *says why* for each token, with the irreducible composition
honestly labelled "distributed." (`runs/disassembly/runtime_explain_summary.json`.)

## Next steps

- **Optimise the idiom mix on the smallest host** further — the levers above plateau on GPT-2, but the head-room is
  largest where the decompilable fraction is highest.
- **Factual corpora** where the knowledge table moves the next-token metric (on tiny-Shakespeare relations don't
  appear, so the table is a capability demo, not a metric lift).
- The ceiling itself is the result: pylm makes "the decompilable fraction" a running artifact, and the levers above
  show it is robust — you cannot memorise your way past the composition.
