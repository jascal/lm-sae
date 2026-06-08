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
| `pylm/build.py` | decompiler (corpus route) | `transformers` *tokenizer* (a flat-file BPE, not a net) |
| `pylm/capture.py` | decompiler (model route, via MI) | `torch` + `transformers` — *reads the model* to extract flat files |
| `pylm/validate.py` | validator | `torch` only behind `--no-model` — the ground-truth ceiling, **not part of pylm** |

So pylm itself runs with **no neural network and no ML dependency** — the goal's constraint, satisfied and auditable.

## The program (small) and the store (flat data)

**The program** — `PyLM.predict` in `lm.py`, **~49 lines** of plain Python. The reused instructions:
- **induction (in-context copy)** — the keystone idiom (an inline-cache macro): the longest local context (≥ 2
  tokens) that has recurred earlier *in this sequence* → predict the token that followed it. Pure list scan.
- **n-gram backoff** — 4-gram → trigram → bigram → unigram successor lookup. Pure dict lookup.

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

## The ceiling — why it stops short of 100% (the forge tax, made runnable)

pylm reproduces ~half the model and no more, and it *can't* reach 100%: the un-reproduced fraction is the entangled
**composition** the [flagship](DECOMPILATION.md) shows does not factor through any clean basis — the model's accuracy
lives in full-context composition that a flat local-context lookup + an in-context-copy macro cannot hold (distilling
the model into local n-gram tables provably discards it). pylm turns "the decompilable fraction" from a number into a
**running artifact**: ~half of a real LLM *is* a small symbolic program over flat knowledge; the rest is the
irreducible core the forge tax measures.

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

## Next steps

- **Smaller hosts pushed further** (Pythia-14m, already 56%): optimise the idiom mix where the decompilable fraction
  is highest — how *fully* can the smallest LLM be reduced to a small program?
- **Factual corpora** where the knowledge table moves the next-token metric (on tiny-Shakespeare relations don't
  appear, so the table is a capability demo, not a metric lift).
- The ceiling itself is the result: pylm makes "the decompilable fraction" a running artifact, and the levers above
  show it is robust — you cannot memorise your way past the composition.
