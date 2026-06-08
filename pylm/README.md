# pylm — a decompiled language model in pure Python (no neural-net code or concepts)

> **The narrative, results, and design rationale live in the sister-effort thread:
> [`docs/PYLM_TRACK.md`](../docs/PYLM_TRACK.md).** This README is the module quick-start.

**Goal.** Fully decompile and reimplement a whole small LLM's *behaviour* as a **small pure-Python program** plus
**flat-file knowledge stores** — the program's literal end-goal ("the model IS the database"; the accreting-VM
framing). No matrix, no attention, no layer — only the catalogued idioms expressed as plain Python over flat data,
validated against the corpus *and* against the real model.

**Dependencies (the constraint, satisfied).** `lm.py` — the decompiled LM that *runs* — imports **only `json` and
`pathlib`** (pure stdlib, zero ML). The neural net is touched only by the one-time **decompiler** (`build.py`
tokenizer / `capture.py` reads the model) and the **validator** (`--no-model` skips it) — never by the running pylm.

**The split (bias: small code, flat data).**
- **The program** — [`lm.py`](lm.py), `PyLM.predict`: deliberately tiny (currently **43 lines of code**). The reused
  instructions:
  - **induction (in-context copy)** — the keystone idiom (an inline-cache macro): if the local context has recurred
    earlier *in this sequence*, predict what followed it. Pure list scan.
  - **n-gram backoff** — trigram → bigram → unigram successor lookup. Pure dict lookup.
  - *(knowledge relations + structural rules: next steps.)*
- **The data** — [`store.json`](store.json) (~1 MB), a flat n-gram successor table built from the corpus by
  [`build.py`](build.py) over the target model's token ids (the BPE tokenizer is a flat-file artifact, not a net).

**Result** (GPT-2, tiny-Shakespeare, 4000 held-out positions; ~49 LOC of pure Python):

| decompiler | pylm corpus top-1 | GPT-2 ceiling | **pylm↔model agreement (decompilable fraction)** |
|---|---|---|---|
| corpus-fit (`build.py`) | 31.9% | 34.6% | **35%** |
| **model-capture (`capture.py`, MI)** | 29.0% | 34.6% | **49%** |

Capturing the model's own predictions (vs fitting the corpus) raises the decompilable fraction **35% → 49%** — a
49-line pure-Python program + flat store reproduces gpt2's *exact* next token half the time, with no neural net in
the running artifact. Induction carries real weight (the longest in-context match, `induction-3`, is the most accurate
idiom at **68%**). The gap to 100% is the entangled core the [forge tax](../docs/DECOMPILATION.md) measures — pylm
turns "the decompilable fraction" from a metric into a **running artifact**. Full narrative:
[`docs/PYLM_TRACK.md`](../docs/PYLM_TRACK.md).

**Run.** corpus route: `python pylm/build.py --model gpt2`; model route: `python pylm/capture.py --model gpt2`; then
`python pylm/validate.py --model gpt2` (add `--no-model` for corpus-only, no torch).
