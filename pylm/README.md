# pylm — a decompiled language model in pure Python (no neural-net code or concepts)

**Goal.** Fully decompile and reimplement a whole small LLM's *behaviour* as a **small pure-Python program** plus
**flat-file knowledge stores** — the program's literal end-goal ("the model IS the database"; the accreting-VM
framing). No matrix, no attention, no layer — only the catalogued idioms expressed as plain Python over flat data,
validated against the corpus *and* against the real model.

**The split (bias: small code, flat data).**
- **The program** — [`lm.py`](lm.py), `PyLM.predict`: deliberately tiny (currently **43 lines of code**). The reused
  instructions:
  - **induction (in-context copy)** — the keystone idiom (an inline-cache macro): if the local context has recurred
    earlier *in this sequence*, predict what followed it. Pure list scan.
  - **n-gram backoff** — trigram → bigram → unigram successor lookup. Pure dict lookup.
  - *(knowledge relations + structural rules: next steps.)*
- **The data** — [`store.json`](store.json) (~1 MB), a flat n-gram successor table built from the corpus by
  [`build.py`](build.py) over the target model's token ids (the BPE tokenizer is a flat-file artifact, not a net).

**First result** (GPT-2, tiny-Shakespeare, 4000 held-out positions; [`validate.py`](validate.py)):

| | value |
|---|---|
| program size | **43 LOC** of pure Python |
| data size | ~1 MB flat n-gram file |
| pylm corpus top-1 next-token accuracy | **29.7%** |
| GPT-2 corpus top-1 (the ceiling) | 34.6% |
| **pylm reproduces** | **86%** of the model's accuracy |
| **pylm ↔ model top-1 agreement (decompilable fraction)** | **35%** |

Per-instruction (share @ accuracy): `induction-3` 11% **@68%** · `trigram` 34%@32% · `induction-1` 25%@21% ·
`bigram` 22%@19% · `induction-2` 7%@29% · `unigram` 2%@3%. The longer the in-context match, the more accurate it is —
induction (the catalog's keystone) carries genuine predictive weight as plain Python; the n-gram store carries the
bulk.

**Why the gap matters.** pylm does *not* reach 100% — and it can't: the un-reproduced fraction is the entangled core
the [forge tax](../docs/DECOMPILATION.md) measures (composition that doesn't factor through any clean basis). pylm
turns "the decompilable fraction" from a metric into a **running artifact**: how much of a real LLM is a small
symbolic program over flat knowledge, and how much is irreducible.

**Run.** `python pylm/build.py --model gpt2` then `python pylm/validate.py --model gpt2` (add `--no-model` for
corpus-only, no torch). Outputs `runs/pylm/validate_summary.json`.
