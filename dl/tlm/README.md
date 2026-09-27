# TLM — the Type–Link–Merge grammar as a runnable Datalog program

`tlm.dl` is the universal grammar of [docs/UNIVERSAL_GRAMMAR.md](../../docs/UNIVERSAL_GRAMMAR.md) written as
Soufflé Datalog. It is the normative artifact. `scripts/grammar/tlm.py` only stages facts, runs Soufflé, and
prints what the program derived.

| rule | Datalog | measured basis (4 LLMs × 13 languages) |
|---|---|---|
| **Type**: features from two universal axes | `type_of/4` from the `type_axis` register | category geometry agrees across models (RSA 0.92–0.96); rank-4 code; LEACE-causal in 52/52 cells |
| **Link**: one head per word; relators head what they introduce | `link/4` (`relator_rel`: `case`, `mark`) | relator heads beat UD content heads 47/52; P 40/52, C 45/52; D 13/52 and T 25/52, so not promoted |
| **Merge**: binary, complements → adjuncts → subject, nearest first | `merge_edge/5`, `span/6` | binary grouping beats flat links 37/52; X-bar bar levels add nothing (14/52) |
| **Order**: per-language linearisation, derived | `order_stat/4` | not stipulated; counted |

The `type_axis` register is generated from the measured category axes (`runs/grammar/*_categories_*`). The
relator set is the measured one. Nothing else is hand-tuned.

## Run

```bash
# needs souffle on PATH and the UD PUD treebanks in data/ud_pud/ (see docs/UNIVERSAL_GRAMMAR.md § Reproduce)
.venv/bin/python scripts/grammar/tlm.py run --langs en de ja ar --sent 3   # invariants, ORDER, TYPE, parallel trees
.venv/bin/python scripts/grammar/tlm.py gate                               # differential gate, all 13 × 1000 sentences
```

Example output (held-out PUD sentences):

```
INVARIANTS (each must be empty)          ORDER: share of links where the head comes first
  bad_head_count            0  ok          lang  relator>complement  verb>object  verb>subject  noun>adjective
  cycle                     0  ok          en    0.97                0.96         0.03          0.00
  bad_root_count            0  ok          de    1.00                0.41         0.18          0.00
  relator_without_complement 0 ok          ja    0.00                0.00         0.00          0.00
  determiner_heads          0  ok          tr    0.01                0.00         0.00          0.00
  uncovered                 0  ok          zh    0.32                1.00         0.00          0.00
                                           ar    0.99                0.97         0.61          1.00
PARALLEL SENTENCE #290
  en: [[Drop [the mic]] .]
  de: [[[Lass [das Mikro]] fallen] .]
  ja: [[[マイク を] 落とす] 。]
```

The same Link rules, spelled out left-to-right with one head-direction choice, give prepositions and VO in
en/es/ru/ar and postpositions and OV in ja/ko/tr/hi. Greenberg's correlation falls out of the counts rather than
being written in. German (0.41, verb-second with verb-final clauses) and Chinese (0.32, prepositions plus
postpositional localisers) are the known mixed cases.

## What it proves and what it doesn't

- **Invariants** (empty relations over every parsed sentence): one head per word, acyclic, single root, each
  promoted relator heads exactly one complement, determiners never head, and the root's maximal projection covers
  every word. These show the three rules always build a well-formed tree. They do not show the tree is *right*.
- **Differential gate** (`tlm.py gate`): Link heads and Merge leaf-to-leaf distances are identical to the Python
  trees the LLM measurements were made with (`ug_trees.function_head` / `_phrase_tree`, relators = {case, mark})
  on all 13,000 PUD sentences. This proves the program *is* the tested theory. Like germandata's gates, it proves
  agreement between implementations, not linguistic correctness.
- **Input.** The prototype restructures gold UD analyses; it is not a parser. Type comes from gold UPOS mapped
  through the measured axes. A parser (or the LLM itself, via the probes) would supply those facts in a full
  system.

Tag: the rules' *content* is `empirical` (probe measurements); the program's *well-formedness* on PUD is checked
by the invariant relations, and its *equivalence* to the measured theory by the gate.

Implementation note: every `count` names every variable, with single-use ones written `_x`. Over a join, Soufflé
treats `_` as existential and counts distinct bindings of the named variables. An earlier draft's `_` silently
turned link counts into sentence counts (English prepositions came out at 31%).
