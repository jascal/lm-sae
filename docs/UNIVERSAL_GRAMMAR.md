# A universal grammar that four LLMs share — and it is simpler than X-bar

**Question.** Across LLMs and across languages, is there one grammar the models' internal geometry agrees on —
and is it simpler to state than Chomsky's X-bar theory?

**Answer (empirical, stated domain below).** Yes. Four LLMs from three families (Qwen2.5-0.5B/1.5B, Llama-3.2-1B,
Gemma-2-2B), read on the same 1000 sentences translated into 13 languages (UD PUD: en de es fr ru zh ja ko tr hi ar
id fi), share one grammar. It has three rules:

> **TYPE–LINK–MERGE grammar**
> 1. **Type.** Every word gets a type from a small universal space (~4–8 dimensions). Its two main axes are
>    *referent ↔ relator* (names, numbers, nouns, adjectives vs. verbs, auxiliaries, adpositions, subordinators,
>    conjunctions) and *content ↔ function*.
> 2. **Link.** Every word hangs from exactly one head, and the link has a type (subject, object, modifier …) from a
>    universal inventory. **Relators are heads**: an adposition heads its noun (*[of → house]*) and a subordinator
>    heads its clause (*[that → he left]*). **Determiners are not heads** (*the* hangs from *house*, not vice versa).
> 3. **Merge, one at a time.** A head takes its dependents one by one, closest-bound first (complement, then
>    modifiers, then subject). There are no bar levels, no specifier/complement positions, and no empty projections.
>
> Word order is a separate, per-language choice of how to spell the tree out left to right.

Compared with X-bar, that drops every piece of X-bar machinery that the models do not show:

| X-bar claim | what the four models show | cells (model × language) |
|---|---|---|
| intermediate bar levels X′/XP on every head | **no gain**: X-bar fits no better than bare Merge (tie within seed noise) | X-bar > Merge in 14/52 |
| the DP hypothesis (determiner heads the noun phrase) | **rejected**: promoting determiners to heads makes the fit *worse* | 13/52 in favour |
| TP (auxiliary heads the clause) | **no effect** either way | 25/52 |
| PP and CP (adposition / complementizer heads) | **supported** | 40/52, 45/52 |
| Chomsky's ±N/±V category square | **half**: the ±N split is the main axis; ±V does not appear | parallelogram cos 0.05–0.16 |
| content words head (UD's choice) | **rejected** in favour of relator heads | function-head > UD in 47/52 |

So the grammar the models agree on is *dependency links with relator heads, merged binarily* — the part of Minimalist
Merge that survives, without the X-bar scaffolding. It can be taught in three sentences.

Tag discipline: everything here is **empirical** (linear probes, rank correlations and one causal erasure on four
0.5–2.6 B decoder LMs; 13 languages of one parallel news/Wikipedia corpus). Nothing here is `proved`. "The models do
not show X" means "a linear readout of the mid-layer geometry gains nothing from X", not "X is absent".

---

## 1. Grammar is learned, shared across languages, and shared across models

**Grouping.** For each theory we train a structural probe (Hewitt & Manning 2019; rank 128) whose squared distances
should equal the theory's word-to-word tree distances. It is trained on all 13 languages and scored on held-out
sentences. The statistic is the rank correlation between probe distance and tree distance, after partialling out
(a) word order and (b) every theory's per-word *centrality* (how high a word sits, which is predictable from the word
type alone). What remains is *which words group together*. This is context, not vocabulary.

| model | embedding layer (lexical-only control) | mid layer |
|---|---|---|
| Qwen2.5-0.5B (L12) | 0.06–0.07 | **0.37–0.41** |
| Llama-3.2-1B (L8) | 0.06–0.07 | **0.40–0.43** |
| Qwen2.5-1.5B (L14) | 0.06–0.07 | **0.39–0.43** |
| Gemma-2-2B (L13) | 0.07 | **0.38–0.41** |
| Qwen2.5-0.5B **untrained** (L12) | — | 0.01 |

The embedding layer and the untrained network carry essentially no grouping, so the structure is learned from data.

**Categories and relations transfer to unseen languages.** A probe is trained on 12 languages and tested on the 13th
(leave-one-language-out; mean over the 13; non-punctuation, non-numeral words):

| model | UPOS | relation (deprel) |
|---|---|---|
| Qwen2.5-0.5B | 0.71 | 0.55 |
| Llama-3.2-1B | 0.73 | 0.58 |
| Qwen2.5-1.5B | 0.73 | 0.57 |
| Gemma-2-2B | 0.73 | 0.58 |
| untrained | 0.23 (≈ majority) | 0.08 |

Transfer is highest within Indo-European (0.8–0.9) and lowest to ja/ko (0.4–0.5), where both script and word order
differ. Tree *edges* (UUAS from a minimum spanning tree) do **not** transfer above the adjacent-word baseline
(0.46–0.49 vs 0.50). The shared structure is graded grouping, not a directly recoverable tree. This is an honest limit
of a symmetric distance probe on left-to-right models: a word cannot see its later dependents.

**The category code is small, and the same in every model.** A rank-*k* bottleneck probe trained on 12 languages
reaches 95% of its held-out-language accuracy at k = 4 and saturates at k ≈ 8 (Qwen-0.5B: 0.38 / 0.53 / 0.68 / 0.71 at
k = 1 / 2 / 4 / 8). The class-centroid geometry (RSA between category-distance matrices, same language) agrees across
models:

| | Llama-1B | Qwen-1.5B | Gemma-2B | Qwen-0.5B embedding layer | untrained |
|---|---|---|---|---|---|
| Qwen-0.5B | **0.95** | **0.96** | **0.92** | 0.26 | 0.52 |
| Llama-1B | | **0.96** | **0.95** | 0.26 | 0.52 |
| Qwen-1.5B | | | **0.94** | 0.25 | 0.54 |

In every model the top two axes are the same (per-language rank agreement 0.92–0.99; cross-model axis agreement
0.7–0.9):

- **axis 1 (41–48%)**: PROPN, NUM, NOUN, ADJ (+) ↔ AUX, SCONJ, VERB, CCONJ, ADP (−). This is *referent vs relator*,
  and it is the ±N split of Chomsky (1970): [+N] = {N, A}, [−N] = {V, P}.
- **axis 2 (24–31%)**: VERB, NOUN, ADJ (+) ↔ CCONJ, DET, NUM (−). This is *content vs function*.

Chomsky's second feature, ±V (predicted parallelogram N−A ∥ P−V), does not appear (cos 0.05).

**Causally used.** A closed-form LEACE eraser (Belrose et al. 2023) removes all linearly decodable UPOS information.
It is fitted on the *other 12 languages* and applied to the held-out language's residual stream. Erasing it hurts
next-token prediction more than a same-rank random erasure in **52 of 52** model × language cells (§4). The code the
languages share is one the model *uses*, in languages it was not fitted on.

## 2. Which tree? Five rival theories, one geometry

Every theory is derived deterministically from the same gold UD tree, so the theories differ *only* in the structural
claims they add (`scripts/grammar/ug_trees.py`):

| theory | adds |
|---|---|
| `linear` | nothing — word order only (null) |
| `ud` | content-head dependency tree (Universal Dependencies) |
| `fhead` | function words (adposition, determiner, auxiliary/copula, subordinator) head their content word; the subject moves to the top of the verbal chain (SUD-like) |
| `merge` | `fhead` + binary Merge: a head combines with its complements (nearest first), then adjuncts, then the specifier; no unary nodes |
| `xbar` | `merge` + X-bar bar levels: every head projects X0 → X′ → XP even when nothing attaches |

Own grouping fit (mid layer, 13 languages pooled; paired sentence-bootstrap 95% CIs are ±0.005; probe-seed variance
≤ 0.002):

| model | ud | fhead | merge | xbar |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.372 | 0.405 | **0.411** | 0.408 |
| Llama-3.2-1B | 0.400 | 0.421 | **0.431** | 0.427 |
| Qwen2.5-1.5B | 0.395 | 0.421 | **0.428** | 0.425 |
| Gemma-2-2B | 0.380 | 0.399 | **0.406** | 0.401 |

Consistency across the 52 model × language cells:

| contrast | cells in favour | mean Δ |
|---|---|---|
| `fhead` > `ud` (relator/function heads) | **47/52** | +0.024 |
| `merge` > `fhead` (binary grouping) | 37/52 | +0.007 |
| `xbar` > `merge` (bar levels) | 14/52 | −0.004 |

The ordering is the same at other depths: Llama L5/L11, Qwen-0.5B L8/L16, Qwen-1.5B L9/L19, and Gemma L9/L17 (where
xbar and merge tie). At the embedding layer the ordering is reversed (ud highest), so the preference is contextual,
not lexical. The large effect is headedness; binary grouping is a small real refinement; bar levels are not supported.

**Which function words are heads?** Each class is promoted on its own over UD:

| promoted | cells in favour | mean Δ |
|---|---|---|
| subordinator (`mark`, C) | **45/52** | +0.009 |
| adposition / case marker (`case`, P) | **40/52** | +0.008 |
| auxiliary / copula (T) | 25/52 | +0.001 |
| determiner (D) | 13/52 | **−0.004** |

The words that *relate one phrase to another* (P, C) behave as heads. The determiner does not.

## 3. How this was guarded against artifacts

- **Lexical artifact.** An early version of the theory test, which controlled only for word order, showed the same
  "phrase structure wins" pattern *at the embedding layer*. The reason is that function-head and X-bar trees make
  word height predictable from word type. The centrality control removes this: the embedding layer drops to ≈ 0.07
  for all theories and the untrained network to ≈ 0.01.
- **Optimiser artifact.** The first structural probes (plain Adam) spiked and, on Qwen-1.5B, failed outright (every
  theory ≈ 0.04). All reported numbers use gradient clipping plus cosine decay, which converges identically at
  20 and 40 epochs. Theory results from the unstable probe were discarded. The one surviving file from that era,
  `qwen05_sweep_summary.json` (used only to pick layers from the category-probe curves), still carries old-optimiser
  `struct` numbers; don't quote those.
- **Train/test.** PUD is parallel, so test sentences (index % 5 == 0) are unseen *in every language*. Language
  statistics are unlabelled per-language z-scores.
- **Runtime.** fieldrun's own residual stream (rebuilt from its final-norm-folded `--source-dump` writes, divided by
  γ) matches HF `hidden_states[12]` at mean cosine **0.9996** (min 0.997) over 6,099 positions in en/de/es/ja/zh
  (`runs/grammar/fieldrun_parity_summary.json`). The measurements are properties of the model, not of one runtime.

## 4. Semantics, causality and codex's fieldrun analyzer

**Codex's shared-write-subspace test, now on real multilingual text** (`scripts/disassembly/fieldrun_grammar.py`,
Qwen2.5-0.5B-Instruct via fieldrun, 40 PUD sentences per language). The top-16 layer-write subspace overlaps
**0.68** across en/de/es/ja/zh, against 0.74 within one language (document halves) and 0.02 for random subspaces.
The shared subspace is real. But read through the *unembedding*, those directions do not favour closed-class words
(0–4%; their extreme tokens are junk). The universal grammar lives in the relational **geometry** that probes and
LEACE read, not in logit-readout directions, consistent with FINDINGS' "the core is not the readout subspace".

**Proto-roles (Dowty 1991).** Is the shared *relation* code syntactic (subject/object) or semantic (agent/patient)?
A subject-vs-object probe is trained on active clauses and applied to passive subjects (syntactic subjects,
semantic patients). Position 0 means "looks like a subject" and 1 means "looks like an object". Preverbal passive
subjects sit at 0.12–0.15, but a left-to-right model has not yet seen the passive verb at that point. Post-verbal
passive subjects (n = 34, pooled) sit at **0.47** (embedding layer 0.34). Once the passive is visible, the
representation moves halfway toward "patient". The relation code mixes grammatical function with proto-role.
Suggestive only; the sample is small.

**Causal erasure, all models** (`ug_causal.py`; ΔNLL in nats on the first subword of held-out words, mean over the 13
languages; the LEACE eraser is always fitted without the target language):

| model | erase shared UPOS code: function / content | same-rank random erasure | languages shared > random |
|---|---|---|---|
| Qwen2.5-0.5B | +0.37 / +0.33 | −0.02 / +0.04 | **13/13** |
| Llama-3.2-1B | +0.23 / +0.74 | +0.07 / +0.20 | **13/13** |
| Qwen2.5-1.5B | +0.43 / +0.37 | +0.15 / +0.11 | **13/13** |
| Gemma-2-2B | +0.20 / +0.32 | +0.02 / +0.05 | **13/13** |

The hypothesis that the shared code is mainly a *function-word skeleton* (function-word targets hurt more) is **not**
supported: content-word prediction suffers as much or more (Llama: 3×). It is a general category code. Qwen-1.5B's
random control is itself large in ko/tr, so that model's margin is the least clean.

**Proto-roles across models.** Post-verbal passive-subject position (0 = like an active subject, 1 = like an object;
n = 34 pooled): Qwen-0.5B 0.47, Llama-1B 0.55, Qwen-1.5B 0.43, Gemma-2B 0.26. Preverbal: 0.13–0.15 in every model.
All four models move a visibly-passive subject partway toward "patient". The relation code is partly semantic in
every model, most in Llama and least in Gemma. Small n; suggestive.

## 5. A runnable prototype: `dl/tlm/tlm.dl`

The three rules are implemented as a Soufflé Datalog program ([dl/tlm/README.md](../dl/tlm/README.md)).
`scripts/grammar/tlm.py run` parses PUD sentences with it, checks well-formedness invariants (all empty), prints
parallel sentences as Merge trees, and *derives* each language's word-order parameter. Greenberg's VO↔preposition /
OV↔postposition correlation falls out of the counts. `tlm.py gate` proves the program's Link heads and Merge
distances identical to the Python trees these measurements used, on all 13 × 1000 sentences.

## 6. Scope and open questions

- Four decoder LLMs, 0.5–2.6 B parameters, one mid layer each (plus depth checks); linear probes. Larger models,
  encoders and non-transformers (Mamba) are open.
- One parallel corpus (news and Wikipedia). Theory trees are *our* deterministic conversions of UD. A different
  conversion (e.g. where the subject attaches, or adjunct order) could shift the small Merge-vs-fhead margin; the large
  headedness margin was stable across every variant tried.
- The ±V feature, dependency *edges* across languages (UUAS), and the proto-role effect are open or weak.
- Next steps: Minimum Description Length probing (Voita & Titov 2020) to put the Occam comparison in bits; a
  mixed-effects model (theory × model × language); LEACE erasure of the *relation* and *tree* codes (not just
  categories); and interchange interventions on head identity.

## Reproduce

```bash
# data: UD PUD treebanks → data/ud_pud/ (gitignored)
for p in English:en German:de Spanish:es French:fr Russian:ru Chinese:zh Japanese:ja Korean:ko Turkish:tr \
         Hindi:hi Arabic:ar Indonesian:id Finnish:fi; do L=${p%%:*}; c=${p##*:}
  curl -sS -o data/ud_pud/${c}_pud.conllu \
    https://raw.githubusercontent.com/UniversalDependencies/UD_${L}-PUD/master/${c}_pud-ud-test.conllu; done
cd scripts/grammar
../../.venv/bin/python ug_extract.py --model Qwen/Qwen2.5-0.5B --langs en de es fr ru zh ja ko tr hi ar id fi --layers 0,12 --out ../../data/ug
../../.venv/bin/python ug_theory.py --model-dir ../../data/ug/Qwen2.5-0.5B --layer 12 --src all --out ../../runs/grammar/qwen05_theory_L12_summary.json
../../.venv/bin/python ug_theory.py --set headedness --model-dir ../../data/ug/Qwen2.5-0.5B --layer 12 --src all --out ../../runs/grammar/qwen05_headedness_L12_summary.json
../../.venv/bin/python ug_probe.py loo --model-dir ../../data/ug/Qwen2.5-0.5B --layer 12 --out ../../runs/grammar/qwen05_loo_L12_summary.json
../../.venv/bin/python ug_categories.py --model-dir ../../data/ug/Qwen2.5-0.5B --layer 12 --out ../../runs/grammar/qwen05_categories_L12_summary.json
../../.venv/bin/python ug_causal.py --model Qwen/Qwen2.5-0.5B --model-dir ../../data/ug/Qwen2.5-0.5B --layer 12 --out ../../runs/grammar/qwen05_causal_L12_summary.json
../../.venv/bin/python ug_semantic.py --model-dir ../../data/ug/Qwen2.5-0.5B --layer 12 --out ../../runs/grammar/qwen05_protorole_L12_summary.json
../../.venv/bin/python ug_summary.py        # cross-model tables → runs/grammar/universal_grammar_summary.json
```

Models × layers used: Qwen2.5-0.5B L12, Llama-3.2-1B L8, Qwen2.5-1.5B L14, Gemma-2-2B L13 (plus embedding layer 0
and `--random-init`). Every run writes a `runs/grammar/*_summary.json`.
