# Talema — a language with three grammar rules, built on the universal grammar four LLMs share

Talema is a constructed language designed to be the **smallest exact spelling of the universal grammar** measured in
[UNIVERSAL_GRAMMAR.md](UNIVERSAL_GRAMMAR.md) (the Type–Link–Merge form, TLM). Its grammar is three exceptionless
rules. They convert a TLM tree to a sentence and back, and do nothing else. Its vocabulary is drawn from
**English, German and Spanish, weighted by how often each word is used** in the three languages. It is written in the
plain Latin alphabet.

```
en  She was 84 years old.            de  Sie war 84 Jahre alt.
    veto pare 84-a ba pe sela .          veto pare 84-a ba pe sela .
    old  year 84   be SUBJ she            (identical: both sources have the same UG tree)
```

## 1. The alphabet

14 consonants and 5 vowels, all plain ASCII, pronounced as in Spanish or German:

`p t k b d g m n l r s f v h` · `a e i o u`

A **word** is a **root** plus an **ending**. A root is a consonant with optional vowel–consonant pairs, C(VC)*:
`l`, `mit`, `sofir`. The ending is a run of vowels. Every root ends in a consonant, so the ending always
begins at the last consonant. Names, numbers and words outside the lexicon are written as their source lemma, a
hyphen, then the ending: `Aldrin-a`, `84-a`.

## 2. The grammar: three rules

| rule | says | encodes |
|---|---|---|
| **R1 Link** | Every argument is introduced by a relator word. Prepositions and subordinators are relators, as in TLM. A subject gets the particle **p-**, an object **t-**, an indirect object its own particle. Every other dependent is a bare modifier. | what each link *means* |
| **R2 Order** | A word comes first, then its dependents: complements, then modifiers, then the subject (TLM Merge order). No exceptions. | the order |
| **R3 Arity** | A word's ending counts its dependents: `a` 0, `e` 1, `i` 2, `o` 3, `u` 4, then base 5 (`ea` = 5, `ee` = 6…). | the bracketing |

That is the whole grammar. There is no inflection, agreement, case, gender, or tense morphology. Tense, number and
similar distinctions exist only where the source used a separate word (*will*, *have*, numerals).

**Reading a sentence** is a stack machine. Take a word, read its ending as a number *n*, and the next *n*
phrases are its dependents:

```
"I don't know why I chose her"  →  sevu te koho te sela vaha pe ma fa dara pe ma

sevu   know    u = 4 dependents
├─ te     OBJ     e = 1
│  └─ koho   choose  o = 3
│     ├─ te    OBJ  → sela   she   (a = 0)
│     ├─ vaha  why               (a = 0)
│     └─ pe    SUBJ → ma     I
├─ fa     not     (a = 0)
├─ dara   do      (a = 0)        English "do"-support survives as a word
└─ pe     SUBJ → ma  I
```

### Why these three, and why exactly three

A sentence in Talema must determine a labelled, ordered tree, and a tree like that has three independent kinds of
information: its **shape** (who depends on whom), its **order** (which dependent comes first) and its **labels**
(subject or object). Each rule encodes one of them, with no exceptions:

- Drop R3 and `V N A` could mean `[V [N A]]` or `[V N A]`.
- Drop R2 and the same tree could be spelled many ways.
- Drop R1 and `V N N` no longer says which noun is the subject.

So no rule can be removed without the string ceasing to determine the tree. That argument is informal. What is
**checked** is that the three rules are sufficient: `conlang.py gate` spells every UG tree from all 3,000 English,
German and Spanish PUD sentences and parses it back to the identical tree (3,000/3,000).

### Why head-first

R2 has to pick one direction. Talema picks the one the three source languages favour, weighted by their ORDER
parameters from the TLM prototype:

| head before dependent | en | de | es | mean |
|---|---|---|---|---|
| relator before its complement (prepositions) | 0.97 | 1.00 | 1.00 | 0.99 |
| verb before its object | 0.96 | 0.41 | 0.90 | 0.76 |

So Talema has prepositions and verb–object order like its sources. The price of a *single* direction is that the
subject, which merges last, also follows the verb: Talema is verb–object–subject, like Malagasy. Putting the subject
first (SVO, as in all three sources) would take a **fourth** rule ("specifier first"). Talema keeps three; an SVO
dialect is one rule away. Determiners also follow their noun (`pare la`, "the year"), like Scandinavian or Romanian
articles.

## 3. The vocabulary

`conlang/lexicon.tsv` holds 6,000 concepts, built by `scripts/conlang/lexicon.py`.

1. **Frequencies.** wordfreq gives how often each *word form* is used. The UD treebanks of each language (en
   EWT+GUM, de GSD+HDT, es AnCora) give which lemma each form belongs to. A lemma's frequency is the sum over its
   forms, normalised per language so English, German and Spanish count equally.
2. **Concepts.** A curated core of about 130 closed-class words (articles, pronouns, prepositions, conjunctions,
   auxiliaries: `conlang/core.tsv`) is aligned by hand, because the bilingual dictionaries handle them poorly. Open-class
   words are aligned through the MUSE en↔de and en↔es dictionaries: each English lemma claims its most frequent
   German and Spanish translation of the same word class, and leftover German or Spanish lemmas join the concept of
   their best English translation. **A concept's weight is the sum of its words' frequencies in all three
   languages.** 3,488 concepts have a word in all three.
3. **Word length follows weight.** The 14 heaviest concepts get one-consonant roots (`l` the, `d` of, `b` be,
   `n` in…), the next 980 get CVC roots (`mit` with, `kas` that, `sel` she), and the rest get CVCVC. Given those
   tier sizes, giving the shortest roots to the most frequent concepts minimises the expected root length. That
   follows directly from the rearrangement inequality.
4. **Word shape follows the sources, weighted by frequency.** Within its tier, each concept (heaviest first) takes
   the free root most similar to its source words. Similarity is edit similarity after mapping each language's
   spelling to the Talema alphabet, and it is weighted by the concept's frequency in each language. For example,
   *with / mit / con* gives `mit`, *we / wir / nosotros* gives `vor`, and *my / mein / mi* gives `men`.

The most frequent words:

| root | meaning | en / de / es | | root | meaning | en / de / es |
|---|---|---|---|---|---|---|
| `l` | the | the / der / el | | `kas` | that (conj.) | that / dass / que |
| `p` | SUBJ particle | — | | `mit` | with | with / mit / con |
| `t` | OBJ particle | — | | `for` | for | for / für / para |
| `d` | of | of / von / de | | `tad` | you | you / du / tú |
| `b` | be | be / sein / ser, estar | | `sof` | on | on / auf / sobre |
| `n` | in | in / in / en | | `vir` | will | will / werden / ir |
| `s` | and | and / und / y | | `por` | by | by / durch / por |
| `k` | a | a / ein / uno | | `ken` | can | can / können / poder |
| `h` | he | he / er / él | | `sel` | she | she / sie / ella |
| `g` | to | to / zu / a | | `vor` | we | we / wir / nosotros |
| `m` | I | I / ich / yo | | `ber` | but | but / aber / pero |
| `v` | have | have / haben / haber, tener | | `men` | my | my / mein / mi |
| `r` | his | his / sein / su | | `des` | say | say / sagen / decir |
| `f` | not | not / nicht / no | | `mus` | must | must / müssen / deber |

## 4. How it compares

These numbers come from spelling the same UG trees from all 1,000 PUD sentences per language (`conlang.py compare`,
`runs/conlang/conlang_summary.json`). An "order rule" is one (head class, relation) direction rule that differs from
the language's default direction.

| | English | German | Spanish | **Talema** |
|---|---|---|---|---|
| order rules (head class × relation) | 90 | 86 | 62 | **1** |
| links placed correctly by those rules | 93.4% | 87.8% | 93.4% | **100%** |
| order exceptions left over | 1,170 | 2,135 | 1,318 | **0** |
| order entropy (bits per link) | 0.24 | 0.36 | 0.24 | **0** |
| sentences with discontinuous phrases | 4.7% | 13.5% | 6.3% | **0%** |
| inflectional feature values in use | 50 | 49 | 57 | **0** |
| surface forms per lemma | 1.20 | 1.25 | 1.42 | **1** |
| lexicon coverage (non-name tokens) | 91.9% | 85.9% | 93.1% | — |
| mean letters per word | 4.7 | 5.8 | 4.8 | **3.5 / 3.4 / 3.3** (from en / de / es) |
| grammar rules | — | — | — | **3** |

Talema's words are shorter than its sources' (the frequency-ordered roots), and its sentences are about as long in
characters (held-out PUD: 111 vs 108 characters from English, 117 vs 121 from German, 114 vs 119 from Spanish),
even though role particles add about 3 words per sentence.

**Convergence.** The same PUD sentence rendered from English, German and Spanish shares on average 38% of its
concept roots (Jaccard). Where the sources build the same UG tree, Talema is identical (the example at the top). Where
it diverges, it's because the source languages themselves say different things (*tenía 84 años*, "had 84 years").

## 5. Examples

Held-out PUD sentences, rendered from each source language (`conlang.py translate --sent 6 --max-len 9`). The third line is a
word-by-word gloss in the order Talema speaks it.

```
#125
  en │ She was 84 years old.
     → │ veto pare 84-a ba pe sela .
       │ old year 84 be SUBJ she
  de │ Sie war 84 Jahre alt.
     → │ veto pare 84-a ba pe sela .
       │ old year 84 be SUBJ she
  es │ Tenía 84 años.
     → │ ve te pare 84-a .
       │ have OBJ year 84

#290
  en │ Drop the mic.
     → │ derope te mic-e la .
       │ drop OBJ mic the
  de │ Lass das Mikro fallen.
     → │ leni te Mikro-e la te fala .
       │ let OBJ Mikro the OBJ fall
  es │ Dejar caer el micrófono.
     → │ lari te kabera te micrófono-e la .
       │ leave OBJ caer OBJ micrófono the

#545
  en │ Aldrin has been married three times.
     → │ mabu ba va time tura pe Aldrin-a .
       │ marry be have time three SUBJ Aldrin
  de │ Aldrin war dreimal verheiratet.
     → │ marado dreimal-a ba pe Aldrin-a .
       │ married dreimal be SUBJ Aldrin
  es │ Aldrin se ha casado tres veces.
     → │ mabu va ha vese tura pe Aldrin-a .
       │ marry have he vez three SUBJ Aldrin

#555
  en │ Its importance resides in two facts.
     → │ residi ne heke tova pe potane susa .
       │ reside in fact two SUBJ importance its
  de │ Seine Bedeutung beruht auf zwei Fakten.
     → │ ruheni sofe Fakt-e tova pe benege ra .
       │ beruhen on Fakt two SUBJ meaning his
  es │ La importancia reside en dos hechos.
     → │ residi ne heke tova pe potane la .
       │ reside in fact two SUBJ importance the
```

## 6. Limits

- **Talema ↔ UG form is exact; source language → UG form is lossy.** Inflection, agreement, gender and case are
  dropped (the UG form is lemmas). Lemmas map many-to-one onto concepts, so *ser* and *estar* both become `b`, and
  some senses merge wrongly (Spanish clitic *la* is lemmatised as *él* and becomes `h` "he").
- **Coverage.** The 6,000-concept lexicon covers 86%–93% of non-name word tokens; the rest are written literally
  (`Marktplatz-i`). Names and numerals are always literal.
- **Alignment noise.** MUSE and the treebank lemmatisers are imperfect, so some rare concepts are misaligned. The core
  table fixes the closed classes; open-class entries are automatic.
- **Verb–object–subject order and postposed determiners** are unfamiliar to speakers of the three source languages.
  They are the price of a single order rule.
- **This is a prototype for reading and writing, not a spoken language.** It has no prosody or phonology beyond the
  alphabet, and translation depends on a gold UD parse of the source sentence.

## Reproduce

```bash
.venv/bin/pip install wordfreq            # word frequencies (en/de/es)
mkdir -p data/lexicon && for p in en-de de-en en-es es-en; do
  curl -sS -o data/lexicon/$p.txt https://dl.fbaipublicfiles.com/arrival/dictionaries/$p.txt; done   # MUSE
.venv/bin/python scripts/conlang/lexicon.py --out conlang/lexicon.tsv      # build the lexicon (~6 min)
.venv/bin/python scripts/conlang/conlang.py translate --sent 6              # parallel examples
.venv/bin/python scripts/conlang/conlang.py gate                            # round trip on 3,000 sentences
.venv/bin/python scripts/conlang/conlang.py compare --out runs/conlang/conlang_summary.json
```

Needs `souffle`, the UD PUD treebanks in `data/ud_pud/`, and the sibling treebank checkouts listed in
`scripts/conlang/lexicon.py` (glossa, germandata, satzklar-model). Data licences: MUSE dictionaries CC BY-NC 4.0;
wordfreq data CC BY-SA 4.0; UD treebanks under their own licences. `conlang/lexicon.tsv` is derived from these.
