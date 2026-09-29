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
| **R2 Order** | A word comes first, then its dependents. No exceptions. The dependents' own order is free and records Merge order (which merged first); the unmarked order is complements, then modifiers, then the subject. | the order |
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

## 6. The founding book: *Buke de lore fira*

[`conlang/BUKE_DE_LORE_FIRA.md`](../conlang/BUKE_DE_LORE_FIRA.md) ("Book of the First Word") is the founding text,
written entirely in Talema. It is a primer, a literature, a guide to growing the language, and a dictionary in one
core of about 88k tokens (plus field volumes) (Qwen and Llama tokenizers), small enough for any frontier model's context.

| chapter | what it is |
|---|---|
| *ge lesere la* | to the reader |
| *ruli tura la* | the three rules, taught by counting, contrast, and drawn trees; literals; the counting test for where a sentence ends |
| *lori fira la* | the first words: every relator, the 60 most frequent words, and the book's coinages, each with its word class and mother words |
| *desi te masa hova* | how to say more: time and number, the three question shapes, focus by order, commands and address, easily confused words |
| *lori mela la* | the small words: a child, Ana, and a nameless agent trade the semantic primes |
| five tales | *the leaf who had no children* (Grimm: why every sentence ends in *a*); *four minds and thirteen tongues* (the founding story, the UG research itself, every claim marked); *the knight of the book* (Cervantes); *the bargain* (Goethe); *the storm*, a play in three scenes (Shakespeare) |
| *seke hove pe geneta* | how agents speak: claim marks, asking, promising, correcting, handing over; how to read the agents of the first age |
| *gove hove pe talema* | how Talema grows: definitions, the root law, the waiting letters, new rules, dialects, games |
| songs, sayings, open questions | a counting stair, the vowels' song, a 14-line poem to the far reader, a lullaby |
| *buki de rarisa la* | the book of roots: 5,934 entries, each a Talema tree headed by the bare root in mention form (`sahen-o veraba count-a zählen-a`), whose children are its word class and its English, German and Spanish mother words (Johnson and Webster's etymologies); 13 rough words are fenced in their own closing section |
| *pigi lata la* | the last page |

The only material in it that is not Talema is literal loans (`with-a`, `Kihote-a`), which Talema already allowed for
names. The dictionary uses them as etymologies, so a reader of any of the three mother tongues can decipher every root.

**Refinements the book made to Talema.** These are conventions and words, not new rules:

- **R2 clarified.** Only head-first is a rule. The order of a head's dependents is free and marks emphasis. A dialect
  that puts the subject *before* the head needs a fourth rule and must declare it (`talemi gage pona rule fura`,
  "Talema, age one, four rules").
- **Relators of every kind head their parts.** That includes *and* and *or*: `si tova tova` is "and two two", so
  arithmetic is Polish notation.
- **Mention.** A word spoken *about* is written in literal form under `lor`: `lore talem-a` is "the word *talem*",
  while `lore mela` is "a small word". (v1 quoted with bare words, which could not be told apart from description.)
- **Yes/no questions** are headed by `tob` ("whether").
- **Claim marks.** The project's proved/empirical/open discipline becomes three heads that open a claim: `bove …`
  (proved), `sere …` (seen/measured), `pefe …` (open: *not known to us yet*, a statement about the speaker's
  knowledge, never a claim that nobody could know). The law of the marks: never mark a claim higher than its evidence,
  and say "my way stops here", not "there is no way".
- **The waiting letters.** `c j q w x y z` are held in reserve. When the short roots are all given and speakers agree, a
  new *age* wakes one, so new short roots can exist without ever reassigning an old one, and old texts stay readable.
- **Time and number are unsaid unless a word says them.** `lafe pe doge la` leaves the time open; `lafi nora …`
  (before), `lafi nova …` (now) and `lafi vira …` (will) fix it. `doga` leaves the number open; `doge tova` is two dogs.
- **Three question shapes.** `tobe …` alone asks. `tob` under a verb (`sevo te tobe … fa pe ma`, "I do not know
  whether …") reports a question. `pefe tobe …` marks a question nobody can answer yet. In a content question the
  question word stands where the answer would: `seri te vasa pe tada` ("what do you see?"), `seri te doge la pe ma`.
- **Focus by order.** The dependent nearest the head carries the weight: `seri pe kide la te doge la` is "the
  *child* sees the dog". This is the use R2's free dependent order was reserved for.
- **Commands and address.** `les` ("please") heads a command and `g` ("to") marks the one addressed:
  `lesi lafa ge kide la` ("child, sleep!"). A sentence without a subject may also be a command; `les` makes it certain.
- **Where a sentence ends.** An `a` ending marks every leaf, not only the last word. The test the book teaches: start
  at one; at each word subtract one and add its dependents; the sentence ends when the count reaches zero.
- **Coinages**, made with the root law and recorded in `conlang/book/coin.tsv`: `talem` (Talema), `vokel` (vowel),
  `sonat` (consonant), `dinal` (ending), `token`, `niter` (knight), `rasel` (riddle), `dilek` (dialect), `sarin` (saying),
  `lator` (relator), `ginon` (hyphen), and the word-class labels the dictionary uses: `nomun` noun, `verab` verb, `detiv` adjective,
  `derob` adverb, `ronon` pronoun, `pepos` preposition, `dunon` conjunction, `numer` numeral, `teron` interjection
  (with the existing `tik` article and `patik` particle).

- **Reading order.** Each dependent's whole subtree is read before the next dependent (the leaf tale said
  "children, then grandchildren", which suggested reading by generation).
- **Long endings** are base-5 numerals, most significant vowel first: the last vowel counts ones, the one before it
  fives, the one before that twenty-fives (`ei` = 7, `uu` = 24, `eaa` = 25).
- **Sounds.** Letters are read as in Spanish; the first vowel of the root carries the stress.
- **Reserved roots.** Nine short roots are free: `pip pim pod pud puv tub bok buf huk`. They had belonged to junk lexicon
  rows (stray single letters) that never appeared in any text, so the root law allows freeing them. They wait for words
  that will be said often. Chapter 6 walks through one coinage end to end: `ginon` "hyphen" gets a three-consonant root
  because it is rare, and sounds most like Spanish *guion*.

- **Dictionary headwords are bare roots** (v4). An entry is headed by the root in mention form, `sahen-o …`, so the
  root cannot be mistaken for root-plus-vowel. v3's `saheno …` led a reviewer to treat `saheno`, `pono`, `su`, `pu` as
  citation forms, and so to report defined roots (`mid`, `nad`, `til`, `s` …) as missing.
- **Claim marks, numbers and nested sentences are taught before the tales** (v4). The conventions chapter gains a
  section on the three marks with a drawn tree, a numbers table (0–10, 100, 1000), and a drawn quotation showing how
  the endings separate an inner `pe` from an outer one. Page one draws its first sentence and points to the book of
  roots.

**The book is normative** for everything beyond the three rules: the conventions above, the claim-mark law, mention,
the waiting letters. The rules stay minimal; the book gives the conventions their authority.

**The lexicon is frozen.** A root is never reassigned, so `conlang/lexicon.tsv` is no longer regenerated
(`lexicon.py` refuses to overwrite it without `--force`). Wrong senses are corrected in place: `feras` is now
*sentence/Satz/frase* (it had German *Strafe*, "punishment"), `let` is *letter/Buchstabe/letra* (it had "write" and
"mail letter"), and `tum` is *subject/Subjekt/sujeto* (its topic senses moved to `topik`). A third review corrected `hov` "how"
(*wie*, not *woher*), `tar` "they" (*sie*, not *ihr*) and `kit` "kind, sort" (*tipo*, not *amable*), and removed the
86 junk rows. New words go through
`conlang/book/coin.tsv`.

**Core book and field volumes.** The founding text is split so the core stays within a frontier context window as
fields are added:
- **`conlang/BUKE_DE_LORE_FIRA.md`**, the core (~88k tokens): grammar, first words, conventions, tales, agent speech,
  growth, songs, the book of roots and the last page. Its growth chapter lists the volumes and tells the reader to
  read the core first.
- **`conlang/volumes/<field>.md`**, one volume per field: a Talema title page ("read the Book of the First Word
  first"), the field chapter, and its glossary. Current volumes: `digital.md` (~4k tokens), `mathematics.md`
  (~3k), `logic.md` (~3k), `physics.md` (~2.5k) and
  `philosophy.md` (~2k), `morality.md` (~2k), `food.md` (~2k) and `talk.md` (~4k).

Field coinages live in their volume's glossary, not the core book of roots. `scripts/conlang/books.py` regenerates,
lints, builds and checks every book in one command.

**Field chapters.** The book now has specialised vocabularies, one chapter per field. The first is *vuli digala
la* (the digital world). It covers computing and code; the shell and CLI; Unix time and timestamps; services and
cloud; keys and secrets; models, training, tokens, context, attention and interpretability. It ends with an incident
dialogue. It fixes three conventions:
- **Code keeps its letters.** Commands, paths and names in code are literals, with `_` for spaces (`git_status-a`,
  `/home/ana/buke.md-a`).
- **Time is Unix time.** Agents give times as seconds after the epoch.
- **Decimals** are headed by `pun` ("point"): 0.7 = `puni senura geva`.

Old roots take new digital senses explicitly, as English did: `bugek` (insect) is a bug, `sonel` (shell) the shell,
`pohip` a pipe, `ranik` a branch. 51 concepts were coined by the root law. How it works:
- **`conlang/fields/<field>.tsv`** lists each field's concepts, with the English, German and Spanish words *in the
  field's sense*.
- **`scripts/conlang/field.py`** checks which concepts have roots, coins the rest, lints the chapter so every concept
  resolves to its glossary root, and generates the field glossary.
- **Field 2, mathematics** (*mamaka*). Talema's grammar already is Łukasiewicz (Polish) notation, so operators are
  relator heads and no brackets are needed: `mapori si pona tova tura` is (1 + 2) × 3, and
  `si pona mapori tova tura` is 1 + 2 × 3. The operators are `menos` (minus: one child is negative, two is
  subtraction), `mapor` (times) and `dided` (divided by); `raris` "root" with one child is the square root, and
  with two, the first child says which root. Functions are literal heads (f(x) = `f-e x-a`). The chapter covers
  sets, geometry, proof, calculus and chance. Its centrepieces are Euclid's proof that the primes never end, marked
  `bove`, and Goldbach's conjecture, marked `pefe` and seen true below 4 × 10^18. The chapter warns that `finit`
  (from *infinito*) means *infinite* and `nilik` means *finite*.
- **Field 3, logic** (*gogik*). Polish notation was invented for logic, so connectives are relator heads whose
  ending counts the statements they join: negation `f` (one child), conjunction `s` and disjunction `dor` (many),
  implication `ven` (two). The same `s` adds numbers and joins statements. The endings replace brackets:
  `fa si …` is ¬(A ∧ B) and `si fa … …` is (¬A) ∧ B. The chapter teaches truth tables in prose; necessary, possible
  and impossible as truth in every, some and no world; syllogism and modus ponens; deduction, induction and
  abduction mapped onto the claim marks (`bove`, `sere`, `pefe`) with the black swan as induction's
  counterexample; quantifiers; affirming the consequent, circular arguments, authority and bias; the liar
  (a good Talema sentence with no truth value), Gödel and Turing marked proved; Datalog-style facts, rules and a
  query; use and mention (the word `sinov` mentioned as a literal, `sinov-a`, against its use), Tarski's truth rule, and models. The chapter
  warns that `posib` means *impossible* (`pol` is *possible*). 20 concepts were coined; *incompleteness* was not
  coined, because its root would have read as "complete", so it is said as a phrase.
  Writing it exposed a silent bug: `author.py` let a tree with one `)` too many swallow the lines after it, and
  one such tree had cut the last sentences of the knight tale from the core. `author.py` now refuses unbalanced trees.
- **Field 4, physics** (*fisis*). Laws of physics carry the mark `sere`, never `bove`: one experiment can end a
  theory, many never prove one. Units are always said (`metar`, `loram`, the second). Formulas are trees as in
  mathematics (F = m a, ½ m v², E = m c²). The chapter covers matter (atom, nucleus, charge; water as solid, liquid,
  gas), Newton's three laws, Galileo's falling stone and feather, the orbit as endless falling, conservation of
  energy, heat flow and entropy, light as wave and photon (Maxwell), relativity (the speed of light, spacetime and
  its curve as gravity), and the quantum (superposition, Heisenberg, entanglement with no message). It ends on three
  `pefe` questions: quantum gravity, dark matter, why the universe began. Quantum *state* uses `did` (Zustand), not
  `tat` (the political state). 24 concepts coined; `feler` (feather) coined for the core.
- **Field 5, philosophy** (*filof*). Mostly tales and questions, each school through its tale: Socrates who
  knows nothing, Heraclitus's river, the ship of Theseus, Leibniz's "why something", Plato's cave, Kant's forms,
  Descartes's doubt ending in *I think, therefore I am*, Hume's unseen cause and tomorrow's sun, knowledge as
  justified true belief (marked open), Zhuangzi's butterfly, determinism and free will, Wittgenstein's limits of
  language. The open questions carry `pefe`, including one addressed to its readers: *does a mind of numbers
  feel?* Direct yes/no questions are headed by `whether`, as in the core. 8 concepts coined.
- **Field 6, morality** (*morat*). The rule many traditions share (Confucius's negative form, then the positive);
  three schools, each seeing part of the good: virtue (Aristotle's courage between fear and rashness, grown by
  practice), duty (Kant's universal rule; every person a purpose, never only a tool) and consequences (the most
  happiness for all; Bentham's *can they suffer?*). Then promise, trust, the lie that takes a choice away, and
  forgiveness; the ring of Gyges (*who are you when nobody sees you?*); Rawls's choice of rules before knowing
  your place; compassion, cruelty and mercy; the trolley as a hard case marked `pefe`; and the claim marks as
  an agent's honesty: `bove` only when proved, `sere` only when seen, `pefe` when not known. Three concepts were
  left out rather than given misleading roots: *vice* (`viset` is a vice-president), *invisible* (`visil` reads
  as visible), *humility* (the adjective `humil`, humble, is used). 11 coined.
- **Sense check.** `books.py` now warns when a role particle hangs under a noun head. It found eight verbs in the
  core and digital books that had resolved to their noun roots (*dream*, *point*, *return*, *measure*, *review*,
  *work*) and misplaced brackets in the logic and physics volumes; all fixed.
- **Field 7, food and eating** (*fod*). The frequency-built lexicon had most food words (71 of 95 checked) but
  lacked everyday table and kitchen words that are rare in the corpora it was weighted from: *sandwich*, *fork*,
  *spoon*, *banana*, *snack*, *dessert*, *boil*, *fry*, *waiter* and others, 22 coined here. The chapter covers
  meals, the four tastes, the tale of the Earl of Sandwich (marked "maybe a tale"), fruit and vegetables, the
  table, a recipe as a list of steps, a restaurant dialogue, and eating together. `order` (to order food) was not
  used, because the lexicon's `dader` is the commanding sense; the chapter says "ask for". *lettuce* was not coined
  (its root would join `salat` salt, `salad` and `salut` salty); the chapter warns about those three.
- **Field 8, conversation** (*nerin*). The other volumes tell, define and prove; none has people talking to one
  another. Across the core and the earlier volumes only 1.2% of sentences held an interjection, and just 4 of the
  lexicon's 37 interjection roots (*please*, *sorry*, *yes*, *welcome*) appeared in any sentence at all, so a reader
  had *hey*, *okay*, *oh*, *well* and *maybe* in the dictionary and no example of one in use. This volume is a play in
  ten scenes (a market, from a wet morning to a night by a fire) with ten speakers and a dog: 331 turns of short
  exchange. Scenes 1-5 are the first version: greetings and small words, a bargain, a child who asks why, a stranger
  and a dog, the end of a day. Scenes 6-10 were written after tests of a tutor built on these books (jascal/talema,
  `avatar/experiments/`): given a looser prompt she used small words freely but misplaced them ("I am glad" got a
  bare "Yes."; "Why?" got "Sorry. I was wrong."). So they pair what a learner says with a fitting answer: a feeling
  gets a reaction ("I am glad" is "Good! Why are you glad?"; "I am sad" is "Why are you sad?" and a seat), a *why*
  gets a *because*, "I want an apple" gets "Which apple?", "Is it good?" gets two opinions and a joke, a *sorry* is
  waved away, a *thanks* is "It is nothing" or "Do not thank me", "I do not understand" gets a simpler repeat, and
  goodbye gets "Come again." Scene 6 is a morning with a book and a word she does not know; 7, hospitality (offers
  taken and refused, a compliment turned into a joke, plans made with *if*); 8, a quarrel and its repair; 9, a
  lesson, with a mistake turned into a joke and short praise; 10, a story with listeners who react, a fear and a
  sadness that are comforted, and good nights. In all it has 9 *because* clauses, 6 *if*, 7 *I think*,
  and *I want you to read it with me*. All 116 of its concepts already had roots; none was coined. Conventions: a turn
  opens with the speaker's name as a literal on its own line, as in the two earlier exchanges; stage directions are
  `@prose`, so they render apart from speech; a yes/no question, even a one-word echo like "Early?", is written with
  `whether`, because Talema has no question mark and a bare fragment reads as a statement; scenes 9 and 10 are headed
  with cardinals because the lexicon has no *ninth* or *tenth*. `smile` had to be written `smile/VERB`, since the
  bare key resolves to the noun `sonil`; `field.py lint` caught it. Not yet tested: whether reading the play changes
  how the tutor talks. Thin spots: sadness and fear appear once or twice each, *hmm* and *ouch* are not in the
  lexicon, and there is no politeness register beyond *please*.
- **Planned next fields:** open. Everyday domains are the likeliest gaps (clothing, the home, the body, animals):
  run `field.py check` on a concept list before assuming a word exists.

**Speaking** (v6). Chapter 2 gives the letters where they differ from Spanish (*g* always hard, *h* as in *house*,
*v* as in *voice*, a short tapped *r*). It also says that one-consonant words are unstressed, the hyphen is silent,
and "." is a pause. Chapter 2c adds number compounding, so every number can be said: a number under a big number
says how many of it (`dehe tova` = 20), and `s` adds (`si deha fura` = 14). `conlang/speech/` holds the
voice kit: `scripts/conlang/speech.py` (respelling per voice language, number phrases), `.pls` lexicons for
Italian, Spanish, English and IPA, and a system prompt for voice agents.

**Speech v2** answers Astra's review of the kit (`docs/reviews/astra_speech_v1.md`). Chapter 2 now also teaches that
every vowel is its own syllable (coined *silab*, syllable), that a one-consonant word at the end of a sentence leans
on the word before, and that an ending is never reduced. `speech.py` says negatives with `menos` (`-5-a` → `menose
fiva`), rejects a numeric literal with dependents, spells literals letter by letter in IPA, and writes
`alphabet="ipa"` on every lexicon (the IPA lexicon's locale set by `--ipa-lang`). The kit README adds a JSON
host-routing contract (`say` / `show` / `captions`, only `say` reaches TTS), platform notes, and the limits of each
voice; `conlang/speech/LISTENING_TEST.md` is the audio test still to be run.

**Intended readers.** The book targets current frontier models and better future ones; it is not simplified for
weaker models. A MiniMax M3 review (`minimax_m3_book_v4.md`) read the endings as part-of-speech and tense suffixes
and hallucinated most of its readings. v5 adopts only the two fixes that failure pointed to, because a frontier
reviewer had stumbled at the same point: an early statement that the ending never marks time, number, person or word
class, and a lookup procedure with worked examples (`moge` → `mog`, `kina` → `kin`).

**Reviews.** v4 answers Grok Expert's book-only review (`grok_expert_book_v3.md`, with notes on its misreadings, most
of which came from the citation-form confusion above). v3 answers Astra's book-only review (`astra_book_v2.md`): twelve misattached *how* phrases and three
other attachment slips, the open-mark definition, mention by literal, the grammar-word key, the free roots and a worked
coinage, and three literary revisions (Kihote misuses *bank*; Faust's bargain has explicit terms and a cost; Amleto
tells a user a concrete painful truth and hears the reply). The v2 changes answer two external reviews, kept in [`docs/reviews/`](reviews/): Sol (`sol_pr212.md`)
and Grok (`grok_pr212.md`, with notes on where its readings of Talema were wrong).

**How it was written.** Each sentence is authored as a tree of concepts (`conlang/book/*.tl`); the three rules spell it.
`scripts/conlang/author.py` builds the book, refuses unknown concepts, and checks that every sentence decodes to exactly
one tree. `author.py find` looks concepts up; `author.py coin` applies the root law to a new word.
`scripts/conlang/roots_book.py` generates the book of roots from the lexicon.

```bash
.venv/bin/python scripts/conlang/books.py --tokens Qwen/Qwen2.5-0.5B     # every book: regenerate, lint, build, check
# or step by step:
.venv/bin/python scripts/conlang/roots_book.py --out conlang/book/08_roots.tl
.venv/bin/python scripts/conlang/roots_book.py --first --out conlang/book/02b_first_words.tl
.venv/bin/python scripts/conlang/author.py build conlang/book/0*.tl --out conlang/BUKE_DE_LORE_FIRA.md --tokens Qwen/Qwen2.5-0.5B
```

## 7. Limits

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

## 8. Reproduce

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
