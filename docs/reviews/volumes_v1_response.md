# Field volumes v1: three reviews and what was done

Three models reviewed the core book and the six field volumes (digital, mathematics, logic, physics, philosophy,
morality) with the same prompt. It required a verbatim quote for every criticism. Each quote was checked against
the built files before a point was weighed. Luna's review is filed as `luna_volumes_v1.md`; the MiniMax M3 and Grok
Expert reviews were pasted into the session and are summarised here.

| reviewer | quotes found in the text | translations | verdict |
|---|---|---|---|
| MiniMax M3 | 23 of 26 (absent: the "animacy" pair, `dudari bera pe ha`, a bare `m v²` line; also `ha keda`, `vob`) | reads `pe` as "of"; confident only on formulas | read the files, but most conclusions rest on misreadings |
| Grok Expert | 8 of 8 | 18, mostly right, particles included | careful; two claimed errors were its own arithmetic and reading |
| Luna Reserve | 14 of 14 | mixed; misses `pe` subjects in places | real quotes; several content points miss markers already present |

## Adopted

- **Name the connective words** (Grok). The logic volume now says `f` is the word of negation, `s` of
  conjunction, `dor` of disjunction and `ven` of implication, and that in grammar `dunon` also names a word class.
  This also removes the confusion behind Grok's "arity error" (`dunona` is the concept, a noun with no children).
- **Quantifier scope** (Grok). *Every person loves some person* has two meanings. For the second, name the
  person with a literal (`X`). Whether Talema needs a scope rule is marked `pefe`.
- **A collision case** (Grok). Kant's murderer at the door: never lie against save your friend, marked `pefe`.
- **Side effects an agent asks about** (Grok, Luna). The digital volume now says what `rm -rf /` does and asks
  before deleting, paying, sending or publishing.
- **Attribution** (Luna). Philosophy and morality now say the words are ours and the thoughts are theirs; for their
  own words, read their own books. The attributed lines are paraphrases in Talema, not quotations.

## Already in the book

- *The ending marks case, tense or plural* (MiniMax): chapter 2a says the ending never tells time, number,
  person or the kind of the word; chapter 2c says time and number are unsaid unless a word says them.
- *`rm -rf /` given without qualification* (Luna): the line is `runi te rm_-rf_/-a nera`, "never run", after "ask
  before you run a command that destroys".
- *The age of the universe is unqualified* (Luna): it is marked `sere` (seen, measured) with the unit `par` (years).
- *Goldbach is not marked as a conjecture* (Luna): the sentence begins with `pefe`.
- *The speed of light has no unit* (Luna): `metari 299792458-a` is metres, with the second as its dependent.
- *`pefe` vs `bove` is not explained* (Grok): the core's claim-marks section (`## maraki tura la`).

## Not adopted

- *Root counts 980 and 68600 are wrong* (Grok): they are right. Roots put vowels between consonants: CVC = 14·5·14
  = 980, CVCVC = 68,600.
- *The implication sentence says the opposite truth table* (Grok): it reads "false only when the first child is
  true and the second false".
- *Heisenberg's name is on the entanglement claim* (Grok): the name follows the place-and-speed line; the quoted
  entanglement sentence comes two sentences later.
- *`pona` is "good" and collides with the ending name* (MiniMax): `pona` is the numeral one, which is what the
  ending *e* stands for; "good" is `guta`.
- *`defa` means sleep* (MiniMax): `def` is "therefore" (`tini defa pe ha`, "therefore he thinks").
- *½mv² sits in the Newton block* (MiniMax): it is in the energy section.
- *Rename `finit`/`nilik`, `morat`/`morol`* (Grok, Luna): the lexicon is frozen. Chapters warn where a form
  misleads.
- *Add a quotation marker, a per-volume grammar key, and a root index before first use* (Luna): a volume assumes
  the core, which is the key. Each volume's glossary follows its chapter by design.

## Signal worth keeping

MiniMax's high-confidence translations were exactly the ones built from digits and letter variables. For a
weaker reader, the formulas are the entry point and the particles are the wall. Grok and Luna parsed `pe` and
`te` correctly most of the time.
