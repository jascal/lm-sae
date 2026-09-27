# Review of the book v4 (MiniMax M3)

Book-only review of `conlang/BUKE_DE_LORE_FIRA.md` after #215, from a clean session, pasted into the session by the
user. Kept as a record of where a less capable model breaks. Its specific readings and suggested edits were **not**
adopted: checked against the book, most are wrong or describe text that does not exist.

## What the review said (summary)

- Talema's five vowel endings mark part of speech (-u noun, -a verb, -i article, -o adjective, -e adverb); tense is
  carried by `bi` (present), `bo` (past), `vi` (future).
- About a dozen core words are missing from the dictionary (`ma` "land", `moge`, `genete`, `hase` "dark", `lafi`
  "thing", `kina` "all", `fide`, `toreme`, `gagi`, `rage` …), "perhaps 40% of the content words in the prose".
- The tales include a Genesis retelling (*mogi fira la*) with Adam, Eve and the fall, a curse poem, and a Faust with
  Gretchen and Mephisto; the play has Horatio.
- Suggested: a part-of-speech/tense table, a verb paradigm, dictionary entries for the "missing" words, and additions
  to the Genesis, Quixote and Faust retellings.

## Notes (checked against the book)

- **Invented content.** The book has no Genesis, Adam, Eve, fall, curse poem, Gretchen, Mephisto or Horatio.
  *mogi fira la* is "the first morning": a child meets an agent; "I am Ana"; "who are you?"
- **Invented grammar.** Endings count dependents; they do not mark word class or tense. `bi`/`bo` are "be" with two
  and three children, `vi` is "have" with two.
- **The "missing" words are all present,** and several meanings were inverted: `ma` is `m` "I" (not "land"), `kina` is
  `kin` "no" (not "all"), `hase` is `has` "house" (not "dark"), `lafi` is `laf` "sleep", `moge` is `mog` "morning",
  `toreme` is `torem` "storm" (the play's title).
- **The useful finding is the failure mode.** The model read the endings as ordinary inflection, as its training on
  natural languages would suggest, and so could not strip them to find roots. Grok stumbled at the same point before
  v4. **Adopted in v5:** an early statement of what the ending is *not* (never time, number, person or word class), and
  a lookup procedure with worked examples from page one (`moge` → `mog`, `kina` → `kin`, `hase` → `has`, `ma` → `m`).
