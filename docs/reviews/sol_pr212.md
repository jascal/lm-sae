# Review of PR #212: *Buke de lore fira*

**I can learn Talema’s sentence structure from the book alone, and I can read much of it with the help of its dictionary. I cannot read it confidently as a sole reference.** The counted endings, head first order, and `pe`/`te` contrast are taught well: the progression from `Ana-a` to the four dependent `seru` sentence, followed by its drawn tree, works particularly well ([book, lines 28–59](conlang/BUKE_DE_LORE_FIRA.md#L28-L59)). The stories then reuse those patterns enough to make them familiar.

I got stuck on *meaning*, rather than tree parsing. The dictionary lists source words without word classes or sense notes, even where the source words conflict. Some conventions also arrive after the reader has needed them. The book says every word begins with a consonant, for example, while already using `Ana-a`, `14-a`, and `a-a`; the English design document explains the hyphenated literal form, but the book does not explain it at that point ([book, lines 21–38](conlang/BUKE_DE_LORE_FIRA.md#L21-L38); [design, lines 21–24](docs/CONLANG.md#L21-L24)).

## Passages I could read

| Talema | My reading |
|---|---|
| `seru te bofe la nova ne vule la pe Ana-a` | “Ana sees the sun now, in the world.” The `u` on `seru` correctly announces four dependents ([line 57](conlang/BUKE_DE_LORE_FIRA.md#L57)). |
| `bi si Ana-a Lut-a pe pepe kase labiha` | “Ana and Lut are people who laugh.” The conjunction heads its two names ([line 118](conlang/BUKE_DE_LORE_FIRA.md#L118)). |
| `bove bi fura pe si tova tova` | “Proved: two and two are four.” This also demonstrates the claim mark ([line 346](conlang/BUKE_DE_LORE_FIRA.md#L346)). |
| `sere pagi te tebi 3000-a de 3000-a pe voge la` | “Observed: the code passed 3,000 of 3,000 tests” ([line 350](conlang/BUKE_DE_LORE_FIRA.md#L350)). |

I did not find a clear failure of the **three structural rules** in these examples. The more consequential problems are that a valid tree can say less, or something different, than the surrounding text appears to intend:

- **`lafe mide mela` is intended as “Sleep, small mind.”** Its tree makes “small mind” a bare dependent of “sleep”; nothing marks it as the person addressed, and nothing distinguishes a command from a statement with an omitted subject ([book, line 557](conlang/BUKE_DE_LORE_FIRA.md#L557); [authored intent](conlang/book/07_songs.tl#L46)). This is ambiguous, rather than a counting error.
- **The ending `a` does not, by itself, announce the end of a sentence.** It marks *any* leaf; `la` inside the `seru` example is one. A reader knows the sentence has ended when the entire tree’s outstanding dependent count reaches zero. The claim that the ending knows whether the sentence ends needs that qualification ([book, lines 44–70](conlang/BUKE_DE_LORE_FIRA.md#L44-L70)).
- **“Proved: Talema’s words are the shortest” overstates the result.** The English design argues for the shortest *expected root length given the chosen tier sizes and frequency ordering*. That narrower result does not prove an unrestricted shortest language ([book, line 430](conlang/BUKE_DE_LORE_FIRA.md#L430); [design, vocabulary section](docs/CONLANG.md#L108-L113)).

## Design and vocabulary

The three rules consistently recover an ordered **tree**. They do not settle every distinction a reader needs to recover the intended **message**. This is a reasonable small grammar, provided the book states its conventions and limits precisely.

- **Tense and plural:** Their absence is consistent with the design. Words such as “before,” “will,” numerals, and “many” can express them when needed. An unmarked event or noun leaves time and number unspecified; the book should say so explicitly ([design, lines 30–35](docs/CONLANG.md#L30-L35)).
- **Questions:** `tob` supplies a yes/no question *content*, and `rak` (“ask”) can make it an act of asking. The book should distinguish those functions from `pef` (“open/unknown”), which also heads some question shaped passages ([book, lines 340–354](conlang/BUKE_DE_LORE_FIRA.md#L340-L354)).
- **Word formation:** The root shape and counts are clear. The criteria “short if often used” and “when speakers agree” leave future coiners without a reproducible choice or a way to record a new age’s assignments ([book, lines 416–446](conlang/BUKE_DE_LORE_FIRA.md#L416-L446)).
- **Dictionary senses:** `feraso` lists English *sentence*, German *Strafe* (“punishment”), and Spanish *frase* (“phrase”); a learner cannot tell which sense governs this central grammar word ([line 5736](conlang/BUKE_DE_LORE_FIRA.md#L5736)). `leto` lists *letter / schreiben / carta*: the latter two point toward “write” and “mail letter,” although the primer uses it for an alphabetic letter ([line 4053](conlang/BUKE_DE_LORE_FIRA.md#L4053)). `tumo` is used for grammatical “subject,” but its German and Spanish entries chiefly suggest “thing/topic” ([line 1290](conlang/BUKE_DE_LORE_FIRA.md#L1290)). These look like alignment errors or unresolved senses, not useful poetic ambiguity.
- **Confusable entries:** Both `senur` and `nuler` offer “zero / null / cero,” while `g` and `pat` both offer “to.” Their distinct roles may be justified, but the dictionary does not label those roles ([zero entries](conlang/BUKE_DE_LORE_FIRA.md#L3895); [particle entry](conlang/BUKE_DE_LORE_FIRA.md#L583)).

## Concrete changes

1. **Add a short early example of `Ana-a`, `14-a`, and `a-a` as hyphenated literals.** This resolves the apparent exception to the consonant rule when it first appears.
2. **Rewrite the end of sentence explanation as “the sentence ends when the first tree is complete; its final word has ending `a`.”** This teaches the usable parsing test.
3. **Mark dictionary entries with a word class and a brief Talema sense phrase, starting with `feras`, `let`, and `tum`.** Source lemmas alone do not disambiguate these core terms.
4. **Add paired examples of an unmarked event and one marked with a time word, and of an unmarked noun and one with a number word.** This makes the deliberate tense and plural limits learnable from the book.
5. **Add one direct yes/no question, one reported “whether” clause, and one `pef` open claim, with their functions explained in Talema.** Their similar shapes currently invite confusion.
6. **Give commands and address a stated convention, then rewrite `lafe mide mela` using it.** A future reader should be able to identify whom the lullaby addresses.
7. **Narrow the “shortest words” proved claim to its stated frequency and tier assumptions.** The evidence then matches the strength of the claim.

The book’s strongest feature is its progression from visible trees to stories and then to a dictionary and guide for future writers. Those parts are worth keeping. Its main review issue is that **unique parsing is presented as though it also guarantees a unique, recoverable meaning**; a few sense labels and precise conventions would make the book much closer to the standalone reference it aims to be.
