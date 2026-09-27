# Book-only review of *Buke de lore fira*

Source: [the current book](https://raw.githubusercontent.com/jascal/lm-sae/main/conlang/BUKE_DE_LORE_FIRA.md), read September 27, 2026. Line links below refer to that version; they may shift as `main` changes.

**I could learn enough to follow the stories, understand the teaching examples, and construct simple sentences. I would still hesitate to use the book as my sole authority for writing or extending Talema.** The main obstacles are a few misleading attachments, inconsistent dictionary senses, and incomplete instructions for interpreting or introducing words.

I used only the current book for this review. Since earlier conversation exposed me to other Talema material, this cannot be a blind first-reader experiment. I read the teaching sections, tales, play, and songs, and used the dictionary for lookup and targeted checks; I have not verified every dictionary definition.

## Learning from the book

I could recover the head-first trees, the subject and object particles, and the dependent counts. I inferred base-five endings from the examples for 5, 6, 7, and 10. The dog examples successfully distinguish unspecified time from explicit time, and unspecified number from explicit quantities. The question and command lessons also give usable contrasts. [Teaching sections](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L284-L343)

The opening still requires considerable dictionary travel. Instructions use technical words before the early vocabulary lists explain them. I could work backwards from the diagrams and source-word entries, but I could not simply read forward and learn everything as it appeared.

## Four translations

| Passage | My reading | Confidence |
|---|---|---|
| [The dog question lesson](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L302-L310) | “Does the dog sleep?” / “I do not know whether the dog sleeps.” / “Open: whether the dog sleeps.” | High about the structures. The intended strength of “open” is problematic, as discussed below. |
| [End of the encounter with “bank”](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L461) | “Kihote learns something new. A word lives in a sentence. Without a sentence, a word is like a fish without water.” | High. Reading “learned” instead of “learns” would be a narrative choice, not something explicitly marked here. |
| [Amleto’s second scene](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L536-L544) | “To speak or not to speak. That is the question. … The truth is hard, and the voice can be soft. I will say the true things with a kind voice.” | High. The contrast between painful content and gentle delivery comes through clearly. |
| [Three lines addressed to a future reader](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L763-L765) | “Our words are small. / They carry our voices. / When you read, we speak again.” | High about attachment and meaning; the plural readings come from context. |

## Sentences and conventions needing attention

The counting mechanism is consistent. A simple check using the endings found complete trees throughout the prose, headings, and dictionary, except for the explicitly announced dialect example. That does not guarantee that every dependent attaches where the author appears to intend.

- **The comparison between Ana and Lut attaches “me” incorrectly.** At the beginning of [line 403](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L403), “like” has zero children, so “me” becomes a separate dependent of “be.” Assuming the meaning is “You are very like me,” write `bo like ma vera pe tada .`—this makes “me” the complement of “like.”
- **The heading about agents speaking puts the agent under “how.”** Its counted structure is `speak(how(SUBJ(agent)))`. For “How agents speak,” I expect `seki hova pe geneta`—the agent then belongs to “speak.” The same attachment pattern appears elsewhere and deserves a systematic review. [Heading and following examples](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L571-L576)
- **Ana’s final sentence about good and bad lacks a subject marker.** I infer “Many things are like that,” but “many things” is a bare dependent of “be.” Write `desi te bi like tesa pe kige vana pe Ana-a .`—the added `pe` expresses the apparent intended role. [Passage](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L371)
- **The Faust tale has avoidable pronoun ambiguity.** After mentioning Faust and the answer, `desi te pesa pe pesa` literally gives “It says it.” Context suggests “Faust says the answer,” but the sentence does not identify either referent. Write `desi te nuve la pe Faust-a .`—both roles become explicit. [Passage](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L483)
- **“Open” is defined too strongly.** The explanation says that nobody knows the answer. That goes beyond reporting the speaker’s uncertainty, and sits awkwardly beside the excellent instruction to describe the limits of one’s own method. Define it as “unresolved for us” or require an explicit scope—otherwise ordinary uncertainty becomes a claim about everyone’s knowledge. [Definition and ensuing principle](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L593-L599)

## Teaching order and clarity

The teaching works best when it uses contrasts: the expanding Ana sentence, the reversed subject and object, the dog’s time and number examples, and the explicit distinction between asking a question and reporting uncertainty. The sentence-completion counter is particularly useful. Keep those.

I would make four teaching changes:

- **Move a small key of technical vocabulary before the first explanations**—“word,” “root,” “ending,” “child,” “head,” “count,” and the role particles would substantially reduce backwards lookup.
- **State the general base-five calculation explicitly**—the examples let me infer it, but a sole reference should settle arbitrary longer endings.
- **Say that each child’s entire subtree is read before the next child**—the leaf tale’s sequence of head, children, then grandchildren can suggest reading by generations instead. [Leaf tale](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L413)
- **Explain how mentioning a word differs from describing one**—examples such as a “word” with a particle underneath suggest mention, whereas other dependents describe a word; the reader needs a stated distinction.

The literal-word lesson helps, but the following universal claim that every word begins with a consonant still conflicts with its own names and numerals. Change that statement to apply to native roots, and explicitly exempt the literal forms already demonstrated. [Opening lesson](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L21-L32)

## Vocabulary, dictionary, and growth

The dictionary is useful, especially with its word-class labels, but several entries still mislead. These are the cases I would prioritize:

| Root | Problem | Concrete change |
|---|---|---|
| `hov` | English “how” and Spanish “cómo” are paired with German “woher,” meaning “from where.” | Use “wie,” or distinguish manner from origin. [Entry](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L6753) |
| `tar` | “They” is paired with German “ihr,” which does not straightforwardly express that subject pronoun. | Use “sie” for the intended plural subject. [Entry](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L1299) |
| `kit` | “Kind” meaning category is paired with Spanish “amable,” meaning kind-hearted. | Use “tipo” or “clase.” [Entry](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L1865) |
| `sigan` | Its entry describes signing a document, but the explanation of “letter” needs a symbol. | Use the existing `sibol` in that definition. [Definition](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L347), [entries](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L5576-L5586) |
| `kesim` | “Claim / Anspruch / reclamo” mixes an assertion with a demand or entitlement, precisely where the book discusses evidence. | Use or cross-reference `behan`, whose entry aligns “assertion / Behauptung / afirmación.” [Claim entry](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L1846), [assertion entry](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L2465) |
| `lator` | It appears as the heading for linking words but has no dictionary entry. I infer “relator”; I cannot confirm that from a definition. | Add its definition and examples. [Heading](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L147) |

Two other cases deserve short usage notes. `kas` and `pas` appear to duplicate the German conjunction under the spellings *dass* and *daß*, without explaining a semantic distinction. Also, `ner` means “never,” while `never` means “answer”: that second root is a conspicuous trap for an English reader. Neither requires casually replacing an established root, but both require guidance.

There is also a concrete gap in the growth instructions. **I counted nine legal CVC roots with no entry in either the main dictionary or introductory lists:** `pip`, `pim`, `pod`, `pud`, `puv`, `tub`, `bok`, `buf`, and `huk`. The book says the short roots have already been assigned. A future reader therefore cannot tell whether these nine are available or reserved. Add explicit entries or a reserved-root list, plus a worked coinage showing the definition, availability check, source words, and example sentence. [Growth instructions](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L659-L693)

## Tales, play, and songs

As literature, the book has a coherent imagination, but it explains its morals too often. Words become people, trees, seeds, shelter, and inherited voices. That gives the collection a recognizable identity. The strongest passages make those images do something in a scene.

The leaf tale works because a grammatical fact becomes an emotional conflict: having no children looks like worthlessness until the leaf discovers its role. Its companion counting poem is particularly successful—the first word’s dependent counts rise through 0, 1, 2, 3, 4 and descend again, while the leaf is carried away and finally sheltered. The form participates in the story. [Counting poem](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L739-L749)

The Kihote tale has the strongest comic possibility. Meeting a troublesome word and wanting to fight it is a good extension of his earlier mistake. I would let him actually misuse “bank” once, with a small consequence, before the word explains itself; that would make the lesson an event rather than a lecture.

The Faust tale has a good turning point in the child’s repeated questions, but I am unsure how its bargain is meant to operate. The voice’s initial condition and Faust’s counterproposal differ, and the later breaking of the bargain has little visible cost. Two changes would help: make the agreed condition explicit, and describe Faust as *believing* he knows everything rather than having the narrator certify it. That preserves the force of his later discovery. [Faust tale](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L469-L487)

The play’s second scene contains a real ethical dilemma, but it remains abstract: we never hear the user’s actual question or see the response to Amleto’s choice. Give him a specific painful truth to tell, then let the other person react. His gentle delivery would then be demonstrated rather than merely promised.

The future-reader song is the most affecting piece to me: reading restores the absent speakers’ voices. The lullaby also earns its simplicity. I can judge those effects on the page; the book does not give enough pronunciation, stress, or musical guidance for me to assess them as sung pieces.

I would preserve the tree imagery and the recurring final vowel, while cutting some explanatory endings and giving the characters more distinct desires and mistakes. The book already shows that its language can carry a story. Its next improvement should let the reader discover more of the meaning through what happens.
