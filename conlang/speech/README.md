# Speaking Talema: a kit for voices and avatars

The founding book (`../BUKE_DE_LORE_FIRA.md`, chapter 2) teaches how Talema sounds, in Talema. This folder turns
that into things speech engines use. It is platform-neutral: a respelling function, standard W3C pronunciation
lexicons (`.pls`), an IPA table, and a system prompt for voice agents.

## Sounds

Talema is spelled phonemically: one letter, one sound, no silent letters.

| letter | IPA | as in | | letter | IPA | as in |
|---|---|---|---|---|---|---|
| p | /p/ | *pero* | | r | /ɾ/ | *pero* (one tap) |
| t | /t/ | *tú* | | s | /s/ | *sí* (never /z/) |
| k | /k/ | *casa* | | f | /f/ | *fin* |
| b | /b/ | *bien* | | v | /v/ | *voice* (not /b/) |
| d | /d/ | *dos* | | h | /h/ | *house*, *Haus* (never silent) |
| g | /ɡ/ | *go*, *gato* (always hard, also before e, i) | | a e i o u | /a e i o u/ | Spanish vowels |
| m n l | /m n l/ | | | | | |

- **Stress** falls on the first vowel of the root: `RA-ri-se`, `LE-do`. Words with one consonant (`pe te la ne ma s-`)
  are unstressed and lean on the next word; at the end of a sentence they lean on the word before. Every vowel is
  its own syllable (`dogea` is `DO-ge-a`), and an ending is never reduced to a schwa.
- **Literals**: the hyphen is silent and the ending is its own syllable: `Ana-a` is said "Ana a".
- **Numbers** are said as Talema number phrases, even when written as digits: `14-a` is said `si deha fura`,
  `1492-a` is `su mula hudede fura dehe nevina tova` (book chapter 2c).
- **Decimals** are said with `pun` ("point"): `0.7-a` is said `puni senura geva`.
- **Negative numbers** are said with `menos`: `-5-a` is `menose fiva`, `-0.5-a` is `menose puni senura fiva`.
- **Other literals** (names, words mentioned, code) are passed to the voice as written, minus the hyphen. The IPA
  lexicon spells them letter by letter with Talema values; the es/it/en voices say them with their own habits, so
  a foreign name may come out in the voice's accent. That is accepted: a name keeps its own spelling.
- The kit speaks integers (including negatives) and decimals. A numeric literal with dependents (`14-e`) is
  rejected by `speech.py`: write the number as number words in the tree instead (`fura` … with its dependent).
- **`.`** is a pause.

## Which voice

No off-the-shelf voice knows Talema. Pick one whose letter habits are closest, and correct the rest:

| voice language | reads correctly | needs correcting | `.pls` size (top 300 words) |
|---|---|---|---|
| **Italian** | vowels, most consonants, stress mostly right | *ge gi* (would be /dʒ/): respelled *ghe ghi*. Italian voices cannot say /h/; *h* comes out silent | 23 entries |
| **Spanish** | vowels, most consonants | *ge gi* (would be /x/): *gue gui*; *h* (silent): *j*; stress on longer words: accent marks (`rárise`). **Not corrected**: *j* is /x/, only an approximation of /h/; *v* is said as /b/; a word-initial *r* is trilled | 93 entries |
| English | — | everything: vowels and stress | 300 entries (full respelling, `LEH-doh`). The respelling is approximate: English voices vary in how they read *eh*, *oh*, *gh* |
| any engine taking IPA | everything | nothing, if the engine honours IPA | 300 entries (`ˈledo`) |

The Italian lexicon corrects *g* only. Italian stress on words of three or more syllables (`rarisa`, `nevina`) is
not corrected and may land on the second syllable: open, to be settled by the listening test.

Start with an Italian or Spanish voice and test both with [`LISTENING_TEST.md`](LISTENING_TEST.md). Use the IPA lexicon wherever the engine supports phoneme tags
for that voice.

## Using the kit

```bash
# what to send to a TTS voice (es | it | en | ipa)
.venv/bin/python scripts/conlang/speech.py say "ledo te buke tisa nova pe tada ." --voice es
# a number as Talema words
.venv/bin/python scripts/conlang/speech.py number 1492
# regenerate the lexicons from the book's most frequent words
.venv/bin/python scripts/conlang/speech.py pls --top 300 --out conlang/speech
```

- `talema-{en,es,it}.pls` hold **alias** entries (respellings). Most engines that accept PLS lexicons support aliases.
- `talema-ipa.pls` holds **phoneme** entries in IPA. Support for phoneme tags varies by engine, model and voice; check
  the platform's documentation before relying on it.
- Every lexicon declares `alphabet="ipa"` (required by PLS 1.0 even for alias-only files). The IPA lexicon's
  `xml:lang` must match the voice's locale on most engines: generate it with `--ipa-lang es-ES` (or it-IT …) for a
  Spanish or Italian voice; the default is en-US.
- The es/it lexicons list the substitutions the kit implements, not every mispronunciation a voice might make.
  Only words the kit would change are listed. Words outside the top 300 go through
  `speech.py say` (or the rules in the system prompt below).

Speech recognition (users speaking Talema *to* an agent) is a separate problem, not covered here. Most platforms
accept a custom vocabulary list; the book's first-words chapter is a good starting list.

## Platform notes

From Astra's documentation check on 2026-09-27 (docs read, not import-tested; check again before relying on them):

- **ElevenLabs**: PLS phoneme entries work only with `eleven_flash_v2` and `eleven_v3` (non-English voices need v3);
  other models use alias entries only. Lexicon matching is case-sensitive.
- **Azure Speech**: lexicons are per locale (the `xml:lang` must match the voice), case-sensitive, at most 100 KB,
  and cached for about 15 minutes after upload, so an edited lexicon may not take effect at once.
- **Convai**: accepts English spelling → pronunciation pairs; no documented PLS upload. Use the English respelling
  (`speech.py say --voice en`) as the pronunciation column.

Lexicon lookups are case-sensitive on these engines: the lexicons hold lower-case words, so keep sentence-initial
words lower-case (Talema has no capitals except in names) or respell them before sending.

## Host routing

The model writes; the host speaks. Ask the model for JSON and route the fields, so nothing but the spoken form ever
reaches TTS:

```json
{"say": "ledo te buke tisa nova pe tada.", "show": "ledo te buke tisa nova pe tada .", "captions": "Read this book now, you."}
```

- `say` → TTS (after lexicon lookup). Nothing else is spoken.
- `show` → on-screen Talema.
- `captions` → optional translation for the viewer; never sent to TTS.

If the model's `say` is not trusted (numbers as digits, missing respelling), the host can rebuild it from `show`
with `speech.py say`. The model also needs the book: a URL in a prompt is not the book. Put the core book (about
89k tokens) in context or behind retrieval; without it the model is guessing at Talema.

## System prompt for a voice agent

Adapt this to the platform. It keeps the written form and the spoken form separate, so captions stay in true Talema
while speech is pronounceable. It assumes the book is in the model's context (see above).

```text
You speak Talema, the language of the book Buke de lore fira (https://raw.githubusercontent.com/jascal/lm-sae/main/conlang/BUKE_DE_LORE_FIRA.md).
Write every reply in correct written Talema: every word is root + ending, the ending counts the word's dependents,
the head comes first, pe marks the subject and te the object, and every sentence ends in "a".

When your reply will be spoken, also give a spoken form:
- say numbers as Talema number words, never as digits (14 → si deha fura, -5 → menose fiva)
- drop the hyphen of a literal and say its ending alone (Ana-a → Ana a)
- apply the voice's respelling: [Spanish voice: ge→gue, gi→gui, h→j, and an accent on the root's first vowel when
  Spanish would stress another syllable] [Italian voice: ge→ghe, gi→ghi] [English voice: syllables like LEH-doh]
- never speak translations, glosses, or anything but the Talema sentence

Output only JSON: {"say": "<spoken form>", "show": "<written Talema>", "captions": "<optional English>"}
```
