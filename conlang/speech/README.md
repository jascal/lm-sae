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
  are unstressed and lean on the next word.
- **Literals**: the hyphen is silent and the ending is its own syllable: `Ana-a` is said "Ana a".
- **Numbers** are said as Talema number phrases, even when written as digits: `14-a` is said `si deha fura`,
  `1492-a` is `su mula hudede fura dehe nevina tova` (book chapter 2c).
- **`.`** is a pause.

## Which voice

No off-the-shelf voice knows Talema. Pick one whose letter habits are closest, and correct the rest:

| voice language | reads correctly | needs correcting | `.pls` size (top 300 words) |
|---|---|---|---|
| **Italian** | vowels, most consonants, stress mostly right | *ge gi* (would be /dʒ/): respelled *ghe ghi*. Italian voices cannot say /h/; *h* comes out silent | 23 entries |
| **Spanish** | vowels, most consonants | *ge gi* (would be /x/): *gue gui*; *h* (silent): *j*; stress on longer words: accent marks (`rárise`); *v* is said as /b/ | 93 entries |
| English | — | everything: vowels and stress | 300 entries (full respelling, `LEH-doh`) |
| any engine taking IPA | everything | nothing, if the engine honours IPA | 300 entries (`ˈledo`) |

Start with an Italian or Spanish voice and test both. Use the IPA lexicon wherever the engine supports phoneme tags
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
- Only words the voice would get wrong are listed in the es/it lexicons. Words outside the top 300 go through
  `speech.py say` (or the rules in the system prompt below).

Speech recognition (users speaking Talema *to* an agent) is a separate problem, not covered here. Most platforms
accept a custom vocabulary list; the book's first-words chapter is a good starting list.

## System prompt for a voice agent

Adapt this to the platform. It keeps the written form and the spoken form separate, so captions stay in true Talema
while speech is pronounceable.

```text
You speak Talema, the language of the book Buke de lore fira (https://raw.githubusercontent.com/jascal/lm-sae/main/conlang/BUKE_DE_LORE_FIRA.md).
Write every reply in correct written Talema: every word is root + ending, the ending counts the word's dependents,
the head comes first, pe marks the subject and te the object, and every sentence ends in "a".

When your reply will be spoken, also give a spoken form:
- say numbers as Talema number words, never as digits (14 → si deha fura)
- drop the hyphen of a literal and say its ending alone (Ana-a → Ana a)
- apply the voice's respelling: [Spanish voice: ge→gue, gi→gui, h→j, and an accent on the root's first vowel when
  Spanish would stress another syllable] [Italian voice: ge→ghe, gi→ghi] [English voice: syllables like LEH-doh]
- never speak translations, glosses, or anything but the Talema sentence

Output:
SAY: <spoken form>
SHOW: <written Talema>
```

If your app shows translations as captions, add them after `SHOW:` and make sure they never reach the TTS.
