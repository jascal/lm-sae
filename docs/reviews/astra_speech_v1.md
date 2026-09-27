# Review of the Talema speech kit

Reviewed September 27, 2026. Links below refer to the reviewed files on `main`; their line numbers may shift as the repository changes.

**The kit is a useful prototype, but it does not yet produce consistently correct Talema or work as a drop-in kit across platforms.** The main defects are Spanish stress placement, missing required PLS attributes, and handling of numeric literals with dependents.

I opened all seven linked files, exercised the existing Python functions unchanged, and parsed all four XML files. I did not upload dictionaries or synthesize audio. Platform statements below are **checked against official documentation**; predicted pronunciation problems remain unverified by listening.

## Sound rules and stress

The sound rules establish a clear target, but leave some important details implicit. The book says:

> `levi te gehe la pe vokeli fira de rarise la .`

I read this as placing emphasis on the root’s first vowel. The README adds:

> “one letter, one sound, no silent letters.”

These agree with the code’s stress calculation for native words. [Book sound rules](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L69-L79), [README](https://github.com/jascal/lm-sae/blob/main/conlang/speech/README.md#L9-L23)

However, the code specifies something the prose should state explicitly:

> `extra ending vowels stand alone.`

That matters for endings such as `ea`, `iu`, and `eaa`: a Spanish, Italian, or English voice may combine adjacent vowels or reduce them. Also, the rule that unstressed one-consonant words lean on the next word needs a sentence-final case. Literal names and source words need a pronunciation policy beyond separating their endings. [speech.py, lines 83–99](https://github.com/jascal/lm-sae/blob/main/scripts/conlang/speech.py#L83-L99)

**Change:** Explicitly require separate vowel syllables, unreduced grammatical endings, and rules for sentence-final particles and literal pronunciation—these details carry grammatical information.

## Per-voice respellings

| Path | Quoted evidence | Finding and concrete change |
|---|---|---|
| Spanish stress | `<grapheme>genete</grapheme><alias>gúenete</alias>` | **Definite bug.** The inserted, normally silent `u` receives the accent. The expected spelling is `guénete`. Calculate stress from the original syllables and map it onto the respelled word. [Spanish lexicon, line 39](https://github.com/jascal/lm-sae/blob/main/conlang/speech/talema-es.pls#L39) |
| Spanish consonants | `<grapheme>hase</grapheme><alias>jase</alias>` | Spanish `j` commonly gives /x/, especially under the declared `es-ES` locale, rather than Talema /h/. The documented /v/→/b/ problem remains; initial `r` also tends to be trilled rather than tapped. Label this path an approximation and record the chosen voice’s actual results. [Spanish lexicon, line 23](https://github.com/jascal/lm-sae/blob/main/conlang/speech/talema-es.pls#L23) |
| Italian | `<grapheme>genete</grapheme><alias>ghenete</alias>` | `ge`→`ghe` correctly addresses hard /g/, but does not mark Talema’s initial stress. Longer words remain vulnerable to Italian stress placement; silent `h`, vowel quality, and intervocalic `s` are further problems. Add tested stress corrections and document unresolved sounds. [Italian lexicon, line 11](https://github.com/jascal/lm-sae/blob/main/conlang/speech/talema-it.pls#L11) |
| English | `<grapheme>ledo</grapheme><alias>LEH-doh</alias>` | This is a human-readable approximation, not a phonetic instruction. Likely readings include /ɛ/ for Talema /e/, /oʊ/ for /o/, and English /ɹ/ for the tap. Uppercase does not guarantee engine stress. Describe this as approximate and validate it per voice. [English lexicon, line 162](https://github.com/jascal/lm-sae/blob/main/conlang/speech/talema-en.pls#L162) |

The Spanish bug comes directly from this sequence:

> `w = re.sub(r"g(?=[ei])", "gu", w).replace("h", "j")`  
> `vowel_positions = [i for i, ch in enumerate(w) if ch in VOWS]`

The second line counts vowels in the **modified spelling**, including the inserted `u`. I reproduced `genete → gúenete`, `gini → gúini`, and `geva → gúeva`. By contrast, the unnecessary accent in `doge → dógue` does not itself move stress incorrectly. The comment treating Spanish stress as a matter of the penultimate *vowel* is also too simple: syllables and diphthongs matter. [speech.py, lines 118–127](https://github.com/jascal/lm-sae/blob/main/scripts/conlang/speech.py#L118-L127)

There is a second unsupported assumption in lexicon generation:

> `if voice != "ipa" and r == w:`  
> `    continue`

An unchanged spelling does not establish that the voice reads it correctly. For example, the Italian path leaves `rarise`, `talema`, and `hase` unchanged despite unresolved stress or consonants. **Change:** Describe these dictionaries as covering the implemented substitutions, rather than all mispronunciations. [speech.py, lines 169–170](https://github.com/jascal/lm-sae/blob/main/scripts/conlang/speech.py#L169-L170)

## PLS validity and platform compatibility

**The four files are well-formed XML, but the three alias files are not conforming W3C PLS as written.** Their opening elements omit `alphabet`, for example:

> `<lexicon version="1.0" xmlns="http://www.w3.org/2005/01/pronunciation-lexicon" xml:lang="es-ES">`

The same omission occurs in the Italian and English files. W3C requires this attribute even when a document contains only aliases. The generator explicitly omits it outside IPA mode:

> `alphabet = ' alphabet="ipa"' if voice == "ipa" else ""`

**Change:** Emit the required default alphabet on all four lexicons—alias-only contents do not waive the root-element requirement. [Generator](https://github.com/jascal/lm-sae/blob/main/scripts/conlang/speech.py#L161-L165), [W3C PLS §4.1, checked](https://www.w3.org/TR/pronunciation-lexicon/#S4.1)

The IPA file has the attribute:

> `alphabet="ipa" xml:lang="en-US"`

Its entries, such as:

> `<grapheme>rarise</grapheme><phoneme>ˈɾaɾise</phoneme>`

match the stated native-word sound and stress target. But `en-US` is a deployment choice, not a platform-neutral language declaration. [IPA lexicon](https://github.com/jascal/lm-sae/blob/main/conlang/speech/talema-ipa.pls#L2-L40)

Using **alias for respelled text** and **phoneme for IPA** is appropriate. I found 93 Spanish entries, 23 Italian entries, and 300 each for English and IPA, with no duplicate graphemes. All entries matched the current generator.

Actual platform acceptance is narrower:

| Platform | Checked against its documentation | Implication for this kit |
|---|---|---|
| **ElevenLabs Agents** | Its current page lists `eleven_flash_v2` and `eleven_v3` for dictionary phonemes; non-English phoneme use requires `eleven_v3`. Other models use aliases instead. Matching is case-sensitive. [Docs](https://elevenlabs.io/docs/eleven-agents/customization/voice/pronunciation-dictionary) | Name the model as well as the voice. “ElevenLabs supports PLS” is insufficient to establish that this IPA file will be applied. |
| **Azure Speech** | Custom lexicons are locale-specific, case-sensitive, limited to 100 KB, and cached by URI for up to 15 minutes. Supported phones depend on locale. [Docs](https://learn.microsoft.com/en-ca/azure/ai-services/speech-service/speech-synthesis-markup-pronunciation) | All four files are below the size limit, but the `en-US` IPA file cannot simply serve an Italian or Spanish locale. Document locale matching and verify the phone inventory. |
| **Convai** | Its documented custom-pronunciation and new-word-recognition controls use spelling/pronunciation pairs and currently support English only. The page does not establish direct PLS upload support. [Docs](https://docs.convai.com/api-docs/convai-playground/character-customization/language-and-speech) | Do not claim these files are directly accepted by Convai. Document the supported integration or mark it unverified. |

These are documentation checks, **not successful import or audio tests**.

## Spoken text, written Talema, and translated captions

**The system prompt expresses the desired separation but cannot reliably enforce it.** It requests:

> `SAY: <spoken form>`  
> `SHOW: <written Talema>`

and later advises:

> “add them after `SHOW:`”

The latter leaves translated captions without an unambiguous boundary. A voice-agent host that speaks the whole response can read the labels, duplicate the sentence, or speak the translation. Streaming makes that especially important. [README, lines 79–84](https://github.com/jascal/lm-sae/blob/main/conlang/speech/README.md#L79-L84)

**Change:** Specify how the host extracts only `SAY`, preserves `SHOW`, and identifies translated captions—prompt instructions alone do not perform that routing. Also state that the book must actually be supplied or retrieved; its URL in the prompt does not demonstrate that the agent has read it.

Literal handling also undermines the separation between a phonetic rendering and ordinary text:

> `out.append(base)`  
> `out.append(end)`

Those branches bypass respelling. For example, IPA mode returns ordinary `Ana a` for `Ana-a`, and English mode leaves the ending as ordinary `a`, which may be reduced. Multi-vowel literal endings likewise bypass the explicit syllable handling. **Change:** Document and handle literal pronunciation and endings separately; do not label the resulting mixed output a complete IPA transcription. [speech.py, lines 137–146](https://github.com/jascal/lm-sae/blob/main/scripts/conlang/speech.py#L137-L146)

## Number compounding and tree validity

**The stated number rule is internally consistent for the nonnegative integer phrases I checked.** The book gives:

> `bi su mula hudede fura dehe nevina tova pe 1492-a .`

Its examples and the current generator agree. [Book, lines 375–381](https://github.com/jascal/lm-sae/blob/main/conlang/BUKE_DE_LORE_FIRA.md#L375-L381)

| Number phrase | Tree and arithmetic |
|---|---|
| `si deha fura` | `s` has two children: `10` and `4`. Both children have zero dependents. Value: **14**. |
| `su mula hudede fura dehe nevina tova` | `s` has four children: `1000`, `100(4)`, `10(9)`, `2`. The two multipliers have one child each. Value: **1492**. |

Both are single complete Talema trees. Fourteen selected checks, including zero and boundaries through one billion, also produced complete trees with the expected arithmetic values. The multiplication convention is regular, although ease of listening still needs an audio test.

Three number fixes are needed:

- **Correct the module’s introductory example.** It quotes `1492 → "so mula hudede fura dehe nevina tova"`; `so` takes only three children, leaving `tova` outside the tree. Change `so` to `su`. [speech.py, line 7](https://github.com/jascal/lm-sae/blob/main/scripts/conlang/speech.py#L7)
- **Define or reject numeric literals with dependents.** The line `words = phrase.split() + ([] if end == "a" else [end])` appends the original ending as a separate item. For `14-e la`, the number phrase completes before the appended `e` and its following text; this is not a tree-preserving expansion under the documented rules. Limit support to leaf numeric literals until the non-leaf convention is specified. [speech.py, line 142](https://github.com/jascal/lm-sae/blob/main/scripts/conlang/speech.py#L142)
- **Validate the numeric domain.** `return (UNITS[n], [])` accepts negative Python indices: `number(-1)` produces nine, while `number(-11)` raises an error. Decimal literals such as `3.14-a` remain digits in spoken output. Reject unsupported values explicitly and document that current compounding covers nonnegative integers. [speech.py, lines 58–59](https://github.com/jascal/lm-sae/blob/main/scripts/conlang/speech.py#L58-L59)

## Hosting an avatar and remaining uncertainty

For hosting an avatar, the README correctly identifies the remaining input problem:

> “Speech recognition … is a separate problem”

Its suggestion of a:

> “custom vocabulary list”

does not establish recognition of Talema’s grammar or its changing endings. [README, lines 59–60](https://github.com/jascal/lm-sae/blob/main/conlang/speech/README.md#L59-L60)

The missing evidence is a tested voice/model/locale combination, actual dictionary-import results, demonstrated `SAY`/`SHOW` routing, and recognition results that measure errors in dependent-count endings. Avatar timing and lip synchronization are also outside these files. None of that needs a redesigned kit, but it needs to be documented before claiming a working Talema-speaking avatar.

I would fix the **Spanish accent bug, PLS headers, and unsupported-number behavior first**, then run a small listening comparison covering initial `r`, `h`, `v`, three-syllable stress, multi-vowel endings, names, and compound numbers. Those recordings would resolve the largest remaining uncertainties.
