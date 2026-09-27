"""Speaking Talema: numbers as words, and text a speech engine will read correctly.

Talema is spelled phonemically (docs/CONLANG.md §1), but off-the-shelf TTS voices bring their own
language's letter rules. This module turns written Talema into a string a given voice will pronounce as
Talema, and exports the same knowledge as W3C pronunciation dictionaries (.pls) and an IPA table:

  number(n)            the Talema number phrase for n, e.g. 1492 → "su mula hudede fura dehe nevina tova"
  decimal(s)           a decimal: 0.7 → "puni senura geva" (pun, "point", heads the whole part and each digit)
  spoken(text, voice)  written Talema → what to send to a TTS voice ("es", "it", "en" or "ipa")
  pls                  write .pls dictionaries (alias respellings per voice, and IPA) for the most frequent words
  ipa                  print the letter table

Rules applied (the book's speaking conventions, chapter 2):
  - the hyphen of a literal is silent, and its ending is its own syllable: Ana-a → "Ana a"
  - digit literals are spoken as Talema number phrases: 14-a → "si deha fura"
  - "." is a pause; stress falls on the first vowel of the root; one-consonant words (pe te la ne s …) are unstressed
  - per voice: g is always hard, h is [h], v is [v]; each voice language gets the respelling it needs

  .venv/bin/python scripts/conlang/speech.py say "ledo te buke tisa nova pe tada ." --voice en
  .venv/bin/python scripts/conlang/speech.py pls --top 300 --out conlang/speech
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONS, VOWS = "ptkbdgmnlrsfvh", "aeiou"
IPA = {"p": "p", "t": "t", "k": "k", "b": "b", "d": "d", "g": "ɡ", "m": "m", "n": "n", "l": "l", "r": "ɾ",
       "s": "s", "f": "f", "v": "v", "h": "h", "a": "a", "e": "e", "i": "i", "o": "o", "u": "u"}
# number roots (conlang/lexicon.tsv)
UNITS = ["senur", "pon", "tov", "tur", "fur", "fiv", "sak", "gev", "doh", "nevin"]
TEN, HUNDRED, THOUSAND, MILLION, AND = "deh", "huded", "mul", "mok", "s"


def _num(v: int) -> str:
    return "".join(VOWS[int(d)] for d in _base5(v))


def _base5(v: int) -> str:
    out = ""
    while True:
        out = str(v % 5) + out
        v //= 5
        if v == 0:
            return out


def _spell(tree) -> list[str]:
    head, kids = tree
    return [head + _num(len(kids))] + [w for k in kids for w in _spell(k)]


def number_tree(n: int):
    """A number as a Talema tree. A number word under a big number word says how many of it (dehe tova = 20);
    s ("and") adds its parts (si deha fura = 14). 0 is senur."""
    if n < 10:
        return (UNITS[n], [])
    parts = []
    for base, root in ((10 ** 6, MILLION), (1000, THOUSAND), (100, HUNDRED), (10, TEN)):
        q, n = divmod(n, base)
        if q:
            parts.append((root, [] if q == 1 else [number_tree(q)]))
    if n:
        parts.append((UNITS[n], []))
    return parts[0] if len(parts) == 1 else (AND, parts)


def number(n: int) -> str:
    return " ".join(_spell(number_tree(n)))


POINT = "pun"


def decimal_tree(text: str):
    """A decimal: pun ("point") heads the whole part, then each digit after the point, one by one:
    0.7 → puni senura geva; 3.14 → puno tura pona fura. One tree, like every number phrase."""
    whole, frac = text.split(".")
    return (POINT, [number_tree(int(whole or 0))] + [(UNITS[int(d)], []) for d in frac])


def decimal(text: str) -> str:
    return " ".join(_spell(decimal_tree(text)))


# ── written → spoken ──────────────────────────────────────────────────────────────────────────────────────
def _split(word: str) -> tuple[str, str]:
    """(root, ending) of a native word."""
    k = len(word)
    while k and word[k - 1] in VOWS:
        k -= 1
    return word[:k], word[k:]


def _syllables(word: str) -> list[str]:
    """Talema syllables: each consonant with the vowel after it; extra ending vowels stand alone."""
    out, i = [], 0
    while i < len(word):
        if word[i] in CONS and i + 1 < len(word) and word[i + 1] in VOWS:
            out.append(word[i:i + 2])
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def _stressed(word: str) -> int | None:
    """Index of the stressed syllable: the first syllable of a root with 2+ consonants; None if unstressed."""
    root, _ = _split(word)
    return 0 if sum(c in CONS for c in root) >= 2 else None


def _respell(word: str, voice: str) -> str:
    syl, st = _syllables(word), _stressed(word)
    if voice == "ipa":
        return "".join(("ˈ" if i == st else "") + "".join(IPA[c] for c in s) for i, s in enumerate(syl))
    if voice == "en":
        v = {"a": "ah", "e": "eh", "i": "ee", "o": "oh", "u": "oo"}
        out = []
        for i, s in enumerate(syl):
            c, x = (s[0], s[1]) if len(s) == 2 else ("", s)
            c = "gh" if c == "g" and x in "ei" else c
            r = c + v[x]
            out.append(r.upper() if i == st else r)
        return "-".join(out)
    # es / it: letters mostly already right; fix g before e/i, h, and stress where the voice would miss it
    w = word
    if voice == "es":
        w = re.sub(r"g(?=[ei])", "gu", w).replace("h", "j")
    elif voice == "it":
        w = re.sub(r"g(?=[ei])", "gh", w)
    nv = sum(ch in VOWS for ch in word)
    if voice == "es" and st is not None and nv >= 2:
        # Spanish stresses the second-to-last vowel of a vowel-final word; mark the root's first vowel when it differs
        # the u of gue/gui is silent (it only keeps g hard), so it is not a vowel for stress
        vowel_positions = [i for i, ch in enumerate(w) if ch in VOWS
                           and not (ch == "u" and i and w[i - 1] == "g" and i + 1 < len(w) and w[i + 1] in "ei")]
        first = vowel_positions[0]
        if first != vowel_positions[-2]:
            w = w[:first] + {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú"}[w[first]] + w[first + 1:]
    return w


def spoken(text: str, voice: str = "es") -> str:
    out = []
    for tok in text.split():
        if tok == ".":
            out.append("." if voice != "ipa" else "|")
            continue
        if "-" in tok:
            base, end = tok.rsplit("-", 1)
            base = base.replace("_", " ")
            if base.isdigit() or re.fullmatch(r"\d*\.\d+", base):
                phrase = number(int(base)) if base.isdigit() else decimal(base)
                words = phrase.split() + ([] if end == "a" else [end])
                out.extend(_respell(w, voice) if w not in VOWS else w for w in words)
            else:
                out.append(base)
                out.append(end)            # the ending is its own syllable
            continue
        out.append(_respell(tok, voice))
    s = " ".join(out)
    return s.replace(" .", ".")


# ── exports ───────────────────────────────────────────────────────────────────────────────────────────────
def frequent_words(top: int) -> list[str]:
    text = (ROOT / "conlang" / "BUKE_DE_LORE_FIRA.md").read_text(encoding="utf-8")
    body = text.split("## buki de rarisa la")[0]                 # the prose, not the dictionary
    toks = [t for t in re.findall(r"[a-z]+", body) if re.fullmatch(f"[{CONS}]([{VOWS}][{CONS}])*[{VOWS}]+", t)]
    return [w for w, _ in Counter(toks).most_common(top)]


def write_pls(words: list[str], voice: str, path: Path) -> None:
    lang = {"en": "en-US", "es": "es-ES", "it": "it-IT", "ipa": "en-US"}[voice]
    alphabet = ' alphabet="ipa"' if voice == "ipa" else ""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<lexicon version="1.0" xmlns="http://www.w3.org/2005/01/pronunciation-lexicon"{alphabet} xml:lang="{lang}">',
             "  <!-- Talema pronunciations, generated by scripts/conlang/speech.py; do not edit by hand -->"]
    for w in words:
        r = _respell(w, voice)
        if voice != "ipa" and r == w:
            continue                                   # the voice already reads it right
        tag = "phoneme" if voice == "ipa" else "alias"
        lines.append(f"  <lexeme><grapheme>{w}</grapheme><{tag}>{r}</{tag}></lexeme>")
    lines.append("</lexicon>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("say")
    q.add_argument("text")
    q.add_argument("--voice", default="es", choices=["es", "it", "en", "ipa"])
    q = sub.add_parser("number")
    q.add_argument("n", help="an integer, or a decimal such as 0.7")
    q = sub.add_parser("pls")
    q.add_argument("--top", type=int, default=300)
    q.add_argument("--out", type=Path, default=ROOT / "conlang" / "speech")
    sub.add_parser("ipa")
    a = p.parse_args()
    if a.cmd == "say":
        print(spoken(a.text, a.voice))
    elif a.cmd == "number":
        print(decimal(a.n) if "." in a.n else number(int(a.n)))
    elif a.cmd == "ipa":
        for c in CONS + VOWS:
            print(f"{c}\t/{IPA[c]}/")
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        words = frequent_words(a.top)
        for voice in ("en", "es", "it", "ipa"):
            write_pls(words, voice, a.out / f"talema-{voice}.pls")
        print(f"wrote talema-{{en,es,it,ipa}}.pls for the {len(words)} most frequent words → {a.out}")


if __name__ == "__main__":
    main()
