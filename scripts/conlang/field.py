"""Field chapters: the specialised vocabularies of the founding text (digital world, mathematics, then logic …).

Each field is a companion volume, conlang/volumes/<field>/ (00_title.tl, 01_chapter.tl, 02_words.tl), built by
scripts/conlang/books.py into conlang/volumes/<field>.md.

A field is a concept list, conlang/fields/<field>.tsv (en, de, es, class, group): the concepts in the sense
the field uses them. This tool

  check     says which concepts already have a root (in the frozen lexicon or the book's coinages) and which do not
  coin      coins the missing ones by the root law (author.py coin) and records them in conlang/book/coin.tsv,
            tagged with the field
  lint      checks a field chapter (.tl) against the glossary: every concept key the chapter uses must resolve to
            the root the glossary gives it (otherwise use key/CLASS or =root in the chapter)
  glossary  writes the field's glossary chapter: every concept in dictionary form (bare root, kind, the FIELD's
            mother words), grouped. An old root used in a new sense (bugek, the insect, as a bug in code) is
            listed with the field sense, so the glossary is also the record of the field's metaphors

  .venv/bin/python scripts/conlang/field.py check conlang/fields/digital.tsv
  .venv/bin/python scripts/conlang/field.py coin conlang/fields/digital.tsv
  .venv/bin/python scripts/conlang/field.py glossary conlang/fields/digital.tsv --out conlang/volumes/digital/02_words.tl
  .venv/bin/python scripts/conlang/field.py lint conlang/fields/digital.tsv conlang/volumes/digital/01_chapter.tl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "conlang"))
import author  # noqa: E402
from roots_book import CLASS_WORD  # noqa: E402

# group names (the concept file's last column) → the Talema heading for that section of the glossary
GROUP_HEAD = {"computing": "(and computer code)", "unix": "(shell the)", "security": "(and key secret)",
              "time": "time", "cloud": "(and cloud service)", "ai": "(and model mind)",
              # mathematics
              "number": "number", "operation": "(act/NOUN (of number))", "algebra": "(and equation unknown/NOUN)",
              "structure": "(and set/NOUN element)", "geometry": "geometry", "proof": "(and proof theorem)",
              "calculus": "calculus", "chance": "(and chance probability)",
              # logic
              "truth": "(and true false)", "connective": "connective", "argument": "(and argument conclusion)",
              "quantifier": "(and quantifier predicate)", "limit": "(and paradox (limit/NOUN (of logic)))",
              "fallacy": "fallacy", "rule": "(and rule symbol)",
              # physics
              "measure": "(and measurement theory)", "matter": "matter", "motion": "(and motion force)",
              "energy": "energy", "field": "(and light wave)", "cosmos": "(and space universe)",
              "quantum": "quantum",
              # philosophy
              "wonder": "wonder/NOUN", "being": "(and being change)", "knowing": "(and knowledge doubt)",
              "mind": "(and mind self)", "language": "language",
              # morality
              "meal": "meal", "bread": "bread", "meat": "meat", "plant": "vegetable", "fruit": "fruit",
              "drink": "drink/NOUN", "sweet": "(and sugar salt)", "table": "table", "kitchen": "kitchen",
              "restaurant": "restaurant",
              "good": "(and good/NOUN evil)", "virtue": "virtue", "trust": "(and promise/NOUN trust/NOUN)",
              "harm": "harm", "person": "(and dignity person)", "theory": "consequence"}


def concepts(path: Path) -> list[dict]:
    rows = path.read_text(encoding="utf-8").splitlines()
    cols = rows[0].split("\t")
    return [dict(zip(cols, r.split("\t"))) for r in rows[1:] if r.strip()]


def root_of(lex: author.Lex, c: dict) -> str | None:
    """The root a concept already has: a coinage under the same key, else a lexicon row with the same English
    word and class. Concepts named with an underscore (network_neural) are phrases, not roots."""
    for root, (cl, en, *_rest) in lex.roots.items():
        src = _rest[-1]
        if src == "coin" and en.split("|")[0] == c["en"]:
            return root
    hits = lex.key.get(f"{c['en']}/{c['class']}") or []
    return hits[0][0] if hits else None


def cmd_check(a) -> None:
    lex = author.Lex()
    have, miss = [], []
    for c in concepts(a.tsv):
        r = root_of(lex, c)
        (have if r else miss).append(f"{c['en']}={r}" if r else c["en"])
    print(f"{len(have)} have a root, {len(miss)} do not")
    print("have:", " ".join(have))
    print("missing:", " ".join(miss))


def cmd_coin(a) -> None:
    from lexicon import candidates, sim, stem
    from wordfreq import word_frequency
    field = a.tsv.stem
    made = []
    for c in concepts(a.tsv):
        lex = author.Lex()
        if "_" in c["en"] or root_of(lex, c):
            continue                                   # a phrase, or already has a root
        words = [("en", c["en"]), ("de", c["de"].split("_")[0]), ("es", c["es"].split("_")[0])]
        srcs = [(max(word_frequency(w, lang), 1e-9), stem(w, lang)) for lang, w in words]
        tot = sum(f for f, _ in srcs)
        best = None
        for L in range(3, 6):
            free = [r for r in candidates([s for _, s in srcs], L) if r not in lex.roots]
            if free:
                best = max(free, key=lambda r: (sum(f / tot * sim(r, s) for f, s in srcs), -len(r)))
                break
        with (ROOT / "conlang" / "book" / "coin.tsv").open("a", encoding="utf-8") as f:
            f.write(f"{c['en']}\t{best}\t{c['class']}\t\t{c['de'].replace('_', ' ')}\t{c['es'].replace('_', ' ')}"
                    f"\tfield: {field} ({c['group']})\n")
        made.append(f"{c['en']}={best}")
    print(f"coined {len(made)}:", " ".join(made))


def cmd_lint(a) -> None:
    lex = author.Lex()
    code = "\n".join(author.strip_comments(line) for line in a.chapter.read_text(encoding="utf-8").splitlines())
    used = set(author.TOK.findall(code))                   # trees only: English comments are not concept keys
    bad = []
    for c in concepts(a.tsv):
        want = root_of(lex, c)
        for key in (c["en"], f"{c['en']}/{c['class']}"):
            if want and key in used and lex.resolve(key, "lint")[0] != want:
                got = lex.resolve(key, "lint")[0]
                bad.append(f"{key}: chapter says {got} ({lex.roots[got][1]}), glossary says {want}")
    print("\n".join(bad) if bad else f"{a.chapter.name}: every field concept resolves to its glossary root")
    sys.exit(1 if bad else 0)


def cmd_glossary(a) -> None:
    lex = author.Lex()
    field = a.tsv.stem
    out = [f"# Field glossary: {field}. Generated by scripts/conlang/field.py glossary; do not edit by hand.",
           f"## (word (of (world {a.head})))", "",
           "(have (OBJ (meaning (of (world this)))) (in (list this)) (SUBJ (word the)))   "
           "# In this list the words have the meaning of this world.",
           ""]
    groups: dict[str, list[dict]] = {}
    for c in concepts(a.tsv):
        groups.setdefault(c["group"], []).append(c)
    for g, cs in groups.items():
        out.append(f"## {GROUP_HEAD.get(g, g)}")
        for c in cs:
            r = root_of(lex, c)
            if not r:
                continue                               # phrases are taught in the chapter, not listed as roots
            words = [c["en"], c["de"].replace("_", " "), c["es"].replace("_", " ")]
            kids = [CLASS_WORD.get(c["class"], "nomun") + "a"] + [f"{w.replace(' ', '_')}-a" for w in words]
            from conlang import numeral
            out.append("@raw " + " ".join([f"{r}-{numeral(len(kids))}"] + kids))
        out.append("")
    a.out.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{a.out}: {sum(len(v) for v in groups.values())} concepts in {len(groups)} groups")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("coin", cmd_coin), ("glossary", cmd_glossary), ("lint", cmd_lint)):
        q = sub.add_parser(name)
        q.add_argument("tsv", type=Path)
        if name == "lint":
            q.add_argument("chapter", type=Path)
        if name == "glossary":
            q.add_argument("--out", type=Path, required=True)
            q.add_argument("--head", default="digital/ADJ the", help="Talema words naming the field's world")
        q.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
