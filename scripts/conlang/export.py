"""Export the Talema books as a parallel corpus (the Hugging Face dataset jallanscott/talema, github.com/jascal/talema).

Every sentence of every book is one record: its Talema text (exactly as the built book prints it), the concept tree it
was written as, and the English line the author wrote beside the tree. German and Spanish are machine translations
of that English line (Helsinki-NLP opus-mt), marked as such. The dictionaries (book of roots, field glossaries) become a
lexicon with the trilingual source words of every root.

  .venv/bin/python scripts/conlang/export.py --out ../talema            # sentences + lexicon + books + sources
  .venv/bin/python scripts/conlang/export.py --out ../talema --no-mt    # translate nothing new (keeps existing ones)
  .venv/bin/python scripts/conlang/export.py --out ../talema --retranslate   # translate every line again

A line already translated in <out>/data/sentences.jsonl keeps its translation, and only new English lines go
through the model. Retranslating everything is not idempotent: the model batches its input, so adding lines can
re-roll a borderline one (a German line flipped into a hallucinated Parliament sentence that way).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "conlang"))
import author  # noqa: E402
from books import FIELD_HEAD  # noqa: E402,F401  (the volumes books.py builds)

MT = {"de": "Helsinki-NLP/opus-mt-en-de", "es": "Helsinki-NLP/opus-mt-en-es"}


def comment(line: str) -> str:
    """The English after the tree's '#' (outside quotes), or ''."""
    code = author.strip_comments(line)
    rest = line[len(code):].strip()
    return rest[1:].strip() if rest.startswith("#") else ""


def tree_text(t) -> str:
    head, kids = t
    return head if not kids else "(" + " ".join([head] + [tree_text(k) for k in kids]) + ")"


def books() -> list[tuple[str, list[Path], Path]]:
    out = [("core", sorted((ROOT / "conlang" / "book").glob("*.tl")), ROOT / "conlang" / "BUKE_DE_LORE_FIRA.md")]
    for vol in sorted((ROOT / "conlang" / "volumes").iterdir()):
        if vol.is_dir():
            out.append((vol.name, sorted(vol.glob("*.tl")), ROOT / "conlang" / "volumes" / f"{vol.name}.md"))
    return out


def sentences(lex: author.Lex, book: str, files: list[Path]) -> list[dict]:
    """Walk the sources the way author.build does, keeping each tree's English line and section."""
    recs = []
    for f in files:
        mode, section, section_tree, buf, notes, start = "prose", "", "", [], [], 0
        for n, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            s = raw.strip()
            if s.startswith("## "):
                t = author.parse_trees(author.strip_comments(s[3:]), f.name)
                section = " ".join(w for tr in t for w in author.spell(lex, tr, f.name))
                section_tree = " ".join(tree_text(tr) for tr in t)
                continue
            if s.startswith("@"):
                word = s.split()[0]
                if word in ("@verse", "@prose"):
                    mode = word[1:]
                if word == "@raw":                   # a dictionary line: exported through the lexicon instead
                    continue
                if word == "@dialect":
                    recs.append(dict(id=f"{book}/{f.stem}/{n:04d}", book=book, chapter=f.stem, section=section,
                                     section_tree=section_tree, kind="dialect", talema=s[9:].strip(), tree="",
                                     en="", source_line=n))
                continue
            body = author.strip_comments(raw)
            if not body.strip():
                continue
            if not buf:
                start = n
            buf.append(body)
            notes.append(comment(raw))
            text = " ".join(buf)
            if text.count("(") == text.count(")"):
                trees = author.parse_trees(text, f"{f.name}:{start}")
                en = " ".join(x for x in notes if x)
                for i, tr in enumerate(trees):
                    recs.append(dict(id=f"{book}/{f.stem}/{start:04d}" + (f".{i}" if len(trees) > 1 else ""),
                                     book=book, chapter=f.stem, section=section, section_tree=section_tree,
                                     kind=mode, talema=" ".join(author.spell(lex, tr, f.name)) + " .",
                                     tree=tree_text(tr), en=en if len(trees) == 1 else "", source_line=start))
                buf, notes = [], []
    return recs


def lexicon(lex: author.Lex) -> list[dict]:
    """Every root: the frozen lexicon, then the book's coinages (core or field)."""
    out = []
    rows = (ROOT / "conlang" / "lexicon.tsv").read_text(encoding="utf-8").splitlines()
    cols = rows[0].split("\t")
    for r in rows[1:]:
        d = dict(zip(cols, r.split("\t")))
        out.append(dict(root=d["root"], cls=d["class"], en=d["en"], de=d["de"], es=d["es"], tier=int(d["tier"]),
                        weight=float(d["weight"]), source="lexicon"))
    rows = (ROOT / "conlang" / "book" / "coin.tsv").read_text(encoding="utf-8").splitlines()
    cols = rows[0].split("\t")
    for r in rows[1:]:
        d = dict(zip(cols, r.split("\t")))
        note = d.get("note", "") or ""
        field = note.split("field: ")[1].split(" ")[0] if note.startswith("field: ") else "core"
        out.append(dict(root=d["root"], cls=d["class"], en=d["key"], de=d["de"], es=d["es"], tier=None, weight=None,
                        source=f"coin:{field}"))
    return out


def translate(texts: list[str], model_name: str, bs: int = 32) -> list[str]:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(dev).eval()
    out = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=4, max_new_tokens=256)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-mt", action="store_true")
    p.add_argument("--retranslate", action="store_true", help="ignore the translations already in --out")
    a = p.parse_args()
    lex = author.Lex()
    (a.out / "data").mkdir(parents=True, exist_ok=True)

    recs, bad = [], 0
    for book, files, md in books():
        rs = sentences(lex, book, files)
        text = md.read_text(encoding="utf-8")
        for r in rs:                                 # every exported sentence must be in the built book verbatim
            if r["kind"] != "dialect" and r["talema"][:-2] not in text:
                bad += 1
        recs += rs
        dest = a.out / ("books" if book == "core" else "books/volumes")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(md, dest / md.name)
        src = a.out / "source" / ("core" if book == "core" else f"volumes/{book}")
        src.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, src / f.name)
    if bad:
        sys.exit(f"{bad} exported sentences are not in their built book")

    have = {}                                        # (language, English line) -> the translation already published
    prev = a.out / "data" / "sentences.jsonl"
    if prev.exists() and not a.retranslate:
        for old in prev.read_text(encoding="utf-8").splitlines():
            r = json.loads(old)
            for lang in MT:
                if r.get("en") and r.get(f"{lang}_mt"):
                    have.setdefault((lang, r["en"]), r[f"{lang}_mt"])
    translated = 0
    for lang, model in MT.items():
        todo = sorted({r["en"] for r in recs if r["en"] and (lang, r["en"]) not in have})
        tr = dict(zip(todo, translate(todo, model))) if todo and not a.no_mt else {}
        translated += len(todo)
        for r in recs:
            r[f"{lang}_mt"] = tr.get(r["en"]) or have.get((lang, r["en"]), "")
    with (a.out / "data" / "sentences.jsonl").open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    lx = lexicon(lex)
    with (a.out / "data" / "lexicon.jsonl").open("w", encoding="utf-8") as fh:
        for r in lx:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for f in ("lexicon.tsv", "core.tsv", "book/coin.tsv"):
        shutil.copy(ROOT / "conlang" / f, a.out / "source" / Path(f).name)
    shutil.copytree(ROOT / "conlang" / "speech", a.out / "speech", dirs_exist_ok=True)
    kit = a.out / "speech" / "README.md"                 # the core book sits under books/ in the dataset
    kit.write_text(kit.read_text(encoding="utf-8").replace("(`../BUKE_DE_LORE_FIRA.md`",
                                                           "(`../books/BUKE_DE_LORE_FIRA.md`"), encoding="utf-8")
    kinds = {}
    for r in recs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"{len(recs)} sentences ({kinds}); {sum(1 for r in recs if r['en'])} with English; "
          f"{translated} translations made (per language: only lines not already in {a.out.name}); {len(lx)} lexicon entries")


if __name__ == "__main__":
    main()
