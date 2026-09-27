"""Build every Talema book, and check them all.

The founding text is a core book plus companion field volumes:

  conlang/BUKE_DE_LORE_FIRA.md        the core: grammar, first words, conventions, tales, agent speech, growth,
                                      songs, the book of roots, the last page (sources: conlang/book/*.tl)
  conlang/volumes/<field>.md          one volume per field: title page, the field chapter, its glossary
                                      (sources: conlang/volumes/<field>/*.tl, concepts: conlang/fields/<field>.tsv)

Keeping fields in volumes keeps the core within a frontier context window as fields are added; a reader loads
the core and whichever volumes a task needs.

Steps: regenerate the core's generated chapters (book of roots, first words) and each field glossary; lint each
field chapter against its glossary; build every book (author.py refuses unknown concepts and any sentence that
does not decode to one tree); check that every sentence ends in "a"; report sizes.

  .venv/bin/python scripts/conlang/books.py [--tokens Qwen/Qwen2.5-0.5B]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
S = ROOT / "scripts" / "conlang"
# the Talema words that name each field's world, for its glossary heading
FIELD_HEAD = {"digital": "digital/ADJ the", "mathematics": "mathematics"}


def run(*args: str) -> str:
    r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(args)}\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def endings_ok(md: Path) -> tuple[int, list[str]]:
    """Every sentence ends in "a" (tree drawings and the one declared dialect line aside)."""
    sents, inblock = [], False
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            inblock = not inblock
            continue
        if inblock or not line or line.startswith("---"):
            continue
        sents += [s.strip() for s in line.lstrip("# ").rstrip().split(" .") if s.strip()]
    return len(sents), [s for s in sents if not s.endswith("a") and not s.startswith("pe ma seri")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", nargs="*", default=[], help="HF tokenizer names to report sizes with")
    a = p.parse_args()

    run(str(S / "roots_book.py"), "--out", "conlang/book/08_roots.tl")
    run(str(S / "roots_book.py"), "--first", "--out", "conlang/book/02b_first_words.tl")
    books = [("core", sorted((ROOT / "conlang" / "book").glob("*.tl")), ROOT / "conlang" / "BUKE_DE_LORE_FIRA.md")]
    for vol in sorted((ROOT / "conlang" / "volumes").iterdir()):
        if not vol.is_dir():
            continue
        tsv = ROOT / "conlang" / "fields" / f"{vol.name}.tsv"
        run(str(S / "field.py"), "glossary", str(tsv), "--out", str(vol / "02_words.tl"),
            "--head", FIELD_HEAD.get(vol.name, vol.name))
        print(run(str(S / "field.py"), "lint", str(tsv), str(vol / "01_chapter.tl")))
        books.append((vol.name, sorted(vol.glob("*.tl")), ROOT / "conlang" / "volumes" / f"{vol.name}.md"))

    bad_any = False
    for name, files, out in books:
        extra = ["--tokens", *a.tokens] if a.tokens else []
        print(f"[{name}] " + run(str(S / "author.py"), "build", *map(str, files), "--out", str(out), *extra)
              .replace("\n", " · "))
        n, bad = endings_ok(out)
        print(f"[{name}] {n} sentences; not ending in a: {len(bad)}")
        bad_any |= bool(bad)
    sys.exit(1 if bad_any else 0)


if __name__ == "__main__":
    main()
