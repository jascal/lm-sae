"""Write Talema by writing its trees. The three rules then spell the text, and the result is checked.

Source files (conlang/book/*.tl) hold sentences as trees of concepts:

    (know (OBJ (choose (OBJ she) why (SUBJ I))) not (SUBJ I))

  word          a concept by its English key (conlang/lexicon.tsv, then conlang/book/coin.tsv)
  word/CLASS    the same, when the key names several concepts (that/SCONJ, one/NUM, so/ADV)
  =root         a root directly
  "text"        a literal: a name, a number, or a quoted source word (spelled  text-<ending>)
  84            a number literal
  SUBJ OBJ DAT  the role particles
  # ...         comment to end of line (inside or outside trees)

Layout lines: `## (tree)` a heading; `@verse` one sentence per line; `@prose` running paragraphs
(blank line = new paragraph); `@tree` draw the next sentence indented by depth as well; `@raw text` a line
of pre-built Talema (the dictionary); `@dialect text` a line in a declared dialect, printed but exempt
from the standard one-tree check. Layout never changes a sentence; it only places it.

  author.py build conlang/book/*.tl --out conlang/TALEMA.md    render + round-trip check + stats
  author.py find WORD...                                        look a concept up by English/German/Spanish
  author.py coin --en X --de Y --es Z [--class NOUN]            the root the coinage law gives a new word
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "conlang"))
from conlang import VOWELS, decode, numeral  # noqa: E402

LEX = ROOT / "conlang" / "lexicon.tsv"
COIN = ROOT / "conlang" / "book" / "coin.tsv"
ROLES = {"SUBJ": "p", "OBJ": "t", "DAT": None}
CONS = "ptkbdgmnlrsfvh"


# ── lexicon (+ coinages) ──────────────────────────────────────────────────────────────────────────────
class Lex:
    def __init__(self):
        self.rows = []                       # (root, class, en, de, es, source)
        for line in LEX.read_text(encoding="utf-8").splitlines()[1:]:
            c = line.split("\t")
            self.rows.append((c[0], c[1], c[2], c[3], c[4], "lexicon"))
        if COIN.exists():
            for line in COIN.read_text(encoding="utf-8").splitlines()[1:]:
                if line.strip() and not line.startswith("#"):
                    c = (line.split("\t") + [""] * 7)[:7]
                    self.rows.append((c[1], c[2], c[0] + ("|" + c[3] if c[3] else ""), c[4], c[5], "coin"))
        self.key, self.roots = {}, {}
        for root, cl, en, de, es, src in self.rows:
            if root in self.roots and src == "coin":
                raise SystemExit(f"coin.tsv: root {root} is already taken by {self.roots[root][2]}")
            self.roots[root] = (cl, en, de, es, src)
            for w in filter(None, en.split("|")):
                self.key.setdefault(w, []).append((root, cl))
                self.key.setdefault(f"{w}/{cl}", []).append((root, cl))
        for r in ("SUBJ", "OBJ", "DAT"):
            name = {"SUBJ": "<subject>", "OBJ": "<object>", "DAT": "<to-whom>"}[r]
            ROLES[r] = next(root for root, cl, en, *_ in self.rows if en == name)

    def resolve(self, atom: str, where: str) -> tuple[str, bool]:
        """→ (spelling stem, is_literal)."""
        if atom in ROLES:
            return ROLES[atom], False
        if atom.startswith("="):
            if atom[1:] not in self.roots:
                raise SystemExit(f"{where}: {atom} is not a root")
            return atom[1:], False
        if atom.startswith('"'):
            return atom.strip('"').replace(" ", "_"), True
        if re.fullmatch(r"-?\d+([.,]\d+)?", atom):
            return atom, True
        base, _, cl = atom.partition("/")
        hits = self.key.get(base.lower() + ("/" + cl if cl else ""))
        if not hits:
            raise SystemExit(f"{where}: no concept '{atom}' (try author.py find, or coin it)")
        return hits[0][0], False             # rows are in weight order: the heaviest sense unless /CLASS


# ── source → trees ────────────────────────────────────────────────────────────────────────────────────
TOK = re.compile(r'\(|\)|"[^"]*"|[^\s()]+')


def parse_trees(text: str, where: str):
    toks = TOK.findall(text)
    i = 0

    def tree():
        nonlocal i
        t = toks[i]
        if t == "(":
            i += 1
            head = toks[i]
            i += 1
            kids = []
            while toks[i] != ")":
                kids.append(tree())
            i += 1
            return (head, kids)
        if t == ")":
            raise SystemExit(f"{where}: unexpected ')'")
        i += 1
        return (t, [])
    out = []
    while i < len(toks):
        out.append(tree())
    return out


def spell(lex: Lex, t, where: str) -> list[str]:
    head, kids = t
    stem, lit = lex.resolve(head, where)
    word = f"{stem}-{numeral(len(kids))}" if lit else stem + numeral(len(kids))
    return [word] + [w for k in kids for w in spell(lex, k, where)]


def draw(lex: Lex, t, where: str, depth: int = 0) -> list[str]:
    head, kids = t
    stem, lit = lex.resolve(head, where)
    word = f"{stem}-{numeral(len(kids))}" if lit else stem + numeral(len(kids))
    return ["    " * depth + word] + [line for k in kids for line in draw(lex, k, where, depth + 1)]


def strip_comments(line: str) -> str:
    out, q = [], False
    for ch in line:
        if ch == '"':
            q = not q
        if ch == "#" and not q:
            break
        out.append(ch)
    return "".join(out)


def build(files: list[Path], lex: Lex):
    """→ (markdown lines, list of rendered sentences)."""
    md, sentences = [], []
    for f in files:
        mode, para, pending_tree = "prose", [], False

        def flush():
            nonlocal para
            if para:
                md.append(" ".join(para) if mode == "prose" else "  \n".join(para))
                md.append("")
                para = []
        buf, start = [], 0
        lines = f.read_text(encoding="utf-8").splitlines()
        for n, raw in enumerate(lines, 1):
            where = f"{f.name}:{n}"
            s = raw.strip()
            if buf and (s.startswith("## ") or s.startswith("@")):
                raise SystemExit(f"{f.name}:{start}: unbalanced parentheses (the tree runs into line {n})")
            if s.startswith("## "):
                flush()
                t = parse_trees(strip_comments(s[3:]), where)
                md.append("## " + " ".join(w for tr in t for w in spell(lex, tr, where)))
                md.append("")
                continue
            if s.startswith("@"):
                flush()
                word = s.split()[0]
                if word == "@verse":
                    mode = "verse"
                elif word == "@prose":
                    mode = "prose"
                elif word == "@tree":
                    pending_tree = True
                elif word == "@raw":
                    md.append(s[5:])
                    sentences.append(s[5:])
                elif word == "@dialect":                 # printed, deliberately NOT checked: a dialect line
                    md.append(s[9:] + "  ")
                elif word == "@break":
                    md.append("---")
                    md.append("")
                continue
            body = strip_comments(raw)
            if not body.strip():
                if not buf:
                    flush()
                continue
            if not buf:
                start = n
            buf.append(body)
            text = " ".join(buf)
            if text.count(")") > text.count("("):
                raise SystemExit(f"{f.name}:{start}: unbalanced parentheses (one ')' too many)")
            if text.count("(") == text.count(")"):
                for tr in parse_trees(text, where):
                    words = spell(lex, tr, where)
                    sentence = " ".join(words)
                    sentences.append(sentence)
                    if pending_tree:
                        flush()
                        md.append("```")
                        md.extend(draw(lex, tr, where))
                        md.append("```")
                        md.append("")
                        pending_tree = False
                    para.append(sentence + " .")
                buf = []
        if buf:
            raise SystemExit(f"{f.name}:{start}: unbalanced parentheses (the tree never closes)")
        flush()
    return md, sentences


def check(sentences: list[str]) -> None:
    """Every sentence must decode to exactly one tree (the grammar's promise)."""
    for s in sentences:
        for part in [s]:
            try:
                decode(part)
            except (StopIteration, ValueError) as e:
                raise SystemExit(f"does not decode as one tree: {s!r} ({e})")


# ── commands ──────────────────────────────────────────────────────────────────────────────────────────
def unknown_keys(files: list[Path], lex: Lex) -> list[str]:
    """Every atom that does not resolve, across all files, so a draft can be fixed in one pass."""
    bad = []
    for f in files:
        text = "\n".join(strip_comments(l)[3:] if l.strip().startswith("## ") else
                         ("" if l.strip().startswith("@") else strip_comments(l))
                         for l in f.read_text(encoding="utf-8").splitlines())
        for tok in TOK.findall(text):
            if tok in "()" or tok.startswith('"') or tok in ROLES or re.fullmatch(r"-?\d+([.,]\d+)?", tok):
                continue
            base, _, cl = tok.partition("/")
            if tok.startswith("="):
                ok = tok[1:] in lex.roots
            else:
                ok = (base.lower() + ("/" + cl if cl else "")) in lex.key
            if not ok:
                bad.append(f"{f.name}: {tok}")
    return sorted(set(bad))


def cmd_build(a) -> None:
    lex = Lex()
    bad = unknown_keys(a.files, lex)
    if bad:
        raise SystemExit("unknown concepts:\n  " + "\n  ".join(bad))
    md, sentences = build(a.files, lex)
    check(sentences)
    text = "\n".join(md).rstrip() + "\n"
    a.out.write_text(text, encoding="utf-8")
    words = [w for s in sentences for w in s.split()]
    roots = {w.rstrip(VOWELS) for w in words if "-" not in w}
    lits = sum(1 for w in words if "-" in w)
    print(f"{a.out}: {len(sentences)} sentences · {len(words)} words · {len(roots)} distinct roots · "
          f"{lits} literals · {len(text)} chars")
    if a.tokens:
        from transformers import AutoTokenizer
        for name in a.tokens:
            tok = AutoTokenizer.from_pretrained(name)
            print(f"  tokens ({name}): {len(tok(text)['input_ids'])}")


def cmd_find(a) -> None:
    lex = Lex()
    for q in a.words:
        ql = q.lower()
        hits = [r for r in lex.rows if any(ql == w.lower() for col in r[2:5] for w in col.split("|"))]
        if not hits:
            hits = [r for r in lex.rows if any(w.lower().startswith(ql) for col in r[2:5] for w in col.split("|"))][:8]
        print(f"{q}: " + ("  ".join(f"{r[0]}[{r[1]}] {r[2]}/{r[3]}/{r[4]}" for r in hits[:8]) or "—"))


def cmd_coin(a) -> None:
    sys.path.insert(0, str(ROOT / "scripts" / "conlang"))
    from lexicon import candidates, sim, stem
    from wordfreq import word_frequency
    lex = Lex()
    srcs = [(max(word_frequency(w, l), 1e-9), stem(w, l)) for l, w in (("en", a.en), ("de", a.de), ("es", a.es)) if w]
    tot = sum(f for f, _ in srcs)
    for L in range(a.min_len, 6):
        pool = candidates([s for _, s in srcs], L)
        free = [r for r in pool if r not in lex.roots]
        if free:
            best = max(free, key=lambda r: (sum(f / tot * sim(r, s) for f, s in srcs), -len(r)))
            score = sum(f / tot * sim(best, s) for f, s in srcs)
            print(f"{best}\t(similarity {score:.2f}; {len(free)} free candidates at {L} consonants)")
            return
    print("no free root")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("build")
    q.add_argument("files", nargs="+", type=Path)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--tokens", nargs="*", help="HF tokenizer names to count tokens with")
    q.set_defaults(fn=cmd_build)
    q = sub.add_parser("find")
    q.add_argument("words", nargs="+")
    q.set_defaults(fn=cmd_find)
    q = sub.add_parser("coin")
    q.add_argument("--en", default="")
    q.add_argument("--de", default="")
    q.add_argument("--es", default="")
    q.add_argument("--min-len", type=int, default=3)
    q.set_defaults(fn=cmd_coin)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
