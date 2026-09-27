"""Driver for the conlang (grammar: dl/conlang/linearize.dl, lexicon: conlang/lexicon.tsv).

Python stages facts, runs Soufflé, spells what the program derived, and parses strings back:

  translate  English/German/Spanish UD sentences (PUD by default) → conlang, with glosses
  gate       round trip: UG tree → string → parsed tree must be identical, on every sentence
  compare    how many rules each language needs to spell the UG form, next to the conlang's three
             (order rules, discontinuity, inflection), plus lexicon coverage, word length and how
             closely the en/de/es versions of the same sentence agree

Spelling: word = root + arity numeral (vowels a e i o u = 0–4, base 5 above, most significant first).
A name, number or word outside the lexicon is written as its source lemma + "-" + numeral, so the
string still decodes exactly.

  .venv/bin/python scripts/conlang/conlang.py translate --sent 5
  .venv/bin/python scripts/conlang/conlang.py gate
  .venv/bin/python scripts/conlang/conlang.py compare --out runs/conlang/conlang_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "grammar"))
from ug_extract import PUD, read_conllu  # noqa: E402

DL = ROOT / "dl" / "conlang" / "linearize.dl"
LEX = ROOT / "conlang" / "lexicon.tsv"
VOWELS = "aeiou"
CLASSES = {"AUX": "VERB"}


# ── lexicon ───────────────────────────────────────────────────────────────────────────────────────────
class Lexicon:
    def __init__(self, path: Path = LEX):
        self.by_key, self.gloss, self.role = {}, {}, {}
        for row in path.read_text(encoding="utf-8").splitlines()[1:]:
            root, cl, en, de, es, *_ = row.split("\t")
            if cl == "ROLE":
                name = en.strip("<>")
                self.role[{"subject": "SUBJ", "object": "OBJ", "to-whom": "DAT"}[name]] = root
                self.gloss[root] = name.upper()
                continue
            g = (en or de or es).split("|")[0]
            self.gloss[root] = "I" if g == "i" else g
            for lang, cell in (("en", en), ("de", de), ("es", es)):
                for w in filter(None, cell.split("|")):
                    self.by_key.setdefault((lang, w, cl), root)
                    self.by_key.setdefault((lang, w, "*"), root)

    def root(self, lang: str, lemma: str, upos: str) -> str | None:
        if upos in ("PROPN", "PUNCT", "SYM", "X") or not lemma.isalpha():
            return None
        lem, cl = lemma.lower(), CLASSES.get(upos, upos)
        # possessives are DET in one treebank and PRON in another: try the sibling class before any class
        sib = {"DET": "PRON", "PRON": "DET"}.get(cl)
        return (self.by_key.get((lang, lem, cl)) or (sib and self.by_key.get((lang, lem, sib)))
                or self.by_key.get((lang, lem, "*")))


def numeral(n: int) -> str:
    digits = []
    while True:
        digits.append(VOWELS[n % 5])
        n //= 5
        if n == 0:
            return "".join(reversed(digits))


def parse_numeral(v: str) -> int:
    n = 0
    for ch in v:
        n = n * 5 + VOWELS.index(ch)
    return n


# ── Datalog ───────────────────────────────────────────────────────────────────────────────────────────
def load(lang: str, split: str = "all", n: int = 0):
    sents = read_conllu(PUD / f"{lang}_pud.conllu")
    out = [(si, s) for si, s in enumerate(sents) if split == "all" or (split == "test") == (si % 5 == 0)]
    return out[:n] if n else out


def lemma_of(w: dict) -> str:
    return w.get("lemma") or w["form"]


def run_datalog(batch: list[tuple[str, int, dict]], out: Path) -> dict:
    """batch: (lang, index, sentence). Returns per-sentence cwords and conlang-tree parents."""
    rows, lrows = [], []
    for s_id, (lang, si, sent) in enumerate(batch, 1):
        lrows.append(f"{s_id}\t{lang}")
        for i, w in enumerate(sent["words"], 1):
            lem = lemma_of(w).replace("\t", " ")
            rows.append(f"{s_id}\t{i}\t{lem}\t{w['upos']}\t{w['head'] + 1}\t{w['deprel']}")
    (out / "word.facts").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "lang.facts").write_text("\n".join(lrows) + "\n")
    (out / "type_axis.facts").write_text("")
    subprocess.run(["souffle", "-F", str(out), "-D", str(out), str(DL)], check=True)
    for bad in ("pos_clash", "unplaced", "bad_head_count", "cycle", "bad_root_count"):
        if (out / f"{bad}.csv").read_text().strip():
            raise SystemExit(f"Datalog invariant violated: {bad}")
    res = defaultdict(lambda: {"words": [], "parent": {}})
    for line in (out / "cword.csv").read_text(encoding="utf-8").splitlines():
        s, p, n, a, kind, lem, upos = line.split("\t")
        res[int(s)]["words"].append((int(p), int(n), int(a), kind, lem, upos))
    for line in (out / "cparent.csv").read_text().splitlines():
        s, n, h, _ = line.split("\t")
        res[int(s)]["parent"][int(n)] = int(h)
    for r in res.values():
        r["words"].sort()
    return res


def spell(lex: Lexicon, lang: str, words) -> tuple[list[str], list[str]]:
    toks, gloss = [], []
    for _, _, a, kind, lem, upos in words:
        if kind != "word":
            root = lex.role[kind]
            toks.append(root + numeral(a))
            gloss.append(kind)
            continue
        root = lex.root(lang, lem, upos)
        if root:
            toks.append(root + numeral(a))
            gloss.append(lex.gloss[root])
        else:                                            # name / number / out-of-lexicon: literal lemma
            lit = lem.replace(" ", "_")
            toks.append(f"{lit}-{numeral(a)}")
            gloss.append(lit)
    return toks, gloss


def tree_from_datalog(lex: Lexicon, lang: str, r) -> tuple:
    """The UG form at concept level: (label, children...) with children in Merge (= spoken) order."""
    toks, _ = spell(lex, lang, r["words"])
    labels = {n: t.rsplit("-", 1)[0] + "-" if "-" in t else t.rstrip(VOWELS) for t, (_, n, *_ ) in zip(toks, r["words"])}
    order = {n: p for p, n, *_ in r["words"]}
    kids = defaultdict(list)
    for n, h in r["parent"].items():
        kids[h].append(n)
    root = next(n for _, n, *_ in r["words"] if n not in r["parent"])

    def build(n):
        return (labels[n],) + tuple(build(c) for c in sorted(kids[n], key=order.get))
    return build(root)


def decode(text: str) -> tuple:
    """String → tree, using nothing but the three rules: numeral = arity, head first, preorder."""
    toks = text.split()
    it = iter(toks)

    def read():
        t = next(it)
        if "-" in t:
            label, num = t.rsplit("-", 1)
            label += "-"
        else:
            k = len(t)
            while k > 0 and t[k - 1] in VOWELS:
                k -= 1
            label, num = t[:k], t[k:]
        return (label,) + tuple(read() for _ in range(parse_numeral(num)))
    tree = read()
    if next(it, None) is not None:
        raise ValueError("trailing words: the string is not one tree")
    return tree


# ── commands ──────────────────────────────────────────────────────────────────────────────────────────
def cmd_translate(a) -> None:
    lex = Lexicon()
    by = defaultdict(dict)
    for lang in a.langs:
        for si, s in load(lang, "test"):
            by[si][lang] = s
    chosen = [si for si in sorted(by) if len(by[si]) == len(a.langs)
              and max(len(v["words"]) for v in by[si].values()) <= a.max_len][: a.sent]
    batch = [(lang, si, by[si][lang]) for si in chosen for lang in a.langs]
    with tempfile.TemporaryDirectory() as tmp:
        res = run_datalog(batch, Path(tmp))
    for k, (lang, si, sent) in enumerate(batch, 1):
        toks, gloss = spell(lex, lang, res[k]["words"])
        if lang == a.langs[0]:
            print(f"\n#{si}")
        print(f"  {lang} │ {sent['text']}")
        print(f"     → │ {' '.join(toks)} .")
        print(f"       │ {' '.join(g.upper() if g in ('SUBJ', 'OBJ', 'DAT') else g for g in gloss)}")


def cmd_gate(a) -> None:
    lex = Lexicon()
    batch = [(lang, si, s) for lang in a.langs for si, s in load(lang, "all", a.n)]
    with tempfile.TemporaryDirectory() as tmp:
        res = run_datalog(batch, Path(tmp))
    ok = 0
    for k, (lang, si, _) in enumerate(batch, 1):
        toks, _ = spell(lex, lang, res[k]["words"])
        if decode(" ".join(toks)) == tree_from_datalog(lex, lang, res[k]):
            ok += 1
        elif ok < 3:
            print(f"  mismatch {lang}#{si}")
    print(f"round trip: {ok}/{len(batch)} UG trees → string → identical tree")
    sys.exit(0 if ok == len(batch) else 1)


def nonprojective(heads: list[int]) -> bool:
    arcs = [(min(i, h), max(i, h)) for i, h in enumerate(heads) if h >= 0]
    return any(a1 < a2 < b1 < b2 for a1, b1 in arcs for a2, b2 in arcs)


def cmd_compare(a) -> None:
    lex = Lexicon()
    out = {"languages": {}, "conlang": {}}
    # ── rules each natural language needs to spell the UG form ──
    for lang in ("en", "de", "es"):
        sents = [s for _, s in load(lang)]
        with tempfile.TemporaryDirectory() as tmp:
            res = run_datalog([(lang, i, s) for i, s in enumerate(sents)], Path(tmp))
            links = [l.split("\t") for l in (Path(tmp) / "link.csv").read_text(encoding="utf-8").splitlines()]
        upos = {(k, i + 1): w["upos"] for k, s in enumerate(sents, 1) for i, w in enumerate(s["words"])}
        # order: direction of each UG link as a function of (head class, relation)
        groups = defaultdict(Counter)
        for s, i, h, r in links:
            s, i, h = int(s), int(i), int(h)
            if h == 0 or upos[(s, i)] == "PUNCT":
                continue
            groups[(upos[(s, h)], r)]["after" if h < i else "before"] += 1
        total = sum(sum(c.values()) for c in groups.values())
        default = max(("after", "before"), key=lambda d: sum(c[d] for c in groups.values()))
        explicit = [g for g, c in groups.items() if max(c, key=c.get) != default]
        correct = sum(max(c.values()) for c in groups.values())
        h_bits = -sum(sum(c.values()) / total * sum(v / sum(c.values()) * math.log2(v / sum(c.values()))
                                                    for v in c.values() if v) for c in groups.values())
        # morphology: surface forms per lemma, feature values in use
        forms, feats = defaultdict(set), set()
        for line in (PUD / f"{lang}_pud.conllu").read_text(encoding="utf-8").splitlines():
            c = line.split("\t")
            if len(c) == 10 and c[0].isdigit() and c[3] not in ("PUNCT", "PROPN", "NUM"):
                forms[(c[2].lower(), c[3])].add(c[1].lower())
                feats.update(f for f in c[5].split("|") if f != "_")
        multi = [len(v) for v in forms.values()]
        # lexicon coverage and word length
        words = [w for s in sents for w in s["words"] if w["upos"] not in ("PUNCT", "PROPN", "NUM", "SYM", "X")]
        covered = sum(1 for w in words if lex.root(lang, lemma_of(w), w["upos"]))
        conlang_len = [len(t) for k in res for t in spell(lex, lang, res[k]["words"])[0] if "-" not in t]
        src_len = [len(w["form"]) for w in words]
        out["languages"][lang] = {
            "order_rules": 1 + len(explicit), "order_groups": len(groups),
            "order_accuracy_with_those_rules": correct / total, "order_exceptions": total - correct,
            "order_entropy_bits_per_link": h_bits,
            "nonprojective_sentences": sum(nonprojective([w["head"] for w in s["words"]]) for s in sents) / len(sents),
            "morph_feature_values": len(feats), "forms_per_lemma": sum(multi) / len(multi),
            "lemmas_with_several_forms": sum(1 for m in multi if m > 1) / len(multi),
            "lexicon_coverage": covered / len(words),
            "mean_word_letters_source": sum(src_len) / len(src_len),
            "mean_word_letters_conlang": sum(conlang_len) / len(conlang_len),
        }
        print(lang, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out["languages"][lang].items()}))
    out["conlang"] = {"order_rules": 1, "order_accuracy": 1.0, "order_exceptions": 0, "order_entropy_bits_per_link": 0.0,
                      "nonprojective_sentences": 0.0, "morph_feature_values": 0, "forms_per_lemma": 1.0,
                      "grammar_rules": 3}
    # ── convergence: the same PUD sentence from en / de / es ──
    by = defaultdict(dict)
    batch = []
    for lang in ("en", "de", "es"):
        for si, s in load(lang):
            batch.append((lang, si, s))
    with tempfile.TemporaryDirectory() as tmp:
        res = run_datalog(batch, Path(tmp))
    for k, (lang, si, _) in enumerate(batch, 1):
        by[si][lang] = spell(lex, lang, res[k]["words"])[0]
    jac = []
    for si, d in by.items():
        bags = {l: Counter(t.rstrip(VOWELS) for t in toks if "-" not in t) for l, toks in d.items()}
        for x, y in (("en", "de"), ("en", "es"), ("de", "es")):
            inter, union = sum((bags[x] & bags[y]).values()), sum((bags[x] | bags[y]).values())
            jac.append(inter / union if union else 1.0)
    out["convergence"] = {"root_bag_jaccard_mean": sum(jac) / len(jac)}
    out["convergence"]["note"] = "Jaccard of concept roots between the en/de/es versions of each PUD sentence"
    print("convergence", json.dumps(out["convergence"]))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("translate")
    q.add_argument("--langs", nargs="+", default=["en", "de", "es"])
    q.add_argument("--sent", type=int, default=4)
    q.add_argument("--max-len", type=int, default=12)
    q.set_defaults(fn=cmd_translate)
    q = sub.add_parser("gate")
    q.add_argument("--langs", nargs="+", default=["en", "de", "es"])
    q.add_argument("--n", type=int, default=0)
    q.set_defaults(fn=cmd_gate)
    q = sub.add_parser("compare")
    q.add_argument("--out", type=Path)
    q.set_defaults(fn=cmd_compare)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
