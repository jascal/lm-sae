"""Driver for the Type–Link–Merge grammar prototype (dl/tlm/tlm.dl, the normative program).

Python only stages facts, runs Soufflé, and prints what the program derived:

  run   parse UD PUD sentences with the TLM rules; print the invariant checks, the per-language ORDER
        parameters, and the same parallel sentence as a bracketed Merge tree in several languages.
  gate  differential gate: the Datalog Link and Merge trees must equal the python trees the
        measurements used (ug_trees.function_head / _phrase_tree with relators = {case, mark}),
        word-for-word heads and leaf-to-leaf Merge distances, on every sentence.

The TYPE register (type_axis.facts) is generated from the measured category axes in
runs/grammar/*_categories_L*_summary.json: each category's coordinate on the two shared axes,
max-abs-normalised per model and averaged over the four models.

  .venv/bin/python scripts/grammar/tlm.py run --langs en de ja ar --sent 5
  .venv/bin/python scripts/grammar/tlm.py gate --langs en de es fr ru zh ja ko tr hi ar id fi
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ug_trees as T  # noqa: E402
from ug_extract import PUD, read_conllu  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DL = ROOT / "dl" / "tlm" / "tlm.dl"
RUNS = ROOT / "runs" / "grammar"
CAT_RUNS = ["qwen05_categories_L12", "llama1b_categories_L8", "qwen15_categories_L14", "gemma2b_categories_L13"]
RELATORS = {"case", "mark"}


def type_axes() -> dict[str, tuple[float, float]]:
    """Mean (axis1, axis2) coordinate per category across the four models (per-model max-abs normalised)."""
    acc = defaultdict(list)
    for name in CAT_RUNS:
        r = json.loads((RUNS / f"{name}_summary.json").read_text())
        ax = [r["axes"][0]["coords"], r["axes"][1]["coords"]]
        scale = [max(abs(v) for v in a.values()) for a in ax]
        for c in ax[0]:
            if c in ax[1]:
                acc[c].append((ax[0][c] / scale[0], ax[1][c] / scale[1]))
    return {c: tuple(np.mean(v, 0)) for c, v in acc.items()}


def stage(langs: list[str], n_sent: int, out: Path, test_only: bool) -> dict:
    """Write word/lang/type_axis facts; return {s: (lang, sentence index, sentence)}."""
    rows, lrows, index = [], [], {}
    s_id = 0
    for lang in langs:
        taken = 0
        for si, sent in enumerate(read_conllu(PUD / f"{lang}_pud.conllu")):
            if (test_only and si % 5 != 0) or (n_sent and taken >= n_sent):
                continue
            s_id += 1
            taken += 1
            index[s_id] = (lang, si, sent)
            lrows.append(f"{s_id}\t{lang}")
            for i, w in enumerate(sent["words"], 1):
                rows.append(f"{s_id}\t{i}\t{w['form']}\t{w['upos']}\t{w['head'] + 1}\t{w['deprel']}")
    (out / "word.facts").write_text("\n".join(rows) + "\n")
    (out / "lang.facts").write_text("\n".join(lrows) + "\n")
    (out / "type_axis.facts").write_text("".join(f"{c}\t{a:.4f}\t{b:.4f}\n" for c, (a, b) in type_axes().items()))
    return index


def souffle(out: Path) -> None:
    subprocess.run(["souffle", "-F", str(out), "-D", str(out), str(DL)], check=True)


def read(out: Path, rel: str) -> list[list[str]]:
    p = out / f"{rel}.csv"
    return [line.split("\t") for line in p.read_text().splitlines() if line] if p.exists() else []


def links(out: Path) -> dict[int, dict[int, tuple[int, str]]]:
    L = defaultdict(dict)
    for s, i, h, r in read(out, "link"):
        L[int(s)][int(i)] = (int(h), r)
    return L


def merge_graph(out: Path):
    E = defaultdict(list)
    for s, h, k, ch, ck in read(out, "merge_edge"):
        E[int(s)].append(((int(h), int(k)), (int(ch), int(ck))))
    return E


def bracket(node, kids, forms) -> str:
    """Render a Merge node as brackets; the driver only formats what Datalog derived."""
    h, k = node
    if k == 0:
        return forms[h - 1]
    a, b = kids[node]                       # (h, k-1) and maxproj of the k-th dependent
    left, right = (a, b) if min_word(a, kids) < min_word(b, kids) else (b, a)
    return f"[{bracket(left, kids, forms)} {bracket(right, kids, forms)}]"


def min_word(node, kids):
    h, k = node
    return h if k == 0 else min(min_word(c, kids) for c in kids[node])


def cmd_run(a) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        index = stage(a.langs, a.n, out, test_only=True)
        souffle(out)
        print(f"TLM parsed {len(index)} sentences ({', '.join(a.langs)})")
        print("\nINVARIANTS (each must be empty)")
        for rel in ("bad_head_count", "cycle", "bad_root_count", "relator_without_complement", "determiner_heads",
                    "uncovered"):
            n = len(read(out, rel))
            print(f"  {rel:28s} {n:5d}  {'ok' if n == 0 else 'VIOLATED'}")
        print("\nORDER parameter: share of links where the head comes first")
        stats = defaultdict(dict)
        for lang, rel, hf, n in read(out, "order_stat"):
            stats[lang][rel] = int(hf) / int(n)
        names = {"_fcomp": "relator>complement", "obj": "verb>object", "nsubj": "verb>subject", "amod": "noun>adjective",
                 "det": "noun>determiner"}
        print("  " + "lang".ljust(6) + "".join(v.ljust(20) for v in names.values()))
        for lang in a.langs:
            print("  " + lang.ljust(6) + "".join(f"{stats[lang].get(r, float('nan')):<20.2f}" for r in names))
        types = defaultdict(lambda: defaultdict(int))
        for s, i, ref, lex in read(out, "type_of"):
            types[index[int(s)][2]["words"][int(i) - 1]["upos"]][(ref, lex)] += 1
        print("\nTYPE features by category (from the measured axes)")
        for c in sorted(types, key=lambda c: -sum(types[c].values()))[:12]:
            (ref, lex), _ = max(types[c].items(), key=lambda kv: kv[1])
            print(f"  {c:6s} {ref:9s} {lex}")
        E, L = merge_graph(out), links(out)
        by_sent = defaultdict(dict)
        for s, (lang, si, sent) in index.items():
            by_sent[si][lang] = (s, sent)
        demo = [si for si, d in sorted(by_sent.items()) if len(d) == len(a.langs)
                and max(len(v[1]["words"]) for v in d.values()) <= a.max_len][: a.sent]
        for si in demo:
            print(f"\nPARALLEL SENTENCE #{si}")
            for lang in a.langs:
                s, sent = by_sent[si][lang]
                kids = defaultdict(list)
                for parent, child in E[s]:
                    kids[parent].append(child)
                forms = [w["form"] for w in sent["words"]]
                root = next(i for i, (h, _) in L[s].items() if h == 0)
                k = max([kk for (hh, kk) in kids if hh == root], default=0)
                print(f"  {lang}: {bracket((root, k), kids, forms)}")


def cmd_gate(a) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        index = stage(a.langs, a.n, out, test_only=False)
        souffle(out)
        L, E = links(out), merge_graph(out)
        head_bad = dist_bad = 0
        for s, (lang, si, sent) in index.items():
            heads = [w["head"] for w in sent["words"]]
            rels = [w["deprel"] for w in sent["words"]]
            n = len(heads)
            ref_heads, _ = T.function_head(heads, rels, RELATORS)
            dl_heads = [L[s][i + 1][0] - 1 for i in range(n)]
            if dl_heads != list(ref_heads):
                head_bad += 1
                if head_bad <= 3:
                    print(f"  head mismatch {lang}#{si}: dl={dl_heads} py={list(ref_heads)}")
            # leaf-to-leaf distances in the Datalog Merge graph vs the python phrase tree
            nodes = {}
            adj = defaultdict(list)
            for p, c in E[s]:
                for x in (p, c):
                    nodes.setdefault(x, len(nodes))
                adj[nodes[p]].append(nodes[c])
                adj[nodes[c]].append(nodes[p])
            for i in range(1, n + 1):
                nodes.setdefault((i, 0), len(nodes))
            D = np.zeros((n, n))
            for i in range(n):
                src = nodes[(i + 1, 0)]
                dist, frontier = {src: 0}, [src]
                while frontier:
                    nxt = []
                    for u in frontier:
                        for v in adj[u]:
                            if v not in dist:
                                dist[v] = dist[u] + 1
                                nxt.append(v)
                    frontier = nxt
                for j in range(n):
                    D[i, j] = dist.get(nodes[(j + 1, 0)], -1)
            ref = T._phrase_tree(heads, rels, bar_levels=False, func=RELATORS)
            if not np.array_equal(D, ref):
                dist_bad += 1
                if dist_bad <= 3:
                    print(f"  merge-distance mismatch {lang}#{si}")
        total = len(index)
        print(f"gate: {total} sentences · Link heads equal {total - head_bad}/{total} · "
              f"Merge distances equal {total - dist_bad}/{total}")
        sys.exit(1 if head_bad or dist_bad else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run")
    q.add_argument("--langs", nargs="+", default=["en", "de", "ja", "ar"])
    q.add_argument("--n", type=int, default=200, help="held-out sentences per language")
    q.add_argument("--sent", type=int, default=3, help="parallel sentences to print")
    q.add_argument("--max-len", type=int, default=12)
    q.set_defaults(fn=cmd_run)
    q = sub.add_parser("gate")
    q.add_argument("--langs", nargs="+", default=["en", "de", "es", "fr", "ru", "zh", "ja", "ko", "tr", "hi", "ar", "id", "fi"])
    q.add_argument("--n", type=int, default=0, help="sentences per language (0 = all 1000)")
    q.set_defaults(fn=cmd_gate)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
