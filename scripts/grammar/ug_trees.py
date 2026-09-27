"""Rival grammar theories as word-to-word distance matrices over the same UD sentence.

Every theory is derived deterministically from the gold UD tree, so they differ ONLY in
the structural claims each theory adds — which is what the probe comparison tests.

  linear  : word order only, d(i,j) = |i-j|                                   (null: no syntax)
  ud      : content-head dependency tree (Universal Dependencies)             (Tesnière / UD)
  fhead   : function-head dependency — adpositions, determiners, auxiliaries/copulas,
            subordinators head their content word; subject moves to the top of the
            verbal chain (SUD-style)                                           (functional-head DG)
  merge   : binary Merge over the function-head tree — a head combines with its
            complements (nearest first), then adjuncts, then the specifier; no unary nodes
                                                                               (bare phrase structure)
  xbar    : `merge` plus X-bar's bar levels: every head projects X0 → X' → XP even when
            nothing attaches (unary projections), complements inside X', adjuncts adjoin
            to X', specifier at XP                                             (X-bar theory)

Complexity order: linear < ud ≈ fhead < merge < xbar (each adds structure to the one before).
"""
from __future__ import annotations

import numpy as np

FUNC = {"case", "det", "aux", "cop", "mark"}                   # function words promoted to heads
SPEC = {"nsubj", "csubj", "expl"}                               # specifier of the clause
COMP = {"obj", "iobj", "ccomp", "xcomp", "_fcomp"}              # _fcomp: content complement of a promoted head
# verbal chain order, top first: mark (C) > aux/cop (T) > verb; nominal: case (P) > det (D) > noun
RANK = {"mark": 3, "aux": 2, "cop": 2, "case": 3, "det": 2}


def linear(heads, deprel, upos):
    n = len(heads)
    i = np.arange(n)
    return np.abs(i[:, None] - i[None, :]).astype(np.float32)


def _bfs_all(adj: list[list[int]]) -> np.ndarray:
    n = len(adj)
    D = np.zeros((n, n), dtype=np.float32)
    for s in range(n):
        dist = {s: 0}
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        for v, d in dist.items():
            D[s, v] = d
    return D


def _tree_dist(heads) -> np.ndarray:
    adj = [[] for _ in heads]
    for i, h in enumerate(heads):
        if h >= 0:
            adj[i].append(h)
            adj[h].append(i)
    return _bfs_all(adj)


def ud(heads, deprel, upos):
    return _tree_dist(heads)


def function_head(heads, deprel, func=FUNC):
    """Return (heads', rel') of the function-head (SUD-like) tree, promoting the relations in `func`."""
    n = len(heads)
    heads, rel = list(heads), list(deprel)
    for c in range(n):
        funcs = [f for f in range(n) if heads[f] == c and rel[f] in func]
        if not funcs:
            continue
        # top of chain first: higher RANK, then farther from the content word
        funcs.sort(key=lambda f: (-RANK[rel[f]], -abs(f - c)))
        top_parent, top_rel = heads[c], rel[c]
        chain = funcs + [c]
        for a, b in zip(chain, chain[1:]):
            heads[b] = a
            rel[b] = "_fcomp"
        heads[chain[0]], rel[chain[0]] = top_parent, top_rel
        # the subject is the specifier of the highest aux/cop (T), below any subordinator (C)
        t_heads = [f for f in funcs if deprel[f] in {"aux", "cop"}]
        if t_heads:
            for s in range(n):
                if heads[s] == c and rel[s] in SPEC:
                    heads[s] = t_heads[0]
    return heads, rel



def fhead(heads, deprel, upos):
    return _tree_dist(function_head(heads, deprel)[0])


def _phrase_tree(heads, deprel, bar_levels: bool) -> np.ndarray:
    """Leaf-to-leaf distances in a binary headed phrase-structure tree built from the function-head tree."""
    fh, rel = function_head(heads, deprel)
    n = len(fh)
    kids = [[] for _ in range(n)]
    root = None
    for i, h in enumerate(fh):
        if h >= 0:
            kids[h].append(i)
        else:
            root = i
    adj: list[list[int]] = [[] for _ in range(n)]      # nodes 0..n-1 are the word leaves

    def new_node() -> int:
        adj.append([])
        return len(adj) - 1

    def link(a: int, b: int) -> None:
        adj[a].append(b)
        adj[b].append(a)

    def project(h: int) -> int:
        """Build h's maximal projection; return its node id."""
        comps = sorted((k for k in kids[h] if rel[k] in COMP), key=lambda k: abs(k - h))
        specs = sorted((k for k in kids[h] if rel[k] in SPEC), key=lambda k: abs(k - h))
        adjs = sorted((k for k in kids[h] if rel[k] not in COMP and rel[k] not in SPEC), key=lambda k: abs(k - h))
        cur = h
        if bar_levels:                                      # X' always exists above X0
            bar = new_node()
            link(bar, cur)
            cur = bar
            for k in comps:                                 # complements are sisters inside X'
                link(cur, project(k))
        else:
            for k in comps:
                node = new_node()
                link(node, cur)
                link(node, project(k))
                cur = node
        for k in adjs:                                      # adjunction: a new segment per adjunct
            node = new_node()
            link(node, cur)
            link(node, project(k))
            cur = node
        if bar_levels:                                      # XP always exists; specifier is its daughter
            xp = new_node()
            link(xp, cur)
            for k in specs:
                link(xp, project(k))
            cur = xp
        else:
            for k in specs:
                node = new_node()
                link(node, cur)
                link(node, project(k))
                cur = node
        return cur

    import sys
    sys.setrecursionlimit(10000)
    project(root)
    # attach any disconnected leaves (malformed trees) to keep distances finite
    D = _bfs_all(adj)[:n, :n]
    return D


def merge(heads, deprel, upos):
    return _phrase_tree(heads, deprel, bar_levels=False)


def xbar(heads, deprel, upos):
    return _phrase_tree(heads, deprel, bar_levels=True)


THEORIES = {"linear": linear, "ud": ud, "fhead": fhead, "merge": merge, "xbar": xbar}


def _promote(func):
    return lambda heads, deprel, upos: _tree_dist(function_head(heads, deprel, set(func))[0])


# which function words behave as heads? promote one class at a time over the UD baseline
HEADEDNESS = {"linear": linear, "ud": ud, "P:case": _promote({"case"}), "D:det": _promote({"det"}),
              "T:aux+cop": _promote({"aux", "cop"}), "C:mark": _promote({"mark"}),
              "PCT:no-det": _promote({"case", "mark", "aux", "cop"}), "fhead": fhead}
SETS = {"theories": THEORIES, "headedness": HEADEDNESS}


if __name__ == "__main__":
    # "the cat that I saw will sit on the mat"
    words = ["the", "cat", "that", "I", "saw", "will", "sit", "on", "the", "mat"]
    heads = [1, 6, 4, 4, 1, 6, -1, 9, 9, 6]
    rels = ["det", "nsubj", "obj", "nsubj", "acl", "aux", "root", "case", "det", "obl"]
    for name, f in THEORIES.items():
        D = f(heads, rels, None)
        print(f"{name:6s} cat–sit={D[1, 6]:.0f} will–sit={D[5, 6]:.0f} on–mat={D[7, 9]:.0f} "
              f"the–cat={D[0, 1]:.0f} cat–mat={D[1, 9]:.0f}")
