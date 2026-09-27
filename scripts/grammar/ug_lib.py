"""Shared loaders + probes for the universal-grammar track (see docs/UNIVERSAL_GRAMMAR.md).

Split: PUD sentences are parallel across languages, so a sentence index is the
same content in every language. Test = index % 5 == 0 (200 sentences); a probe
trained on any language never sees the test content in any language.

Normalisation: each language is z-scored with its OWN train-split mean/std
(unlabelled statistics only — no target labels are used). This removes the
language-identity offset, which is the standard condition for asking whether
the *rest* of the geometry is shared. `norm="source"` instead applies the
source language's statistics to every target (reported as a stricter variant).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
        "PUNCT", "SCONJ", "SYM", "VERB", "X"]
TRIVIAL = {"PUNCT", "NUM", "SYM", "X"}  # categories shared tokens can carry across languages for free
DEPREL = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj",
          "cop", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj",
          "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum",
          "root", "vocative", "xcomp"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Lang:
    lang: str
    X: torch.Tensor          # [n_words, d] normalised, on DEV
    sent: np.ndarray
    widx: np.ndarray
    upos: np.ndarray         # int index into UPOS
    deprel: np.ndarray       # int index into DEPREL
    head: np.ndarray
    mwt: np.ndarray
    test: np.ndarray         # bool per word
    stats: tuple[torch.Tensor, torch.Tensor]


def info(model_dir: Path) -> dict:
    return json.loads((model_dir / "info.json").read_text())


def load(model_dir: Path, lang: str, layer: int, stats=None) -> Lang:
    m = np.load(model_dir / f"{lang}.meta.npz")
    X = torch.from_numpy(np.load(model_dir / f"{lang}.L{layer}.npy").astype(np.float32)).to(DEV)
    test = (m["sent"] % 5) == 0
    if stats is None:
        tr = X[torch.from_numpy(~test).to(DEV)]
        stats = (tr.mean(0), tr.std(0) + 1e-4)
    X = ((X - stats[0]) / stats[1]).half()   # fp16 on device; probes cast minibatches to fp32
    return Lang(lang, X, m["sent"], m["widx"], np.array([UPOS.index(u) for u in m["upos"]]),
                np.array([DEPREL.index(d) if d in DEPREL else DEPREL.index("dep") for d in m["deprel"]]),
                m["head"], m["mwt"], test, stats)


# ── category probes ──────────────────────────────────────────────────────────────────────────────────────

def fit_linear(X: torch.Tensor, y: np.ndarray, n_cls: int, wd: float = 1e-3, steps: int = 300,
               batch: int = 8192) -> torch.nn.Linear:
    torch.manual_seed(0)
    lin = torch.nn.Linear(X.shape[1], n_cls).to(DEV)
    yt = torch.from_numpy(y).to(DEV)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        b = torch.randint(0, X.shape[0], (min(batch, X.shape[0]),), device=DEV)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(X[b].float()), yt[b]).backward()
        opt.step()
    return lin


def category_eval(lin, L: Lang, attr: str, names: list[str]) -> dict:
    keep = L.test & ~L.mwt
    with torch.no_grad():
        pred = lin(L.X[torch.from_numpy(keep).to(DEV)].float()).argmax(1).cpu().numpy()
    gold = getattr(L, attr)[keep]
    nontriv = np.array([names[g] not in TRIVIAL for g in gold]) if attr == "upos" else \
        np.array([names[g] != "punct" for g in gold])
    return {"acc": float((pred == gold).mean()), "acc_nontrivial": float((pred == gold)[nontriv].mean()),
            "majority": float(np.bincount(gold).max() / len(gold)), "n": int(keep.sum())}


def train_category(L: Lang, attr: str, n_cls: int):
    keep = ~L.test & ~L.mwt
    return fit_linear(L.X[torch.from_numpy(keep).to(DEV)], getattr(L, attr)[keep], n_cls)


# ── structural (distance) probe, Hewitt & Manning 2019 ───────────────────────────────────────────────────

def tree_distances(heads: np.ndarray, deprel=None, upos=None) -> np.ndarray:
    n = len(heads)
    adj = [[] for _ in range(n)]
    for i, h in enumerate(heads):
        if h >= 0:
            adj[i].append(h)
            adj[h].append(i)
    D = np.zeros((n, n), dtype=np.float32)
    for s in range(n):
        seen, frontier, d = {s}, [s], 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        D[s, v] = d
                        nxt.append(v)
            frontier = nxt
    return D


def sentences(L: Lang, split: str, max_len: int = 60):
    """Yield (word index array, heads, upos) per sentence in the split."""
    want = L.test if split == "test" else ~L.test
    order = np.flatnonzero(want)
    bounds = np.flatnonzero(np.diff(L.sent[order])) + 1
    for idx in np.split(order, bounds):
        if 2 <= len(idx) <= max_len:
            yield idx, L.head[idx], L.upos[idx]


def batch_sents(L: Lang, split: str, target=tree_distances):
    """Padded tensors: X [S, N, d], D [S, N, N], mask [S, N], plus raw per-sentence info."""
    sents = list(sentences(L, split))
    N = max(len(s[0]) for s in sents)
    S, d = len(sents), L.X.shape[1]
    X = torch.zeros(S, N, d, device=DEV, dtype=torch.half)
    D = torch.zeros(S, N, N, device=DEV)
    M = torch.zeros(S, N, device=DEV)
    for k, (idx, heads, _) in enumerate(sents):
        n = len(idx)
        X[k, :n] = L.X[torch.from_numpy(idx).to(DEV)]
        D[k, :n, :n] = torch.from_numpy(target(heads, [DEPREL[r] for r in L.deprel[idx]], None))
        M[k, :n] = 1
    return X, D, M, sents


def pred_dist(B: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    Z = X.float() @ B
    return ((Z[:, :, None, :] - Z[:, None, :, :]) ** 2).sum(-1)


def fit_structural(L: Lang, rank: int = 128, epochs: int = 40, target=tree_distances, batch: int = 32,
                   seed: int = 0) -> torch.Tensor:
    X, D, M, _ = batch_sents(L, "train", target)
    torch.manual_seed(seed)
    B = (torch.randn(X.shape[2], rank, device=DEV) * 0.01).requires_grad_()
    opt = torch.optim.Adam([B], lr=1e-3)
    steps = epochs * ((X.shape[0] + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)   # + clipping: the L1-on-squared-
    for _ in range(epochs):                                           # distance loss spikes without them
        perm = torch.randperm(X.shape[0], device=DEV)
        for s in range(0, X.shape[0], batch):
            b = perm[s:s + batch]
            m2 = M[b][:, :, None] * M[b][:, None, :]
            n2 = m2.sum((1, 2)).clamp(min=1)
            loss = (((pred_dist(B, X[b]) - D[b]).abs() * m2).sum((1, 2)) / n2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([B], 1.0)
            opt.step()
            sched.step()
    return B.detach()


def mst_edges(P: np.ndarray) -> set[tuple[int, int]]:
    n = len(P)
    inT, best, parent = [False] * n, np.full(n, np.inf), [-1] * n
    best[0] = 0
    edges = set()
    for _ in range(n):
        u = min((i for i in range(n) if not inT[i]), key=lambda i: best[i])
        inT[u] = True
        if parent[u] >= 0:
            edges.add((min(u, parent[u]), max(u, parent[u])))
        for v in range(n):
            if not inT[v] and P[u, v] < best[v]:
                best[v], parent[v] = P[u, v], u
    return edges


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def structural_eval(B: torch.Tensor, L: Lang) -> dict:
    """UUAS (punctuation excluded, as H&M) + mean Spearman of predicted vs tree distance; linear-chain baseline."""
    X, D, M, sents = batch_sents(L, "test")
    with torch.no_grad():
        P = pred_dist(B, X).cpu().numpy()
    D = D.cpu().numpy()
    punct = UPOS.index("PUNCT")
    hit = tot = hit_chain = 0
    rhos = []
    for k, (idx, heads, upos) in enumerate(sents):
        keep = np.flatnonzero(upos != punct)
        if len(keep) < 2:
            continue
        gold = set()
        for i, h in enumerate(heads):
            if h >= 0 and upos[i] != punct and upos[h] != punct:
                gold.add((min(i, h), max(i, h)))
        pk = P[k][np.ix_(keep, keep)]
        pred = {(keep[a], keep[b]) for a, b in mst_edges(pk)}
        chain = {(keep[j], keep[j + 1]) for j in range(len(keep) - 1)}
        hit += len(pred & gold)
        hit_chain += len(chain & gold)
        tot += len(gold)
        n = len(idx)
        iu = np.triu_indices(n, 1)
        if n >= 5:
            rhos.append(spearman(P[k][:n, :n][iu], D[k][:n, :n][iu]))
    return {"uuas": hit / max(tot, 1), "uuas_linear_chain": hit_chain / max(tot, 1),
            "spearman": float(np.nanmean(rhos)), "n_edges": tot}


def pool(langs: list[Lang], max_train_sents: int = 0) -> Lang:
    """Concatenate languages into one training Lang (sentence ids offset per language).

    `max_train_sents` keeps only the first N train sentences of each language (tree probes pad to
    [S, N, d], so a 12-language pool is capped to fit the GPU)."""
    parts = []
    for k, L in enumerate(langs):
        keep = np.ones(len(L.sent), dtype=bool)
        if max_train_sents:
            train_ids = np.unique(L.sent[~L.test])[:max_train_sents]
            keep = L.test | np.isin(L.sent, train_ids)
        parts.append((k, L, keep))
    cat = lambda f: np.concatenate([getattr(L, f)[keep] for _, L, keep in parts])  # noqa: E731
    X = torch.cat([L.X[torch.from_numpy(keep).to(DEV)] for _, L, keep in parts])
    sent = np.concatenate([L.sent[keep] + 100_000 * k for k, L, keep in parts])
    return Lang("+".join(L.lang for L in langs), X, sent, cat("widx"), cat("upos"), cat("deprel"),
                cat("head"), cat("mwt"), cat("test"), langs[0].stats)
