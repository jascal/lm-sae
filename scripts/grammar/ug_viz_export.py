"""Export coordinates for the universal-grammar explorer page (docs/UNIVERSAL_GRAMMAR.md, views 1-4).

  categories : a rank-4 category bottleneck per model (trained on the train split of all languages),
               its space rotated to the principal axes of the class centroids, then every model
               Procrustes-rotated onto the first model's frame (rotation only — agreement after a
               rigid rotation is the cross-model claim). Points are HELD-OUT sentences' words.
  trees      : parallel test sentences; each word's function-head structural-probe vector (probe
               trained on all languages' train split), jointly PCA'd to 3D per sentence across
               languages, with the UD and function-head edges.
  scoreboard : copied from runs/grammar/universal_grammar_summary.json.

  .venv/bin/python scripts/grammar/ug_viz_export.py --out runs/grammar/viz/ug_viz.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import ug_lib as U
import ug_trees as T
from ug_categories import CATS, fit_bottleneck
from ug_theory import scaled, theory_scale

ROOT = Path(__file__).resolve().parents[2]
MODELS = [("Qwen2.5-0.5B", "Qwen2.5-0.5B", 12), ("Llama-3.2-1B", "Llama-3.2-1B", 8),
          ("Qwen2.5-1.5B", "Qwen2.5-1.5B", 14), ("Gemma-2-2B", "gemma-2-2b", 13),
          ("Qwen2.5-0.5B untrained", "Qwen2.5-0.5B-randinit", 12)]
SHOW = [c for c in CATS]


def forms(lang: str) -> np.ndarray:
    return np.load(ROOT / "data" / "ug" / "Qwen2.5-0.5B" / f"{lang}.meta.npz")["form"]


def category_space(md: Path, layer: int, langs: list[str], per_lang: int, rng) -> dict:
    data = {l: U.load(md, l, layer) for l in langs}
    P = U.pool(list(data.values()))
    keep = ~P.test & ~P.mwt
    net = fit_bottleneck(P.X[torch.from_numpy(keep).to(U.DEV)], P.upos[keep], 4)
    W = net[0].weight.detach()
    pts, cents = {}, {}
    allc = []
    for l, L in data.items():
        sel = L.test & ~L.mwt & np.isin(L.upos, [U.UPOS.index(c) for c in SHOW])
        idx = np.flatnonzero(sel)
        Z = (W @ L.X[torch.from_numpy(idx).to(U.DEV)].float().T).T.cpu().numpy()
        cents[l] = {c: Z[L.upos[idx] == U.UPOS.index(c)].mean(0) for c in SHOW if (L.upos[idx] == U.UPOS.index(c)).sum() >= 5}
        allc += list(cents[l].values())
        pick = rng.choice(len(idx), min(per_lang, len(idx)), replace=False)
        pts[l] = (Z[pick], L.upos[idx][pick], idx[pick])
    # rotate to principal axes of the pooled class centroids
    M = np.stack([np.mean([cents[l][c] for l in langs if c in cents[l]], 0) for c in SHOW])
    mu = M.mean(0)
    _, _, vt = np.linalg.svd(M - mu, full_matrices=False)
    R = vt.T
    return {"pts": {l: ((z - mu) @ R, u, i) for l, (z, u, i) in pts.items()},
            "cents": {l: {c: (v - mu) @ R for c, v in cl.items()} for l, cl in cents.items()},
            "M": (M - mu) @ R}


def procrustes(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation Q (and uniform scale s) minimising |s·A Q − B|."""
    U_, S, Vt = np.linalg.svd(A.T @ B)
    Q = U_ @ Vt
    s = S.sum() / (A ** 2).sum()
    return Q, s


def trees(md: Path, layer: int, langs: list[str], n_sent: int) -> list[dict]:
    data = {l: U.load(md, l, layer) for l in langs}
    src = U.pool(list(data.values()), 150)
    fn = T.THEORIES["fhead"]
    B = U.fit_structural(src, target=scaled(fn, theory_scale(src, fn)))
    # parallel test sentences short enough to read, present in every language
    common = set.intersection(*[set(np.unique(data[l].sent[data[l].test])) for l in langs])
    lens = {s: max((data[l].sent == s).sum() for l in langs) for s in common}
    chosen = sorted([s for s in common if 8 <= lens[s] <= 14])[:n_sent]
    out = []
    for s in chosen:
        per = {}
        vecs = []
        for l in langs:
            L = data[l]
            idx = np.flatnonzero(L.sent == s)
            Z = (L.X[torch.from_numpy(idx).to(U.DEV)].float() @ B).cpu().numpy()
            rels = [U.DEPREL[r] for r in L.deprel[idx]]
            fh = T.function_head(L.head[idx], rels)[0]
            per[l] = {"words": [str(f) for f in forms(l)[idx]], "upos": [U.UPOS[u] for u in L.upos[idx]],
                      "ud": [int(h) for h in L.head[idx]], "fhead": [int(h) for h in fh], "Z": Z}
            vecs.append(Z)
        allv = np.concatenate(vecs)
        mu = allv.mean(0)
        _, _, vt = np.linalg.svd(allv - mu, full_matrices=False)
        for l in langs:
            per[l]["xyz"] = np.round((per[l].pop("Z") - mu) @ vt[:3].T, 3).tolist()
        out.append({"sent": int(s), "langs": per})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-lang", type=int, default=250)
    p.add_argument("--n-sent", type=int, default=6)
    p.add_argument("--tree-langs", nargs="+", default=["en", "de", "ja", "zh", "ar", "tr"])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    langs = U.info(ROOT / "data" / "ug" / "Qwen2.5-0.5B")["langs"]
    langs = list(langs)
    form = {l: forms(l) for l in langs}
    rng = np.random.default_rng(0)
    spaces = {}
    for name, d, layer in MODELS:
        spaces[name] = category_space(ROOT / "data" / "ug" / d, layer, langs, a.per_lang, rng)
        torch.cuda.empty_cache()
    ref = spaces[MODELS[0][0]]["M"]
    cat = {"cats": SHOW, "langs": langs, "models": {}}
    for name, sp in spaces.items():
        Q, s = procrustes(sp["M"], ref)
        fit = float(1 - ((s * sp["M"] @ Q - ref) ** 2).sum() / (ref ** 2).sum())
        m = {"procrustes_r2": fit, "points": {}, "centroids": {}}
        for l, (z, u, i) in sp["pts"].items():
            m["points"][l] = {"xyzw": np.round(s * z @ Q, 3).tolist(), "upos": [U.UPOS[x] for x in u],
                              "form": [str(form[l][j]) for j in i]}
            m["centroids"][l] = {c: np.round(s * v @ Q, 3).tolist() for c, v in sp["cents"][l].items()}
        cat["models"][name] = m
        print(f"{name}: Procrustes R² to {MODELS[0][0]} = {fit:.3f}", flush=True)
    out = {"categories": cat, "trees": trees(ROOT / "data" / "ug" / "Qwen2.5-0.5B", 12, a.tree_langs, a.n_sent)}
    summ = ROOT / "runs" / "grammar" / "universal_grammar_summary.json"
    if summ.exists():
        out["scoreboard"] = json.loads(summ.read_text())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
