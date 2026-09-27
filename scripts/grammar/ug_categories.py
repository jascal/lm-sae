"""How are universal word categories laid out? (universal-grammar track)

  bottleneck : held-out-language UPOS accuracy of a rank-k probe (d→k→17), trained on the other
               languages pooled — how many dimensions does a SHARED category code need?
  centroids  : per-language class centroids at the layer (per-language z-scored); RSA (Spearman of
               centroid distance matrices) between every language pair — is the category layout
               the same everywhere? Centroids are saved so ug_categories_xmodel can compare models.
  features   : Chomsky's (1970) category features N=[+N,-V] V=[-N,+V] A=[+N,+V] P=[-N,-V] predict a
               parallelogram: N−A ∥ P−V (the ±V axis) and N−P ∥ A−V (the ±N axis). Cosines are compared
               against every other pairing of the same four centroids.

  .venv/bin/python scripts/grammar/ug_categories.py --model-dir data/ug/Qwen2.5-0.5B --layer 12 \
      --out runs/grammar/qwen05_categories_L12_summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import ug_lib as U

CATS = ["NOUN", "PROPN", "PRON", "VERB", "AUX", "ADJ", "ADV", "ADP", "DET", "NUM", "CCONJ", "SCONJ", "PART"]


def fit_bottleneck(X: torch.Tensor, y: np.ndarray, k: int, steps: int = 400, batch: int = 8192):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], k, bias=False), torch.nn.Linear(k, len(U.UPOS))).to(U.DEV)
    yt = torch.from_numpy(y).to(U.DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        b = torch.randint(0, X.shape[0], (min(batch, X.shape[0]),), device=U.DEV)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(X[b].float()), yt[b]).backward()
        opt.step()
    return net


def centroids(L: U.Lang) -> dict[str, np.ndarray]:
    out = {}
    keep = ~L.mwt
    for c in CATS:
        m = keep & (L.upos == U.UPOS.index(c))
        if m.sum() >= 10:
            out[c] = L.X[torch.from_numpy(m).to(U.DEV)].float().mean(0).cpu().numpy()
    return out


def dist_matrix(C: dict[str, np.ndarray], cats: list[str]) -> np.ndarray:
    V = np.stack([C[c] for c in cats])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return 1 - V @ V.T


def rsa(A: np.ndarray, B: np.ndarray) -> float:
    iu = np.triu_indices(len(A), 1)
    return U.spearman(A[iu], B[iu])


def features(C: dict[str, np.ndarray]) -> dict:
    """Parallelogram test on N, V, A, P centroids (P = ADP)."""
    if not all(c in C for c in ("NOUN", "VERB", "ADJ", "ADP")):
        return {}
    four = {"N": C["NOUN"], "V": C["VERB"], "A": C["ADJ"], "P": C["ADP"]}
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))  # noqa: E731
    # the three ways to split {N,V,A,P} into two pairs; each gives two parallelogram predictions
    pairings = {}
    for (w, x), (y, z) in [(("N", "A"), ("P", "V")), (("N", "P"), ("A", "V")), (("N", "V"), ("A", "P"))]:
        pairings[f"{w}-{x}||{y}-{z}"] = cos(four[w] - four[x], four[y] - four[z])
        pairings[f"{w}-{y}||{x}-{z}"] = cos(four[w] - four[y], four[x] - four[z])
    return pairings


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--langs", nargs="+")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 32, 64])
    p.add_argument("--axes-k", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    inf = U.info(a.model_dir)
    langs = a.langs or list(inf["langs"])
    data = {l: U.load(a.model_dir, l, a.layer) for l in langs}
    res = {"model": inf["model"], "random_init": inf.get("random_init", False), "layer": a.layer, "langs": langs}

    # 1. bottleneck curve (leave-one-language-out)
    curve = {}
    for k in a.ks:
        accs = {}
        for t in langs:
            P = U.pool([data[l] for l in langs if l != t])
            keep = ~P.test & ~P.mwt
            net = fit_bottleneck(P.X[torch.from_numpy(keep).to(U.DEV)], P.upos[keep], k)
            accs[t] = U.category_eval(net, data[t], "upos", U.UPOS)["acc_nontrivial"]
        curve[k] = {"mean": float(np.mean(list(accs.values()))), "per_lang": accs}
        print(f"k={k} held-out UPOS (non-trivial) {curve[k]['mean']:.3f}", flush=True)
    res["bottleneck"] = curve

    # 2. centroid layout + cross-language RSA
    C = {l: centroids(data[l]) for l in langs}
    # categories attested (>=10 tokens) in at least 3/4 of the languages; each pair compares its shared ones
    cats = [c for c in CATS if sum(c in C[l] for l in langs) >= 0.75 * len(langs)]
    pairs = {}
    for x, y in itertools.combinations(langs, 2):
        common = [c for c in cats if c in C[x] and c in C[y]]
        pairs[f"{x}-{y}"] = rsa(dist_matrix(C[x], common), dist_matrix(C[y], common))
    res["centroid_cats"] = cats
    res["centroids"] = {l: {c: v.tolist() for c, v in C[l].items()} for l in langs}
    res["rsa_langs"] = {"mean": float(np.mean(list(pairs.values()))), "min": float(min(pairs.values())),
                        "pairs": pairs}
    print(f"cross-language centroid RSA mean {res['rsa_langs']['mean']:.3f} min {res['rsa_langs']['min']:.3f}")

    # 3. ±N/±V features
    res["features"] = {l: features(C[l]) for l in langs}
    keys = list(next(iter(res["features"].values())))
    res["features_mean"] = {k: float(np.mean([res["features"][l][k] for l in langs if res["features"][l]]))
                            for k in keys}
    print("parallelogram cosines (mean over langs):",
          json.dumps({k: round(v, 3) for k, v in res["features_mean"].items()}))
    # 4. what the shared axes separate: bottleneck on ALL languages, PCA of class centroids inside it
    P = U.pool(list(data.values()))
    keep = ~P.test & ~P.mwt
    net = fit_bottleneck(P.X[torch.from_numpy(keep).to(U.DEV)], P.upos[keep], a.axes_k)
    W = net[0].weight.detach()
    proj = {l: {c: (W @ torch.from_numpy(v).to(U.DEV)).cpu().numpy() for c, v in C[l].items()} for l in langs}
    M = np.stack([np.mean([proj[l][c] for l in langs if c in proj[l]], 0) for c in cats])
    mu = M.mean(0)
    _, sv, vt = np.linalg.svd(M - mu, full_matrices=False)
    axes = []
    for j in range(min(a.axes_k, len(sv))):
        v = vt[j] * (np.sign((M[cats.index("NOUN")] - mu) @ vt[j]) or 1)   # orient: NOUN positive
        coord = (M - mu) @ v
        agree = []
        for l in langs:
            have = [i for i, c in enumerate(cats) if c in proj[l]]
            agree.append(U.spearman(np.stack([proj[l][cats[i]] for i in have]) @ v, coord[have]))
        order = sorted(zip(cats, coord.tolist()), key=lambda kv: -kv[1])
        axes.append({"var_frac": float(sv[j] ** 2 / (sv ** 2).sum()), "coords": dict(order),
                     "per_lang_rank_agreement": float(np.mean(agree))})
        print(f"axis {j} ({axes[-1]['var_frac']:.0%}, lang agreement {axes[-1]['per_lang_rank_agreement']:.2f}): "
              + " ".join(f"{c}{v:+.1f}" for c, v in order))
    res["axes"] = axes
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
