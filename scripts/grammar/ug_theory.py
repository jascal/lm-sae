"""Which grammar theory's tree does the model's geometry encode? (universal-grammar track)

For each theory T in ug_trees.THEORIES a structural probe is trained (on --src, train split)
to make squared probe distance match T's word-to-word distances. Every probe is then scored
on the SAME held-out sentences (test split, all languages) against every theory:

  beyond_linear[P][T]  partial Spearman(probe_P, T | word order)
  grouping[P][T]       partial Spearman(probe_P, T | word order + every theory's per-word
                       centrality). Centrality (row-mean distance: how high a word sits) is
                       predictable from the word type alone; what is left is WHICH words group
                       together. Validated on the embedding layer (token identity only), where
                       it is ~0 — the lexical-artifact control.
  encompass[P][A|B]    grouping partial of probe_P with A, additionally controlling B.

The headline statistic is own-grouping, grouping[T][T]: how well the geometry can be made to
realise T's constituency. Paired sentence-bootstrap CIs are given for every pairwise
difference (all probes are scored on identical sentences).

  .venv/bin/python scripts/grammar/ug_theory.py --model-dir data/ug/Qwen2.5-0.5B --layer 12 \
      --out runs/grammar/qwen05_theory_L12_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import ug_lib as U
import ug_trees as T


def scaled(fn, scale):
    return lambda heads, deprel, upos: fn(heads, deprel, upos) / scale


def theory_scale(L: U.Lang, fn) -> float:
    vals = []
    for idx, heads, _ in U.sentences(L, "train"):
        D = fn(heads, [U.DEPREL[r] for r in L.deprel[idx]], None)
        vals.append(D[np.triu_indices(len(idx), 1)])
    return float(np.concatenate(vals).mean())


def ranks(x: np.ndarray) -> np.ndarray:
    r = x.argsort().argsort().astype(np.float64)
    return (r - r.mean()) / (r.std() + 1e-12)


def partial(p, a, controls: list[np.ndarray]) -> float:
    """Correlation of p and a after regressing both on the control columns."""
    C = np.stack(controls, 1)
    beta_p, *_ = np.linalg.lstsq(C, p, rcond=None)
    beta_a, *_ = np.linalg.lstsq(C, a, rcond=None)
    rp, ra = p - C @ beta_p, a - C @ beta_a
    return float((rp * ra).sum() / np.sqrt((rp ** 2).sum() * (ra ** 2).sum() + 1e-12))


def theory_columns(L: U.Lang):
    """Per held-out sentence (n>=5): rank columns of every theory's distances and centralities."""
    X, _, _, sents = U.batch_sents(L, "test")
    keep, cols = [], []
    for k, (idx, heads, _) in enumerate(sents):
        n = len(idx)
        if n < 5:
            continue
        iu = np.triu_indices(n, 1)
        rels = [U.DEPREL[r] for r in L.deprel[idx]]
        c = {"n": n}
        for t, fn in T.THEORIES.items():
            D = fn(heads, rels, None)
            c[t] = ranks(D[iu])
            if t != "linear":
                m = D.mean(1)
                c[f"c_{t}"] = ranks((m[:, None] + m[None, :])[iu])
        keep.append(k)
        cols.append(c)
    return X, keep, cols


def probe_column(B, X, keep, cols):
    with torch.no_grad():
        P = U.pred_dist(B, X).cpu().numpy()
    return [ranks(P[k][: c["n"], : c["n"]][np.triu_indices(c["n"], 1)]) for k, c in zip(keep, cols)]


def stats(probe: list[np.ndarray], cols: list[dict]) -> dict:
    C = {k: np.concatenate([c[k] for c in cols]) for k in cols[0] if k != "n"}
    p = np.concatenate(probe)
    th = [t for t in T.THEORIES if t != "linear"]
    cen = [C[f"c_{t}"] for t in th]
    return {"spearman": {t: float((p * C[t]).mean()) for t in T.THEORIES},
            "beyond_linear": {t: partial(p, C[t], [C["linear"]]) for t in th},
            "grouping": {t: partial(p, C[t], [C["linear"]] + cen) for t in th},
            "encompass": {f"{a}|{b}": partial(p, C[a], [C[b], C["linear"]] + cen)
                          for a in th for b in th if a != b}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--src", default="en", help="training language, or 'all' (150 train sentences per language)")
    p.add_argument("--langs", nargs="+")
    p.add_argument("--set", choices=list(T.SETS), default="theories", help="theories | headedness variants")
    p.add_argument("--seed", type=int, default=0, help="probe init seed (for seed-variance checks)")
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    T.THEORIES = T.SETS[a.set]
    inf = U.info(a.model_dir)
    langs = a.langs or list(inf["langs"])
    data = {l: U.load(a.model_dir, l, a.layer) for l in langs}
    src = U.pool(list(data.values()), 150) if a.src == "all" else data[a.src]
    probes = {t: U.fit_structural(src, rank=a.rank, target=scaled(fn, theory_scale(src, fn)), seed=a.seed)
              for t, fn in T.THEORIES.items()}
    th = [t for t in T.THEORIES if t != "linear"]
    res = {"model": inf["model"], "random_init": inf.get("random_init", False), "layer": a.layer, "set": a.set,
           "seed": a.seed,
           "src": a.src, "langs": langs, "per_lang": {}}
    cols_all, probe_all, owner = [], {t: [] for t in T.THEORIES}, []
    for l in langs:
        X, keep, cols = theory_columns(data[l])
        pc = {t: probe_column(probes[t], X, keep, cols) for t in T.THEORIES}
        res["per_lang"][l] = {t: stats(pc[t], cols) for t in th}
        cols_all += cols
        owner += [l] * len(cols)
        for t in T.THEORIES:
            probe_all[t] += pc[t]
    res["pooled"] = {t: stats(probe_all[t], cols_all) for t in th}
    # paired bootstrap over sentences (resampled within language); all probes see the same resample
    rng = np.random.default_rng(0)
    owner = np.array(owner)
    by_lang = [np.flatnonzero(owner == l) for l in langs]
    boots = {t: [] for t in th}
    for _ in range(a.boot):
        sel = np.concatenate([rng.choice(ix, len(ix)) for ix in by_lang])
        cen = [np.concatenate([cols_all[i][f"c_{u}"] for i in sel]) for u in th]
        lin = np.concatenate([cols_all[i]["linear"] for i in sel])
        for t in th:
            boots[t].append(partial(np.concatenate([probe_all[t][i] for i in sel]),
                                    np.concatenate([cols_all[i][t] for i in sel]), [lin] + cen))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    res["own_grouping"] = {t: {"value": res["pooled"][t]["grouping"][t], "ci95": ci(boots[t])} for t in th}
    res["own_grouping_diff"] = {
        f"{x}-{y}": {"value": res["own_grouping"][x]["value"] - res["own_grouping"][y]["value"],
                     "ci95": ci(np.array(boots[x]) - np.array(boots[y]))}
        for i, x in enumerate(th) for y in th[i + 1:]}
    print(json.dumps({t: round(v["value"], 3) for t, v in res["own_grouping"].items()}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
