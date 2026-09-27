"""Is the category layout the same across MODELS? (universal-grammar track)

Reads ug_categories.py summaries (one per model) and, for every language, correlates the
models' category-centroid distance matrices (RSA) and their shared-axis coordinates.
Different models have unrelated bases, so only basis-free quantities (distances, rank orders
of categories along each model's own principal axes) are compared.

  .venv/bin/python scripts/grammar/ug_categories_xmodel.py runs/grammar/*_categories_L*_summary.json \
      --out runs/grammar/xmodel_categories_summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import ug_lib as U
from ug_categories import dist_matrix, rsa


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summaries", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    runs = {}
    for f in a.summaries:
        r = json.loads(f.read_text())
        tag = r["model"].split("/")[-1] + ("-randinit" if r.get("random_init") else "") + f"@L{r['layer']}"
        runs[tag] = r
    res = {"runs": list(runs), "pairs": {}}
    for x, y in itertools.combinations(runs, 2):
        langs = [l for l in runs[x]["langs"] if l in runs[y]["langs"]]
        per = {}
        for l in langs:
            Cx = {c: np.array(v) for c, v in runs[x]["centroids"][l].items()}
            Cy = {c: np.array(v) for c, v in runs[y]["centroids"][l].items()}
            common = [c for c in runs[x]["centroid_cats"] if c in Cx and c in Cy]
            per[l] = rsa(dist_matrix(Cx, common), dist_matrix(Cy, common))
        ax = []   # for each of x's axes: best-matching axis of y (axes may permute across models)
        for j, axj in enumerate(runs[x]["axes"]):
            best = (0.0, -1)
            for k, axk in enumerate(runs[y]["axes"]):
                cx, cy = axj["coords"], axk["coords"]
                common = [c for c in cx if c in cy]
                rho = U.spearman(np.array([cx[c] for c in common]), np.array([cy[c] for c in common]))
                best = max(best, (abs(rho), k))   # axis sign is arbitrary when NOUN sits near 0
            ax.append(best[0])
        res["pairs"][f"{x} vs {y}"] = {"rsa_mean": float(np.mean(list(per.values()))), "rsa_per_lang": per,
                                       "axis_rank_agreement": ax}
        print(f"{x:28s} vs {y:28s} centroid RSA {np.mean(list(per.values())):.3f}  "
              f"axis agreement {' '.join(f'{v:+.2f}' for v in ax)}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
