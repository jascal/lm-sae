"""Cross-lingual transfer of category / relation / tree probes (universal-grammar track).

  sweep : train on --src at every extracted layer, test on every language → pick layers.
  matrix: at one --layer, train on EVERY language and test on every language (src×tgt matrix).
  loo   : at one --layer, train on all languages but one (pooled) and test on the held-out one —
          the multi-source test of a SHARED code (an English-only probe also learns English specifics).

A probe trained on language A and tested zero-shot on language B can only
succeed if B's words sit in the same place in the shared geometry — the operational
meaning of "universal" here. Baselines: majority class, the embedding layer (0),
the linear-chain tree (UUAS), and the same probe on a random-init model if extracted.

  .venv/bin/python scripts/grammar/ug_probe.py sweep --model-dir data/ug/Qwen2.5-0.5B --src en \
      --out runs/grammar/qwen05_sweep_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ug_lib as U


def probe_all(md: Path, layer: int, srcs: list[str], tgts: list[str], tasks: list[str], rank: int) -> dict:
    data = {l: U.load(md, l, layer) for l in sorted(set(srcs) | set(tgts))}
    out = {}
    for s in srcs:
        S = data[s]
        row = {}
        if "upos" in tasks:
            lin = U.train_category(S, "upos", len(U.UPOS))
            row["upos"] = {t: U.category_eval(lin, data[t], "upos", U.UPOS) for t in tgts}
        if "deprel" in tasks:
            lin = U.train_category(S, "deprel", len(U.DEPREL))
            row["deprel"] = {t: U.category_eval(lin, data[t], "deprel", U.DEPREL) for t in tgts}
        if "struct" in tasks:
            B = U.fit_structural(S, rank=rank)
            row["struct"] = {t: U.structural_eval(B, data[t]) for t in tgts}
        out[s] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["sweep", "matrix", "loo"])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--src", default="en")
    p.add_argument("--layer", type=int)
    p.add_argument("--langs", nargs="+")
    p.add_argument("--tasks", nargs="+", default=["upos", "deprel", "struct"])
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    inf = U.info(a.model_dir)
    langs = a.langs or list(inf["langs"])
    res = {"model": inf["model"], "random_init": inf.get("random_init", False), "mode": a.mode,
           "langs": langs, "rank": a.rank}
    if a.mode == "sweep":
        res["src"] = a.src
        res["layers"] = {}
        for layer in inf["layers"]:
            r = probe_all(a.model_dir, layer, [a.src], langs, a.tasks, a.rank)[a.src]
            summ = {}
            for task, key in (("upos", "acc_nontrivial"), ("deprel", "acc_nontrivial"), ("struct", "uuas")):
                if task in r:
                    others = [r[task][t][key] for t in langs if t != a.src]
                    summ[task] = {"in_lang": r[task][a.src][key], "xling_mean": float(np.mean(others))}
            res["layers"][layer] = {"summary": summ, "detail": r}
            print(layer, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in summ.items()}),
                  flush=True)
    elif a.mode == "loo":
        res["layer"] = a.layer
        data = {l: U.load(a.model_dir, l, a.layer) for l in langs}
        res["loo"] = {}
        for t in langs:
            rest = [data[l] for l in langs if l != t]
            row = {}
            if "upos" in a.tasks:
                row["upos"] = U.category_eval(U.train_category(U.pool(rest), "upos", len(U.UPOS)), data[t], "upos", U.UPOS)
            if "deprel" in a.tasks:
                row["deprel"] = U.category_eval(U.train_category(U.pool(rest), "deprel", len(U.DEPREL)), data[t],
                                                "deprel", U.DEPREL)
            if "struct" in a.tasks:
                row["struct"] = U.structural_eval(U.fit_structural(U.pool(rest, 150), rank=a.rank, epochs=20), data[t])
            res["loo"][t] = row
            print(t, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in row.items()}), flush=True)
    else:
        res["layer"] = a.layer
        res["matrix"] = probe_all(a.model_dir, a.layer, langs, langs, a.tasks, a.rank)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
