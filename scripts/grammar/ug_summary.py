"""Aggregate the universal-grammar runs into the cross-model tables of docs/UNIVERSAL_GRAMMAR.md.

The universality statistic is SIGN CONSISTENCY: for each theory contrast, in how many of the
(model × language) cells does the difference point the same way — a claim is only called
universal when it holds in (nearly) every cell, not just on average.

  .venv/bin/python scripts/grammar/ug_summary.py --out runs/grammar/universal_grammar_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "runs" / "grammar"
MODELS = {"qwen05": ("Qwen2.5-0.5B", 12), "llama1b": ("Llama-3.2-1B", 8), "qwen15": ("Qwen2.5-1.5B", 14),
          "gemma2b": ("gemma-2-2b", 13)}


def load(name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def contrasts(kind: str, pairs: list[tuple[str, str]]) -> dict:
    out = {}
    for x, y in pairs:
        cells, means = [], {}
        for tag, (_, layer) in MODELS.items():
            r = load(f"{tag}_{kind}_L{layer}_summary.json")
            if r is None:
                continue
            means[tag] = r["own_grouping"][x]["value"] - r["own_grouping"][y]["value"]
            for l, v in r["per_lang"].items():
                cells.append(v[x]["grouping"][x] - v[y]["grouping"][y])
        if cells:
            out[f"{x} - {y}"] = {"per_model": means, "cells": len(cells),
                                 "positive": int((np.array(cells) > 0).sum()), "mean": float(np.mean(cells))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=R / "universal_grammar_summary.json")
    a = p.parse_args()
    res = {"models": MODELS}
    res["theory_table"] = {}
    for tag, (_, layer) in MODELS.items():
        for L, key in ((0, "embedding"), (layer, "mid")):
            r = load(f"{tag}_theory_L{L}_summary.json")
            if r:
                res["theory_table"].setdefault(tag, {})[key] = {t: v["value"] for t, v in r["own_grouping"].items()}
    r = load("qwen05rand_theory_L12_summary.json")
    if r:
        res["theory_table"]["qwen05-randinit"] = {"mid": {t: v["value"] for t, v in r["own_grouping"].items()}}
    res["theory_contrasts"] = contrasts("theory", [("fhead", "ud"), ("merge", "fhead"), ("xbar", "merge"),
                                                   ("xbar", "fhead")])
    res["headedness_contrasts"] = contrasts("headedness", [("P:case", "ud"), ("C:mark", "ud"), ("T:aux+cop", "ud"),
                                                           ("D:det", "ud"), ("PCT:no-det", "fhead")])
    res["loo"] = {}
    for tag, (_, layer) in list(MODELS.items()) + [("qwen05rand", ("", 12))]:
        r = load(f"{tag}_loo_L{layer}_summary.json")
        if r:
            res["loo"][tag] = {task: {"mean": float(np.mean([v[task][k] for v in r["loo"].values()])),
                                      **({"chain_mean": float(np.mean([v[task]["uuas_linear_chain"] for v in r["loo"].values()]))}
                                         if task == "struct" else {})}
                               for task, k in (("upos", "acc_nontrivial"), ("deprel", "acc_nontrivial"), ("struct", "uuas"))
                               if task in next(iter(r["loo"].values()))}
    res["categories"] = {}
    for tag, (_, layer) in list(MODELS.items()) + [("qwen05rand", ("", 12))]:
        r = load(f"{tag}_categories_L{layer}_summary.json")
        if r:
            res["categories"][tag] = {"bottleneck": {k: v["mean"] for k, v in r["bottleneck"].items()},
                                      "rsa_langs": r["rsa_langs"]["mean"],
                                      "axes": [{"var": ax["var_frac"], "agree": ax["per_lang_rank_agreement"],
                                                "top": list(ax["coords"])[:3], "bottom": list(ax["coords"])[-3:]}
                                               for ax in r["axes"]],
                                      "features": r["features_mean"]}
    x = load("xmodel_categories_summary.json")
    if x:
        res["xmodel_categories"] = {k: {"rsa": v["rsa_mean"], "axes": v["axis_rank_agreement"]} for k, v in x["pairs"].items()}
    res["causal"], res["protorole"] = {}, {}
    for tag, (_, layer) in MODELS.items():
        r = load(f"{tag}_causal_L{layer}_summary.json")
        if r:
            res["causal"][tag] = {c: r[f"mean_{c}"] for c in ("leace_shared", "leace_random")}
        r = load(f"{tag}_protorole_L{layer}_summary.json")
        if r:
            res["protorole"][tag] = {"preverbal_mean": float(np.mean(list(r["passive_position"].values()))),
                                     "postverbal": r.get("postverbal_passive_position"),
                                     "n_postverbal_passive": r["postverbal_pooled"].get("nsubj:pass:post", [0, 0])[1]}
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
