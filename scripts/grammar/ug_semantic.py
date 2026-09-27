"""Syntax or semantics? Dowty proto-roles via passive subjects (universal-grammar track).

A passive subject ("the ball was kicked") is syntactically a subject (nsubj:pass) but
semantically a proto-PATIENT, like an object. A linear probe is trained, pooled over languages
(train split), to separate ACTIVE subjects (nsubj, proto-agents mostly) from objects (obj). It is
then applied to held-out words:

  active nsubj → P(obj)   (should be low)
  obj          → P(obj)   (should be high)
  nsubj:pass   → P(obj)   the question: near obj ⇒ the shared relation code is semantic (roles);
                          near active nsubj ⇒ it is syntactic (grammatical function).

Reported per language (languages with ≥ 20 held-out passive subjects), with the embedding layer
as the lexical control.

  .venv/bin/python scripts/grammar/ug_semantic.py --model-dir data/ug/Qwen2.5-0.5B --layer 12 \
      --out runs/grammar/qwen05_protorole_L12_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import ug_lib as U
from ug_extract import PUD


def full_deprels(lang: str) -> dict[tuple[int, int], str]:
    """(sentence index, word index) → deprel WITH subtype (ug_extract strips subtypes)."""
    out, si, wi = {}, -1, 0
    for line in (PUD / f"{lang}_pud.conllu").read_text(encoding="utf-8").splitlines():
        if line.startswith("# sent_id"):
            si, wi = si + 1, 0
        elif line and not line.startswith("#"):
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:
                continue
            out[(si, wi)] = c[7]
            wi += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--langs", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    inf = U.info(a.model_dir)
    langs = a.langs or list(inf["langs"])
    X_tr, y_tr, test = [], [], {}
    for l in langs:
        L = U.load(a.model_dir, l, a.layer)
        full = full_deprels(l)
        rel = np.array([full[(s, w)] for s, w in zip(L.sent, L.widx)])
        for split, want in (("train", ~L.test), ("test", L.test)):
            sel = want & ~L.mwt
            # heads are sentence-local word indices: a word is post-head when its head precedes it
            post = (L.head >= 0) & (L.head < L.widx)
            groups = {"nsubj": sel & (rel == "nsubj"), "obj": sel & (rel == "obj"), "nsubj:pass": sel & (rel == "nsubj:pass"),
                      "nsubj:post": sel & (rel == "nsubj") & post, "nsubj:pass:post": sel & (rel == "nsubj:pass") & post}
            if split == "train":
                for g, y in (("nsubj", 0), ("obj", 1)):
                    X_tr.append(L.X[torch.from_numpy(groups[g]).to(U.DEV)])
                    y_tr.append(np.full(groups[g].sum(), y))
            else:
                test[l] = {g: L.X[torch.from_numpy(m).to(U.DEV)] for g, m in groups.items()}
    lin = U.fit_linear(torch.cat(X_tr), np.concatenate(y_tr), 2)
    res = {"model": inf["model"], "random_init": inf.get("random_init", False), "layer": a.layer, "per_lang": {}}
    for l, groups in test.items():
        row = {}
        for g, X in groups.items():
            if len(X) == 0:
                continue
            with torch.no_grad():
                pobj = torch.softmax(lin(X.float()), -1)[:, 1].cpu().numpy()
            row[g] = {"n": int(len(X)), "p_obj": float(pobj.mean()), "frac_obj": float((pobj > 0.5).mean())}
        res["per_lang"][l] = row
        if row.get("nsubj:pass", {}).get("n", 0) >= 20:
            print(f"{l}: P(obj)  active-subj {row['nsubj']['p_obj']:.2f}  obj {row['obj']['p_obj']:.2f}  "
                  f"passive-subj {row['nsubj:pass']['p_obj']:.2f} (n={row['nsubj:pass']['n']})", flush=True)
    ok = [l for l, r in res["per_lang"].items() if r.get("nsubj:pass", {}).get("n", 0) >= 20]
    # position of passive subjects between the two poles: 0 = like active subjects, 1 = like objects
    res["passive_position"] = {l: (res["per_lang"][l]["nsubj:pass"]["p_obj"] - res["per_lang"][l]["nsubj"]["p_obj"])
                               / max(res["per_lang"][l]["obj"]["p_obj"] - res["per_lang"][l]["nsubj"]["p_obj"], 1e-6)
                               for l in ok}
    # the causal-model control: only subjects AFTER their verb (the model has seen the passive), pooled
    pool = {g: [] for g in ("nsubj:post", "nsubj:pass:post", "obj")}
    for l, groups in test.items():
        for g in pool:
            if len(groups[g]):
                with torch.no_grad():
                    pool[g].append(torch.softmax(lin(groups[g].float()), -1)[:, 1].cpu().numpy())
    pooled = {g: (float(np.concatenate(v).mean()), int(sum(map(len, v)))) for g, v in pool.items() if v}
    res["postverbal_pooled"] = pooled
    if all(g in pooled for g in pool):
        pos = (pooled["nsubj:pass:post"][0] - pooled["nsubj:post"][0]) / max(pooled["obj"][0] - pooled["nsubj:post"][0], 1e-6)
        res["postverbal_passive_position"] = pos
        print(f"POST-VERBAL only (pooled): P(obj) active-subj {pooled['nsubj:post'][0]:.2f} (n={pooled['nsubj:post'][1]})  "
              f"passive-subj {pooled['nsubj:pass:post'][0]:.2f} (n={pooled['nsubj:pass:post'][1]})  "
              f"obj {pooled['obj'][0]:.2f}  → position {pos:.2f}")
    if ok:
        print(f"passive subject position (0=subject, 1=object), mean over {len(ok)} langs: "
              f"{np.mean(list(res['passive_position'].values())):.2f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
