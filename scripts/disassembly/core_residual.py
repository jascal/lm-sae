"""Where does the entangled core LIVE? — locate the incompressible residual the data-aware low-rank floor leaves behind.

The richer-form survey (#157–164) showed every *frozen* factorization plateaus at a ~⅔d size floor: data-aware low-rank
on pythia-160m is +0.45 NLL at rank 256. This asks *which tokens* that residual error sits on. Data-aware-compress every
composition matrix to rank r, then split the compression ΔNLL by next-token category — PUNCTUATION (clause boundary),
DUPLICATE (induction / copy family), OTHER. If the error concentrates on DUPLICATE, the incompressible core *is* the
copy/induction circuits (→ the preserve-circuits hybrid: keep them verbatim, compress the rest). If it spreads evenly,
the core is diffuse content composition. Cheap (no training): one covariance pass + one whitened-SVD per matrix + eval.

Output: runs/disassembly/core_residual_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from min_to_run import composition_mats  # noqa: E402

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def run_model(mid, args):
    import torch
    from residual_vm import ResidualVM
    t = torch
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); tok = vm.tok
    recs = composition_mats(vm.model)
    mods = [m for (m, _) in recs]; mats = [m.weight for (m, _) in recs]; is_lin = [lin for (_, lin) in recs]

    def eff(W, lin):
        return W.t() if lin else W

    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")[: args.chars]
    ids = tok(text)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16][: args.fit]

    vocab = vm.model.config.vocab_size
    punct_chars = set('.,;:!?"\'()-\n—’“”…`')
    punct_id = np.zeros(vocab, bool)
    for v in range(vocab):
        s = tok.decode([v]).strip()
        if s != "" and all(ch in punct_chars for ch in s):
            punct_id[v] = True

    def cat_nll():                                                                # mean NLL on punct / dup / other
        tot = {"punct": 0.0, "dup": 0.0, "other": 0.0}; cnt = {"punct": 0, "dup": 0, "other": 0}
        with t.no_grad():
            for c in chunks:
                lp = t.log_softmax(vm.model(input_ids=t.tensor([c], device=vm.dev)).logits[0].float(), -1)
                seen = set()
                for p in range(len(c) - 1):
                    seen.add(c[p]); nxt = c[p + 1]
                    cat = "punct" if punct_id[nxt] else ("dup" if nxt in seen else "other")
                    tot[cat] += -float(lp[p, nxt]); cnt[cat] += 1
        return {k: tot[k] / max(cnt[k], 1) for k in tot}, cnt

    # ---- input covariance per matrix (functional / data-aware weighting) ----
    cov = [None] * len(mods); cnt = [0] * len(mods); hs = []
    for j, mod in enumerate(mods):
        def mk(j):
            def hook(m, i):
                x = i[0].detach().reshape(-1, i[0].shape[-1]).float(); g = x.t() @ x
                cov[j] = g if cov[j] is None else cov[j] + g; cnt[j] += x.shape[0]
            return hook
        hs.append(mod.register_forward_pre_hook(mk(j)))
    with t.no_grad():
        for c in chunks:
            vm.model(input_ids=t.tensor([c], device=vm.dev))
    for h in hs:
        h.remove()

    orig = [W.detach().clone() for W in mats]
    base, cnts = cat_nll()

    def set_daware(rank, ridge=1e-3):                                             # data-aware rank-r, write back into weights
        for W, lin, c in zip(mats, is_lin, cov):
            M = eff(W, lin).detach().float()
            C = c.to(M.device); C = C + ridge * C.diag().mean() * t.eye(C.shape[0], device=M.device)
            w, Q = t.linalg.eigh(C); w = w.clamp_min(1e-8)
            Chalf = (Q * w.sqrt()) @ Q.t(); Cinv = (Q * w.rsqrt()) @ Q.t()
            U, S, Vh = t.linalg.svd(Chalf @ M, full_matrices=False)
            P = Cinv @ U
            Mr = (P[:, :rank] * S[:rank]) @ Vh[:rank]
            W.data.copy_((Mr.t() if lin else Mr).to(W.dtype))

    def restore():
        for W, o in zip(mats, orig):
            W.data.copy_(o)

    curve = []
    for rank in sorted({int(x) for x in args.ranks.split(",")}):
        set_daware(rank); comp, _ = cat_nll(); restore()
        d = {k: comp[k] - base[k] for k in base}
        curve.append({"rank": rank, "dNLL_punct": d["punct"], "dNLL_dup": d["dup"], "dNLL_other": d["other"],
                      "concentration_dup_over_other": d["dup"] / (d["other"] + 1e-9)})
    return {"model": mid.split("/")[-1], "baseline": base, "counts": cnts, "curve": curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-160m")
    p.add_argument("--ranks", default="64,128,256")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=80)
    p.add_argument("--chars", type=int, default=200000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
    args = p.parse_args(argv)

    import torch
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = run_model(mid, args)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(r)
            b = r["baseline"]
            print(f"  baseline NLL punct {b['punct']:.2f} · dup {b['dup']:.2f} · other {b['other']:.2f}  "
                  f"(counts {r['counts']})")
            print("  data-aware compression ΔNLL by category — where the incompressible core sits:")
            for c in r["curve"]:
                print(f"    rank {c['rank']:4d}  punct {c['dNLL_punct']:+.3f} · dup {c['dNLL_dup']:+.3f} · "
                      f"other {c['dNLL_other']:+.3f}  (dup/other {c['concentration_dup_over_other']:.2f})")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "core_residual_summary.json"
    sumpath.write_text(json.dumps({"experiment": "where the data-aware low-rank residual concentrates (punct/dup/other)",
                                   "results": results}, indent=2, default=float))
    print(f"\n[done] → {sumpath}")
    return results


if __name__ == "__main__":
    main()
