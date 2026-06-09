"""H-dependence of the active feature count — does k grow with next-token entropy? (the P3 ansatz's δ)

The ansatz has k(d,H) = k_0 (d/d_0)^γ (H/H_0)^δ. We measured the d-scaling (k ∝ d^1.15, `feature_economy.py`). This
measures the H-dependence *within* a fixed model: per token, compute the model's next-token entropy H = −Σ p log p AND
the per-token feature count k (participation ratio of the MLP hidden, averaged over layers). Bin tokens by H and fit
k ∝ H^δ. A clean within-model test that separates the data-entropy axis from the width axis.

Output: runs/disassembly/entropy_dependence_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_kv_sparsity import down_projs  # noqa: E402

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def run_model(mid, args):
    import torch
    from residual_vm import ResidualVM
    t = torch
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); tok = vm.tok; nL = vm.nL
    downs = down_projs(vm.model)
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")[: args.chars]
    ids = tok(text)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16][: args.fit]

    cap = {}
    hs = [mod.register_forward_pre_hook((lambda L: lambda m, i: cap.__setitem__(L, i[0].detach()))(L))
          for L, mod in enumerate(downs)]
    Hs = []; Ks = []
    with t.no_grad():
        for c in chunks:
            cap.clear(); out = vm.model(input_ids=t.tensor([c], device=vm.dev))
            lp = t.log_softmax(out.logits[0].float(), -1)
            ent = -(lp.exp() * lp).sum(-1)                                  # (seq,) next-token entropy (nats)
            kper = t.zeros(len(c), device=vm.dev)
            for L in range(nL):
                h = cap[L][0].float()
                kper += (h.abs().sum(-1) ** 2) / (h.pow(2).sum(-1) + 1e-9)   # per-token PR, summed over layers
            kper /= nL
            Hs.append(ent.cpu().numpy()); Ks.append(kper.cpu().numpy())
    for h in hs:
        h.remove()
    H = np.concatenate(Hs); Kv = np.concatenate(Ks)

    nb = args.bins
    qs = np.quantile(H, np.linspace(0, 1, nb + 1))
    bins = []
    for b in range(nb):
        msk = (H >= qs[b]) & (H <= qs[b + 1])
        if msk.sum() > 5:
            bins.append({"H_mid": float(np.median(H[msk])), "k_mean": float(Kv[msk].mean()), "n": int(msk.sum())})
    hh = np.array([b["H_mid"] for b in bins]); kk = np.array([b["k_mean"] for b in bins])
    ok = (hh > 1e-3) & (kk > 0)
    delta = float(np.polyfit(np.log(hh[ok]), np.log(kk[ok]), 1)[0]) if ok.sum() >= 2 else float("nan")
    corr = float(np.corrcoef(H, Kv)[0, 1])
    return {"model": mid.split("/")[-1], "n_tokens": int(len(H)), "delta_k_vs_H": delta, "corr_H_k": corr, "bins": bins}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-160m,EleutherAI/pythia-410m")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=60)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--chars", type=int, default=200000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/disassembly/entropy_dependence_summary.json"))
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
            print(f"  corr(H,k)={r['corr_H_k']:+.2f} · fit k ∝ H^{r['delta_k_vs_H']:.2f}  ({r['n_tokens']} tokens)")
            print("    H (nats) → k:  " + " · ".join(f"{b['H_mid']:.1f}:{b['k_mean']:.0f}" for b in r["bins"]))
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "H-dependence of the per-token feature count k (the ansatz's δ)", "results": results}, indent=2, default=float))
    print(f"\n[done] → {args.out}")
    return results


if __name__ == "__main__":
    main()
