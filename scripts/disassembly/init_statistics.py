"""Test Grok's mechanism for the d^0.15 excess in k — does the INPUT COVARIANCE STRUCTURE inflate the GELU participation
ratio above the iid baseline?

The checkpoint test showed k (per-token PR of the MLP hidden) is architectural — present at random init, flat over
training. For iid pre-activations z, k = PR(GELU(z)) ≈ c·d_ff (LINEAR in d). The measured k ∝ d^1.15 has a d^0.15 excess.
Grok's hypothesis: it comes from the residual stream x being low-effrank, so z = W↑x is correlated (non-iid), and GELU on
correlated inputs delocalizes more (higher PR). Direct test: per layer, compare
  k_real  = PR(GELU(W↑ x))                       — real residual inputs (structured, low effrank);
  k_shuf  = PR(GELU(W↑ x̃)),  x̃ = columns of x shuffled independently across the batch  — same marginals, NO correlation.
If the structure causes the excess, k_real > k_shuf and the GAP grows with d (k_real ∝ d^1.15, k_shuf ∝ d^1.0). Also
report the input effective rank effrank(Σ_x). Output: runs/disassembly/init_statistics_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_router import up_projs  # noqa: E402

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
LADDER_D = {"pythia-70m": 512, "pythia-160m": 768, "pythia-410m": 1024, "pythia-1b": 2048}


def pr(h):                                                               # per-row participation ratio (|·|₁² / |·|₂²), mean
    import torch
    return float(((h.abs().sum(-1) ** 2) / (h.pow(2).sum(-1) + 1e-9)).mean())


def run_model(mid, args):
    import torch
    import torch.nn.functional as F
    from residual_vm import ResidualVM
    t = torch
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); tok = vm.tok; nL = vm.nL
    ups = up_projs(vm.model)
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")[: args.chars]
    ids = tok(text)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16][: args.fit]

    cap = {}; samples = {L: [] for L in range(nL)}
    hs = [mod.register_forward_pre_hook((lambda L: lambda m, i: cap.__setitem__(L, i[0].detach()))(L))
          for L, mod in enumerate(ups)]
    with t.no_grad():
        for c in chunks:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                if sum(x.shape[0] for x in samples[L]) < args.rows:
                    samples[L].append(cap[L][0].float())                # up-proj input x (post-LN residual)
    for h in hs:
        h.remove()
    g = np.random.default_rng(0)
    kr = []; ksh = []; eff = []; kii = []
    with t.no_grad():
        for L in range(nL):
            x = t.cat(samples[L], 0)[: args.rows]                       # (N, d)
            z = ups[L](x)                                              # pre-activation W↑ x
            kr.append(pr(F.gelu(z)))                                    # real hidden
            perm = t.stack([t.tensor(g.permutation(x.shape[0]), device=x.device) for _ in range(x.shape[1])], 1)
            xsh = t.gather(x, 0, perm)                                  # shuffle each column independently (whiten x's covariance)
            ksh.append(pr(F.gelu(ups[L](xsh))))
            # pure-iid baseline: z̃ iid Gaussian matched to z's MEAN per-coord variance (homogeneous) → Grok's c·d_ff
            ziid = t.randn_like(z) * z.var(0).mean().sqrt()
            kii.append(pr(F.gelu(ziid)))
            xc = (x - x.mean(0)); sv = t.linalg.svdvals(xc.float()); lam = (sv ** 2)
            eff.append(float((lam.sum() ** 2) / (lam.pow(2).sum() + 1e-12)))
    m = ups[0].weight.shape[0] if vm.is_gpt2 else ups[0].weight.shape[0]   # up-proj out dim = d_ff (Conv1D (d,d_ff)→shape[1]; Linear (d_ff,d)→shape[0])
    m = ups[0].weight.shape[1] if vm.is_gpt2 else ups[0].weight.shape[0]
    return {"model": mid.split("/")[-1], "d": LADDER_D.get(mid.split("/")[-1]), "m": int(m),
            "k_real": float(np.mean(kr)), "k_shuf": float(np.mean(ksh)), "k_iid": float(np.mean(kii)),
            "effrank_x": float(np.mean(eff))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-70m,EleutherAI/pythia-160m,EleutherAI/pythia-410m,EleutherAI/pythia-1b")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--chars", type=int, default=160000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/disassembly/init_statistics_summary.json"))
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
            print(f"  d={r['d']} m={r['m']}  k_iid={r['k_iid']:.0f}  k_real={r['k_real']:.0f}  "
                  f"k_shuf={r['k_shuf']:.0f}  (iid/m={r['k_iid'] / r['m']:.2f})  effrank(x)={r['effrank_x']:.0f}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    good = [r for r in results if "k_real" in r and r.get("d")]
    fit = {}
    if len(good) >= 2:
        d = np.array([r["d"] for r in good], float)
        fit["k_iid_exp"] = float(np.polyfit(np.log(d), np.log([r["k_iid"] for r in good]), 1)[0])
        fit["k_real_exp"] = float(np.polyfit(np.log(d), np.log([r["k_real"] for r in good]), 1)[0])
        fit["k_shuf_exp"] = float(np.polyfit(np.log(d), np.log([r["k_shuf"] for r in good]), 1)[0])
        fit["effrank_exp"] = float(np.polyfit(np.log(d), np.log([r["effrank_x"] for r in good]), 1)[0])
        print(f"\n  FIT:  k_iid ∝ d^{fit['k_iid_exp']:.2f}  ·  k_real ∝ d^{fit['k_real_exp']:.2f}  ·  "
              f"k_shuf ∝ d^{fit['k_shuf_exp']:.2f}  ·  effrank(x) ∝ d^{fit['effrank_exp']:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "does input covariance structure cause the d^0.15 excess in k? (real vs column-shuffled)", "results": results, "fit": fit}, indent=2, default=float))
    print(f"  [done] → {args.out}")
    return results


if __name__ == "__main__":
    main()
