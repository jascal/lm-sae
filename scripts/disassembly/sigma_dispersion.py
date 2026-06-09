"""Measure the per-neuron pre-activation variance dispersion {σ_i²} and test the heterogeneous-σ PR formula.

The d^0.15 excess in k is carried by the spread of the per-neuron pre-activation variances σ_i² = Var((W↑x)_i). This
measures that dispersion across the Pythia ladder and tests which closed form for the participation ratio matches the
measured k:
  σ_i² = Var(z_i),  A_i = E|GELU(z_i)|,  B_i = E[GELU(z_i)²]   (per neuron, over tokens)
  k_moment = (Σ_i A_i)² / (Σ_i B_i)      — the correct vector-PR concentration limit  m·(E A)²/(E B)
  k_meanc  = Σ_i A_i²/B_i = m·E[c(σ)]    — Grok's proposed  m·E[c]
  k_actual = mean_token (Σ_i|h_i|)² / (Σ_i h_i²)
Dispersion of {σ_i²} reported as CV = std/mean and as a participation-ratio fraction; fit vs d gives the source exponent.
Output: runs/disassembly/sigma_dispersion_summary.json.
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
                    samples[L].append(cap[L][0].float())
    for h in hs:
        h.remove()
    cv = []; prfrac = []; km = []; kc = []; ka = []
    with t.no_grad():
        for L in range(nL):
            x = t.cat(samples[L], 0)[: args.rows]
            z = ups[L](x); h = F.gelu(z)
            s2 = z.var(0)                                                 # (d_ff,) per-neuron pre-activation variance
            A = h.abs().mean(0); B = h.pow(2).mean(0)                     # per-neuron |GELU| and GELU²
            cv.append(float(s2.std() / (s2.mean() + 1e-9)))              # dispersion: coefficient of variation
            prfrac.append(float((s2.sum() ** 2 / (s2.pow(2).sum() + 1e-12)) / s2.numel()))
            km.append(float((A.sum() ** 2) / (B.sum() + 1e-9)))         # correct moment-ratio PR
            kc.append(float((A.pow(2) / (B + 1e-9)).sum()))             # Grok's m·E[c]
            ka.append(float(((h.abs().sum(-1) ** 2) / (h.pow(2).sum(-1) + 1e-9)).mean()))  # actual per-token PR
    return {"model": mid.split("/")[-1], "d": LADDER_D.get(mid.split("/")[-1]), "m": int(ups[0].weight.shape[0]),
            "sigma2_CV": float(np.mean(cv)), "sigma2_PR_frac": float(np.mean(prfrac)),
            "k_moment": float(np.mean(km)), "k_meanc": float(np.mean(kc)), "k_actual": float(np.mean(ka))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-70m,EleutherAI/pythia-160m,EleutherAI/pythia-410m,EleutherAI/pythia-1b")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--rows", type=int, default=2500)
    p.add_argument("--chars", type=int, default=160000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/disassembly/sigma_dispersion_summary.json"))
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
            print(f"  d={r['d']} m={r['m']}  σ²-dispersion CV={r['sigma2_CV']:.2f} (PRfrac={r['sigma2_PR_frac']:.2f})  "
                  f"| k: actual={r['k_actual']:.0f} moment={r['k_moment']:.0f} mean-c={r['k_meanc']:.0f}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    good = [r for r in results if "k_actual" in r and r.get("d")]
    fit = {}
    if len(good) >= 2:
        d = np.array([r["d"] for r in good], float)
        for key in ("sigma2_CV", "k_actual", "k_moment", "k_meanc"):
            fit[f"{key}_exp"] = float(np.polyfit(np.log(d), np.log([max(r[key], 1e-9) for r in good]), 1)[0])
        print(f"\n  FIT:  σ²-CV ∝ d^{fit['sigma2_CV_exp']:.2f}  ·  k_actual ∝ d^{fit['k_actual_exp']:.2f}  ·  "
              f"k_moment ∝ d^{fit['k_moment_exp']:.2f}  ·  k_meanc ∝ d^{fit['k_meanc_exp']:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "per-neuron pre-activation variance dispersion + heterogeneous-σ PR formula test", "results": results, "fit": fit}, indent=2, default=float))
    print(f"  [done] → {args.out}")
    return results


if __name__ == "__main__":
    main()
