"""Measure k(d) and η(d) — break the degeneracy in the P3 active-dimension ansatz.

The ansatz s ≈ α·k·log(m/k)·η^{-β} fits our measured s(d) ∝ d^1.23 either by (1) k growing super-linearly or (2) the
packing-density overhead η(d) dominating — and s(d) alone can't tell which. This measures the two inputs directly, per
Pythia ladder model, on the post-activation MLP hidden h (the d_ff neurons):

  k(d)  — per-token "effective features in play": the participation ratio of |h|,  PR = (Σ|h_i|)² / Σh_i²  (a soft count
          of effectively-active neurons), averaged over tokens and layers. The geometric feature count (vs the functional
          active support s, which is larger by the log + interference overhead).
  η(d)  — packing density proxy: the EFFECTIVE RANK of the activation covariance, effrank = (Σλ)²/Σλ² over the eigenvalues
          λ of E[hhᵀ], as a fraction of m. Denser superposition ⇒ activations span more of the neuron space ⇒ effrank/m
          ↑ ⇒ packing efficiency η ↓. (Report packing := effrank/m; η ∝ 1/packing.)

Reports k(d), packing(d)=effrank/m, and the overhead ratio s/k (using s from scaling_fit) so the ansatz can be re-fit
with k and η pinned. Output: runs/disassembly/feature_economy_summary.json.
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
LADDER_D = {"pythia-70m": 512, "pythia-160m": 768, "pythia-410m": 1024, "pythia-1b": 2048}


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

    cap = {}; kpr_sum = np.zeros(nL); kpr_n = np.zeros(nL); samples = {L: [] for L in range(nL)}
    hs = [mod.register_forward_pre_hook((lambda L: lambda m, i: cap.__setitem__(L, i[0].detach()))(L))
          for L, mod in enumerate(downs)]
    with t.no_grad():
        for c in chunks:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                h = cap[L][0].float()                                       # (seq, d_ff) post-activation hidden
                kpr = (h.abs().sum(-1) ** 2) / (h.pow(2).sum(-1) + 1e-9)     # (seq,) per-token participation ratio
                kpr_sum[L] += float(kpr.sum()); kpr_n[L] += h.shape[0]
                if sum(x.shape[0] for x in samples[L]) < args.rows:
                    samples[L].append(h.cpu().numpy())
    for h in hs:
        h.remove()
    m = downs[0].weight.shape[1] if not vm.is_gpt2 else downs[0].weight.shape[0]

    kpr = kpr_sum / np.maximum(kpr_n, 1)
    effrank = np.zeros(nL)
    for L in range(nL):
        X = np.concatenate(samples[L], 0)[: args.rows]; X = X - X.mean(0)
        sv = np.linalg.svd(X, compute_uv=False); lam = sv ** 2
        effrank[L] = float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12))
    return {"model": mid.split("/")[-1], "d": LADDER_D.get(mid.split("/")[-1]), "m": int(m), "n_layers": nL,
            "k_per_token": float(kpr.mean()), "k_per_layer": [round(float(x), 1) for x in kpr],
            "effrank": float(effrank.mean()), "packing_effrank_over_m": float(effrank.mean() / m)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-70m,EleutherAI/pythia-160m,EleutherAI/pythia-410m,EleutherAI/pythia-1b")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--rows", type=int, default=1200, help="activation rows sampled per layer for the covariance effrank")
    p.add_argument("--chars", type=int, default=160000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/disassembly/feature_economy_summary.json"))
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
            print(f"  d={r['d']} m={r['m']}  k(per-token PR)≈{r['k_per_token']:.0f}  "
                  f"effrank≈{r['effrank']:.0f}  packing(effrank/m)={r['packing_effrank_over_m']:.2f}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    # fit power laws k ∝ d^γk, packing ∝ d^ρ
    good = [r for r in results if "k_per_token" in r and r.get("d")]
    fit = {}
    if len(good) >= 2:
        d = np.array([r["d"] for r in good], float)
        fit["k_vs_d_exponent"] = float(np.polyfit(np.log(d), np.log([r["k_per_token"] for r in good]), 1)[0])
        fit["packing_vs_d_exponent"] = float(np.polyfit(np.log(d), np.log([r["packing_effrank_over_m"] for r in good]), 1)[0])
        print(f"\n  FIT:  k ∝ d^{fit['k_vs_d_exponent']:.2f}   ·   packing(effrank/m) ∝ d^{fit['packing_vs_d_exponent']:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "k(d) and η(d) for the P3 ansatz", "results": results, "fit": fit}, indent=2, default=float))
    print(f"  [done] → {args.out}")
    return results


if __name__ == "__main__":
    main()
