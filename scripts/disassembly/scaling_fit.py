"""Fit the P3 scaling ansatz against the measured Pythia active-fraction curves.

The ansatz (external, Grok) predicts the per-token effective active dimension s(d,m,H) — the neurons that must be
evaluated to reconstruct the composition — and claims s/m grows with scale. We measured s directly: the top-k MLP-neuron
recovery of *content* (`mlp_kv_sparsity.py`). This reads those ladder curves and fits:
  - s(d): the active dimension at a recovery threshold τ on content ΔNLL (interpolated), absolute and as a fraction s/m;
  - the power laws  s ∝ d^γ_eff  and  s/m ∝ d^φ  (least squares in log-log);
  - the superposition overhead s/r against the functional-rank reference r ≈ d/3 (the data-aware low-rank floor).
Lets us check the ansatz's testable predictions quantitatively. Output: runs/disassembly/scaling_fit_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LADDER = {"pythia-70m": 512, "pythia-160m": 768, "pythia-410m": 1024, "pythia-1b": 2048}   # model → d (residual width)


def crossing(curve, key, tau):
    """interpolate the k at which `key` falls to tau (linear in k between the bracketing points)."""
    pts = sorted(((c["k"], c[key]) for c in curve), key=lambda x: x[0])
    for (k0, v0), (k1, v1) in zip(pts, pts[1:]):
        if v0 >= tau >= v1 and v0 != v1:
            return k0 + (k1 - k0) * (v0 - tau) / (v0 - v1)
    return pts[-1][0] if pts[-1][1] <= tau else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=Path("runs/disassembly/mlp_kv_sparsity_summary.json"))
    p.add_argument("--tau", type=float, default=0.3, help="content ΔNLL recovery threshold defining s")
    p.add_argument("--out", type=Path, default=Path("runs/disassembly/scaling_fit_summary.json"))
    args = p.parse_args(argv)

    data = json.loads(args.summary.read_text())
    by = {r["model"]: r for r in data["results"] if "curve" in r}
    rows = []
    for name, d in LADDER.items():
        if name not in by:
            continue
        m = by[name]["d_ff"]
        s = crossing(by[name]["curve"], "dNLL_other", args.tau)
        rows.append({"model": name, "d": d, "m": m, "s": s, "s_over_m": s / m,
                     "r_third_d": d / 3, "s_over_r": s / (d / 3)})

    d = np.array([r["d"] for r in rows], float); s = np.array([r["s"] for r in rows], float)
    som = np.array([r["s_over_m"] for r in rows], float); sor = np.array([r["s_over_r"] for r in rows], float)
    gamma = float(np.polyfit(np.log(d), np.log(s), 1)[0])               # s ∝ d^gamma
    phi = float(np.polyfit(np.log(d), np.log(som), 1)[0])               # s/m ∝ d^phi
    psi = float(np.polyfit(np.log(d), np.log(sor), 1)[0])               # s/r ∝ d^psi (superposition-overhead growth)

    out = {"experiment": "fit of the P3 active-dimension scaling ansatz to the Pythia ladder", "tau": args.tau,
           "rows": rows, "fit": {"s_vs_d_exponent_gamma": gamma, "frac_vs_d_exponent_phi": phi,
                                 "overhead_s_over_r_exponent_psi": psi}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"  content-recovery active dimension s (τ={args.tau} content ΔNLL), Pythia ladder:")
    for r in rows:
        print(f"    {r['model']:14s} d={r['d']:5d} m={r['m']:5d}  s≈{r['s']:6.0f}  s/m={r['s_over_m']:.2f}  "
              f"s/r(=d/3)={r['s_over_r']:.1f}×")
    print(f"\n  FIT:  s ∝ d^{gamma:.2f}   ·   s/m ∝ d^{phi:.2f}   ·   superposition overhead s/r ∝ d^{psi:.2f}")
    print(f"  [done] → {args.out}")
    return out


if __name__ == "__main__":
    main()
