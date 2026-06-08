"""Is the entangled core LOW-RANK? — effective rank of each layer's residual update + the no-retrain truncation tradeoff.

The entangled core is the composition that doesn't decompile into SAE features / n-grams (the forge tax; the pylm
ceiling). The manifesto's hypothesis: the core is *not* an SAE basis and *not* a dense slab — it is **low-rank in the
right coordinates**. That decides both research horns at once:
  - LOW rank  → the core is decompilable (few reusable directions/templates) AND CPU-simplifiable (a rank-r surrogate
                is cheap: project the layer's output onto its top-r subspace);
  - FULL rank → the entanglement is genuinely dense; simplification needs sparsity/quantisation, not low-rank.

We measure, per layer L, the empirical **residual update** Δ_L = (resid entering L+1) − (resid entering L) — the
layer's total contribution to the residual — over a corpus, then:
  EFFECTIVE RANK — participation ratio of Δ_L's covariance spectrum (and rank to retain 90/95% of the update energy);
  TRUNCATION TRADEOFF — a *no-retrain* low-rank surrogate: project each layer's update onto its top-r PCA subspace
    (a hook, all layers at once) and measure the generic-NLL retained vs the rank fraction → the accuracy↔FLOPs curve.

No weight retraining (the no-retrain constraint). Pure analysis on the frozen model. ResidualVM for load + nll.

Output: runs/disassembly/core_rank_summary.json. Findings -> docs/DECOMPILATION.md / FINDINGS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def participation_ratio(eigs):
    e = np.clip(eigs, 0, None); s = e.sum()
    return float((s * s) / (np.square(e).sum() + 1e-12)) if s > 0 else 0.0


def rank_for_energy(eigs, frac):
    e = np.clip(np.sort(eigs)[::-1], 0, None); c = np.cumsum(e) / (e.sum() + 1e-12)
    return int(np.searchsorted(c, frac) + 1)


def run_model(mid, args):
    import urllib.request
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8]
    fit, ev = chunks[: args.fit], chunks[args.fit: args.fit + args.eval]

    # ---- capture each layer's residual update Δ_L = out − in over the fit corpus; accumulate covariance ----
    cov = {L: np.zeros((d, d)) for L in range(nL)}; nrows = 0
    cap = {}
    hks = [vm.layers[L].register_forward_hook(
        (lambda L: lambda m, i, o: cap.__setitem__(L, ((o[0] if isinstance(o, tuple) else o) - i[0]).detach()))(L))
        for L in range(nL)]
    with t.no_grad():
        for c in fit:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                u = cap[L][0].float().cpu().numpy()                       # (seq, d) layer update
                cov[L] += u.T @ u
            nrows += cap[0].shape[1]
    for h in hks:
        h.remove()
    spectra = {}; eff = {}
    for L in range(nL):
        eigs = np.linalg.eigvalsh(cov[L] / max(nrows, 1)); eigs = np.clip(eigs, 0, None)
        spectra[L] = eigs
        eff[L] = {"participation_ratio": participation_ratio(eigs), "pr_frac": participation_ratio(eigs) / d,
                  "rank90": rank_for_energy(eigs, 0.90), "rank95": rank_for_energy(eigs, 0.95)}

    # ---- PCA bases (top components per layer) for the truncation surrogate ----
    bases = {}
    for L in range(nL):
        w, V = np.linalg.eigh(cov[L]); order = np.argsort(-w)
        bases[L] = t.tensor(V[:, order].astype(np.float32), device=vm.dev)      # columns = components, desc

    def gen_nll(ablate_rank=None, glob=False):
        """generic next-token NLL; ablate_rank=r projects every layer's update onto a rank-r subspace —
        per-layer PCA (glob=False) or the single SHARED global basis Ug (glob=True)."""
        hs = []
        if ablate_rank is not None:
            for L in range(nL):
                Vr = (Ug[:, :ablate_rank] if glob else bases[L][:, :ablate_rank])   # (d, r): shared vs per-layer

                def mk(L, Vr):
                    def hook(m, i, o):
                        out = o[0] if isinstance(o, tuple) else o
                        upd = out - i[0]
                        proj = (upd.float() @ Vr) @ Vr.T                    # project update onto top-r subspace
                        new = i[0] + proj.to(out.dtype)
                        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
                    return hook
                hs.append(vm.layers[L].register_forward_hook(mk(L, Vr)))
        tot = 0.0; k = 0
        try:
            with t.no_grad():
                for c in ev:
                    lp = t.log_softmax(vm.logits(c).float(), -1)
                    y = c[1:]
                    for p in range(len(y)):
                        tot += float(-lp[p, y[p]]); k += 1
        finally:
            for h in hs:
                h.remove()
        return tot / max(k, 1)

    # ---- cross-layer subspace sharing: do the layers write into a SHARED subspace (→ one global low-rank core)? ----
    rs = min(args.share_rank, d // 2)
    stacked = np.concatenate([bases[L][:, :rs].cpu().numpy() for L in range(nL)], axis=1)   # (d, nL*rs)
    Ug, sv, _ = np.linalg.svd(stacked, full_matrices=False)                  # Ug columns = the SHARED global basis
    Ug = t.tensor(Ug.astype(np.float32), device=vm.dev)
    overlaps = []
    for i in range(nL):
        Bi = bases[i][:, :rs].cpu().numpy()
        for j in range(i + 1, nL):
            M = Bi.T @ bases[j][:, :rs].cpu().numpy()
            overlaps.append(float(np.square(M).sum() / rs))         # mean squared principal cosine (1=identical, rs/d=random)
    sharing = {"per_layer_rank": rs, "union_effective_rank": participation_ratio(sv ** 2),
               "no_sharing_upper_bound": min(nL * rs, d), "random_overlap": rs / d,
               "mean_pairwise_overlap": float(np.mean(overlaps)) if overlaps else 0.0}

    base_nll = gen_nll()
    ranks = sorted({min(int(d * int(r) / 100), d) for r in args.rank_fracs.split(",")} | {d})
    ranks = [r for r in ranks if r >= 1]
    curve = []
    for r in ranks:
        nll = gen_nll(ablate_rank=r); gnll = gen_nll(ablate_rank=r, glob=True)
        curve.append({"rank": r, "rank_frac": r / d, "nll": nll, "nll_increase": nll - base_nll,
                      "global_nll_increase": gnll - base_nll})
    return {"model": mid.split("/")[-1], "n_layers": nL, "d_model": d, "base_generic_nll": base_nll,
            "effective_rank_by_layer": {str(L): eff[L] for L in range(nL)},
            "mean_participation_ratio": float(np.mean([eff[L]["participation_ratio"] for L in range(nL)])),
            "mean_pr_frac": float(np.mean([eff[L]["pr_frac"] for L in range(nL)])),
            "mean_rank95": float(np.mean([eff[L]["rank95"] for L in range(nL)])),
            "cross_layer_sharing": sharing, "truncation_curve": curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=40, help="chunks for the covariance/PCA fit")
    p.add_argument("--eval", type=int, default=20, help="chunks for the NLL eval")
    p.add_argument("--share-rank", type=int, default=64, help="per-layer rank for the cross-layer sharing analysis")
    p.add_argument("--rank-fracs", default="2,5,10,20,40,70", help="rank fractions of d to truncate to (percent)")
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
            print(f"  d{r['d_model']} {r['n_layers']}L | mean participation-ratio {r['mean_participation_ratio']:.0f} "
                  f"({r['mean_pr_frac']:.0%} of d) | mean rank95 {r['mean_rank95']:.0f}/{r['d_model']}")
            sh = r["cross_layer_sharing"]
            print(f"  cross-layer sharing: union effective-rank {sh['union_effective_rank']:.0f} vs no-sharing "
                  f"{sh['no_sharing_upper_bound']} (per-layer rank {sh['per_layer_rank']}); mean pairwise overlap "
                  f"{sh['mean_pairwise_overlap']:.2f} (random {sh['random_overlap']:.2f})")
            print("  truncation per-layer (rank-frac → ΔNLL): " +
                  " · ".join(f"{c['rank_frac']:.0%}:{c['nll_increase']:+.2f}" for c in r["truncation_curve"]))
            print("  truncation GLOBAL shared-basis (rank-frac → ΔNLL): " +
                  " · ".join(f"{c['rank_frac']:.0%}:{c['global_nll_increase']:+.2f}" for c in r["truncation_curve"]))
        except Exception as e:  # pragma: no cover
            import traceback; traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "core_rank_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "effective rank of the per-layer residual update + no-retrain low-rank truncation tradeoff",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'truncation_curve' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
