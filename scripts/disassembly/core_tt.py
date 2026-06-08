"""Build the TT/MPS surrogate and measure NLL RETAINED — does the layer stack really run at bond χ≈16?

`core_mps.py` measured the entangled core's cross-layer bond dimension as χ≈16 and FLAT (area-law). That is a claim
about the *coupling* between early and late layers; this script turns it into a **runnable, no-retrain surrogate** and
measures how much of the model's next-token NLL survives when the inter-layer state is forced through a χ-dim bond.

The residual plays two roles: it carries the TOKEN EMBEDDING (full-rank lexical content, injected once) and the
INTER-LAYER COMPUTATION (the contextualisation the area-law says is low-bond). A faithful tensor-train surrogate
**protects the embedding (resid_0) in full** and bottlenecks only the *carried computation* through a χ-dim bond:

    resid_{L+1}  ←  resid_0  +  B_L Bᵀ_L (resid_{L+1} − resid_0)          # keep χ dims of the accumulated update

so each block reads the full token embedding plus a χ-dim summary of everything computed so far — a TT with the
embedding as a global input leg and a χ-dim bond between cores. The last block's output is left full (it feeds the
unembedding directly — no downstream cut). We sweep χ and compare three regimes:

  TT (embedding-protected running bond)  — the surrogate above; the area-law CPU lever.
  RESID (full-residual bottleneck)       — project the WHOLE residual (embedding included) to χ dims: the control that
                                           should FAIL if the low-bond structure is in the computation, not the tokens.
  PER-LAYER (core_rank's lever)          — independently truncate each layer's own update to rank χ: the reference.

No weight retraining. ResidualVM for load + NLL. Output: runs/disassembly/core_tt_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


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

    # ---- pass 1: accumulate covariances of the per-layer update Δ_L, the accumulated update acc_L, and the residual ----
    cov_delta = {L: np.zeros((d, d)) for L in range(nL)}
    cov_acc = {L: np.zeros((d, d)) for L in range(nL)}
    cov_resid = {L: np.zeros((d, d)) for L in range(nL)}
    r0 = {}
    cap = {}
    pre0 = vm.layers[0].register_forward_pre_hook(lambda m, i: r0.__setitem__(0, i[0].detach()))
    hks = [vm.layers[L].register_forward_hook(
        (lambda L: lambda m, i, o: cap.__setitem__(L, (i[0].detach(), (o[0] if isinstance(o, tuple) else o).detach())))(L))
        for L in range(nL)]
    with t.no_grad():
        for c in fit:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            base = r0[0][0].float().cpu().numpy()
            for L in range(nL):
                xin = cap[L][0][0].float().cpu().numpy(); xout = cap[L][1][0].float().cpu().numpy()
                cov_delta[L] += (xout - xin).T @ (xout - xin)
                cov_acc[L] += (xout - base).T @ (xout - base)
                cov_resid[L] += xout.T @ xout
    pre0.remove()
    for h in hks:
        h.remove()

    def topvecs(cov, k):
        w, V = np.linalg.eigh(cov); order = np.argsort(-w)[:k]
        return t.tensor(V[:, order].astype(np.float32), device=vm.dev)
    kmax = min(args.kmax, d)
    Bd = {L: topvecs(cov_delta[L], kmax) for L in range(nL)}           # per-layer update bases
    Ba = {L: topvecs(cov_acc[L], kmax) for L in range(nL)}            # accumulated-update bases (the running bond)
    Br = {L: topvecs(cov_resid[L], kmax) for L in range(nL)}          # full-residual bases (the control)

    def gen_nll(mode=None, chi=None):
        hs = []; r0g = {}
        if mode is not None:
            hs.append(vm.layers[0].register_forward_pre_hook(lambda m, i: r0g.__setitem__(0, i[0])))

            def mk(L, mode):
                def hook(m, i, o):
                    out = o[0] if isinstance(o, tuple) else o
                    if mode == "perlayer":
                        ref = i[0]; B = Bd[L][:, :chi]
                        new = ref + ((out.float() - ref.float()) @ B) @ B.T
                    elif mode == "tt":
                        base = r0g[0]; B = Ba[L][:, :chi]
                        new = base + ((out.float() - base.float()) @ B) @ B.T
                    else:  # resid: bottleneck the whole residual, embedding included
                        B = Br[L][:, :chi]
                        new = (out.float() @ B) @ B.T
                    new = new.to(out.dtype)
                    return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
                return hook
            # tt/resid bottleneck the INTER-layer cuts (0..nL-2); the last output feeds the unembedding, left full.
            # per-layer truncates every layer's own update (0..nL-1).
            last = nL if mode == "perlayer" else nL - 1
            for L in range(last):
                hs.append(vm.layers[L].register_forward_hook(mk(L, mode)))
        tot = 0.0; k = 0
        try:
            with t.no_grad():
                for c in ev:
                    lp = t.log_softmax(vm.logits(c).float(), -1); y = c[1:]
                    for p in range(len(y)):
                        tot += float(-lp[p, y[p]]); k += 1
        finally:
            for h in hs:
                h.remove()
        return tot / max(k, 1)

    base_nll = gen_nll()
    chis = sorted({min(int(x), d) for x in args.chis.split(",")})
    curve = []
    for chi in chis:
        row = {"chi": chi}
        for mode in ("tt", "perlayer", "resid"):
            row[mode] = gen_nll(mode, chi) - base_nll
        curve.append(row)
    return {"model": mid.split("/")[-1], "n_layers": nL, "d_model": d, "base_generic_nll": base_nll,
            "kmax": kmax, "curve": curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--eval", type=int, default=20)
    p.add_argument("--chis", default="4,8,16,32,64,128,256", help="bond dimensions to sweep")
    p.add_argument("--kmax", type=int, default=256, help="max bond basis columns to precompute")
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
            print(f"  d{r['d_model']} {r['n_layers']}L | base NLL {r['base_generic_nll']:.3f}")
            print("  bond χ → ΔNLL (TT=embedding-protected running bond · PER-LAYER=core_rank · RESID=full-resid control)")
            for row in r["curve"]:
                print(f"    χ={row['chi']:4d}  TT {row['tt']:+.3f}   per-layer {row['perlayer']:+.3f}   "
                      f"resid {row['resid']:+.3f}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "core_tt_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "no-retrain tensor-train surrogate — NLL retained vs inter-layer bond dimension χ", "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'curve' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
