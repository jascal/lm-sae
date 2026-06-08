"""Can the core be RETRAINED into a low-rank surrogate below the no-retrain floor? — the test that can falsify "irreducible".

Every other core experiment here is FROZEN + LINEAR + FIXED-BASIS (PCA truncation, SAE features, the running-bond TT)
and they plateau at a constant fraction of the model — but that says nothing about whether a *learned* low-rank
representation exists. This trains one. For each layer L we insert a rank-r bottleneck on the layer's residual update,

    update'  =  (update · W_down[L]) · W_up[L]            W_down ∈ ℝ^{d×r},  W_up ∈ ℝ^{r×d}

initialised from the **top-r PCA of the update** — so at step 0 the surrogate IS the no-retrain core_rank truncation
(same ΔNLL). We then FREEZE the base model and train ONLY the bottleneck factors to minimise next-token NLL on the
corpus (distillation). If the trained rank-r ΔNLL drops well below the PCA-init ΔNLL, the entangled core is
**tractably compressible with learning** — the Θ(d) floor was an artifact of freezing + linearity, not the function.
If training barely helps, the floor is more robust. Either way it directly probes "is detangling the core intractable?"

The only departure from the read-only frame: a few hundred gradient steps on the *bottleneck factors only* (the base
weights stay frozen). Output: runs/disassembly/core_distill_summary.json.
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

    import torch
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d
    for p in vm.model.parameters():
        p.requires_grad_(False)                                       # freeze the base model
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:300000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8]
    fit, train, ev = chunks[: args.fit], chunks[args.fit: args.fit + args.train], chunks[args.fit + args.train: args.fit + args.train + args.eval]

    # ---- per-layer update covariance → PCA bases (the no-retrain init) ----
    cov = {L: np.zeros((d, d)) for L in range(nL)}; cap = {}
    hks = [vm.layers[L].register_forward_hook(
        (lambda L: lambda m, i, o: cap.__setitem__(L, ((o[0] if isinstance(o, tuple) else o) - i[0]).detach()))(L))
        for L in range(nL)]
    with t.no_grad():
        for c in fit:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                u = cap[L][0].float().cpu().numpy(); cov[L] += u.T @ u
    for h in hks:
        h.remove()
    pca = {}
    for L in range(nL):
        w, V = np.linalg.eigh(cov[L]); pca[L] = V[:, np.argsort(-w)].astype(np.float32)   # cols = components desc

    def nll(seqs):
        tot = 0.0; k = 0
        with t.no_grad():
            for c in seqs:
                lp = t.log_softmax(vm.logits(c).float(), -1); y = c[1:]
                for p in range(len(y)):
                    tot += float(-lp[p, y[p]]); k += 1
        return tot / max(k, 1)
    full_nll = nll(ev)

    def fit_rank(r, steps):
        """init rank-r bottleneck from PCA (step0 = no-retrain), train the factors, return (pca_init_nll, distilled_nll)."""
        down = {L: t.tensor(pca[L][:, :r].copy(), device=vm.dev, requires_grad=True) for L in range(nL)}
        up = {L: t.tensor(pca[L][:, :r].T.copy(), device=vm.dev, requires_grad=True) for L in range(nL)}
        hs = []

        def mk(L):
            def hook(m, i, o):
                out = o[0] if isinstance(o, tuple) else o
                upd = (out - i[0]).float()
                low = (upd @ down[L]) @ up[L]
                new = (i[0].float() + low).to(out.dtype)
                return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
            return hook
        for L in range(nL):
            hs.append(vm.layers[L].register_forward_hook(mk(L)))

        def eval_nll(seqs):
            tot = 0.0; k = 0
            with t.no_grad():
                for c in seqs:
                    lp = t.log_softmax(vm.model(input_ids=t.tensor([c], device=vm.dev)).logits[0].float(), -1); y = c[1:]
                    for p in range(len(y)):
                        tot += float(-lp[p, y[p]]); k += 1
            return tot / max(k, 1)
        pca_init = eval_nll(ev)                                        # step 0 == the no-retrain PCA truncation
        params = list(down.values()) + list(up.values())
        opt = torch.optim.Adam(params, lr=args.lr)
        rng = np.random.default_rng(0)
        for s in range(steps):
            c = train[int(rng.integers(0, len(train)))]
            ids = t.tensor([c], device=vm.dev)
            logits = vm.model(input_ids=ids).logits[0]
            loss = t.nn.functional.cross_entropy(logits[:-1].float(), ids[0, 1:])
            opt.zero_grad(); loss.backward(); opt.step()
        distilled = eval_nll(ev)
        for h in hs:
            h.remove()
        return pca_init, distilled

    ranks = sorted({min(int(x), d) for x in args.ranks.split(",")})
    curve = []
    for r in ranks:
        pca_init, distilled = fit_rank(r, args.steps)
        curve.append({"rank": r, "rank_frac": r / d, "pca_init_nll_increase": pca_init - full_nll,
                      "distilled_nll_increase": distilled - full_nll,
                      "recovered_frac": 1 - (distilled - full_nll) / max(pca_init - full_nll, 1e-9)})
        print(f"  rank {r:4d} ({r / d:.0%} d): no-retrain ΔNLL {pca_init - full_nll:+.3f} → distilled ΔNLL "
              f"{distilled - full_nll:+.3f}  (recovered {curve[-1]['recovered_frac']:.0%} of the truncation loss)")
    return {"model": mid.split("/")[-1], "n_layers": nL, "d_model": d, "base_nll": full_nll,
            "steps": args.steps, "curve": curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=30, help="chunks for the PCA init")
    p.add_argument("--train", type=int, default=200, help="chunks for distillation training")
    p.add_argument("--eval", type=int, default=20)
    p.add_argument("--ranks", default="8,16,32,64", help="bottleneck ranks to test")
    p.add_argument("--steps", type=int, default=300, help="distillation gradient steps per rank")
    p.add_argument("--lr", type=float, default=1e-3)
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
            print(f"  d{r['d_model']} {r['n_layers']}L | base NLL {r['base_nll']:.3f} | {r['steps']} distill steps/rank")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "core_distill_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "retrained low-rank surrogate — does training the per-layer bottleneck factors beat the no-retrain PCA floor?",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'curve' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
