"""Can a LEGIBILITY term give you small AND legible? — feature-native distillation (NLL + λ·sparsity), the vision's crux.

#142: NLL-distillation compresses the per-layer update ~30× (small). #143: but it is legibility-NEUTRAL (small ≠
legible). The user's feature-native endgame (train smaller → extract clean features/knowledge/circuits → the small
Python program → runtime explainability) therefore needs the missing ingredient: an explicit legibility objective. This
tests the minimal version — add an **L1 sparsity penalty on the bottleneck codes** (the sae / feature-native term) to
the rank-r distillation, sweep its weight λ, and ask: does the write basis get more interpretable (peak-z / closed-class
up, codes sparse) while NLL stays acceptable? λ=0 reproduces #143 (neutral). If λ>0 raises legibility at little NLL
cost, you *can* train small AND legible — the pipeline's missing ingredient works.

Output: runs/disassembly/feature_native_distill_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_grammar import CLOSED, PUNCT  # noqa: E402


def run_model(mid, args):
    import urllib.request

    import torch
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d; r = args.rank
    for p in vm.model.parameters():
        p.requires_grad_(False)
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:300000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8]
    fit, train, ev = chunks[: args.fit], chunks[args.fit: args.fit + args.train], chunks[args.fit + args.train: args.fit + args.train + args.eval]

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
    pca = {L: np.linalg.eigh(cov[L])[1][:, ::-1][:, :r].astype(np.float32) for L in range(nL)}

    WU = vm.model.get_output_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
    cset = set()
    for w in CLOSED:
        for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            tt = tok(f, add_special_tokens=False)["input_ids"]
            if len(tt) == 1:
                cset.add(tt[0])
    for p_ in PUNCT:
        for f in (p_, " " + p_):
            tt = tok(f, add_special_tokens=False)["input_ids"]
            if len(tt) == 1:
                cset.add(tt[0])

    def legibility(vecs):
        zs = []; cf = []
        for k in range(vecs.shape[1]):
            v = vecs[:, k]; v = v / (np.linalg.norm(v) + 1e-9); lg = WU @ v
            zp = (lg.max() - lg.mean()) / (lg.std() + 1e-9); zn = (lg.mean() - lg.min()) / (lg.std() + 1e-9)
            top = (np.argsort(-lg) if zp >= zn else np.argsort(lg))[:10]
            zs.append(max(zp, zn)); cf.append(np.mean([int(i) in cset for i in top]))
        return float(np.mean(zs)), float(np.mean(cf))

    base_nll = None

    def fit_lambda(lam):
        nonlocal base_nll
        down = {L: t.tensor(pca[L].copy(), device=vm.dev, requires_grad=True) for L in range(nL)}
        up = {L: t.tensor(pca[L].T.copy(), device=vm.dev, requires_grad=True) for L in range(nL)}
        codes_l1 = {}
        hs = []

        def mk(L):
            def hook(m, i, o):
                out = o[0] if isinstance(o, tuple) else o
                codes = (out - i[0]).float() @ down[L]
                codes_l1[L] = codes.abs().mean()
                new = (i[0].float() + codes @ up[L]).to(out.dtype)
                return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
            return hook
        for L in range(nL):
            hs.append(vm.layers[L].register_forward_hook(mk(L)))
        opt = torch.optim.Adam(list(down.values()) + list(up.values()), lr=args.lr)
        rng = np.random.default_rng(0)
        for s in range(args.steps):
            c = train[int(rng.integers(0, len(train)))]; ids = t.tensor([c], device=vm.dev)
            logits = vm.model(input_ids=ids).logits[0]
            ce = t.nn.functional.cross_entropy(logits[:-1].float(), ids[0, 1:])
            loss = ce + lam * sum(codes_l1.values())
            opt.zero_grad(); loss.backward(); opt.step()

        tot = 0.0; k = 0; active = []
        with t.no_grad():
            for c in ev:
                logits = vm.model(input_ids=t.tensor([c], device=vm.dev)).logits[0]
                lp = t.log_softmax(logits.float(), -1); y = c[1:]
                for p in range(len(y)):
                    tot += float(-lp[p, y[p]]); k += 1
        for h in hs:
            h.remove()
        # code sparsity: fraction of bottleneck codes active (|code| > 1% of the layer's max)
        cap2 = {}
        h2 = [vm.layers[L].register_forward_hook(
            (lambda L: lambda m, i, o: cap2.__setitem__(L, ((o[0] if isinstance(o, tuple) else o)[0] - i[0][0]).float() @ down[L]))(L))
            for L in range(nL)]
        with t.no_grad():
            for c in ev[:5]:
                cap2.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
                for L in range(nL):
                    z = cap2[L].abs(); active.append(float((z > 0.01 * z.max()).float().mean()))
        for h in h2:
            h.remove()
        nll = tot / max(k, 1)
        if base_nll is None and lam == 0.0:
            base_nll = nll
        z, cf = legibility(np.concatenate([up[L].detach().cpu().numpy().T.astype(np.float64) for L in range(nL)], axis=1))
        return {"lambda": lam, "nll": nll, "peak_z": z, "closed_frac": cf, "code_active_frac": float(np.mean(active))}

    rows = [fit_lambda(lam) for lam in [float(x) for x in args.lambdas.split(",")]]
    nll0 = next((row["nll"] for row in rows if row["lambda"] == 0.0), rows[0]["nll"])
    for row in rows:
        row["nll_increase_vs_lambda0"] = row["nll"] - nll0
    return {"model": mid.split("/")[-1], "n_layers": nL, "d_model": d, "rank": r, "steps": args.steps, "rows": rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lambdas", default="0,0.01,0.05,0.2", help="L1 weights on the bottleneck codes")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=30)
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--eval", type=int, default=20)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
    args = p.parse_args(argv)

    import torch
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} (rank {args.rank}) ===")
        try:
            r = run_model(mid, args)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(r)
            print("  λ (L1 on codes) → NLL · ΔNLL vs λ0 · peak-z · closed-frac · code-active-frac")
            for row in r["rows"]:
                print(f"    λ={row['lambda']:<5g}  NLL {row['nll']:.3f}  ΔNLL {row['nll_increase_vs_lambda0']:+.3f}  "
                      f"peak-z {row['peak_z']:.2f}  closed {row['closed_frac']:.2f}  active {row['code_active_frac']:.2f}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "feature_native_distill_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "feature-native distillation — does an L1 legibility term give small AND legible?", "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'rows' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
