"""The overcomplete feature-native corner — legible + complete (but not small): SAE on the update vs the tight bottleneck.

#142 compressed the per-layer update to a TIGHT rank-r bottleneck (small + complete) but #143/#144 showed it is
illegible and a sparsity term can't fix a tight bottleneck (no room). The other corner of the triangle is the
**overcomplete** one: an SAE on the update — F > d features, trained with NLL + L1 sparsity — which should be LEGIBLE
(sparse, monosemantic directions) and COMPLETE (reconstruct the update) at the cost of SIZE (F ≫ d). This trains both
on the same object (each layer's residual update) and tabulates all three axes, completing the small/legible/complete
triangle empirically:

  TIGHT (rank-r down·up)     — small (r ≪ d), expected complete + illegible (the #142/#143 corner).
  OVERCOMPLETE (SAE, F > d)  — big (F ≫ d), expected legible (high peak-z, sparse codes) + complete.

Axes measured per config: ΔNLL (completeness), logit-lens peak-z + closed-class fraction of the decode directions
(legibility), mean code L0 fraction (sparsity), and trainable params (size). Base model frozen; ~steps gradient steps.
Output: runs/disassembly/feature_native_sae_summary.json.
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
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d
    for p in vm.model.parameters():
        p.requires_grad_(False)
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:300000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8]
    train, ev = chunks[: args.train], chunks[args.train: args.train + args.eval]

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

    def legibility(W):                                                # W: (k, d) rows = decode directions
        zs = []; cf = []
        idx = np.linspace(0, W.shape[0] - 1, min(W.shape[0], 400)).astype(int)   # sample dirs for speed
        for k in idx:
            v = W[k]; v = v / (np.linalg.norm(v) + 1e-9); lg = WU @ v
            zp = (lg.max() - lg.mean()) / (lg.std() + 1e-9); zn = (lg.mean() - lg.min()) / (lg.std() + 1e-9)
            top = (np.argsort(-lg) if zp >= zn else np.argsort(lg))[:10]
            zs.append(max(zp, zn)); cf.append(np.mean([int(i) in cset for i in top]))
        return float(np.mean(zs)), float(np.mean(cf))

    # ---- capture per-layer update activations (the SAE's training data) ----
    buf = {L: [] for L in range(nL)}; cap = {}
    hks = [vm.layers[L].register_forward_hook(
        (lambda L: lambda m, i, o: cap.__setitem__(L, ((o[0] if isinstance(o, tuple) else o) - i[0]).detach()))(L))
        for L in range(nL)]
    with t.no_grad():
        for c in train:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                buf[L].append(cap[L][0].float())
    for h in hks:
        h.remove()
    U = {L: t.cat(buf[L], 0) for L in range(nL)}                       # (Nrows, d) update activations per layer
    tot_var = {L: float((U[L] - U[L].mean(0)).pow(2).sum()) for L in range(nL)}

    def fit_config(kind, width, lam):
        """kind='tight' (rank-width PCA reconstruction) or 'sae' (F=width SAE trained on MSE+L1). Base frozen."""
        rec_dirs = {}; enc = {}; bdec = {}; benc = {}; l0acc = []; var_exp = []
        if kind == "tight":                                           # optimal rank-r reconstruction == PCA of the update
            for L in range(nL):
                Xc = (U[L] - U[L].mean(0))
                V = t.linalg.eigh(Xc.T @ Xc)[1][:, -width:]            # (d, r) top-r
                rec_dirs[L] = V.T                                      # (r, d) decode rows
                proj = Xc @ V; var_exp.append(float((proj.pow(2).sum()) / (tot_var[L] + 1e-9)))
                bdec[L] = U[L].mean(0)
        else:                                                         # train an SAE on the update: MSE + λ·L1
            for L in range(nL):
                Wd = t.tensor((np.random.randn(width, d) / np.sqrt(d)).astype(np.float32), device=vm.dev, requires_grad=True)
                We = t.tensor(Wd.detach().cpu().numpy().T.copy(), device=vm.dev, requires_grad=True)
                be = t.zeros(width, device=vm.dev, requires_grad=True)
                bd = U[L].mean(0).clone().requires_grad_(True)
                opt = torch.optim.Adam([Wd, We, be, bd], lr=args.lr); X = U[L]; n = X.shape[0]
                rng = np.random.default_rng(L)
                for s in range(args.steps):
                    idx = t.tensor(rng.integers(0, n, min(256, n)), device=vm.dev)
                    x = X[idx]; codes = t.relu((x - bd) @ We + be); rec = codes @ Wd + bd
                    loss = (rec - x).pow(2).mean() + lam * codes.abs().mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                with t.no_grad():                                     # batched eval — avoids materializing (N×F) codes at large F
                    sse = 0.0; l0s = 0.0; nb = 0
                    for bs in range(0, X.shape[0], 512):
                        xb = X[bs:bs + 512]; cb = t.relu((xb - bd) @ We + be); rb = cb @ Wd + bd
                        sse += float((rb - xb).pow(2).sum()); l0s += float((cb > 0).float().sum(-1).mean()) * xb.shape[0]; nb += xb.shape[0]
                    var_exp.append(float(1 - sse / (tot_var[L] + 1e-9)))
                    l0acc.append(float(l0s / max(nb, 1) / width))
                rec_dirs[L] = Wd.detach(); enc[L] = We.detach(); benc[L] = be.detach(); bdec[L] = bd.detach()

        # ---- insert the reconstruction and measure NLL (completeness) ----
        hs = []

        def mk(L):
            def hook(m, i, o):
                out = o[0] if isinstance(o, tuple) else o; upd = (out - i[0]).float()
                if kind == "tight":
                    c = upd - bdec[L]; rec = (c @ rec_dirs[L].T) @ rec_dirs[L] + bdec[L]
                else:
                    codes = t.relu((upd - bdec[L]) @ enc[L] + benc[L]); rec = codes @ rec_dirs[L] + bdec[L]
                return (((i[0].float() + rec).to(out.dtype)),) + tuple(o[1:]) if isinstance(o, tuple) else (i[0].float() + rec).to(out.dtype)
            return hook
        for L in range(nL):
            hs.append(vm.layers[L].register_forward_hook(mk(L)))
        tot = 0.0; k = 0
        with t.no_grad():
            for c in ev:
                lp = t.log_softmax(vm.model(input_ids=t.tensor([c], device=vm.dev)).logits[0].float(), -1); y = c[1:]
                for p in range(len(y)):
                    tot += float(-lp[p, y[p]]); k += 1
        for h in hs:
            h.remove()
        nll = tot / max(k, 1)
        Wdec = np.concatenate([rec_dirs[L].cpu().numpy() for L in range(nL)], axis=0).astype(np.float64)
        z, cf = legibility(Wdec)
        nparams = (nL * 2 * d * width if kind == "tight" else nL * (2 * d * width + width))
        return {"kind": kind, "width": width, "lambda": lam if kind == "sae" else None, "nll": nll,
                "peak_z": z, "closed_frac": cf, "code_L0_frac": (float(np.mean(l0acc)) if l0acc else None),
                "variance_explained": float(np.mean(var_exp)), "trainable_params_M": nparams / 1e6, "width_over_d": width / d}

    np.random.seed(0)
    full = None
    # full-model NLL baseline (no hook)
    tot = 0.0; k = 0
    with t.no_grad():
        for c in ev:
            lp = t.log_softmax(vm.logits(c).float(), -1); y = c[1:]
            for p in range(len(y)):
                tot += float(-lp[p, y[p]]); k += 1
    full = tot / max(k, 1)
    feats = [int(x) for x in str(args.features).split(",")]           # sweep one or more overcomplete widths
    configs = [("tight", args.rank, 0.0)] + [("sae", F, lam) for F in feats for lam in [float(x) for x in args.l1.split(",")]]
    rows = []
    for kind, width, lam in configs:
        m = fit_config(kind, width, lam); m["nll_increase"] = m["nll"] - full; rows.append(m)
        print(f"  {kind:4s} width {width:5d} ({m['width_over_d']:.1f}×d): ΔNLL {m['nll_increase']:+.3f} · "
              f"var-explained {m['variance_explained']:.0%} · peak-z {m['peak_z']:.2f} · closed {m['closed_frac']:.2f} · "
              f"L0 {('%.3f' % m['code_L0_frac']) if m['code_L0_frac'] is not None else '--'} · "
              f"{m['trainable_params_M']:.1f}M params")
    return {"model": mid.split("/")[-1], "n_layers": nL, "d_model": d, "base_nll": full, "steps": args.steps, "configs": rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--rank", type=int, default=32, help="tight bottleneck rank (the small corner)")
    p.add_argument("--features", default="1536", help="overcomplete SAE width(s) F, comma-separated (the legible corner; F>d)")
    p.add_argument("--l1", default="0.02", help="L1 sparsity weight for the SAE")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--eval", type=int, default=20)
    p.add_argument("--steps", type=int, default=400)
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
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "feature_native_sae_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "overcomplete feature-native SAE vs tight bottleneck — completing the small/legible/complete triangle", "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'configs' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
