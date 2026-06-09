"""η(d) via SAE dictionary size — the packing efficiency the feature-economy proxy couldn't pin.

η = m/m' where m' is the number of latent features the model effectively represents. Measure m' as the dictionary width
at which a fixed-sparsity SAE's reconstruction of the MLP hidden saturates: at fixed L0, train top-k SAEs of increasing
width m' ∈ {0.5,1,2,4}·m on a layer's activations and read variance-explained vs width. The width where it saturates is
the effective feature count; m'/m is the overcompleteness, η = m/m'. If m'/m grows with d (η shrinks), denser packing at
scale (the ansatz's mechanism 2); if flat, packing efficiency is scale-stable (consistent with the flat s/k overhead).

Captures activations with the model loaded, FREES the model, then trains SAEs on the cached activations (so a wide SAE +
the base model never coexist on the GPU). Output: runs/disassembly/sae_dictionary_summary.json.
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


def capture_acts(mid, args):
    import torch
    from residual_vm import ResidualVM
    t = torch
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); tok = vm.tok; nL = vm.nL
    downs = down_projs(vm.model); L = nL // 2; m = downs[0].weight.shape[1] if not vm.is_gpt2 else downs[0].weight.shape[0]
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")[: args.chars]
    ids = tok(text)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16][: args.fit]
    rows = []
    cap = {}
    h = downs[L].register_forward_pre_hook(lambda m_, i: cap.__setitem__(0, i[0].detach()))
    with t.no_grad():
        for c in chunks:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            rows.append(cap[0][0].float().cpu())
            if sum(r.shape[0] for r in rows) >= args.rows:
                break
    h.remove()
    X = t.cat(rows, 0)[: args.rows]
    del vm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return X, int(m), L


def train_sae(X, width, L0, args):
    import torch
    t = torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    Xg = X.to(dev); n, d = Xg.shape; mu = Xg.mean(0); tot = float((Xg - mu).pow(2).sum())
    We = (t.randn(d, width, device=dev) / np.sqrt(d)).requires_grad_(True)
    Wd = (t.randn(width, d, device=dev) / np.sqrt(width)).requires_grad_(True)
    be = t.zeros(width, device=dev, requires_grad=True); bd = mu.clone().requires_grad_(True)
    opt = torch.optim.Adam([We, Wd, be, bd], lr=args.lr); rng = np.random.default_rng(0)
    for s in range(args.steps):
        idx = t.tensor(rng.integers(0, n, min(args.batch, n)), device=dev); x = Xg[idx]
        z = t.relu((x - bd) @ We + be)
        thr = z.topk(min(L0, width), dim=-1).values[:, -1:]                # top-L0 gate per row
        zt = z * (z >= thr)
        rec = zt @ Wd + bd
        loss = (rec - x).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with t.no_grad():
        sse = 0.0
        for b in range(0, n, 1024):
            x = Xg[b:b + 1024]; z = t.relu((x - bd) @ We + be)
            thr = z.topk(min(L0, width), dim=-1).values[:, -1:]; zt = z * (z >= thr)
            sse += float((zt @ Wd + bd - x).pow(2).sum())
    del Xg, We, Wd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 1.0 - sse / (tot + 1e-9)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-70m,EleutherAI/pythia-160m,EleutherAI/pythia-410m,EleutherAI/pythia-1b")
    p.add_argument("--widths", default="0.5,1,2,4", help="dictionary widths as multiples of m")
    p.add_argument("--l0", type=int, default=64, help="fixed SAE sparsity (active features per token)")
    p.add_argument("--l0-frac", type=float, default=0.0, help="if >0, scale L0 = round(l0_frac·m) per model (L0 ∝ d) — the clean η")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=120)
    p.add_argument("--rows", type=int, default=4000)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--chars", type=int, default=300000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/disassembly/sae_dictionary_summary.json"))
    args = p.parse_args(argv)

    import torch
    facs = [float(x) for x in args.widths.split(",")]
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        name = mid.split("/")[-1]
        print(f"\n=== {name} ===")
        try:
            X, m, L = capture_acts(mid, args)
            l0 = round(args.l0_frac * m) if args.l0_frac > 0 else args.l0   # scale sparsity to the model (L0 ∝ m ∝ d)
            curve = []
            for f in facs:
                width = int(f * m)
                try:
                    ve = train_sae(X, width, l0, args)
                except torch.cuda.OutOfMemoryError:
                    print(f"  width {f}×m ({width}): OOM — skipped"); torch.cuda.empty_cache(); continue
                curve.append({"width_over_m": f, "width": width, "var_explained": ve})
                print(f"  width {f:>3}×m ({width:6d}) · L0={l0} ({l0 / m:.0%} of m) · var-explained {ve:.3f}")
            results.append({"model": name, "d": LADDER_D.get(name), "m": m, "layer": L, "l0": l0, "curve": curve})
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": name, "error": str(e)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "eta(d) via SAE dictionary-size — var-explained vs overcompleteness at fixed L0",
                                    "results": results}, indent=2, default=float))
    print(f"\n[done] → {args.out}")
    return results


if __name__ == "__main__":
    main()
