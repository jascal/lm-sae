"""Informed router — let the broadcast / cross-layer findings shape the gate (follow-up to mlp_router.py).

The naive learned router (per-layer linear gate from the local input, soft-train / hard-eval) UNDERperformed static
mass-routing. Two fixes, both motivated by what the channel decomposition (`circuit_channels.py`) found:
  (1) GLOBAL CONTEXT — a minority of cross-layer channels are register/topic broadcasts (set early, held across the
      sequence). So give every layer's router an early "context" vector (the residual entering an early layer, carrying
      the broadcast/topic + induction-detection state) alongside the local input — gather-early, route-late.
  (2) STRAIGHT-THROUGH selection — train the *hard* top-E selection it is evaluated with (forward hard, backward through
      the softmax), removing the soft-weight / hard-select mismatch.
Head-to-head against static mass-routing at matched E.

Output: runs/disassembly/mlp_router2_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_experts import kmeans, up_proj_keys  # noqa: E402
from mlp_kv_sparsity import down_projs  # noqa: E402
from mlp_router import up_projs  # noqa: E402

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def run_model(mid, args):
    import torch
    from residual_vm import ResidualVM
    t = torch
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); tok = vm.tok; nL = vm.nL; d = vm.d
    for p in vm.model.parameters():
        p.requires_grad_(False)
    ups = up_projs(vm.model); downs = down_projs(vm.model); K = args.experts
    L_ctx = max(1, nL // 4)                                              # early layer whose residual carries the broadcast/topic state
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")[: args.chars]
    ids = tok(text)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16]
    train, ev = chunks[: args.train], chunks[args.train: args.train + args.fit]

    vocab = vm.model.config.vocab_size
    punct_chars = set('.,;:!?"\'()-\n—’“”…`')
    punct_id = np.zeros(vocab, bool)
    for v in range(vocab):
        s = tok.decode([v]).strip()
        if s != "" and all(ch in punct_chars for ch in s):
            punct_id[v] = True

    onehot = []
    for L in range(nL):
        keys = up_proj_keys(vm.model, L).float(); keys = keys / (keys.norm(dim=1, keepdim=True) + 1e-9)
        lab = kmeans(keys, K, seed=L)
        oh = t.zeros(keys.shape[0], K, device=vm.dev); oh[t.arange(keys.shape[0]), lab] = 1.0
        onehot.append(oh)
    d_ff = onehot[0].shape[0]

    # router: gate from [local MLP input x  ⊕  early global context]  → K experts
    Wr = [t.zeros(2 * d, K, device=vm.dev, requires_grad=True) for _ in range(nL)]
    br = [t.zeros(K, device=vm.dev, requires_grad=True) for _ in range(nL)]
    gate_cache = {}; ctx_vec = {}; topE = [None]; route = ["learned"]

    def mk_ctx():                                                       # capture the early-layer residual (broadcast state)
        def hook(m, i):
            ctx_vec[0] = i[0].detach()
            return None
        return hook
    ctx_layers = sorted({0, L_ctx})                                    # layer 0 = fresh default each fwd; L_ctx overwrites for later layers
    hc = [vm.layers[cl].register_forward_pre_hook(mk_ctx()) for cl in ctx_layers]

    def mk_up(L):
        def hook(m, i):
            x = i[0]
            if route[0] == "mass":
                return None
            g = t.cat([x, ctx_vec[0]], -1) @ Wr[L] + br[L]              # (..,K) gate from local ⊕ global context
            E = min(topE[0], K)
            soft = t.softmax(g, -1)
            idx = g.topk(E, dim=-1).indices
            hard = t.zeros_like(g); hard.scatter_(-1, idx, 1.0)
            gate_cache[L] = hard + (soft - soft.detach())               # straight-through: forward hard, backward soft
            return None
        return hook

    def mk_down(L):
        def hook(m, i):
            h = i[0]
            if route[0] == "mass":
                mass = h.abs() @ onehot[L]; idx = mass.topk(min(topE[0], K), dim=-1).indices
                keep_e = t.zeros_like(mass); keep_e.scatter_(-1, idx, 1.0); keep_n = keep_e @ onehot[L].t()
            else:
                keep_n = gate_cache[L] @ onehot[L].t()
            return (h * keep_n,) + tuple(i[1:])
        return hook
    hu = [mod.register_forward_pre_hook(mk_up(L)) for L, mod in enumerate(ups)]
    hd = [mod.register_forward_pre_hook(mk_down(L)) for L, mod in enumerate(downs)]

    # ---- train the routers at the target E (straight-through), base frozen ----
    route[0] = "learned"; topE[0] = args.trainE; opt = torch.optim.Adam(Wr + br, lr=args.lr); rng = np.random.default_rng(0)
    for s in range(args.steps):
        c = train[int(rng.integers(0, len(train)))]; tid = t.tensor([c], device=vm.dev)
        logits = vm.model(input_ids=tid).logits[0]
        loss = t.nn.functional.cross_entropy(logits[:-1].float(), tid[0, 1:])
        opt.zero_grad(); loss.backward(); opt.step()

    def cat_nll():
        tot = {"punct": 0.0, "dup": 0.0, "other": 0.0}; cnt = {"punct": 0, "dup": 0, "other": 0}
        with t.no_grad():
            for c in ev:
                lp = t.log_softmax(vm.model(input_ids=t.tensor([c], device=vm.dev)).logits[0].float(), -1)
                seen = set()
                for p in range(len(c) - 1):
                    seen.add(c[p]); nxt = c[p + 1]
                    cat = "punct" if punct_id[nxt] else ("dup" if nxt in seen else "other")
                    tot[cat] += -float(lp[p, nxt]); cnt[cat] += 1
        return {k: tot[k] / max(cnt[k], 1) for k in tot}

    for h in hc:
        h.remove()
    for h in hu + hd:
        h.remove()
    base = cat_nll()
    hc = [vm.layers[cl].register_forward_pre_hook(mk_ctx()) for cl in ctx_layers]
    hu = [mod.register_forward_pre_hook(mk_up(L)) for L, mod in enumerate(ups)]
    hd = [mod.register_forward_pre_hook(mk_down(L)) for L, mod in enumerate(downs)]

    curve = []
    with t.no_grad():
        for E in sorted({int(x) for x in args.E.split(",")}):
            topE[0] = E
            route[0] = "learned"; rl = cat_nll()
            route[0] = "mass"; rm = cat_nll()
            curve.append({"E": E, "frac": E / K,
                          "informed_dNLL_other": rl["other"] - base["other"], "informed_dNLL_dup": rl["dup"] - base["dup"],
                          "mass_dNLL_other": rm["other"] - base["other"], "mass_dNLL_dup": rm["dup"] - base["dup"]})
    for h in hc + hu + hd:
        h.remove()
    return {"model": mid.split("/")[-1], "d_ff": int(d_ff), "experts": K, "trainE": args.trainE, "ctx_layer": L_ctx,
            "steps": args.steps, "baseline": base, "curve": curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-160m")
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--E", default="2,4,8,16,32")
    p.add_argument("--trainE", type=int, default=8, help="top-E the router is trained at (straight-through)")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--train", type=int, default=120)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--chars", type=int, default=200000)
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
            b = r["baseline"]
            print(f"  d_ff {r['d_ff']} · K={r['experts']} · ctx-layer {r['ctx_layer']} · baseline content NLL {b['other']:.2f} "
                  f"· router trained@E={r['trainE']} {r['steps']} steps")
            print("  top-E → content ΔNLL:  INFORMED router (global-ctx + straight-through)  vs  static mass-routing")
            for c in r["curve"]:
                print(f"    E {c['E']:3d}/{r['experts']} ({c['frac']:.0%})  informed {c['informed_dNLL_other']:+.3f}  ·  "
                      f"mass {c['mass_dNLL_other']:+.3f}   (Δ {c['informed_dNLL_other'] - c['mass_dNLL_other']:+.3f})")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "mlp_router2_summary.json"
    sumpath.write_text(json.dumps({"experiment": "broadcast/cross-layer-informed router (global ctx + straight-through) vs static mass-routing",
                                   "results": results}, indent=2, default=float))
    print(f"\n[done] → {sumpath}")
    return results


if __name__ == "__main__":
    main()
