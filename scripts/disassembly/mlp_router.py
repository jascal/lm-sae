"""The learned-router swing — can a TRAINED gate route the dense MLP content to few experts where static mass-routing
couldn't (mlp_experts.py)?

mlp_experts showed static MoEfication (cluster + keep top-E experts by activation mass) doesn't make content expert-sparse
— content needs ~50-75% of experts. But that router is a fixed read-off. This trains a small per-layer ROUTER (a linear
gate from the MLP input → K expert scores), with the base frozen, on the next-token loss + a budget penalty that pushes
it to use few experts. Then at eval we hard-select the router's **top-E experts per token**, mask the MLP hidden to them,
and measure content recovery — head-to-head against static mass-routing at the same E. If the learned router recovers
content at a meaningfully LOWER E, conditional (learned) computation is the lever static read-offs missed.

Output: runs/disassembly/mlp_router_summary.json.
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

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def up_projs(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [b.mlp.c_fc for b in model.transformer.h]
    if hasattr(model, "gpt_neox"):
        return [ly.mlp.dense_h_to_4h for ly in model.gpt_neox.layers]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [ly.mlp.up_proj for ly in model.model.layers]
    raise SystemExit("unknown architecture for MLP up-projection")


def run_model(mid, args):
    import torch
    from residual_vm import ResidualVM
    t = torch
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); tok = vm.tok; nL = vm.nL; d = vm.d
    for p in vm.model.parameters():
        p.requires_grad_(False)
    ups = up_projs(vm.model); downs = down_projs(vm.model); K = args.experts
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

    # per-layer router: Linear(d → K). gate from MLP input x; mode controls how the hidden is masked.
    Wr = [t.zeros(d, K, device=vm.dev, requires_grad=True) for _ in range(nL)]
    br = [t.zeros(K, device=vm.dev, requires_grad=True) for _ in range(nL)]
    gate_cache = {}; mode = ["soft"]; topE = [None]; xin = {}

    def mk_up(L):
        def hook(m, i):
            x = i[0]; xin[L] = x                                          # MLP input (d)
            g = x @ Wr[L] + br[L]                                         # (.., K) gate logits
            if mode[0] == "soft":
                gate_cache[L] = t.sigmoid(g)
            else:                                                         # hard top-E experts per token (eval)
                E = min(topE[0], K); idx = g.topk(E, dim=-1).indices
                hard = t.zeros_like(g); hard.scatter_(-1, idx, 1.0); gate_cache[L] = hard
            return None
        return hook

    def mk_down(L):
        def hook(m, i):
            h = i[0]; keep_n = gate_cache[L] @ onehot[L].t()              # (.., d_ff) per-neuron keep weight
            return (h * keep_n,) + tuple(i[1:])
        return hook
    hu = [mod.register_forward_pre_hook(mk_up(L)) for L, mod in enumerate(ups)]
    hd = [mod.register_forward_pre_hook(mk_down(L)) for L, mod in enumerate(downs)]

    # ---- train the routers (base frozen): NLL + λ·budget(mean gate) ----
    mode[0] = "soft"; opt = torch.optim.Adam(Wr + br, lr=args.lr); rng = np.random.default_rng(0)
    for s in range(args.steps):
        c = train[int(rng.integers(0, len(train)))]; tid = t.tensor([c], device=vm.dev)
        logits = vm.model(input_ids=tid).logits[0]
        nll = t.nn.functional.cross_entropy(logits[:-1].float(), tid[0, 1:])
        budget = t.stack([gate_cache[L].mean() for L in range(nL)]).mean()
        loss = nll + args.l1 * budget
        opt.zero_grad(); loss.backward(); opt.step()

    def cat_nll(routed):
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

    # baseline (no masking): detach hooks momentarily
    for h in hu + hd:
        h.remove()
    base = cat_nll(False)
    hu = [mod.register_forward_pre_hook(mk_up(L)) for L, mod in enumerate(ups)]
    hd = [mod.register_forward_pre_hook(mk_down(L)) for L, mod in enumerate(downs)]

    # ---- eval: LEARNED-ROUTER hard top-E vs STATIC mass-routing top-E ----
    def mass_gate(L_idx):                                                 # swap the up-hook to mass-routing for the static control
        pass
    curve = []
    with t.no_grad():
        for E in sorted({int(x) for x in args.E.split(",")}):
            mode[0] = "hard"; topE[0] = E
            r_learned = cat_nll(True)
            # static mass-routing at the same E (override gate with activation mass over experts)
            for hh in hu:
                hh.remove()

            def mk_up_mass(L):
                def hook(m, i):
                    xin[L] = i[0]; return None
                return hook

            def mk_down_mass(L):
                def hook(m, i):
                    h = i[0]; mass = h.abs() @ onehot[L]; idx = mass.topk(min(E, K), dim=-1).indices
                    keep_e = t.zeros_like(mass); keep_e.scatter_(-1, idx, 1.0)
                    return (h * (keep_e @ onehot[L].t()),) + tuple(i[1:])
                return hook
            hu2 = [mod.register_forward_pre_hook(mk_up_mass(L)) for L, mod in enumerate(ups)]
            for hh in hd:
                hh.remove()
            hd2 = [mod.register_forward_pre_hook(mk_down_mass(L)) for L, mod in enumerate(downs)]
            r_mass = cat_nll(True)
            for hh in hu2 + hd2:
                hh.remove()
            hu = [mod.register_forward_pre_hook(mk_up(L)) for L, mod in enumerate(ups)]
            hd = [mod.register_forward_pre_hook(mk_down(L)) for L, mod in enumerate(downs)]
            curve.append({"E": E, "frac": E / K,
                          "learned_dNLL_other": r_learned["other"] - base["other"], "learned_dNLL_dup": r_learned["dup"] - base["dup"],
                          "mass_dNLL_other": r_mass["other"] - base["other"], "mass_dNLL_dup": r_mass["dup"] - base["dup"]})
    for h in hu + hd:
        h.remove()
    return {"model": mid.split("/")[-1], "d_ff": int(d_ff), "experts": K, "l1": args.l1, "steps": args.steps,
            "baseline": base, "curve": curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-160m")
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--E", default="2,4,8,16,32")
    p.add_argument("--l1", type=float, default=0.5, help="budget penalty (push the router to use few experts)")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--train", type=int, default=120)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--steps", type=int, default=400)
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
            print(f"  d_ff {r['d_ff']} · K={r['experts']} · baseline content NLL {b['other']:.2f} · router trained {r['steps']} steps")
            print("  top-E experts → content ΔNLL:  LEARNED router  vs  static mass-routing")
            for c in r["curve"]:
                print(f"    E {c['E']:3d}/{r['experts']} ({c['frac']:.0%})  learned {c['learned_dNLL_other']:+.3f}  ·  "
                      f"mass {c['mass_dNLL_other']:+.3f}   (Δ {c['learned_dNLL_other'] - c['mass_dNLL_other']:+.3f})")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "mlp_router_summary.json"
    sumpath.write_text(json.dumps({"experiment": "learned router vs static mass-routing — conditional expert-sparsity of the MLP content",
                                   "results": results}, indent=2, default=float))
    print(f"\n[done] → {sumpath}")
    return results


if __name__ == "__main__":
    main()
