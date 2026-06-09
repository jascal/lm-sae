"""Router-kernel prototype — the minimal conditional-compute kernel with an INIT-COMPUTED budget.

Putting the scaling results to work (P4). The per-token active budget is architectural: B(d) ≈ 1.4·k(d), and k is the
GELU participation ratio of the MLP hidden — computable from the model itself (no training needed; #checkpoint_k showed it
is even present at random init). This prototype:
  1. measures k on the model → sets the per-token budget B = 1.4·k → expert budget E = round(B·K/m);
  2. trains the informed router (gate from [local MLP input ⊕ early-layer global/broadcast context], straight-through
     hard top-E over K static expert clusters) at that budget, base frozen;
  3. measures how well the routed model preserves the FULL model — KL(full ‖ routed) on held-out text — at active-compute
     fraction E/K, against a random-router control at the same budget.
Demonstrates: store the whole MLP, compute only ~B/m of it per token, and keep the full model's behaviour.

Output: runs/disassembly/router_kernel_summary.json.
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
    import torch.nn.functional as F
    from residual_vm import ResidualVM
    t = torch
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); tok = vm.tok; nL = vm.nL; d = vm.d
    for p in vm.model.parameters():
        p.requires_grad_(False)
    ups = up_projs(vm.model); downs = down_projs(vm.model); K = args.experts; L_ctx = max(1, nL // 4)
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")[: args.chars]
    ids = tok(text)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16]
    train, ev = chunks[: args.train], chunks[args.train: args.train + args.fit]

    onehot = []
    for L in range(nL):
        keys = up_proj_keys(vm.model, L).float(); keys = keys / (keys.norm(dim=1, keepdim=True) + 1e-9)
        lab = kmeans(keys, K, seed=L)
        oh = t.zeros(keys.shape[0], K, device=vm.dev); oh[t.arange(keys.shape[0]), lab] = 1.0
        onehot.append(oh)
    m = onehot[0].shape[0]

    # ---- 1. architectural budget: k = mean per-token PR of the MLP hidden → B = 1.4·k → E experts ----
    cap = {}; hcap = [mod.register_forward_pre_hook((lambda L: lambda mm, i: cap.__setitem__(L, i[0].detach()))(L))
                      for L, mod in enumerate(downs)]
    kk = 0.0; nn = 0
    with t.no_grad():
        for c in ev:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                h = cap[L][0].float(); kk += float(((h.abs().sum(-1) ** 2) / (h.pow(2).sum(-1) + 1e-9)).sum()); nn += h.shape[0]
    for h in hcap:
        h.remove()
    k = kk / max(nn, 1)
    B = 1.4 * k
    E = int(np.clip(round(B * K / m), 1, K))

    # ---- 2. informed router (global ctx + straight-through), trained at budget E ----
    Wr = [t.zeros(2 * d, K, device=vm.dev, requires_grad=True) for _ in range(nL)]
    br = [t.zeros(K, device=vm.dev, requires_grad=True) for _ in range(nL)]
    gate = {}; ctx = {}; route = ["learned"]
    ctx_layers = sorted({0, L_ctx})

    def mk_ctx():
        def hook(mm, i):
            ctx[0] = i[0].detach(); return None
        return hook
    hc = [vm.layers[cl].register_forward_pre_hook(mk_ctx()) for cl in ctx_layers]

    def mk_up(L):
        def hook(mm, i):
            x = i[0]
            if route[0] == "off":
                return None
            if route[0] == "random":
                g = t.randn(x.shape[0] if x.dim() == 2 else x.shape[1], K, device=x.device)
                g = g.unsqueeze(0) if x.dim() == 3 else g
            else:
                g = t.cat([x, ctx[0]], -1) @ Wr[L] + br[L]
            idx = g.topk(E, dim=-1).indices; hard = t.zeros_like(g); hard.scatter_(-1, idx, 1.0)
            gate[L] = hard + (t.softmax(g, -1) - t.softmax(g, -1).detach())   # straight-through
            return None
        return hook

    def mk_down(L):
        def hook(mm, i):
            if route[0] == "off":
                return None
            return (i[0] * (gate[L] @ onehot[L].t()),) + tuple(i[1:])
        return hook
    hu = [mod.register_forward_pre_hook(mk_up(L)) for L, mod in enumerate(ups)]
    hd = [mod.register_forward_pre_hook(mk_down(L)) for L, mod in enumerate(downs)]

    # Grok's P4 objective (hard-budget form): train the router to PRESERVE the full model — KL(full ‖ routed) at top-E=B
    route[0] = "learned"; opt = torch.optim.Adam(Wr + br, lr=args.lr); rng = np.random.default_rng(0)
    for s in range(args.steps):
        c = train[int(rng.integers(0, len(train)))]; tid = t.tensor([c], device=vm.dev)
        with t.no_grad():
            route[0] = "off"; full = t.log_softmax(vm.model(input_ids=tid).logits[0][:-1].float(), -1)
        route[0] = "learned"; routed = t.log_softmax(vm.model(input_ids=tid).logits[0][:-1].float(), -1)
        loss = (full.exp() * (full - routed)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- 3. KL(full ‖ routed) on held-out, vs a random-router control at the same budget ----
    def kl_vs_full(mode):
        tot = 0.0; n = 0
        with t.no_grad():
            for c in ev:
                tid = t.tensor([c], device=vm.dev)
                route[0] = "off"; full = t.log_softmax(vm.model(input_ids=tid).logits[0].float(), -1)
                route[0] = mode; rt = t.log_softmax(vm.model(input_ids=tid).logits[0].float(), -1)
                tot += float((full.exp() * (full - rt)).sum(-1).sum()); n += full.shape[0]
        return tot / max(n, 1)
    kl_learned = kl_vs_full("learned"); kl_random = kl_vs_full("random")
    for h in hc + hu + hd:
        h.remove()
    return {"model": mid.split("/")[-1], "d": d, "m": int(m), "experts_K": K, "k": k, "budget_B": B,
            "expert_budget_E": E, "active_fraction": E / K, "kl_learned_router": kl_learned, "kl_random_router": kl_random}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,EleutherAI/pythia-160m")
    p.add_argument("--experts", type=int, default=64)
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--train", type=int, default=150)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--chars", type=int, default=240000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/disassembly/router_kernel_summary.json"))
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
            print(f"  k={r['k']:.0f} → budget B=1.4k={r['budget_B']:.0f} → E={r['expert_budget_E']}/{r['experts_K']} "
                  f"({r['active_fraction']:.0%} active MLP)")
            print(f"  KL(full ‖ routed):  LEARNED router {r['kl_learned_router']:.3f}  vs  RANDOM router "
                  f"{r['kl_random_router']:.3f}  ({r['kl_random_router'] / max(r['kl_learned_router'], 1e-6):.1f}× worse)")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "router-kernel prototype: init-computed budget B=1.4k, KL preservation of the full model", "results": results}, indent=2, default=float))
    print(f"\n[done] → {args.out}")
    return results


if __name__ == "__main__":
    main()
