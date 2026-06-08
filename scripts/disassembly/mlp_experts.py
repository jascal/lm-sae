"""Runtime (conditional) compression of the dense MLP content — is it expert-sparse even though it's neuron-dense?

Storage tests (low-rank, native top-k, L1-SAE) don't sparsify the MLP content — we don't yet know how to decompose it as
*few features*. But a different axis is runtime/conditional sparsity (MoEfication, Zhang et al.): cluster the d_ff neurons
into K **experts**, and per token only a subset of experts carry the mass. You still *store* every expert (Θ(size)), but
at runtime you only *compute* the active ones — and if the active-expert set per token is small, that is a usable level.

Test: cluster each layer's MLP neurons into K experts (k-means on the up-projection rows = the neurons' input "keys", so
co-activating neurons group together — the standard MoEfication parameter clustering). Then mask the post-activation hidden
to the **top-E experts per token** (by expert activation mass), and measure next-token NLL by category (punct / dup /
**other = content**) vs E. The E/K at which content recovers — and the resulting active-NEURON count — is the runtime
expert-sparsity, to compare against the ~68%-of-neurons the magnitude top-k needed.

Output: runs/disassembly/mlp_experts_summary.json.
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


def up_proj_keys(model, L):
    """per-neuron input 'key' vectors (d_ff, d) for layer L's MLP — co-activating neurons have similar keys."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):           # GPT-2 c_fc Conv1D (d, d_ff)
        return model.transformer.h[L].mlp.c_fc.weight.detach().t()
    if hasattr(model, "gpt_neox"):                                                  # NeoX dense_h_to_4h Linear (d_ff, d)
        return model.gpt_neox.layers[L].mlp.dense_h_to_4h.weight.detach()
    if hasattr(model, "model") and hasattr(model.model, "layers"):                  # Llama up_proj Linear (d_ff, d)
        return model.model.layers[L].mlp.up_proj.weight.detach()
    raise SystemExit("unknown architecture for MLP up-projection")


def kmeans(X, K, iters=25, seed=0):
    """simple Lloyd k-means on rows of X (torch), returns integer labels (len = X.shape[0])."""
    import torch
    t = torch
    n = X.shape[0]; g = np.random.default_rng(seed)
    cen = X[t.tensor(g.choice(n, K, replace=False), device=X.device)].clone()
    lab = t.zeros(n, dtype=t.long, device=X.device)
    for _ in range(iters):
        d2 = (X.pow(2).sum(1, keepdim=True) - 2 * X @ cen.t() + cen.pow(2).sum(1))   # (n,K) sq-dist
        lab = d2.argmin(1)
        for k in range(K):
            m = lab == k
            if m.any():
                cen[k] = X[m].mean(0)
    return lab


def run_model(mid, args):
    import torch
    from residual_vm import ResidualVM
    t = torch
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); tok = vm.tok; nL = vm.nL
    downs = down_projs(vm.model)
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")[: args.chars]
    ids = tok(text)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16][: args.fit]

    vocab = vm.model.config.vocab_size
    punct_chars = set('.,;:!?"\'()-\n—’“”…`')
    punct_id = np.zeros(vocab, bool)
    for v in range(vocab):
        s = tok.decode([v]).strip()
        if s != "" and all(ch in punct_chars for ch in s):
            punct_id[v] = True

    K = args.experts
    onehot = []                                                                      # (d_ff, K) expert membership per layer
    sizes = []
    for L in range(nL):
        keys = up_proj_keys(vm.model, L).float()
        keys = keys / (keys.norm(dim=1, keepdim=True) + 1e-9)                         # cluster by key DIRECTION
        lab = kmeans(keys, K, seed=L)
        oh = t.zeros(keys.shape[0], K, device=vm.dev); oh[t.arange(keys.shape[0]), lab] = 1.0
        onehot.append(oh); sizes.append([int((lab == k).sum()) for k in range(K)])

    keepE = [None]                                                                    # None = full; else top-E experts/token

    def mk(L):
        oh = onehot[L]
        def hook(m, i):
            if keepE[0] is None:
                return None
            h = i[0]                                                                  # (.., d_ff) post-activation hidden
            mass = h.abs() @ oh                                                       # (.., K) expert activation mass
            E = min(keepE[0], mass.shape[-1])
            topE = mass.topk(E, dim=-1).indices
            keep_e = t.zeros_like(mass); keep_e.scatter_(-1, topE, 1.0)               # (..,K) experts kept
            keep_n = keep_e @ oh.t()                                                  # (..,d_ff) neurons in kept experts
            return (h * keep_n,) + tuple(i[1:])
        return hook
    hs = [mod.register_forward_pre_hook(mk(L)) for L, mod in enumerate(downs)]

    def cat_nll():
        tot = {"punct": 0.0, "dup": 0.0, "other": 0.0}; cnt = {"punct": 0, "dup": 0, "other": 0}
        with t.no_grad():
            for c in chunks:
                lp = t.log_softmax(vm.model(input_ids=t.tensor([c], device=vm.dev)).logits[0].float(), -1)
                seen = set()
                for p in range(len(c) - 1):
                    seen.add(c[p]); nxt = c[p + 1]
                    cat = "punct" if punct_id[nxt] else ("dup" if nxt in seen else "other")
                    tot[cat] += -float(lp[p, nxt]); cnt[cat] += 1
        return {k: tot[k] / max(cnt[k], 1) for k in tot}

    keepE[0] = None; base = cat_nll()
    d_ff = onehot[0].shape[0]; avg_expert = d_ff / K
    curve = []
    for E in sorted({int(x) for x in args.E.split(",")}):
        keepE[0] = E; r = cat_nll()
        curve.append({"E": E, "experts_frac": E / K, "active_neurons_approx": int(E * avg_expert),
                      "active_neuron_frac": E * avg_expert / d_ff,
                      "dNLL_other": r["other"] - base["other"], "dNLL_dup": r["dup"] - base["dup"],
                      "dNLL_punct": r["punct"] - base["punct"]})
    for h in hs:
        h.remove()
    return {"model": mid.split("/")[-1], "d_ff": int(d_ff), "experts": K, "baseline": base,
            "expert_sizes_layer0": sizes[0], "curve": curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-160m")
    p.add_argument("--experts", type=int, default=64, help="K experts per MLP (clustered from up-proj keys)")
    p.add_argument("--E", default="1,2,4,8,16,32", help="top-E experts per token to keep")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=50)
    p.add_argument("--chars", type=int, default=140000)
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
            print(f"  d_ff {r['d_ff']} · K={r['experts']} experts (avg size {r['d_ff'] // r['experts']}) · "
                  f"baseline content NLL {b['other']:.2f}")
            print("  top-E experts kept → content recovery (vs the ~68%-of-NEURONS magnitude top-k needed):")
            for c in r["curve"]:
                print(f"    E {c['E']:3d}/{r['experts']} ({c['experts_frac']:.0%} experts ≈ {c['active_neuron_frac']:.0%} neurons)  "
                      f"content {c['dNLL_other']:+.3f} · dup {c['dNLL_dup']:+.3f} · punct {c['dNLL_punct']:+.3f}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "mlp_experts_summary.json"
    sumpath.write_text(json.dumps({"experiment": "runtime expert-sparsity of the MLP content (MoEfication) — top-E experts vs content recovery",
                                   "results": results}, indent=2, default=float))
    print(f"\n[done] → {sumpath}")
    return results


if __name__ == "__main__":
    main()
