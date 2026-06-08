"""The crux of unsquirrelling: is the MLP content computation a SPARSE KEY-VALUE LOOKUP (flat-decompilable, the Θ(size)
flat-knowledge term) or DENSE distributed composition (the irreducible forge tax)?

`content_mechanism.py` showed the entangled core is MLP-carried context-conditioned content prediction. MLPs read as
key-value memories (Geva et al.): hidden_i = act(x·k_i), out = Σ_i hidden_i · v_i — each neuron a (key, value) pair. If
only a FEW neurons fire per token (and recover the prediction), the MLP is a sparse lookup → the content is flat storage,
just large. If it needs MOST of the hidden width, it is dense composition → genuinely computed, the forge tax stays.

Test: at every layer, mask the post-activation MLP hidden to its **top-k neurons per token** (zero the rest — GELU
already ≈0 for inactive ones), and measure next-token NLL by category (punct / dup / **other = content**) vs k. The k at
which content recovers, as a fraction of the hidden width d_ff, is the effective key-value SPARSITY of the content store.

Output: runs/disassembly/mlp_kv_sparsity_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def down_projs(model):
    """the MLP down-projection per layer (its INPUT is the post-activation d_ff hidden = the key-value coefficients)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):           # GPT-2
        return [b.mlp.c_proj for b in model.transformer.h]
    if hasattr(model, "gpt_neox"):                                                  # Pythia / GPT-NeoX
        return [ly.mlp.dense_4h_to_h for ly in model.gpt_neox.layers]
    if hasattr(model, "model") and hasattr(model.model, "layers"):                  # Llama / Qwen
        return [ly.mlp.down_proj for ly in model.model.layers]
    raise SystemExit("unknown architecture for MLP down-projection")


def run_model(mid, args):
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); t = vm.torch; tok = vm.tok
    downs = down_projs(vm.model); d_ff = downs[0].weight.shape[1] if not vm.is_gpt2 else downs[0].weight.shape[0]
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

    keep_k = [None]                                                                  # None = full (no mask)

    def mk(mod):
        def hook(m, i):                                                              # pre-hook on down-proj: mask its input hidden
            if keep_k[0] is None:
                return None
            h = i[0]; k = min(keep_k[0], h.shape[-1])
            idx = h.abs().topk(k, dim=-1).indices
            mask = t.zeros_like(h); mask.scatter_(-1, idx, 1.0)
            return (h * mask,) + tuple(i[1:])
        return hook
    hs = [mod.register_forward_pre_hook(mk(mod)) for mod in downs]

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

    keep_k[0] = None; base = cat_nll()
    curve = []
    for k in sorted({int(x) for x in args.ks.split(",")}):
        keep_k[0] = k; r = cat_nll()
        curve.append({"k": k, "frac_dff": k / d_ff, "dNLL_punct": r["punct"] - base["punct"],
                      "dNLL_dup": r["dup"] - base["dup"], "dNLL_other": r["other"] - base["other"]})
    for h in hs:
        h.remove()
    return {"model": mid.split("/")[-1], "d_ff": int(d_ff), "baseline": base, "curve": curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-160m")
    p.add_argument("--ks", default="4,8,16,32,64,128,256,512,1024", help="top-k neurons per token to keep")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=60)
    p.add_argument("--chars", type=int, default=160000)
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
            print(f"  d_ff {r['d_ff']} · baseline NLL punct {b['punct']:.2f} · dup {b['dup']:.2f} · other {b['other']:.2f}")
            print("  top-k MLP neurons kept → ΔNLL (other=content is the crux):")
            for c in r["curve"]:
                print(f"    k {c['k']:5d} ({c['frac_dff']:.1%} of d_ff)  other {c['dNLL_other']:+.3f} · "
                      f"dup {c['dNLL_dup']:+.3f} · punct {c['dNLL_punct']:+.3f}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "mlp_kv_sparsity_summary.json"
    sumpath.write_text(json.dumps({"experiment": "MLP content as sparse KV lookup vs dense composition — top-k neuron recovery by category",
                                   "results": results}, indent=2, default=float))
    print(f"\n[done] → {sumpath}")
    return results


if __name__ == "__main__":
    main()
