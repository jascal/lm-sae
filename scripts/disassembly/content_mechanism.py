"""What does the entangled core (diffuse CONTENT composition, per core_residual.py) actually COMPUTE — retrieval or stored?

core_residual.py located the incompressible forge tax on generic *content* ("other") next-tokens, not the structural
circuits. This asks the mechanism question (the "explain how we think" half of the north star): is that content
prediction **in-context retrieval** (carried by ATTENTION — soft, context-dependent, decompilable as extended lookup) or
**stored computation** (carried by the MLPs — parametric knowledge)? And how much of it is **context-bound** (needs the
preceding tokens) vs context-free?

Two read-only ablations, ΔNLL split by next-token category (punct / dup / other):
  - mean-ablate ALL attention heads  → if "other" collapses, content is attention-mediated (in-context / retrieval);
  - mean-ablate ALL MLP blocks       → if "other" collapses, content is MLP-stored (parametric compute);
plus a CONTEXT-TRUNCATION sweep (score with only the last k tokens visible) → how context-bound the content is.

Output: runs/disassembly/content_mechanism_summary.json.
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


def run_model(mid, args):
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); t = vm.torch; tok = vm.tok; nL = vm.nL
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")[: args.chars]
    ids = tok(text)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16][: args.fit]
    vm.fit_means(chunks)

    vocab = vm.model.config.vocab_size
    punct_chars = set('.,;:!?"\'()-\n—’“”…`')
    punct_id = np.zeros(vocab, bool)
    for v in range(vocab):
        s = tok.decode([v]).strip()
        if s != "" and all(ch in punct_chars for ch in s):
            punct_id[v] = True

    def cat_nll(forward):
        tot = {"punct": 0.0, "dup": 0.0, "other": 0.0}; cnt = {"punct": 0, "dup": 0, "other": 0}
        with t.no_grad():
            for c in chunks:
                lp = t.log_softmax(forward(c).float(), -1)
                seen = set()
                for p in range(len(c) - 1):
                    seen.add(c[p]); nxt = c[p + 1]
                    cat = "punct" if punct_id[nxt] else ("dup" if nxt in seen else "other")
                    tot[cat] += -float(lp[p, nxt]); cnt[cat] += 1
        return {k: tot[k] / max(cnt[k], 1) for k in tot}

    def plain(c):
        return vm.model(input_ids=t.tensor([c], device=vm.dev)).logits[0]

    base = cat_nll(plain)
    all_heads = [(L, h) for L in range(nL) for h in range(vm.H)]
    with vm.ablate_heads(all_heads):
        no_attn = cat_nll(plain)
    with vm.ablate_mlps(list(range(nL))):
        no_mlp = cat_nll(plain)

    # context-truncation sweep: score each position seeing only the last k tokens (full-attention models)
    ctx_curve = []
    for k in [int(x) for x in args.ctx_window.split(",")]:
        tot = {"punct": 0.0, "dup": 0.0, "other": 0.0}; cnt = {"punct": 0, "dup": 0, "other": 0}
        with t.no_grad():
            for c in chunks:
                seen = set()
                for p in range(len(c) - 1):
                    seen.add(c[p]); nxt = c[p + 1]
                    win = c[max(0, p + 1 - k):p + 1]
                    lp = t.log_softmax(vm.model(input_ids=t.tensor([win], device=vm.dev)).logits[0, -1].float(), -1)
                    cat = "punct" if punct_id[nxt] else ("dup" if nxt in seen else "other")
                    tot[cat] += -float(lp[nxt]); cnt[cat] += 1
        ctx_curve.append({"window": k, **{f"nll_{kk}": tot[kk] / max(cnt[kk], 1) for kk in tot}})

    return {"model": mid.split("/")[-1], "baseline": base,
            "ablate_attention": {k: no_attn[k] - base[k] for k in base},
            "ablate_mlp": {k: no_mlp[k] - base[k] for k in base},
            "context_truncation": ctx_curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-160m")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=60)
    p.add_argument("--chars", type=int, default=160000)
    p.add_argument("--ctx-window", default="1,2,4,8,16,64", help="context-truncation windows to score")
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
            print(f"  baseline NLL punct {b['punct']:.2f} · dup {b['dup']:.2f} · other {b['other']:.2f}")
            print(f"  ablate ATTENTION ΔNLL: punct {r['ablate_attention']['punct']:+.2f} · "
                  f"dup {r['ablate_attention']['dup']:+.2f} · other {r['ablate_attention']['other']:+.2f}")
            print(f"  ablate MLP       ΔNLL: punct {r['ablate_mlp']['punct']:+.2f} · "
                  f"dup {r['ablate_mlp']['dup']:+.2f} · other {r['ablate_mlp']['other']:+.2f}")
            print("  context-truncation (NLL on OTHER content vs window):")
            print("    " + " · ".join(f"k{c['window']}:{c['nll_other']:.2f}" for c in r["context_truncation"]))
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "content_mechanism_summary.json"
    sumpath.write_text(json.dumps({"experiment": "is content composition attention-retrieval or MLP-stored, and how context-bound",
                                   "results": results}, indent=2, default=float))
    print(f"\n[done] → {sumpath}")
    return results


if __name__ == "__main__":
    main()
