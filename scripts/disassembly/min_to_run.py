"""Minimum-to-run frontier — fidelity vs stored size as the transformer weights are low-rank factorized.

The flat pylm program reproduces ~half the model at ~0 matmul; the rest needs compute. This measures the other end of
that frontier: how small can the model's STORED weights get (low-rank factorization of every attention + MLP weight
matrix) while still reproducing the full model, on a fidelity-vs-size curve. No-retrain baseline first (SVD-truncate the
weights to rank r); distillation can push it later. Embeddings/unembedding (the vocab/flat-knowledge) are left intact —
this measures the COMPOSITION weights' compressibility, the part that isn't flat lookup.

Per rank r: SVD-truncate c_attn / attn.c_proj / mlp.c_fc / mlp.c_proj of every layer to rank r, then measure
  FIDELITY  generic next-token NLL, and top-1 AGREEMENT with the unmodified model (the same metric as pylm's
            decompilable fraction — what fraction of the model's tokens the compressed model still predicts);
  SIZE      stored params of the factored transformer weights (r·(m+n) per matrix) vs the full m·n, as a ratio.
GPT-2 only (the anchor; nn.Conv1D weights). Output: runs/disassembly/min_to_run_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_model(mid, args):
    import torch
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:args.corpus_chars]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8][: args.eval]

    # the composition weight matrices (GPT-2 Conv1D: weight shape (in, out))
    mats = []
    for L in range(nL):
        blk = vm.model.transformer.h[L]
        for mod in (blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj):
            mats.append(mod.weight)
    full_params = int(sum(W.shape[0] * W.shape[1] for W in mats))
    orig = [W.detach().clone() for W in mats]
    # precompute SVD of each matrix once
    svds = []
    for W in mats:
        U, S, Vh = torch.linalg.svd(W.detach().float(), full_matrices=False)
        svds.append((U, S, Vh))

    def gen(metric_top1=None):
        tot = 0.0; k = 0; agree = 0; preds = []
        with t.no_grad():
            for ci, c in enumerate(chunks):
                lg = vm.logits(c).float(); lp = t.log_softmax(lg, -1); y = c[1:]
                top1 = lg[:-1].argmax(-1)
                preds.append(top1.cpu())
                for p in range(len(y)):
                    tot += float(-lp[p, y[p]]); k += 1
                if metric_top1 is not None:
                    agree += int((top1 == metric_top1[ci].to(top1.device)).sum())
        return tot / max(k, 1), preds, (agree / max(k, 1) if metric_top1 is not None else None)

    full_nll, full_top1, _ = gen()

    def set_rank(r):
        for (W, (U, S, Vh)) in zip(mats, svds):
            Wr = (U[:, :r] * S[:r]) @ Vh[:r]
            W.data.copy_(Wr.to(W.dtype))

    def restore():
        for W, o in zip(mats, orig):
            W.data.copy_(o)

    # held-out train chunks for distillation (disjoint from eval)
    allc = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    train = allc[args.eval: args.eval + args.train]

    def fit_distill(r, steps):
        """factor every weight W≈A·B at rank r (SVD init), TRAIN the factors to match the full model (others frozen)."""
        for p in vm.model.parameters():
            p.requires_grad_(False)
        A = []; B = []
        for (W, (U, S, Vh)) in zip(mats, svds):
            s = S[:r].sqrt()
            A.append((U[:, :r] * s).detach().clone().requires_grad_(True))
            B.append((s[:, None] * Vh[:r]).detach().clone().requires_grad_(True))
        mods = []
        for L in range(nL):
            blk = vm.model.transformer.h[L]
            mods += [blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj]
        factor_on = [True]; hs = []

        def mk(j):
            def hook(m, i, o):
                return i[0] @ (A[j] @ B[j]).to(i[0].dtype) + m.bias if factor_on[0] else None   # off → teacher
            return hook
        for j, mod in enumerate(mods):
            hs.append(mod.register_forward_hook(mk(j)))
        opt = torch.optim.Adam(A + B, lr=args.lr); rng = np.random.default_rng(0); T = 2.0
        for s in range(steps):
            j = int(rng.integers(0, len(train))); tid = t.tensor([train[j]], device=vm.dev)
            if args.match_teacher:                                    # soft-KL distillation to the full model
                factor_on[0] = False
                with t.no_grad():
                    teach = t.log_softmax(vm.model(input_ids=tid).logits[0][:-1].float() / T, -1)
                factor_on[0] = True
                student = t.log_softmax(vm.model(input_ids=tid).logits[0][:-1].float() / T, -1)
                loss = t.nn.functional.kl_div(student, teach, log_target=True, reduction="batchmean") * (T * T)
            else:                                                     # corpus NLL (capable, not faithful)
                logits = vm.model(input_ids=tid).logits[0]
                loss = t.nn.functional.cross_entropy(logits[:-1].float(), tid[0, 1:])
            opt.zero_grad(); loss.backward(); opt.step()
        factor_on[0] = True
        nll, _, agree = gen(metric_top1=full_top1)
        for h in hs:
            h.remove()
        return nll, agree

    ranks = sorted({int(x) for x in args.ranks.split(",")})
    curve = []
    for r in ranks:
        set_rank(r); nll0, _, agree0 = gen(metric_top1=full_top1); restore()   # no-retrain SVD baseline
        nll_d, agree_d = (fit_distill(r, args.distill_steps) if args.distill_steps > 0 else (None, None))
        stored = int(sum(r * (W.shape[0] + W.shape[1]) for W in mats))
        curve.append({"rank": r, "svd_nll_increase": nll0 - full_nll, "svd_agreement": agree0,
                      "distilled_nll_increase": (nll_d - full_nll) if nll_d is not None else None,
                      "distilled_agreement": agree_d, "stored_params": stored,
                      "compression_ratio": stored / full_params, "params_saved_frac": 1 - stored / full_params})
    return {"model": mid.split("/")[-1], "n_layers": nL, "full_nll": full_nll,
            "transformer_weight_params_M": full_params / 1e6, "distill_steps": args.distill_steps, "curve": curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--eval", type=int, default=24)
    p.add_argument("--ranks", default="8,16,32,64,128,256,512", help="SVD ranks to truncate the weight matrices to")
    p.add_argument("--distill-steps", type=int, default=0, help="if >0, also distill the low-rank factors (train them)")
    p.add_argument("--train", type=int, default=200, help="train chunks for distillation")
    p.add_argument("--corpus-chars", type=int, default=120000, help="chars of corpus to fetch (more = more distill data)")
    p.add_argument("--lr", type=float, default=1e-3, help="distillation learning rate")
    p.add_argument("--match-teacher", action="store_true", help="distill to the full model's top-1 (faithful) vs corpus NLL (capable)")
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
            print(f"  transformer-weight params {r['transformer_weight_params_M']:.0f}M | full NLL {r['full_nll']:.3f} "
                  f"| distill steps {r['distill_steps']}")
            print("  rank → stored% · no-retrain SVD (ΔNLL/agree) · DISTILLED (ΔNLL/agree):")
            for c in r["curve"]:
                dd = (f"distilled ΔNLL {c['distilled_nll_increase']:+.3f} agree {c['distilled_agreement']:.0%}"
                      if c['distilled_nll_increase'] is not None else "distilled --")
                print(f"    rank {c['rank']:4d}  stored {c['compression_ratio']:.0%} ({c['params_saved_frac']:.0%} saved)  "
                      f"| SVD ΔNLL {c['svd_nll_increase']:+.2f} agree {c['svd_agreement']:.0%}  | {dd}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "min_to_run_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "minimum-to-run frontier — fidelity vs stored size under no-retrain low-rank weight factorization", "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'curve' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
