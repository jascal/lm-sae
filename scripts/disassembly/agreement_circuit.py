"""Localize the AGREEMENT circuit — which attention heads carry the head-noun's number across the attractors?

`recursive_syntax.py` showed subject–verb agreement across attractors is a hierarchical dependency carried by attention
composition (ablate attention → the model follows the nearest noun). This finds the *specific* heads: a discovered
"number-mover" circuit for the catalog. Two read-only signals over depth-≥1 agreement stimuli:

  CAUSAL — ablate each head (mean) and measure the drop in the agreement logit-diff lp(correct) − lp(wrong). The heads
    whose ablation most collapses agreement (toward the attractor) are the load-bearing number-movers.
  ATTENTION — do those heads attend FROM the verb position TO the HEAD-noun position (the number-mover signature),
    rather than to the nearest attractor? Mean attention mass verb→head vs verb→attractor for the top heads.

Cross-references the catalog: are the number-movers induction / prev-token / duplicate heads, or a distinct class?
Output: runs/disassembly/agreement_circuit_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recursive_syntax import build_stimuli  # noqa: E402


def run_model(mid, args):
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok
    rng = np.random.default_rng(0)
    depths = list(range(1, args.max_depth + 1))                       # depth ≥ 1: the attractor is present
    stim = build_stimuli(tok, depths, rng)
    sents = [s for d in depths for s in stim[d]]
    if args.max_stim and len(sents) > args.max_stim:                  # subsample to keep the per-head sweep tractable
        sents = [sents[int(i)] for i in rng.choice(len(sents), args.max_stim, replace=False)]

    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + 64] for i in range(0, len(ids), 64) if len(ids[i:i + 64]) >= 8][:24]
    vm.fit_means(chunks)

    # token-id sequence + the correct/wrong verb ids + the HEAD position (token index of the head noun) per stimulus
    items = []
    for s in sents:
        ctx = tok(s["text"], add_special_tokens=False)["input_ids"]
        ci = tok(s["correct"], add_special_tokens=False)["input_ids"][0]
        wi = tok(s["wrong"], add_special_tokens=False)["input_ids"][0]
        head_tok = tok(" " + s["text"].split()[1], add_special_tokens=False)["input_ids"]   # "The {head} ..."
        hpos = next((i for i, x in enumerate(ctx) if x == head_tok[0]), 1) if head_tok else 1
        items.append({"ctx": ctx, "ci": ci, "wi": wi, "hpos": hpos})

    def agree_diff(ctxmgr=None):
        ds = []
        for it in items:
            if ctxmgr is None:
                lp = t.log_softmax(vm.logits(it["ctx"]).float(), -1)[-1]
            else:
                with ctxmgr():
                    lp = t.log_softmax(vm.logits(it["ctx"]).float(), -1)[-1]
            ds.append(float(lp[it["ci"]] - lp[it["wi"]]))
        return float(np.mean(ds))

    base = agree_diff()
    # ---- CAUSAL: per-head ablation drop in the agreement logit-diff ----
    drops = []
    for L in range(vm.nL):
        for h in range(vm.H):
            d = base - agree_diff(lambda L=L, h=h: vm.ablate_heads([(L, h)], mode="mean"))
            drops.append(((L, h), d))                                  # positive drop = head supports agreement
    drops.sort(key=lambda r: -r[1])
    top = drops[: args.top]

    # ---- ATTENTION: do the top heads attend verb→HEAD vs verb→ATTRACTOR (nearest noun)? ----
    def head_attn(L, h):
        vh = va = n = 0.0
        for it in items:
            ctx = it["ctx"]; A = vm.attn(ctx)[L][h].float().cpu().numpy()   # [q,k]
            q = len(ctx) - 1                                          # the verb is predicted at the last position
            vh += A[q, it["hpos"]]                                    # mass on the HEAD noun
            va += A[q, len(ctx) - 1]                                  # mass on the last token (nearest attractor)
            n += 1
        return vh / n, va / n
    top_rows = []
    for (L, h), d in top:
        toh, toa = head_attn(L, h)
        top_rows.append({"head": f"{L}.{h}", "agreement_drop": d, "verb_to_head": float(toh),
                         "verb_to_attractor": float(toa)})

    # ---- catalog cross-reference: are the movers induction / prev-tok / duplicate heads? ----
    probe = [(lambda s: s + s)([int(v) for v in rng.integers(0, min(2000, len(tok)), 20)]) for _ in range(12)]
    named = {}
    for op in ("induction", "prevtok", "duplicate"):
        heads, _ = vm.find_heads(probe, op, top=6)
        named[op] = {f"{L}.{h}" for (L, h) in heads}
    for r in top_rows:
        r["catalog"] = [op for op, hs in named.items() if r["head"] in hs] or ["UNNAMED"]

    return {"model": mid.split("/")[-1], "n_layers": vm.nL, "n_heads": vm.H, "n_stimuli": len(items),
            "base_agreement_diff": base, "top_movers": top_rows,
            "total_positive_drop": float(sum(d for _, d in drops if d > 0))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--max-stim", type=int, default=0, help="subsample stimuli for the per-head sweep (0 = all)")
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
            print(f"  {r['n_stimuli']} stimuli | base agreement logit-diff {r['base_agreement_diff']:+.2f}")
            print("  top number-mover heads (ablation drop in agreement · verb→head vs verb→attractor attn · catalog):")
            for row in r["top_movers"]:
                print(f"    {row['head']:>6}  drop {row['agreement_drop']:+.3f}   "
                      f"v→head {row['verb_to_head']:.2f} v→attr {row['verb_to_attractor']:.2f}   "
                      f"{','.join(row['catalog'])}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "agreement_circuit_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "localize the subject-verb agreement circuit — which attention heads move the head-noun's number",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'top_movers' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
