"""Head-level causal scan for MISSED circuit classes — does the catalog miss head-level circuits the channel
decomposition (`circuit_channels.py`) only saw correlationally?

The channels were correlationally interpretable (boundary / grammar / induction) but single residual-direction ablation
was diffuse — causal circuits live at the COMPONENT level. So this scans there: mean-ablate each attention head and split
the change in next-token NLL by category — PUNCTUATION (clause-boundary), DUPLICATE (induction family), OTHER. A head is
a candidate circuit for a category if its ablation **specifically** raises that category's NLL (targeted, not diffuse).
And a head whose ablation *lowers* a category's NLL was SUPPRESSING it — the signature of **copy-suppression** (a known
class, anti-induction) when it lowers DUPLICATE. This is the catalog methodology (causal, head-level) applied to the
behaviours the channels surfaced, to see which are real, missed circuits — and whether any are known classes.

Output: runs/disassembly/head_category_scan_summary.json.
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
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); t = vm.torch; tok = vm.tok; nL = vm.nL; H = vm.H
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")[: args.chars]
    ids = tok(text)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16][: args.fit]
    vm.fit_means(chunks)                                                          # mean-ablation reference

    PUNCT = set('.,;:!?"\'()-\n—’“”…`')
    vocab = vm.model.config.vocab_size
    punct_id = np.zeros(vocab, bool)
    for v in range(vocab):
        s = tok.decode([v]).strip()
        if s != "" and all(ch in PUNCT for ch in s):
            punct_id[v] = True

    def cat_nll():                                                                # mean NLL on punct / dup / other next-tokens
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

    base = cat_nll()
    rows = []
    for L in range(nL):
        for h in range(H):
            with vm.ablate_heads([(L, h)]):
                ab = cat_nll()
            rows.append({"L": L, "h": h, "d_punct": ab["punct"] - base["punct"],
                         "d_dup": ab["dup"] - base["dup"], "d_other": ab["other"] - base["other"]})
    # targeted score: how much a head's effect concentrates on one category vs the diffuse "other"
    for r in rows:
        r["punct_targeted"] = round(r["d_punct"] - r["d_other"], 3)              # >0 ⇒ punct-specific (boundary circuit)
        r["dup_targeted"] = round(r["d_dup"] - r["d_other"], 3)                  # >0 ⇒ dup-specific (induction)
        r["d_punct"] = round(r["d_punct"], 3); r["d_dup"] = round(r["d_dup"], 3); r["d_other"] = round(r["d_other"], 3)
    top_punct = sorted(rows, key=lambda r: -r["punct_targeted"])[:6]
    top_dup = sorted(rows, key=lambda r: -r["dup_targeted"])[:6]
    copy_suppress = sorted(rows, key=lambda r: r["d_dup"])[:6]                   # most NEGATIVE d_dup = suppressing copies
    return {"model": mid.split("/")[-1], "n_layers": nL, "heads": H, "baseline": base,
            "top_punct_heads": top_punct, "top_dup_heads": top_dup, "copy_suppression_heads": copy_suppress}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-160m")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=40, help="corpus chunks (kept small — nL*H ablations × eval)")
    p.add_argument("--chars", type=int, default=120000)
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
            print("  PUNCT-specific heads (boundary circuit? — ΔNLL punct beyond diffuse):")
            for x in r["top_punct_heads"]:
                print(f"    L{x['L']}.H{x['h']}  punct {x['d_punct']:+.2f} (targeted {x['punct_targeted']:+.2f}) · dup {x['d_dup']:+.2f} · other {x['d_other']:+.2f}")
            print("  DUP-specific heads (induction family):")
            for x in r["top_dup_heads"]:
                print(f"    L{x['L']}.H{x['h']}  dup {x['d_dup']:+.2f} (targeted {x['dup_targeted']:+.2f}) · punct {x['d_punct']:+.2f} · other {x['d_other']:+.2f}")
            print("  COPY-SUPPRESSION candidates (ablation LOWERS dup-NLL):")
            for x in r["copy_suppression_heads"]:
                print(f"    L{x['L']}.H{x['h']}  dup {x['d_dup']:+.2f} · punct {x['d_punct']:+.2f} · other {x['d_other']:+.2f}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "head_category_scan_summary.json"
    sumpath.write_text(json.dumps({"experiment": "head-level causal scan by next-token category (boundary / induction / copy-suppression)",
                                   "results": results}, indent=2, default=float))
    print(f"\n[done] → {sumpath}")
    return results


if __name__ == "__main__":
    main()
