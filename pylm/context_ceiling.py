"""How much of the model is exact-context MEMORIZATION vs genuine COMPOSITION? — the ∞-gram context ceiling.

pylm's flat store is a *bounded* n-gram (4-gram) + induction; it reproduces ~half of the model. This asks the
resource-relevant question behind "cracking the entangled core": **of the half pylm misses, how much is crackable by a
bigger flat store with longer context, and how much is irreducible composition?** It pushes the n-gram to *unbounded*
context — the ∞-gram / longest-suffix predictor over the full training stream — and measures the decompilable fraction
(pylm↔model top-1 agreement) as a function of the context length the flat store is allowed to use.

  For each held-out position, find the LONGEST suffix of the context (up to a cap K) that occurs anywhere in the
  training stream, and predict the token that followed its most recent occurrence (the ∞-gram, capped at K). Agreement
  with the model's top-1, as K grows, rises and SATURATES: the saturation level is the **memorization ceiling** (what a
  flat store can ever reproduce, given any context length); the gap from there to 100% is the **composition** that no
  exact-context lookup captures — the forge tax, quantified as "not crackable by more flat data."

The ∞-gram is pure flat-file lookup (the only torch is the model ceiling — the thing we measure against, not part of
the decompiler). Output: runs/pylm/context_ceiling_summary.json.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--ctx", type=int, default=64, help="context window the model sees")
    p.add_argument("--n-eval", type=int, default=1500, help="held-out positions to score")
    p.add_argument("--caps", default="1,2,3,4,6,8,12,16,24,32", help="context-length caps K for the ∞-gram")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/pylm/context_ceiling_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")
    ids = tok(text)["input_ids"]
    cut = int(len(ids) * (1 - args.holdout)); train, hold = ids[:cut], ids[cut:]

    # ---- ∞-gram via a token string + str.rfind (longest suffix present in train → its most-recent successor) ----
    train_str = " " + " ".join(map(str, train)) + " "                 # space-delimited so matches are token-aligned
    caps = sorted({int(x) for x in args.caps.split(",")})

    def infinigram_pred(ctx, cap):
        """longest suffix (length ≤ cap) of ctx occurring in train → token after its LAST occurrence (None if k=0)."""
        for k in range(min(cap, len(ctx)), 0, -1):
            needle = " " + " ".join(map(str, ctx[-k:])) + " "
            pos = train_str.rfind(needle)                             # most recent occurrence
            if pos != -1:
                after = train_str[pos + len(needle):].split(" ", 1)[0]
                if after:
                    return int(after), k
        return None, 0

    positions = list(range(args.ctx, min(len(hold), args.ctx + args.n_eval)))

    # ---- the model's top-1 at each position (the ground truth we decompile against) ----
    big = any(s in args.model for s in ("1b", "1.4b", "2.8b", "-xl", "-large"))
    m = AutoModelForCausalLM.from_pretrained(args.model, **({"dtype": torch.bfloat16} if big else {})).eval().to(dev)
    model_top1 = []; gold = []
    with torch.no_grad():
        for i in positions:
            ctx = hold[max(0, i - args.ctx):i]
            model_top1.append(int(m(input_ids=torch.tensor([ctx], device=dev)).logits[0, -1].argmax()))
            gold.append(hold[i])

    # ---- ∞-gram agreement + corpus accuracy + matched-suffix length, per cap ----
    curve = []
    for cap in caps:
        agree = corr = found = 0; klens = []
        for j, i in enumerate(positions):
            ctx = hold[max(0, i - args.ctx):i]
            pred, k = infinigram_pred(ctx, cap)
            if pred is None:
                continue
            found += 1; klens.append(k)
            agree += int(pred == model_top1[j]); corr += int(pred == gold[j])
        n = len(positions)
        curve.append({"cap": cap, "decompilable_fraction": agree / n, "corpus_top1": corr / n,
                      "coverage": found / n, "mean_matched_suffix": (sum(klens) / max(len(klens), 1))})

    model_acc = sum(int(model_top1[j] == gold[j]) for j in range(len(positions))) / len(positions)
    ceiling = curve[-1]["decompilable_fraction"]
    result = {"model": args.model, "n_eval": len(positions), "model_corpus_top1": model_acc,
              "memorization_ceiling": ceiling, "composition_residual": 1 - ceiling, "curve": curve}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\n[context-ceiling] {args.model}: model corpus top-1 {model_acc:.1%} | {len(positions)} positions")
    print("  cap K → decompilable fraction (∞-gram↔model) · corpus top-1 · coverage · mean matched-suffix:")
    for c in curve:
        print(f"    K={c['cap']:3d}  decompile {c['decompilable_fraction']:.1%}  corpus {c['corpus_top1']:.1%}  "
              f"cov {c['coverage']:.0%}  suffix {c['mean_matched_suffix']:.1f}")
    print(f"  → memorization ceiling {ceiling:.1%}; composition residual (not crackable by flat context) "
          f"{1 - ceiling:.1%}")
    print(f"[done] → {args.out}")
    return result


if __name__ == "__main__":
    main()
