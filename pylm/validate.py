"""Validate the decompiled pylm against the corpus AND against the real model — the decompilable fraction, literal.

Runs pylm (pure Python) over the held-out corpus token stream and measures top-1 next-token accuracy. Then runs the
*real* model (the only place torch appears — the ground-truth ceiling we're validating against, NOT part of pylm) on
the same stream. Reports:
  - pylm top-1 accuracy vs the corpus (does the program predict the actual next token?);
  - the model's top-1 accuracy vs the corpus (the ceiling);
  - pylm↔model agreement (how often the program's top-1 == the model's top-1) — the decompilable fraction;
  - the per-instruction breakdown (induction / trigram / bigram / unigram: how often each fired and its accuracy);
  - the code-size vs data-size split (the small-code bias made explicit).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import PyLM, program_loc  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=Path("pylm/store.json"))
    p.add_argument("--ids", type=Path, default=Path("pylm/holdout_ids.json"))
    p.add_argument("--ctx", type=int, default=64, help="context window pylm/model see")
    p.add_argument("--n-eval", type=int, default=4000, help="held-out positions to score")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-model", action="store_true", help="skip the torch ceiling (corpus-only validation)")
    p.add_argument("--out", type=Path, default=Path("runs/pylm/validate_summary.json"))
    args = p.parse_args(argv)

    lm = PyLM(args.store)
    hold = json.loads(args.ids.read_text())["holdout_ids"]
    positions = list(range(args.ctx, min(len(hold), args.ctx + args.n_eval)))

    # ---- pylm (pure Python) ----
    pylm_correct = 0; fired = Counter(); fired_correct = Counter(); preds = []
    for i in positions:
        ctx = hold[max(0, i - args.ctx):i]
        pred, instr = lm.predict_explain(ctx)
        preds.append(pred)
        ok = (pred == hold[i]); pylm_correct += ok
        fired[instr] += 1; fired_correct[instr] += ok
    n = len(positions)
    pylm_acc = pylm_correct / n
    breakdown = {k: {"fired": fired[k], "share": fired[k] / n, "accuracy": fired_correct[k] / max(fired[k], 1)}
                 for k in sorted(fired)}

    result = {"model": args.model, "n_eval": n, "pylm_corpus_top1": pylm_acc, "instruction_breakdown": breakdown,
              "program_loc": program_loc(), "store_kb": args.store.stat().st_size / 1024}

    # ---- the real model (torch — ground-truth ceiling, NOT part of pylm) ----
    if not args.no_model:
        import torch
        from transformers import AutoModelForCausalLM
        dev = args.device if torch.cuda.is_available() else "cpu"
        m = AutoModelForCausalLM.from_pretrained(args.model).eval().to(dev)
        model_correct = 0; agree = 0
        with torch.no_grad():
            for j, i in enumerate(positions):
                ctx = hold[max(0, i - args.ctx):i]
                mp = int(m(input_ids=torch.tensor([ctx], device=dev)).logits[0, -1].argmax())
                model_correct += (mp == hold[i]); agree += (preds[j] == mp)
        result["model_corpus_top1"] = model_correct / n
        result["pylm_model_agreement"] = agree / n
        result["decompilable_fraction"] = (agree / n)
        result["pylm_over_model_acc"] = pylm_acc / (model_correct / n + 1e-9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\n[pylm] program: {result['program_loc']} LOC of code | store: {result['store_kb']:.0f} KB of flat data")
    print(f"[pylm] corpus top-1 next-token accuracy: {pylm_acc:.1%}  (over {n} held-out positions)")
    if not args.no_model:
        print(f"[model] {args.model} corpus top-1: {result['model_corpus_top1']:.1%}  → "
              f"pylm reproduces {result['pylm_over_model_acc']:.0%} of the model's accuracy")
        print(f"[decompile] pylm↔model top-1 AGREEMENT: {result['pylm_model_agreement']:.1%} "
              f"(the decompilable fraction — how often the pure-Python program picks the model's token)")
    print("[instructions] " + " · ".join(f"{k} {v['share']:.0%}@{v['accuracy']:.0%}" for k, v in breakdown.items()))
    print(f"[done] → {args.out}")
    return result


if __name__ == "__main__":
    main()
