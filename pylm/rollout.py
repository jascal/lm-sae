"""Sequence-level validation — how far does the decompiled LM track the real model in free generation?

Teacher-forced top-1 (`validate.py`) measures per-position agreement. This measures GENERATION: from held-out seeds,
roll out K tokens greedily with both pylm (pure Python) and the model, and report
  - the FIDELITY HORIZON: mean tokens before pylm's greedy generation first diverges from the model's;
  - per-step agreement on the model's own greedy rollout (teacher-forced on the model's path);
  - pylm's generation vs the actual CORPUS continuation (output-vs-corpus overlap).
The model is the ground-truth ceiling (torch), not part of pylm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import PyLM  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default="pylm/store_captured.json")
    p.add_argument("--ids", default="pylm/holdout_ids.json")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ctx", type=int, default=48)
    p.add_argument("--gen", type=int, default=20, help="tokens rolled out per seed")
    p.add_argument("--seeds", type=int, default=120)
    p.add_argument("--out", type=Path, default=Path("runs/pylm/rollout_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM
    lm = PyLM(args.store)
    hold = json.loads(Path(args.ids).read_text())["holdout_ids"]
    dev = args.device if torch.cuda.is_available() else "cpu"
    big = any(s in args.model for s in ("1b", "1.4b", "-large", "-xl"))
    m = AutoModelForCausalLM.from_pretrained(args.model, **({"dtype": torch.bfloat16} if big else {})).eval().to(dev)

    starts = list(range(args.ctx, len(hold) - args.gen - 1, max(1, (len(hold) - args.ctx - args.gen) // args.seeds)))[: args.seeds]
    horizons = []; step_agree = np.zeros(args.gen); corpus_overlap = []
    with torch.no_grad():
        for st in starts:
            seed = hold[st - args.ctx:st]
            # model greedy rollout
            mctx = list(seed); mgen = []
            for _ in range(args.gen):
                nxt = int(m(input_ids=torch.tensor([mctx[-args.ctx:]], device=dev)).logits[0, -1].argmax())
                mctx.append(nxt); mgen.append(nxt)
            # pylm greedy rollout
            pctx = list(seed); pgen = []
            for _ in range(args.gen):
                nxt = lm.predict(pctx[-args.ctx:]); pctx.append(nxt); pgen.append(nxt)
            div = next((k for k in range(args.gen) if pgen[k] != mgen[k]), args.gen)
            horizons.append(div)
            # per-step agreement teacher-forced on the model's own path
            for k in range(args.gen):
                step_agree[k] += (lm.predict(mctx[:args.ctx + k][-args.ctx:]) == mgen[k])
            # pylm generation vs the actual corpus continuation
            truth = hold[st:st + args.gen]
            corpus_overlap.append(np.mean([pgen[k] == truth[k] for k in range(min(args.gen, len(truth)))]))
    n = len(starts); step_agree /= n
    res = {"model": args.model, "n_seeds": n, "gen_len": args.gen,
           "fidelity_horizon": float(np.mean(horizons)), "fidelity_horizon_median": float(np.median(horizons)),
           "stepwise_agreement": [float(x) for x in step_agree],
           "pylm_vs_corpus_overlap": float(np.mean(corpus_overlap))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, default=float))
    print(f"[rollout] {args.model}: fidelity horizon {res['fidelity_horizon']:.1f}/{args.gen} tokens "
          f"(median {res['fidelity_horizon_median']:.0f}) before pylm's greedy generation diverges from the model")
    print(f"[rollout] step agreement (on the model's path): t1 {step_agree[0]:.0%} … t{args.gen} {step_agree[-1]:.0%} "
          f"(mean {step_agree.mean():.0%}) | pylm gen vs corpus: {res['pylm_vs_corpus_overlap']:.0%}")
    print(f"[done] → {args.out}")
    return res


if __name__ == "__main__":
    main()
