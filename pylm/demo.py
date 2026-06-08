"""Watch the decompiled LM run — generate from a seed, greedy or sampled, with the per-token instruction trace.

Every generated token is attributable to a named idiom (induction / 4-gram / trigram / …) — the decompilation is
interpretable by construction. The model never runs: this is the pure-Python `PyLM.predict` over flat files; the
tokenizer is used only to read the seed in and the ids back out for humans.

  --greedy            argmax (top ranked candidate) — tends to fall into the induction copy-loop (as the real model
                      does under argmax), which makes the keystone idiom visible.
  --temp T            sample over the ranked candidate pool with rank-temperature T (0 ≈ greedy; higher = more
                      diverse) — diverges from the loop while staying a pure-Python idiom mix.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import PyLM  # noqa: E402


def sample_rank(cands, temp, rng):
    """Sample an id from a RANKED candidate list using rank-based weights w_i ∝ exp(-i/temp) (no probs in the store)."""
    if not cands:
        return None
    if temp <= 1e-6 or len(cands) == 1:
        return cands[0]
    w = [math.exp(-i / temp) for i in range(len(cands))]
    return rng.choices(cands, weights=w, k=1)[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default="pylm/store_captured.json")
    p.add_argument("--knowledge", default=None, help="optional flat fact table (pylm/knowledge.json)")
    p.add_argument("--tokenizer", default="gpt2", help="BPE tokenizer for human-readable I/O (a flat file, not a net)")
    p.add_argument("--seed-text", default="First Citizen:\nBefore we proceed any further, hear me speak.")
    p.add_argument("--n", type=int, default=60, help="tokens to generate")
    p.add_argument("--temp", type=float, default=0.0, help="rank-sampling temperature (0 = greedy)")
    p.add_argument("--k", type=int, default=8, help="candidate pool size for sampling")
    p.add_argument("--rng", type=int, default=0)
    p.add_argument("--trace", action="store_true", help="print the per-token instruction trace")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lm = PyLM(args.store, knowledge_path=args.knowledge)
    rng = random.Random(args.rng)

    ctx = tok(args.seed_text)["input_ids"]; gen = []; trace = []
    for _ in range(args.n):
        if args.temp <= 1e-6:
            nxt = lm.predict(ctx)
        else:
            nxt = sample_rank(lm.ranked(ctx, args.k), args.temp, rng)
        trace.append(lm.predict_explain(ctx)[1])
        ctx.append(nxt); gen.append(nxt)

    mode = "greedy" if args.temp <= 1e-6 else f"sampled (temp={args.temp})"
    print(f"=== pylm {mode} — store {Path(args.store).name} ===")
    print("SEED:  " + repr(args.seed_text))
    print("PYLM:  " + repr(tok.decode(gen)))
    print("\ninstructions used:", dict(Counter(trace)))
    if args.trace:
        ab = {"induction": "ind", "trigram": "tri", "bigram": "bi", "quad": "4g", "unigram": "uni"}
        print("trace: " + " ".join(next((ab[k] + t[len(k):] for k in ab if t.startswith(k)), t) for t in trace))


if __name__ == "__main__":
    main()
