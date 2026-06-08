"""Build pylm's flat-file knowledge store from a corpus — the 'database' half of the decompiled LM.

Extracts the n-gram successor tables (trigram → bigram → unigram) over the *target model's* token ids, so pylm
predicts in the same token space the model does (the tokenizer is a flat-file BPE — knowledge, not neural net).
Pruned + top-k so the flat file stays bounded. This is DATA (allowed to be large-ish, flat); the LM PROGRAM
(`lm.py`) stays small.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2", help="tokenizer to share with the target model (flat-file BPE)")
    p.add_argument("--holdout", type=float, default=0.1, help="tail fraction reserved for validation (not in the store)")
    p.add_argument("--tri-topk", type=int, default=3)
    p.add_argument("--bi-topk", type=int, default=8)
    p.add_argument("--uni-topk", type=int, default=64)
    p.add_argument("--tri-min", type=int, default=2, help="min trigram count to keep (prune the tail)")
    p.add_argument("--min-induction", type=int, default=3, help="longest local n-gram the induction macro matches")
    p.add_argument("--out", type=Path, default=Path("pylm/store.json"))
    p.add_argument("--ids-out", type=Path, default=Path("pylm/holdout_ids.json"))
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")
    ids = tok(text)["input_ids"]
    cut = int(len(ids) * (1 - args.holdout))
    train, hold = ids[:cut], ids[cut:]

    quad = defaultdict(Counter); tri = defaultdict(Counter); bi = defaultdict(Counter); uni = Counter()
    for i, c in enumerate(train):
        uni[c] += 1
        if i >= 1:
            bi[train[i - 1]][c] += 1
        if i >= 2:
            tri[(train[i - 2], train[i - 1])][c] += 1
        if i >= 3:
            quad[(train[i - 3], train[i - 2], train[i - 1])][c] += 1

    store = {
        "model": args.model, "min_induction_match": args.min_induction, "min_induction_accept": 2,
        "n_train_tokens": len(train), "source": "corpus",
        "quad": {f"{a},{b},{c}": [d for d, _ in cnt.most_common(args.tri_topk)]
                 for (a, b, c), cnt in quad.items() if cnt.total() >= args.tri_min},
        "tri": {f"{a},{b}": [c for c, _ in cnt.most_common(args.tri_topk)]
                for (a, b), cnt in tri.items() if cnt.total() >= args.tri_min},
        "bi": {str(b): [c for c, _ in cnt.most_common(args.bi_topk)] for b, cnt in bi.items()},
        "uni": [c for c, _ in uni.most_common(args.uni_topk)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(store))
    args.ids_out.write_text(json.dumps({"model": args.model, "holdout_ids": hold}))
    kb = args.out.stat().st_size / 1024
    print(f"[build] {args.model}: {len(train)} train / {len(hold)} holdout tokens | "
          f"tri {len(store['tri'])} / bi {len(store['bi'])} / uni {len(store['uni'])} | store {kb:.0f} KB → {args.out}")


if __name__ == "__main__":
    main()
