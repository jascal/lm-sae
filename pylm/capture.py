"""Capture pylm's flat store FROM THE MODEL (not the corpus) — distill the model's own behaviour into flat files.

`build.py` fits n-grams from the *corpus* — a surrogate. This instead **captures the model**: run the target model
over the corpus (the model is the ultimate MI probe) and distill *its* next-token predictions into the same flat
n-gram-keyed tables, so the store holds **what the model would say** after each local context, not what the corpus
did. pylm (induction + this captured store) is then a decompilation *of the model*, and `validate.py`'s pylm↔model
agreement measures how much of the model's behaviour the small program + captured flat store reproduces.

(The knowledge-relation tables read out of the weights — `relation_decompile` — are the other captured store, for
factual corpora; here, on tiny-Shakespeare, the conditional distillation is the relevant capture.)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grammar import closed_ids, skeleton  # noqa: E402

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--ctx", type=int, default=128, help="window the model sees while being captured")
    p.add_argument("--topk", type=int, default=8, help="model top-k recorded per position")
    p.add_argument("--keep", type=int, default=3, help="successors kept per context key in the store")
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--skel-min", type=int, default=3, help="min count to keep a grammatical-skeleton entry")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("pylm/store_captured.json"))
    p.add_argument("--ids-out", type=Path, default=Path("pylm/holdout_ids.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if "4bit" in args.model.lower():                                  # unsloth/*-bnb-4bit etc. — torch 4-bit load
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        m = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=cfg, device_map=dev).eval()
    else:
        big = any(s in args.model for s in ("1b", "1.4b", "1.5b", "2b", "2.8b", "3b", "7b", "8b", "-xl", "-large"))
        m = AutoModelForCausalLM.from_pretrained(args.model, **({"dtype": torch.bfloat16} if big else {})).eval().to(dev)
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")
    ids = tok(text)["input_ids"]
    cut = int(len(ids) * (1 - args.holdout)); train, hold = ids[:cut], ids[cut:]

    closed = closed_ids(tok)                                             # the model's closed-class id set (grammar)

    # ---- run the model over the train stream; vote the model's top-1 into each local-context key ----
    quad = defaultdict(Counter); tri = defaultdict(Counter); bi = defaultdict(Counter); uni = Counter()
    skel3 = defaultdict(Counter); skel2 = defaultdict(Counter)           # GRAMMAR — content-free skeleton tables
    step = args.ctx
    with torch.no_grad():
        for s in range(0, len(train) - 1, step):
            chunk = train[s:s + step]
            if len(chunk) < 2:
                continue
            top = m(input_ids=torch.tensor([chunk], device=dev)).logits[0].topk(args.topk, -1).indices.cpu().tolist()
            for j in range(len(chunk) - 1):                              # logits[j] predicts chunk[j+1]; context ENDS at chunk[j]
                pred = top[j][0]; g = s + j                              # train[g] = chunk[j] = the last context token
                uni[pred] += 1
                bi[train[g]][pred] += 1                                   # key ends at the last context token, then predict
                if g >= 1:
                    tri[(train[g - 1], train[g])][pred] += 1
                if g >= 2:
                    quad[(train[g - 2], train[g - 1], train[g])][pred] += 1
                w = train[max(0, g - 2):g + 1]                           # the local context ending at the last token
                skel3[skeleton(w, closed, 3)][pred] += 1                  # collapse content → OPEN, keep function-words
                skel2[skeleton(w, closed, 2)][pred] += 1

    store = {
        "model": args.model, "min_induction_match": 3, "min_induction_accept": 2,
        "n_train_tokens": len(train), "source": "model-captured",
        "quad": {f"{a},{b},{c}": [d for d, _ in cnt.most_common(args.keep)]
                 for (a, b, c), cnt in quad.items() if cnt.total() >= args.min_count},
        "tri": {f"{a},{b}": [c for c, _ in cnt.most_common(args.keep)]
                for (a, b), cnt in tri.items() if cnt.total() >= args.min_count},
        "bi": {str(b): [c for c, _ in cnt.most_common(args.keep)] for b, cnt in bi.items()},
        "uni": [c for c, _ in uni.most_common(64)],
        "closed_ids": sorted(closed),
        "skel": {**{f"3:{k}": [d for d, _ in c.most_common(args.keep)]
                    for k, c in skel3.items() if c.total() >= args.skel_min},
                 **{f"2:{k}": [d for d, _ in c.most_common(args.keep)]
                    for k, c in skel2.items() if c.total() >= args.skel_min}},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(store))
    args.ids_out.write_text(json.dumps({"model": args.model, "holdout_ids": hold}))
    print(f"[capture] {args.model}: distilled the model over {len(train)} train tokens | "
          f"quad {len(store['quad'])} / tri {len(store['tri'])} / bi {len(store['bi'])} / "
          f"skel {len(store['skel'])} (closed {len(closed)}) | "
          f"{args.out.stat().st_size / 1024:.0f} KB → {args.out}")


if __name__ == "__main__":
    main()
