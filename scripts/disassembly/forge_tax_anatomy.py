"""What does the content composition COMPUTE? — an anatomy of the forge tax.

compose_core.py + the entangled-core arc located the forge tax (the ~50% of tokens flat retrieval misses) as the
HIGH-rank, MLP-stored, dense content composition. context_ceiling.py measured its SIZE (the ∞-gram memorization
ceiling: how much exact unbounded flat lookup can ever reproduce). Neither says WHAT the residual is. This does.

For every held-out position we run an ORDERED retrieval ladder and assign the token to the first rung that
reproduces the model's top-1 — partitioning the model's behaviour into retrieval of increasing fuzziness, with a
genuinely-computed remainder:

  L0  flat        — pylm's bounded store (4-gram + induction + grammar): the decompilable fraction.
  L1  exact-copy  — UNBOUNDED ∞-gram over the train stream (longest suffix → its successor). Extended exact retrieval.
  L2  soft-assoc  — embedding nearest-neighbour over the train stream: the continuation of the most surface-similar
                    prior context (mean-pooled INPUT token embeddings — no deep computation, so this is a fair test
                    of "is the answer a FUZZY associative lookup?"). Soft / semantic retrieval.
  L3  computed    — none of the above. The genuine composition: not exact-, not soft-retrievable.

Then we CHARACTERISE each bucket — function-vs-content word, token frequency, punctuation/space, model confidence —
so the computed residual gets a linguistic description, not just a number. The decisive question: is the forge tax a
fuzzy associative MEMORY (L2 cracks most of it → "the model IS a soft database") or genuine COMPUTATION (L3 dominates)?

Pure flat-file lookup except the model top-1 we measure against. Run:
    .venv/bin/python scripts/disassembly/forge_tax_anatomy.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CORPORA = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "wikitext": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
}

# A compact English closed-class set (function words); content = open-class. Lowercased, no leading space.
FUNCTION_WORDS = set("""the a an of to in on at by for with from into onto over under and or but nor so yet
as if then than that this these those which who whom whose what when where why how is are was were be been being am
do does did have has had will would shall should can could may might must not no yes it its it's he she they them him
her his their our your my me we you i us he's i'm we're you're not n't 's 're 've 'll 'd up out off down then there
here all any some each both few more most other such only own same too very s t""".split())


def load_streams(model_short: str, corpus: str):
    """Return (train_ids, eval_ids) — a corpus the model could have memorised, split for ∞-gram/soft DB vs eval."""
    url = CORPORA.get(corpus, corpus)
    try:
        txt = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                     timeout=20).read().decode("utf-8", "ignore")
        return ("text", txt)
    except Exception:
        for cand in (f"pylm/holdout_{model_short}.json", "pylm/holdout_ids.json"):
            p = REPO / cand
            if p.exists():
                d = json.loads(p.read_text())
                ids = d["holdout_ids"] if isinstance(d, dict) else d
                return ("ids", [int(x) for x in ids])
    raise SystemExit("no corpus available")


def classify_token(tid: int, tok, freq_rank: dict, common_cut: int) -> str:
    """function / content / punct — a coarse open-vs-closed-class tag for the target token."""
    s = tok.decode([tid]).strip()
    if not s or all(not c.isalnum() for c in s):
        return "punct"
    if s.lower() in FUNCTION_WORDS or freq_rank.get(tid, 10**9) < common_cut:
        return "function"
    return "content"


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    short = args.model.split("/")[-1]
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    kind, data = load_streams(short, args.corpus)
    ids = tok(data)["input_ids"] if kind == "text" else data
    ids = [int(x) for x in ids]
    cut = int(len(ids) * 0.85)
    train, hold = ids[:cut], ids[cut:]
    print(f"[{short}] stream={kind} | train {len(train)} tok | holdout {len(hold)} tok")

    sys.path.insert(0, str(REPO / "pylm"))
    from lm import PyLM
    store = REPO / "pylm" / f"store_{short}.json"
    pylm = PyLM(str(store if store.exists() else REPO / "pylm" / "store.json"))

    # frequency rank (for the function/content split) over the whole stream
    freq = Counter(ids)
    freq_rank = {t: r for r, (t, _) in enumerate(freq.most_common())}

    # ---- L1 ∞-gram: longest suffix of ctx present in train → successor of its most-recent occurrence ----
    train_str = " " + " ".join(map(str, train)) + " "

    def infinigram(ctx, cap):
        for k in range(min(cap, len(ctx)), 1, -1):                  # span >= 2 (1-gram is the unigram prior, not copy)
            needle = " " + " ".join(map(str, ctx[-k:])) + " "
            pos = train_str.rfind(needle)
            if pos != -1:
                after = train_str[pos + len(needle):].split(" ", 1)[0]
                if after:
                    return int(after), k
        return None, 0

    # ---- L2 soft associative retrieval: mean-pooled INPUT-embedding context → NN over the train stream ----
    W = args.soft_window
    wte = None
    model = AutoModelForCausalLM.from_pretrained(args.model).eval().to(dev)
    wte = model.get_input_embeddings().weight.detach().float().cpu().numpy()   # (V, d)
    wte = wte / (np.linalg.norm(wte, axis=1, keepdims=True) + 1e-8)

    def ctx_vec(seq, end):                                          # mean of last-W input embeddings, L2-normed
        a = max(0, end - W)
        v = wte[seq[a:end]].mean(0)
        return v / (np.linalg.norm(v) + 1e-8)

    # build the soft DB from the train stream (subsample for speed)
    db_pos = list(range(W, len(train) - 1))
    if len(db_pos) > args.db_size:
        step = len(db_pos) // args.db_size
        db_pos = db_pos[::step][:args.db_size]
    db_vec = np.stack([ctx_vec(train, j) for j in db_pos]).astype(np.float32)   # (M, d)
    db_next = np.array([train[j] for j in db_pos])                  # successor tokens
    print(f"  soft DB: {len(db_pos)} contexts (window {W})")

    def soft_pred(ctx):
        v = ctx_vec(ctx, len(ctx)).astype(np.float32)
        sims = db_vec @ v                                          # cosine (both normed)
        return int(db_next[int(sims.argmax())])

    # ---- eval: model top-1 + confidence, then assign each position to the first rung that reproduces it ----
    positions = list(range(args.ctx, min(len(hold), args.ctx + args.n_eval)))
    buckets = {b: [] for b in ("flat", "exact_copy", "soft_assoc", "computed")}
    suffix_lens = []
    with torch.no_grad():
        for i in positions:
            ctx = hold[max(0, i - args.ctx):i]
            lg = model(input_ids=torch.tensor([ctx], device=dev)).logits[0, -1].float()
            mt1 = int(lg.argmax()); conf = float(torch.softmax(lg, -1).max())
            rec = {"target": mt1, "conf": conf, "cls": classify_token(mt1, tok, freq_rank, args.common_cut)}
            if pylm.predict(ctx) == mt1:
                buckets["flat"].append(rec); continue
            ig, k = infinigram(ctx, args.cap)
            if ig is not None and ig == mt1:
                rec["suffix"] = k; suffix_lens.append(k); buckets["exact_copy"].append(rec); continue
            if soft_pred(ctx) == mt1:
                buckets["soft_assoc"].append(rec); continue
            buckets["computed"].append(rec)

    n = len(positions)
    forge = n - len(buckets["flat"])

    def describe(recs):
        if not recs:
            return {"n": 0}
        cls = Counter(r["cls"] for r in recs)
        return {"n": len(recs), "frac_of_all": len(recs) / n,
                "frac_of_forge": len(recs) / max(forge, 1),
                "mean_conf": float(np.mean([r["conf"] for r in recs])),
                "function": cls["function"] / len(recs), "content": cls["content"] / len(recs),
                "punct": cls["punct"] / len(recs)}

    desc = {b: describe(v) for b, v in buckets.items()}
    print(f"\n  {n} eval positions | forge tax (non-flat) = {forge} ({forge / n:.1%})")
    print(f"  {'bucket':<12}{'n':>5}{'%all':>7}{'%forge':>8}{'conf':>7}{'func':>7}{'cont':>7}{'punct':>7}")
    for b in ("flat", "exact_copy", "soft_assoc", "computed"):
        d = desc[b]
        if d["n"]:
            print(f"  {b:<12}{d['n']:>5}{d['frac_of_all']:>7.1%}"
                  f"{(d.get('frac_of_forge', 0) if b != 'flat' else 0):>8.1%}"
                  f"{d['mean_conf']:>7.2f}{d['function']:>7.0%}{d['content']:>7.0%}{d['punct']:>7.0%}")
    if suffix_lens:
        print(f"  exact_copy mean matched-suffix: {np.mean(suffix_lens):.1f} tokens")
    print(f"\n  VERDICT: of the forge tax — extended exact-copy {desc['exact_copy'].get('frac_of_forge', 0):.0%}, "
          f"soft-associative {desc['soft_assoc'].get('frac_of_forge', 0):.0%}, "
          f"genuinely computed {desc['computed'].get('frac_of_forge', 0):.0%}")
    comp = desc["computed"]
    if comp["n"]:
        print(f"  the computed residual is {comp['content']:.0%} content / {comp['function']:.0%} function / "
              f"{comp['punct']:.0%} punct, mean model confidence {comp['mean_conf']:.2f}")

    out = {"model": short, "stream": kind, "n_eval": n, "forge_tax_frac": forge / n,
           "buckets": desc, "exact_copy_mean_suffix": float(np.mean(suffix_lens)) if suffix_lens else None,
           "config": {"ctx": args.ctx, "cap": args.cap, "soft_window": W, "db_size": len(db_pos),
                      "common_cut": args.common_cut}}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--corpus", default="shakespeare", help="shakespeare | wikitext | a URL")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=1200)
    p.add_argument("--cap", type=int, default=64, help="max ∞-gram suffix length")
    p.add_argument("--soft-window", type=int, default=8)
    p.add_argument("--db-size", type=int, default=40000)
    p.add_argument("--common-cut", type=int, default=120, help="freq-rank below this counts as function/common")
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", type=Path, default=REPO / "runs/disassembly")
    args = p.parse_args(argv)
    out = run(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    sp = args.outdir / "forge_tax_anatomy_summary.json"
    out["corpus"] = args.corpus
    prior = json.loads(sp.read_text()).get("results", []) if sp.exists() else []
    merged = [out] + [r for r in prior if (r.get("model"), r.get("corpus")) != (out["model"], out["corpus"])]
    sp.write_text(json.dumps({"experiment": "anatomy of the forge tax: exact vs soft retrieval vs computation",
                              "results": merged}, indent=2, default=float))
    print(f"\n[done] {sp}")
    return out


if __name__ == "__main__":
    main()
