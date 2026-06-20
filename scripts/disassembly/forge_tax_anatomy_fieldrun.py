"""Forge-tax anatomy at SCALE — the same ladder, but the model top-1 comes from fieldrun.

forge_tax_anatomy.py showed (GPT-2 124M, Pythia-70m) that ~90% of the forge tax is genuine computation (not exact-
or soft-retrievable) and syntax-heavy. That could be SMALL-MODEL-shaped. This runs the identical retrieval ladder
(flat → exact ∞-gram → soft embedding-NN → computed) against a bigger model served by the **fieldrun** CPU runtime —
Qwen2.5 (0.5B / 1.5B / 7B) or Qwen3-MoE — which HF-on-CPU can't comfortably host. The model's per-position top-1 is
read from `fieldrun --ids … --dump`; its input-embedding matrix (for the soft retriever) is mmap'd straight from the
fieldrun bundle; the flat store is built inline from the train slice. Everything else is identical, so the bucket
distribution is directly comparable to the small-model runs.

The question: as the model grows, does the forge tax stay ~90% computed and syntax-heavy, or does more of it become
retrievable / content (i.e. was the small-model result an artifact of limited capacity)?

Run (Qwen2.5-0.5B bundle already on disk):
    .venv/bin/python scripts/disassembly/forge_tax_anatomy_fieldrun.py \
        --bundle pylm/qwen05b --hf-tokenizer Qwen/Qwen2.5-0.5B
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forge_tax_anatomy import CORPORA, FUNCTION_WORDS  # noqa: E402

FIELDRUN = REPO.parent / "fieldrun" / "target" / "release" / "fieldrun"


def build_store(train, quad_k=2, tri_k=3, bi_k=8, uni_k=64, min_count=2):
    """A flat pylm store (quad/tri/bi/uni successor tables) from the train ids — same schema as pylm/build.py."""
    quad, tri, bi, uni = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter()
    for i in range(len(train) - 1):
        nxt = train[i + 1]
        uni[nxt] += 1
        bi[train[i]][nxt] += 1
        if i >= 1:
            tri[(train[i - 1], train[i])][nxt] += 1
        if i >= 2:
            quad[(train[i - 2], train[i - 1], train[i])][nxt] += 1
    return {
        "model": "fieldrun", "min_induction_match": 3, "min_induction_accept": 2,
        "quad": {f"{a},{b},{c}": [t for t, _ in cnt.most_common(quad_k)]
                 for (a, b, c), cnt in quad.items() if sum(cnt.values()) >= min_count},
        "tri": {f"{a},{b}": [t for t, _ in cnt.most_common(tri_k)]
                for (a, b), cnt in tri.items() if sum(cnt.values()) >= min_count},
        "bi": {str(b): [t for t, _ in cnt.most_common(bi_k)] for b, cnt in bi.items()},
        "uni": [t for t, _ in uni.most_common(uni_k)],
    }


def bundle_embed(bundle_stem: Path):
    """mmap the input-embedding matrix straight out of the fieldrun bundle (.json describes the .bin layout)."""
    meta = json.loads(Path(str(bundle_stem) + ".fieldrun.json").read_text())
    emb = next(a for a in meta["arrays"] if a["name"] == "embed")
    assert emb["dtype"] == "f32", f"embed dtype {emb['dtype']} unsupported"
    V, d = emb["shape"]
    raw = np.memmap(Path(str(bundle_stem) + ".fieldrun.bin"), dtype=np.float32, mode="r",
                    offset=emb["offset"], shape=(V, d))
    wte = np.asarray(raw, dtype=np.float32)
    return wte / (np.linalg.norm(wte, axis=1, keepdims=True) + 1e-8)


def fieldrun_top1(bundle_stem: Path, ids, ctx, n_eval):
    """Run `fieldrun --ids … --dump` → per-position top-1 for positions [ctx, ctx+n_eval). Returns (preds, acc_str)."""
    with tempfile.TemporaryDirectory() as td:
        idf = Path(td) / "ids.json"
        outf = Path(td) / "preds.txt"
        idf.write_text(json.dumps({"holdout_ids": [int(x) for x in ids]}))
        res = subprocess.run([str(FIELDRUN), "--bundle", str(bundle_stem), "--ids", str(idf),
                              "--ctx", str(ctx), "--n-eval", str(n_eval), "--dump", str(outf)],
                             capture_output=True, text=True, check=True)
        preds = [int(x) for x in outf.read_text().split()]
    acc = next((ln for ln in res.stderr.splitlines() + res.stdout.splitlines() if "top-1" in ln), "")
    return preds, acc


def classify_token(tid, tok, freq_rank, common_cut):
    s = tok.decode([tid]).strip()
    if not s or all(not c.isalnum() for c in s):
        return "punct"
    if s.lower() in FUNCTION_WORDS or freq_rank.get(tid, 10**9) < common_cut:
        return "function"
    return "content"


def run(args):
    sys.path.insert(0, str(REPO / "pylm"))
    from lm import PyLM
    from transformers import AutoTokenizer

    bundle = (REPO / args.bundle) if not Path(args.bundle).is_absolute() else Path(args.bundle)
    tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
    url = CORPORA.get(args.corpus, args.corpus)
    txt = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                 timeout=20).read().decode("utf-8", "ignore")
    ids = [int(x) for x in tok(txt)["input_ids"]]
    if args.max_train and len(ids) > args.max_train:
        ids = ids[:args.max_train + 20000]
    cut = int(len(ids) * 0.85)
    train, hold = ids[:cut], ids[cut:]
    print(f"[{args.hf_tokenizer.split('/')[-1]} via fieldrun {bundle.name}] train {len(train)} | holdout {len(hold)} tok")

    pylm = PyLM.__new__(PyLM)            # build a store inline, hand it to a bare PyLM
    store = build_store(train)
    for k in ("quad", "tri", "bi", "uni"):
        setattr(pylm, k, store[k])
    pylm.min_induction, pylm.min_accept = 3, 2
    pylm.knowledge = pylm.grammar = None

    freq = Counter(ids)
    freq_rank = {t: r for r, (t, _) in enumerate(freq.most_common())}
    wte = bundle_embed(bundle)
    print(f"  embed {wte.shape} from bundle | flat store {len(store['tri'])} tri / {len(store['quad'])} quad")

    # model top-1 from fieldrun over positions [ctx, ctx+n_eval) of the holdout stream
    n_eval = min(args.n_eval, len(hold) - args.ctx - 1)
    preds, acc = fieldrun_top1(bundle, hold, args.ctx, n_eval)
    n_eval = min(n_eval, len(preds))
    gold = [hold[args.ctx + j] for j in range(n_eval)]
    my_acc = np.mean([preds[j] == gold[j] for j in range(n_eval)])
    print(f"  fieldrun scored {n_eval} positions | {acc.strip()} | my recomputed top-1-acc {my_acc:.1%} (alignment check)")

    train_str = " " + " ".join(map(str, train)) + " "

    def infinigram(ctx_ids, cap):
        for k in range(min(cap, len(ctx_ids)), 1, -1):
            needle = " " + " ".join(map(str, ctx_ids[-k:])) + " "
            pos = train_str.rfind(needle)
            if pos != -1:
                after = train_str[pos + len(needle):].split(" ", 1)[0]
                if after:
                    return int(after), k
        return None, 0

    W = args.soft_window
    db_pos = list(range(W, len(train) - 1))
    if len(db_pos) > args.db_size:
        db_pos = db_pos[::len(db_pos) // args.db_size][:args.db_size]

    def cvec(seq, end):
        v = wte[seq[max(0, end - W):end]].mean(0)
        return v / (np.linalg.norm(v) + 1e-8)

    db_vec = np.stack([cvec(train, j) for j in db_pos]).astype(np.float32)
    db_next = np.array([train[j] for j in db_pos])

    def soft_pred(ctx_ids):
        sims = db_vec @ cvec(ctx_ids, len(ctx_ids)).astype(np.float32)
        return int(db_next[int(sims.argmax())])

    buckets = {b: [] for b in ("flat", "exact_copy", "soft_assoc", "computed")}
    suffix = []
    for j in range(n_eval):
        i = args.ctx + j
        ctx_ids = hold[max(0, i - args.ctx):i]
        mt1 = preds[j]
        rec = {"target": mt1, "cls": classify_token(mt1, tok, freq_rank, args.common_cut)}
        if pylm.predict(ctx_ids) == mt1:
            buckets["flat"].append(rec); continue
        ig, k = infinigram(ctx_ids, args.cap)
        if ig is not None and ig == mt1:
            suffix.append(k); buckets["exact_copy"].append(rec); continue
        if soft_pred(ctx_ids) == mt1:
            buckets["soft_assoc"].append(rec); continue
        buckets["computed"].append(rec)

    n = n_eval
    forge = n - len(buckets["flat"])

    def describe(recs):
        if not recs:
            return {"n": 0}
        cls = Counter(r["cls"] for r in recs)
        return {"n": len(recs), "frac_of_all": len(recs) / n, "frac_of_forge": len(recs) / max(forge, 1),
                "function": cls["function"] / len(recs), "content": cls["content"] / len(recs),
                "punct": cls["punct"] / len(recs)}

    desc = {b: describe(v) for b, v in buckets.items()}
    print(f"\n  {n} eval positions | forge tax (non-flat) = {forge} ({forge / n:.1%})")
    print(f"  {'bucket':<12}{'n':>5}{'%all':>7}{'%forge':>8}{'func':>7}{'cont':>7}{'punct':>7}")
    for b in ("flat", "exact_copy", "soft_assoc", "computed"):
        d = desc[b]
        if d["n"]:
            print(f"  {b:<12}{d['n']:>5}{d['frac_of_all']:>7.1%}"
                  f"{(d.get('frac_of_forge', 0) if b != 'flat' else 0):>8.1%}"
                  f"{d['function']:>7.0%}{d['content']:>7.0%}{d['punct']:>7.0%}")
    print(f"\n  VERDICT: of the forge tax — exact-copy {desc['exact_copy'].get('frac_of_forge', 0):.0%}, "
          f"soft {desc['soft_assoc'].get('frac_of_forge', 0):.0%}, computed {desc['computed'].get('frac_of_forge', 0):.0%}")
    c = desc["computed"]
    if c["n"]:
        print(f"  computed residual: {c['content']:.0%} content / {c['function']:.0%} function / {c['punct']:.0%} punct")

    return {"model": args.hf_tokenizer.split("/")[-1], "backend": "fieldrun", "bundle": bundle.name,
            "corpus": args.corpus, "n_eval": n, "fieldrun_top1_acc": float(my_acc),
            "forge_tax_frac": forge / n, "buckets": desc,
            "exact_copy_mean_suffix": float(np.mean(suffix)) if suffix else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default="pylm/qwen05b", help="fieldrun bundle stem (rel to lm-sae or absolute)")
    p.add_argument("--hf-tokenizer", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--corpus", default="shakespeare")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--cap", type=int, default=64)
    p.add_argument("--soft-window", type=int, default=8)
    p.add_argument("--db-size", type=int, default=40000)
    p.add_argument("--max-train", type=int, default=300000)
    p.add_argument("--common-cut", type=int, default=120)
    p.add_argument("--outdir", type=Path, default=REPO / "runs/disassembly")
    args = p.parse_args(argv)
    if not FIELDRUN.exists():
        raise SystemExit(f"fieldrun binary not found at {FIELDRUN} (cargo build --release in ../fieldrun)")
    out = run(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    sp = args.outdir / "forge_tax_anatomy_fieldrun_summary.json"
    prior = json.loads(sp.read_text()).get("results", []) if sp.exists() else []
    merged = [out] + [r for r in prior if (r.get("model"), r.get("corpus")) != (out["model"], out["corpus"])]
    sp.write_text(json.dumps({"experiment": "forge-tax anatomy at scale (fieldrun backend)", "results": merged},
                             indent=2, default=float))
    print(f"\n[done] {sp}")
    return out


if __name__ == "__main__":
    main()
