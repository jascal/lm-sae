"""Does the model run a bounded RECURSIVE evaluator? — Dyck-k bracket matching, depth isolated from distance.

The forge-tax anatomy (PR #205) found the computed residual is syntax-heavy. This tests the strongest reading: that
the computed core is a (bounded-depth) recursive evaluator. Dyck-k (balanced brackets, k types) is the canonical
probe — predicting a closing bracket's TYPE requires tracking the open-bracket STACK.

Clean single-forced-close design (avoids the "closing-momentum" confound of full ramps): each prompt ends right
where exactly ONE closer is forced (only the outer bracket is still open), and we read the model's prediction of it.
Two families at MATCHED distance separate true recursion-depth from mere long-distance lookup:
  - DEEP:  `( [ { } ] →`   the outer ) must be recalled across DEEP nesting   (depth d+1, distance 2d+1)
  - FLAT:  `( [] {} () →`  the outer ) recalled across FLAT distractor pairs   (depth 2,   distance 2d+1)
DEEP ≪ FLAT at matched distance ⇒ genuine depth cost (a stack machine); DEEP ≈ FLAT ⇒ only distance matters.

Backends: HF transformers (logits → also type-accuracy among closers + confidence) OR the **fieldrun** runtime
(`--dump` top-1), so the same probe scales GPT-2 → Pythia → Qwen2.5-0.5B/1.5B on CPU.

Run:  .venv/bin/python scripts/disassembly/recursion_depth_probe.py --backend hf --model gpt2
      .venv/bin/python scripts/disassembly/recursion_depth_probe.py --backend fieldrun \
          --bundle pylm/qwen05b --hf-tokenizer Qwen/Qwen2.5-0.5B
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PAIRS = [("(", ")"), ("[", "]"), ("{", "}")]
OPENS = [o for o, _ in PAIRS]
CLOSE = {o: c for o, c in PAIRS}


def gen_deep(d, rng):
    """`t0 t1 … td  c(td) … c(t1) →` : outer close c(t0) forced; nesting depth d+1, distance 2d+1."""
    ts = [OPENS[int(rng.integers(0, len(PAIRS)))] for _ in range(d + 1)]
    prompt = list(ts) + [CLOSE[t] for t in reversed(ts[1:])]
    seq = prompt + [CLOSE[ts[0]]]                             # the forced outer close is the gold next token
    return seq, (len(seq) - 1, CLOSE[ts[0]], 2 * d + 1, d + 1, "deep")


def gen_flat(d, rng):
    """`t0  t1 c(t1) … td c(td) →` : outer close c(t0) forced across d FLAT pairs; depth 2, distance 2d+1."""
    t0 = OPENS[int(rng.integers(0, len(PAIRS)))]
    prompt = [t0]
    for _ in range(d):
        ti = OPENS[int(rng.integers(0, len(PAIRS)))]
        prompt += [ti, CLOSE[ti]]
    seq = prompt + [CLOSE[t0]]
    return seq, (len(seq) - 1, CLOSE[t0], 2 * d + 1, 2, "flat")


def build_stream(tok, dmax, reps, rng, sep="\n"):
    """Concatenate single-forced-close samples into one id stream; one close annotation per sample (absolute index).
    Each bracket is a single ` <b>` token (space-prefixed) so positions are exact."""
    septok = tok(sep, add_special_tokens=False)["input_ids"]
    btok = {ch: tok(" " + ch, add_special_tokens=False)["input_ids"] for ch in OPENS + list(CLOSE.values())}
    for ch, t in btok.items():
        assert len(t) == 1, f"bracket {ch!r} not a single token ({t})"
    btok = {ch: t[0] for ch, t in btok.items()}

    samples = []
    for d in range(0, dmax + 1):
        samples += [gen_deep(d, rng) for _ in range(reps)]
        if d >= 1:
            samples += [gen_flat(d, rng) for _ in range(reps)]
    rng.shuffle(samples)

    ids, closes = [], []                                      # (abs_idx_of_close, gold_id, dist, depth, fam)
    for seq, (idx, gold, dist, depth, fam) in samples:
        base = len(ids)
        ids += [btok[ch] for ch in seq]
        closes.append((base + idx, btok[gold], dist, depth, fam))
        ids += septok
    return ids, closes, btok


def eval_hf(model_name, ids, closes, btok, device="cpu"):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)
    close_ids = [btok[c] for c in CLOSE.values()]
    out, done = [], set()
    win, step = 900, 800
    with torch.no_grad():
        for s in range(0, len(ids), step):
            chunk = ids[s:s + win]
            if len(chunk) < 2:
                continue
            lg = m(input_ids=torch.tensor([chunk], device=device)).logits[0].float()
            for (q, gold, dist, depth, fam) in closes:
                if q in done or not (s + 1 <= q < s + len(chunk)):
                    continue
                done.add(q)
                lp = lg[q - 1 - s]
                conf = float(torch.softmax(lp, -1).max())
                top1 = int(lp.argmax())
                closer_argmax = close_ids[int(np.argmax([float(lp[ci]) for ci in close_ids]))]
                out.append((fam, dist, depth, int(top1 == gold), int(closer_argmax == gold), conf))
    return out


def eval_fieldrun(bundle, ids, closes, ctx=48):
    from forge_tax_anatomy_fieldrun import fieldrun_top1
    stem = (REPO / bundle) if not Path(bundle).is_absolute() else Path(bundle)
    n_eval = len(ids) - ctx
    preds, acc = fieldrun_top1(stem, ids, ctx, n_eval)
    out = []
    for (q, gold, dist, depth, fam) in closes:
        if ctx <= q < ctx + len(preds):
            out.append((fam, dist, depth, int(preds[q - ctx] == gold), -1, -1.0))
    return out, acc


def report(rows, label):
    print(f"\n=== {label} | {len(rows)} forced-close positions ===")
    by = defaultdict(list)
    for fam, dist, depth, strict, typ, conf in rows:
        by[(fam, dist)].append((strict, typ, conf))
    has_type = any(t >= 0 for *_, t, _ in rows)
    has_conf = any(c >= 0 for *_, c in rows)
    print(f"  {'distance':>9} {'depth':>6} {'DEEP':>14} {'FLAT(d=2)':>14}"
          f"   (strict top-1; {'[type-acc]; conf' if has_type else ''})")
    dists = sorted({d for _, d in by})
    summary = {"deep": {}, "flat": {}}

    def cell(lst):
        if not lst:
            return "—", None
        sa = float(np.mean([s for s, _, _ in lst]))
        ta = float(np.mean([t for _, t, _ in lst if t >= 0])) if has_type else None
        ca = float(np.mean([c for _, _, c in lst if c >= 0])) if has_conf else None
        return f"{sa:.2f}" + (f"[{ta:.2f}]" if ta is not None else "") + f"(n{len(lst)})", (sa, ta, ca)

    for d in dists:
        dl, fl = by.get(("deep", d), []), by.get(("flat", d), [])
        dt, dv = cell(dl); ft, fv = cell(fl)
        depth = (d + 1) // 2                                  # distance 2k+1 ⇒ deep nesting depth k+1
        if dv:
            summary["deep"][d] = {"acc": dv[0], "type_acc": dv[1], "conf": dv[2], "depth": depth, "n": len(dl)}
        if fv:
            summary["flat"][d] = {"acc": fv[0], "type_acc": fv[1], "conf": fv[2], "n": len(fl)}
        cf = f"  conf {dv[2]:.2f}" if (dv and dv[2] is not None) else ""
        print(f"  {d:>9} {depth:>6} {dt:>14} {ft:>14}{cf}")

    rel = [d for d in dists if summary["deep"].get(d, {}).get("acc", 0) >= 0.5]
    dl_ = (max(rel) + 1) // 2 if rel else 0
    print(f"  → deepest reliable DEEP match (acc≥0.5): distance {max(rel) if rel else 0} ≈ nesting depth {dl_}")
    summary["deep_depth_limit"] = dl_
    # depth-isolation: deep vs flat at matched distance (>1)
    diffs = [summary["deep"][d]["acc"] - summary["flat"][d]["acc"]
             for d in dists if d in summary["deep"] and d in summary["flat"]]
    if diffs:
        md = float(np.mean(diffs))
        print(f"  depth isolation (deep − flat acc at matched distance): mean {md:+.2f}  "
              f"→ {'DEPTH adds cost' if md < -0.05 else ('distance-only' if abs(md) <= 0.05 else 'deep EASIER')}")
        summary["deep_minus_flat"] = md
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["hf", "fieldrun"], default="hf")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--bundle", default="pylm/qwen05b")
    p.add_argument("--hf-tokenizer", default=None)
    p.add_argument("--dmax", type=int, default=8)
    p.add_argument("--reps", type=int, default=60)
    p.add_argument("--ctx", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=REPO / "runs/disassembly")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tname = args.hf_tokenizer or (args.model if args.backend == "hf" else "Qwen/Qwen2.5-0.5B")
    tok = AutoTokenizer.from_pretrained(tname)
    rng = np.random.default_rng(args.seed)
    ids, closes, btok = build_stream(tok, args.dmax, args.reps, rng)
    label = args.model if args.backend == "hf" else Path(args.bundle).name
    print(f"[{label}] stream {len(ids)} tok | {len(closes)} forced closes | dmax {args.dmax}")

    if args.backend == "hf":
        rows = eval_hf(args.model, ids, closes, btok)
    else:
        rows, acc = eval_fieldrun(args.bundle, ids, closes, args.ctx)
        print(f"  {acc.strip()}")
    summary = report(rows, label)

    out = {"backend": args.backend, "model": label, "tokenizer": tname, "dmax": args.dmax,
           "n_closes": len(rows), "summary": summary}
    args.outdir.mkdir(parents=True, exist_ok=True)
    sp = args.outdir / "recursion_depth_probe_summary.json"
    prior = json.loads(sp.read_text()).get("results", []) if sp.exists() else []
    merged = [out] + [r for r in prior if r.get("model") != out["model"]]
    sp.write_text(json.dumps({"experiment": "Dyck-k recursion-depth probe (deep vs flat, single forced close)",
                              "results": merged}, indent=2, default=float))
    print(f"\n[done] {sp}")
    return out


if __name__ == "__main__":
    main()
