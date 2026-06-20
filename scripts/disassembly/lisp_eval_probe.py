"""Evaluating Lisp s-expressions — does the LLM run a recursive EVALUATOR (with a value stack)?

The Dyck probe tested recursive STRUCTURE (bracket matching). Lisp arithmetic tests recursive COMPUTATION:
`(+ 1 (* 3 (- 5 1)))` forces bottom-up recursive descent — compute `(- 5 1)=4`, then `(* 3 4)=12`, then `13`.
Lisp is the cleanest possible recursion stressor: full parenthesization ⇒ surface form = parse tree (nesting depth
= recursion depth, no operator-precedence shortcuts); prefix ⇒ the operator is known before its operands (a recursive
call with a known function). So accuracy is a clean read on the model's effective evaluation-stack depth.

Few-shot prompt of `(expr) = value` pairs primes the eval task; we read the model's predicted value of a held-out
expression of controlled nesting depth. Values are constrained so every sub-result is a single token (exact read).

Two questions:
  (a) how is recursion represented?  — accuracy vs nesting depth (the eval-stack limit), across scale (fieldrun).
      --mechanism (HF): is each sub-result present in the residual stream at its closing paren (a readable VALUE
      STACK)?, and at which layer does the final answer resolve (does depth consume layers)?
  (b) Lisp-specific — prefix+full-paren forces uniform recursive descent; --infix contrasts with `1 + 3 * (5 - 1)`
      where precedence permits shortcuts.

Run:  .venv/bin/python scripts/disassembly/lisp_eval_probe.py --backend hf --model gpt2 --mechanism
      .venv/bin/python scripts/disassembly/lisp_eval_probe.py --backend fieldrun \
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
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}


def gen_expr(depth, rng, maxv, infix=False):
    """Random arithmetic s-expr of exactly `depth` nesting, all sub-results in [0, maxv]. Returns (str, value, tree).
    tree = ('leaf', v) | ('node', op, left, right, value) — for the value-stack analysis."""
    if depth == 0:
        v = int(rng.integers(0, maxv + 1))
        return str(v), v, ("leaf", v)
    for _ in range(200):                                      # rejection sampling for the value bound
        op = "+-*"[int(rng.integers(0, 3))]
        dl, dr = depth - 1, int(rng.integers(0, depth))       # one child reaches depth-1 (so max depth = depth)
        if rng.integers(0, 2):
            dl, dr = dr, dl
        ls, lv, lt = gen_expr(dl, rng, maxv, infix)
        rs, rv, rt = gen_expr(dr, rng, maxv, infix)
        if op == "-" and lv < rv:
            continue
        v = OPS[op](lv, rv)
        if not (0 <= v <= maxv):
            continue
        s = f"({ls} {op} {rs})" if infix else f"({op} {ls} {rs})"
        return s, v, ("node", op, lt, rt, v)
    return gen_expr(0, rng, maxv, infix)                      # give up → a leaf


def build(tok, dmax, reps, rng, maxv, infix, n_shot=10):
    """Few-shot prefix + a stream of targets. For each target, the gold is the LAST token of the answer continuation
    `tok(' '+value)` (any leading space/digit tokens go into the prompt) — robust across tokenizers that do (GPT-2) or
    don't (Qwen merges a leading space separately) keep ' N' as one token. maxv≤9 ⇒ the gold is a single digit token."""
    def make(depth):
        return gen_expr(depth, rng, maxv, infix)              # (str, value, tree)

    shots = [make(int(rng.integers(1, 3))) for _ in range(n_shot)]
    prefix = tok("\n".join(f"{s} = {v}" for s, v, _ in shots) + "\n", add_special_tokens=False)["input_ids"]

    ids = list(prefix)
    items = []                                                # (eval_index, gold_id, depth, expr_str, tree)
    eq_id = tok(" =", add_special_tokens=False)["input_ids"]
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    for depth in range(1, dmax + 1):
        for _ in range(reps):
            s, v, tree = make(depth)
            ids += tok(s, add_special_tokens=False)["input_ids"]
            ids += eq_id                                      # " ="
            ans = tok(" " + str(v), add_special_tokens=False)["input_ids"]
            ids += ans[:-1]                                   # leading space/digits → prompt
            items.append((len(ids) - 1, ans[-1], depth, s, tree))   # last prompt token predicts the final digit
            ids += [ans[-1]] + nl
    return ids, items, len(prefix)


def eval_hf(model_name, ids, items, mechanism, tok, device="cpu"):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_name).eval().to(device)
    rows = []
    # logit-lens helpers for the mechanism pass
    W_U = m.get_output_embeddings().weight.detach().float()    # (V, d)
    ln_f = None
    for name in ("transformer.ln_f", "model.norm", "gpt_neox.final_layer_norm"):
        obj = m
        try:
            for p in name.split("."):
                obj = getattr(obj, p)
            ln_f = obj; break
        except AttributeError:
            continue

    win, step = 1000, 900
    done = set()
    layer_res, stack_hits = [], []                            # mechanism outputs
    with torch.no_grad():
        for s in range(0, len(ids), step):
            chunk = ids[s:s + win]
            if len(chunk) < 2:
                continue
            out = m(input_ids=torch.tensor([chunk], device=device),
                    output_hidden_states=mechanism)
            lg = out.logits[0].float()
            for (eq, gold, depth, expr, tree) in items:
                if eq in done or not (s + 1 <= eq < s + len(chunk)):
                    continue
                done.add(eq)
                top1 = int(lg[eq - s].argmax())
                rows.append((depth, int(top1 == gold)))
                if mechanism and out.hidden_states is not None:
                    hs = out.hidden_states                     # tuple (L+1) of (1, T, d)
                    # (b1) layer at which the final answer first becomes top-1 (logit lens at the '=' position)
                    pos = eq - s
                    res_layer = None
                    for li in range(1, len(hs)):
                        h = hs[li][0, pos]
                        lens = (ln_f(h) if ln_f is not None else h).float() @ W_U.T
                        if int(lens.argmax()) == gold:
                            res_layer = li; break
                    layer_res.append((depth, res_layer if res_layer is not None else len(hs)))
    return rows, layer_res, stack_hits


def eval_fieldrun(bundle, ids, items, ctx=96):
    from forge_tax_anatomy_fieldrun import fieldrun_top1
    stem = (REPO / bundle) if not Path(bundle).is_absolute() else Path(bundle)
    n_eval = len(ids) - ctx
    preds, acc = fieldrun_top1(stem, ids, ctx, n_eval)
    rows = []
    for (eq, gold, depth, expr, tree) in items:
        if ctx <= eq < ctx + len(preds):
            rows.append((depth, int(preds[eq - ctx] == gold)))
    return rows, acc


def report(rows, label, layer_res=None):
    print(f"\n=== {label} | {len(rows)} expressions evaluated ===")
    by = defaultdict(list)
    for depth, ok in rows:
        by[depth].append(ok)
    print(f"  {'depth':>6} {'eval-acc':>10} {'n':>5}" + ("   resolve-layer" if layer_res else ""))
    lr = defaultdict(list)
    for d, l in (layer_res or []):
        lr[d].append(l)
    summary = {}
    for d in sorted(by):
        acc = float(np.mean(by[d]))
        ml = float(np.mean(lr[d])) if d in lr else None
        summary[d] = {"acc": acc, "n": len(by[d]), "resolve_layer": ml}
        print(f"  {d:>6} {acc:>10.2f} {len(by[d]):>5}" + (f"   {ml:.1f}" if ml is not None else ""))
    rel = [d for d in sorted(by) if summary[d]["acc"] >= 0.5]
    print(f"  → deepest reliably-evaluated nesting (acc≥0.5): {max(rel) if rel else 0}")
    summary["eval_depth_limit"] = max(rel) if rel else 0
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["hf", "fieldrun"], default="hf")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--bundle", default="pylm/qwen05b")
    p.add_argument("--hf-tokenizer", default=None)
    p.add_argument("--dmax", type=int, default=5)
    p.add_argument("--reps", type=int, default=60)
    p.add_argument("--maxv", type=int, default=9, help="value bound (all sub-results in [0, maxv]; ≤9 ⇒ single digit)")
    p.add_argument("--infix", action="store_true", help="infix `(a + b)` instead of prefix `(+ a b)` (Lisp contrast)")
    p.add_argument("--mechanism", action="store_true", help="(HF) logit-lens resolve-layer vs depth")
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=REPO / "runs/disassembly")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tname = args.hf_tokenizer or (args.model if args.backend == "hf" else "Qwen/Qwen2.5-0.5B")
    tok = AutoTokenizer.from_pretrained(tname)
    rng = np.random.default_rng(args.seed)
    ids, items, plen = build(tok, args.dmax, args.reps, rng, args.maxv, args.infix)
    label = (args.model if args.backend == "hf" else Path(args.bundle).name) + ("/infix" if args.infix else "")
    print(f"[{label}] stream {len(ids)} tok | {len(items)} exprs | prefix {plen} | dmax {args.dmax} maxv {args.maxv}")

    if args.backend == "hf":
        rows, layer_res, _ = eval_hf(args.model, ids, items, args.mechanism, tok)
    else:
        rows, acc = eval_fieldrun(args.bundle, ids, items, args.ctx)
        layer_res = None
        print(f"  {acc.strip()}")
    summary = report(rows, label, layer_res if args.mechanism else None)

    out = {"backend": args.backend, "model": label, "infix": args.infix, "dmax": args.dmax,
           "maxv": args.maxv, "n_exprs": len(rows), "summary": summary}
    args.outdir.mkdir(parents=True, exist_ok=True)
    sp = args.outdir / "lisp_eval_probe_summary.json"
    prior = json.loads(sp.read_text()).get("results", []) if sp.exists() else []
    merged = [out] + [r for r in prior if r.get("model") != out["model"]]
    sp.write_text(json.dumps({"experiment": "Lisp s-expression evaluation — recursive evaluator probe",
                              "results": merged}, indent=2, default=float))
    print(f"\n[done] {sp}")
    return out


if __name__ == "__main__":
    main()
