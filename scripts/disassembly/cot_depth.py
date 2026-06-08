"""Does chain-of-thought buy back the within-pass recursion depth? — bracket matching, direct vs CoT, by nesting depth.

`recursion_depth.py` showed a single forward pass caps at shallow nesting (the TC⁰ bound, though distribution is the
binding constraint in practice). Theory's escape hatch: the **decode loop** is TC⁰ *per step* but Turing-complete
*across steps*, so generating intermediate tokens (chain-of-thought) trades within-pass depth for sequence length and
should lift the depth ceiling. This tests it directly on the canonical bounded-recursion task — **Dyck bracket matching**
at nesting depth d — with an instruction-tuned model that can actually reason (Qwen2.5-1.5B-Instruct, fits the 8 GB card):

  DIRECT — "Is this balanced? <seq>. Reply Yes/No." → ~one forward pass of decision; expected to fade with depth.
  COT    — "...work through it bracket by bracket tracking the stack, then answer." → the loop processes one level per
           step; expected to hold accuracy deeper.

Stimuli: fully-nested bracket strings of depth d (balanced), half corrupted to a mismatched closer (unbalanced) — so
the model must track WHICH bracket type is open at each level (a real stack), and chance = 50%. Accuracy vs depth for
both modes. Output: runs/disassembly/cot_depth_summary.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

PAIRS = {"(": ")", "[": "]", "{": "}"}
OPENS = list(PAIRS)
CLOSES = list(PAIRS.values())


def make(d, rng, balanced):
    opens = [OPENS[int(i)] for i in rng.integers(0, len(OPENS), d)]
    closes = [PAIRS[o] for o in reversed(opens)]
    if not balanced:                                                  # corrupt one closer to a mismatched type
        j = int(rng.integers(0, d)); wrong = [c for c in CLOSES if c != closes[j]]
        closes[j] = wrong[int(rng.integers(0, len(wrong)))]
    return "".join(opens) + "".join(closes)


def parse_yesno(text, after=None):
    seg = text
    if after and after in text:
        seg = text.split(after)[-1]
    m = re.search(r"\b(yes|no)\b", seg, re.I)
    return m.group(1).lower() if m else None


def run_model(mid, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).eval().to(dev)
    rng = np.random.default_rng(0)
    depths = [int(x) for x in args.depths.split(",")]

    DIRECT = ("Is this sequence of brackets correctly balanced and properly matched? "
              "Reply with only one word: Yes or No.\n\n{seq}")
    COT = ("Is this sequence of brackets correctly balanced and properly matched? "
           "Work through it one bracket at a time, pushing each opener on a stack and checking each closer matches "
           "the top. On the final line write exactly 'Answer: Yes' or 'Answer: No'.\n\n{seq}")

    def ask(prompt, max_new):
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, n_in:], skip_special_tokens=True)

    by_depth = {}
    for d in depths:
        seqs = [(make(d, rng, b), b) for b in ([True, False] * (args.n // 2))]
        dc = cc = dn = cn = 0
        for seq, bal in seqs:
            gold = "yes" if bal else "no"
            da = parse_yesno(ask(DIRECT.format(seq=seq), args.direct_tokens))
            ca = parse_yesno(ask(COT.format(seq=seq), args.cot_tokens), after="Answer:")
            if da is not None:
                dn += 1; dc += int(da == gold)
            if ca is not None:
                cn += 1; cc += int(ca == gold)
        by_depth[d] = {"direct_acc": dc / max(dn, 1), "cot_acc": cc / max(cn, 1),
                       "direct_n": dn, "cot_n": cn, "n": len(seqs)}
        print(f"  depth {d:2d}: direct {by_depth[d]['direct_acc']:.0%} (n={dn}) · "
              f"CoT {by_depth[d]['cot_acc']:.0%} (n={cn})")
    return {"model": mid.split("/")[-1], "depths": depths, "by_depth": {str(d): by_depth[d] for d in depths}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--depths", default="2,4,6,8,10,12")
    p.add_argument("--n", type=int, default=20, help="stimuli per depth (half balanced)")
    p.add_argument("--direct-tokens", type=int, default=8)
    p.add_argument("--cot-tokens", type=int, default=400)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
    args = p.parse_args(argv)

    import torch
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = run_model(mid, args)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(r)
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "cot_depth_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "does chain-of-thought buy back recursion depth — Dyck bracket matching, direct vs CoT by nesting depth",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'by_depth' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
