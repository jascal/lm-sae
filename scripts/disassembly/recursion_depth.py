"""The recursion DEPTH LIMIT — distance vs nesting, and does the nesting ceiling scale with the number of layers?

Theory (TC⁰): a single forward pass is a fixed-depth circuit, so two "depths" behave differently —
  DISTANCE — how far the head noun is (PP modifiers: "the key near the dogs near the tables is"). Attention can jump
    directly across arbitrary distance in ONE layer, so distance is NOT layer-bounded; it fades from interference
    (more attractors competing for attention mass). Expect a *gradual* decay.
  NESTING — genuine center-embedding ("the key that the dogs that the cat sees chase is"): each embedded clause must be
    resolved before the outer verb, so it needs ~one layer of sequential composition per level (stack-like). This is
    the layer-bounded recursion; expect a *sharper* ceiling, and a ceiling that GROWS with the number of layers.

So the experiment runs both conditions over depth and across model sizes, predicting nesting breaks sooner than
distance and the nesting ceiling rises with layer count (gpt2-small 12L < medium 24L < large 36L). Both score the
SAME thing: the outer verb's agreement with the HEAD noun (correct) vs the opposite/attractor number (wrong).

Read-only; ResidualVM for load + logits. Output: runs/disassembly/recursion_depth_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recursive_syntax import NOUNS, VERBS, build_stimuli, single  # noqa: E402

# transitive, number-marked inner verbs for the embedded relative clauses (singular, plural)
TRANS = [("chases", "chase"), ("sees", "see"), ("likes", "like"), ("holds", "hold"),
         ("watches", "watch"), ("knows", "know"), ("hates", "hate"), ("feeds", "feed")]


def build_nested(tok, depths, rng):
    """Center-embedded agreement: 'The {N0} that the {N1} ... that the {Nd} {Vd} ... {V1}' → predict the OUTER verb,
    which agrees with the HEAD N0 across d nested clauses (each inner Vi agrees with its own subject Ni)."""
    nouns = [(s, p) for (s, p) in NOUNS if single(tok, " " + s) and single(tok, " " + p)]
    verbs = [(a, b) for (a, b) in VERBS if single(tok, a) and single(tok, b)]
    trans = [(s, p) for (s, p) in TRANS if single(tok, " " + s) and single(tok, " " + p)]
    stim = {d: [] for d in depths}
    for d in depths:
        for (sg, pl) in nouns:
            for hnum in (0, 1):                                          # 0 = singular head, 1 = plural head
                head = sg if hnum == 0 else pl
                onum = 1 - hnum                                          # inner subjects: opposite number (attractors)
                opp = [(p if hnum == 0 else s) for (s, p) in nouns if (s, p) != (sg, pl)]
                inner = [opp[int(i)] for i in rng.integers(0, len(opp), d)]
                tverbs = [trans[int(i)][onum] for i in rng.integers(0, len(trans), d)]   # agree with inner subjects
                clauses = "".join(f" that the {n}" for n in inner)
                verbtail = "".join(f" {v}" for v in reversed(tverbs))   # center-embedding: innermost verb first
                text = "The " + head + clauses + verbtail
                for (vs, vp) in verbs:
                    correct = vs if hnum == 0 else vp                   # OUTER verb agrees with the HEAD
                    wrong = vp if hnum == 0 else vs
                    stim[d].append({"text": text, "correct": correct, "wrong": wrong, "hnum": hnum})
    return stim


def run_model(mid, args):
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok
    rng = np.random.default_rng(0)
    depths = list(range(args.max_depth + 1))
    conditions = {"distance": build_stimuli(tok, depths, rng), "nesting": build_nested(tok, depths, rng)}

    def score(sents):
        diffs = []
        for s in sents:
            ctx = tok(s["text"], add_special_tokens=False)["input_ids"]
            ci = tok(s["correct"], add_special_tokens=False)["input_ids"][0]
            wi = tok(s["wrong"], add_special_tokens=False)["input_ids"][0]
            lp = t.log_softmax(vm.logits(ctx).float(), -1)[-1]
            diffs.append(float(lp[ci] - lp[wi]))
        diffs = np.array(diffs)
        return {"acc": float((diffs > 0).mean()), "logit_diff": float(diffs.mean()), "n": int(len(diffs))}

    out = {"model": mid.split("/")[-1], "n_layers": vm.nL, "d_model": vm.d, "depths": depths, "conditions": {}}
    for cond, stim in conditions.items():
        bd = {d: score(stim[d]) for d in depths}
        accs = [bd[d]["acc"] for d in depths]; diffs = [bd[d]["logit_diff"] for d in depths]
        ceiling = max([d for d in depths if bd[d]["acc"] >= args.thresh], default=-1)
        cross0 = next((d for d in depths if bd[d]["logit_diff"] <= 0), -1)
        out["conditions"][cond] = {"by_depth": {str(d): bd[d] for d in depths}, "accuracy_curve": accs,
                                   "logit_diff_curve": diffs, "ceiling_depth": ceiling, "logit_cross0_depth": cross0}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large")
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--thresh", type=float, default=0.75, help="accuracy threshold defining the depth ceiling")
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
            print(f"  d{r['d_model']} {r['n_layers']}L | depths {r['depths']}")
            for cond in ("distance", "nesting"):
                c = r["conditions"][cond]
                print(f"  {cond:8s} ceiling(acc≥{args.thresh:.0%}) depth {c['ceiling_depth']} · logit crosses 0 at {c['logit_cross0_depth']}")
                print("            acc:    " + " ".join(f"{a:>5.0%}" for a in c["accuracy_curve"]))
                print("            Δlogit: " + " ".join(f"{x:>+5.1f}" for x in c["logit_diff_curve"]))
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "recursion_depth_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "recursion depth limit — distance (interference-bounded) vs nesting (layer-bounded), and whether the nesting ceiling scales with layers",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'conditions' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
