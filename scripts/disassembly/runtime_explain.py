"""The auditable artifact — per-token runtime explanation at BOTH levels: the symbolic pylm idiom AND the model circuit.

This realises the "auditable corner" of the small/legible/complete triangle: an explainer that, for every predicted
token, says (a) what the decompiled program (pylm) predicts and via which named idiom — induction / n-gram / grammar /
knowledge — and (b) which mechanism in the *actual model* carries it, with attention evidence. For an induction
prediction it locates the model's induction head and reports its attention mass from the query to the copy-source
position (the token after the earlier match) — confirming the symbolic idiom against the real circuit. Content
predictions (n-gram / grammar) are attributed to the distributed MLP/composition bulk (no single attention signature),
and knowledge to the readout. So every token is attributable end-to-end: program idiom ↔ model circuit ↔ evidence.

Combines pylm (the decompiled program), the operator/circuit catalog (induction / prev-token / duplicate / number-mover,
located behaviourally), and ResidualVM (the model + attention probing). Output: runs/disassembly/runtime_explain_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def induction_source(ctx, min_match, min_accept):
    """Recompute pylm's induction match: the copy-SOURCE key position (token after the earlier occurrence of the tail)."""
    for span in range(min_match, min_accept - 1, -1):
        if len(ctx) <= span:
            continue
        tail = ctx[-span:]
        for i in range(len(ctx) - span - 1, -1, -1):
            if ctx[i:i + span] == tail:
                return i + span, span                                  # key the induction head should attend to
    return None, 0


def run_model(mid, args):
    import urllib.request

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pylm"))
    from lm import PyLM
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); tok = vm.tok
    lm = PyLM(args.store)

    # locate the model's induction heads behaviourally (the circuit the symbolic 'induction' idiom maps to)
    rng = np.random.default_rng(0)
    probe = [(lambda s: s + s)([int(v) for v in rng.integers(0, min(2000, len(tok)), 20)]) for _ in range(12)]
    ind_heads, imass = vm.find_heads(probe, "induction", top=3)
    Lh, Hh = ind_heads[0]                                              # the top induction head

    def explain_seq(ids, trace=False):
        """per-position: model top-1, pylm idiom+pred, agreement, and (for induction) the model induction-head evidence."""
        rows = []
        for p in range(args.min_ctx, len(ids) - 1):
            ctx = ids[: p + 1]
            mtop = int(vm.logits(ctx)[-1].argmax())
            pred, idiom = lm.predict_explain(ctx)
            agree = (pred == mtop)
            ev = None; circuit = "MLP/content (distributed)"
            if idiom.startswith("induction"):
                src, span = induction_source(ctx, lm.min_induction, lm.min_accept)
                if src is not None:
                    att = vm.attn(ctx)[Lh][Hh].float().cpu().numpy()    # [q,k]
                    ev = float(att[len(ctx) - 1, src])                  # query=last pos → copy-source key
                    circuit = f"induction head {Lh}.{Hh} (attn {ev:.2f}→pos {src} '{tok.decode([ids[src]]).strip()}')"
            elif idiom.startswith("knowledge"):
                circuit = "knowledge readout (fact table)"
            elif idiom == "grammar":
                circuit = "grammar scaffold (closed-class write)"
            rows.append({"pos": p, "model_top1": tok.decode([mtop]), "pylm": tok.decode([pred]),
                         "idiom": idiom, "agree": agree, "circuit": circuit, "induction_attn": ev})
            if trace:
                mark = "✓" if agree else "✗"
                print(f"    [{p:3d}] model '{tok.decode([mtop])}' | pylm '{tok.decode([pred])}' [{idiom}] {mark}  "
                      f"← {circuit}")
        return rows

    # ---- demo: a repetitive prompt so induction fires and the head-evidence shows ----
    demo = "The cat sat on the mat. The dog ran in the park. The cat sat on the"
    print(f"  induction heads located: {', '.join(f'{L}.{h}' for L, h in ind_heads)} (mass {imass.max():.2f})")
    print(f"  demo trace — '{demo}'")
    drows = explain_seq(tok(demo)["input_ids"], trace=True)

    # ---- summary over held-out prose: attribution mix + induction circuit-evidence ----
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[120000:135000]
    ids = tok(txt)["input_ids"][: args.n_eval + args.min_ctx]
    rows = explain_seq(ids)
    from collections import Counter
    by_idiom = Counter(r["idiom"] for r in rows)
    agree_by = {}
    for r in rows:
        agree_by.setdefault(r["idiom"], [0, 0]); agree_by[r["idiom"]][0] += int(r["agree"]); agree_by[r["idiom"]][1] += 1
    ind_ev = [r["induction_attn"] for r in rows if r["induction_attn"] is not None]
    attribution = {k: {"share": v / len(rows), "agreement": agree_by[k][0] / agree_by[k][1]} for k, v in by_idiom.items()}
    print(f"  attribution over {len(rows)} held-out tokens (idiom → share@agreement):")
    for k in sorted(attribution, key=lambda k: -attribution[k]["share"]):
        print(f"    {k:14s} {attribution[k]['share']:.0%} @ {attribution[k]['agreement']:.0%}")
    if ind_ev:
        print(f"  induction circuit-evidence: head {Lh}.{Hh} mean attn-to-source {np.mean(ind_ev):.2f} "
              f"over {len(ind_ev)} induction tokens (the symbolic idiom confirmed in the model)")
    return {"model": mid.split("/")[-1], "induction_heads": [f"{L}.{h}" for L, h in ind_heads],
            "demo": demo, "demo_trace": drows, "n_eval": len(rows), "attribution": attribution,
            "induction_mean_attn_to_source": float(np.mean(ind_ev)) if ind_ev else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--store", default="pylm/store_grammar.json")
    p.add_argument("--min-ctx", type=int, default=8)
    p.add_argument("--n-eval", type=int, default=300)
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
    sumpath = args.outdir / "runtime_explain_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "auditable runtime — per-token explanation at both the pylm-idiom and model-circuit level", "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'attribution' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
