"""recursion-explain — show explain detail ONLY for the parts of a forward pass doing RECURSIVE computation.

Everything the Dyck + Lisp probes measured gives us a per-position signature of recursive computation, which we can
gate on: a token is "doing recursion" iff it is (1) DEFERRED — its prediction only resolves in the late layers
(logit-lens), as recursive evaluation does, *not* an early read-out; (2) COMPUTED — not reproducible by flat in-context
copy / n-gram (the forge-tax ladder), so not retrieval; (3) BINDING — it attends back across a NESTED earlier span (the
frame it is folding), not just the previous token. Positions that fail the gate emit nothing. So on flat text the tool
is silent; on a recursive computation it lights up exactly the folding steps — and because the gate is the model's own
internal signature (no parser), it works on non-Lisp recursion too.

For each lit position it shows the **value stack** read straight out of the residual: the logit-lens trajectory
(what the model "has computed so far" at each layer), the resolve layer (how much network it spent), and which earlier
span it folds. On Lisp s-exprs this should surface the intermediate sub-results (e.g. (- 5 1)->4, (* 3 4)->12, ->13).

Run:  .venv/bin/python scripts/disassembly/recursion_explain.py --model Qwen/Qwen2.5-1.5B
      .venv/bin/python scripts/disassembly/recursion_explain.py --model Qwen/Qwen2.5-1.5B --text "(+ 1 (* 3 (- 5 1)))"
      add --show-all to see the gate value at every position (tuning/validation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# few-shot eval prime so an arithmetic s-expr is actually evaluated (else the model just continues the text)
PRIME = "(+ 2 3) = 5\n(* 2 4) = 8\n(- 9 4) = 5\n(+ 1 (* 2 3)) = 7\n(- 8 (+ 1 2)) = 5\n"
DEMOS = [
    ("lisp-recursion", PRIME + "(+ 1 (* 3 (- 5 1))) ="),
    ("flat-control", "The cat sat on the mat and then it went back to the warm house."),
    ("nested-language", "The key that the man that the dog liked lost was found."),
]


def _lm_parts(model):
    """(final_norm, unembed_weight, n_layers) arch-generic."""
    W_U = model.get_output_embeddings().weight.detach().float()
    norm = None
    for name in ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm"):
        obj = model
        ok = True
        for p in name.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False; break
        if ok:
            norm = obj; break
    return norm, W_U


def explain(model, tok, text, show_all=False, defer=0.6, reach_min=3, conc_min=0.20, topk=3):
    import torch
    norm, W_U = _lm_parts(model)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    toks = [tok.decode([i]) for i in ids]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True, output_attentions=True)
    logits = out.logits[0].float()
    hs = out.hidden_states                                    # (nL+1) × (1, T, d)
    nL = len(hs) - 1
    att = out.attentions                                      # nL × (1, H, T, T)
    late = list(range(2 * nL // 3, nL))                       # late-layer attention = where binding/return lives
    # MAX over (late layers, heads): binding is done by a single head — averaging over H dilutes it by 1/H.
    Aback = np.stack([att[L][0].float().numpy() for L in late]).max(axis=(0, 1))   # (T, T)
    Aback[:, :2] = 0.0                                         # zero the attention SINK (pos 0/1) — else it dominates

    def lens(vec):
        h = norm(vec) if norm is not None else vec
        return (h.float() @ W_U.T)

    # paren-nesting depth per position (only meaningful for bracketed input; a display aid, NOT used in the gate)
    depth, d = [], 0
    for t in toks:
        d += t.count("(") - t.count(")")
        depth.append(max(d, 0))

    rows = []
    for p in range(len(ids) - 1):
        final = int(logits[p].argmax())
        # (1) DEFERRED: first layer whose logit-lens argmax == the final prediction
        res = nL
        for L in range(1, nL + 1):
            if int(lens(hs[L][0, p]).argmax()) == final:
                res = L; break
        deferred = res / nL
        # (2) COMPUTED: is `final` NOT an in-context copy (longest-suffix induction)? then it's computed, not retrieved
        copy = False
        ctx = ids[:p + 1]
        for span in range(min(6, len(ctx) - 1), 1, -1):
            tail = ctx[-span:]
            for i in range(len(ctx) - span - 1, -1, -1):
                if ctx[i:i + span] == tail and i + span < len(ctx):
                    copy = (ids[i + span] == final); break
            if copy:
                break
        computed = not copy
        # (3) BINDING: dominant NON-SINK back-attention target, its reach, and how CONCENTRATED it is (a real bind,
        # not diffuse). Flat text binds locally/diffusely; recursion binds strongly to a distant pending antecedent.
        if p > 2:
            back = int(np.argmax(Aback[p, :p])); conc = float(Aback[p, back])
        else:
            back, conc = p, 0.0
        reach = p - back
        # gate (general, no parser): a CONCENTRATED, distant, non-sink bind (flat text binds diffusely/locally;
        # recursion binds strongly to the pending antecedent it is folding), to a token that is computed + resolved
        # late. Concentration is the key discriminator that silences flat prose.
        lit = computed and (reach >= reach_min) and (conc >= conc_min) and (deferred >= defer)
        rows.append((p, toks[p], depth[p], final, res, deferred, computed, back, reach, conc, lit))

    # ---- emit ----
    name_w = max((len(repr(t)) for _, t, *_ in rows), default=6)
    print(f"\n=== recursion-explain: {text!r}  ({nL} layers) ===")
    lit_rows = [r for r in rows if r[-1]]
    if not lit_rows and not show_all:
        print("  (no recursive computation detected — silent)")
        return {"text": text, "n_lit": 0}
    hdr = "  pos  token".ljust(8 + name_w) + "  depth  resolve  defer  reach  conc   folds-back-to     value-stack (logit-lens late layers)"
    print(hdr)
    for (p, t, dep, final, res, deferred, computed, back, reach, conc, lit) in rows:
        if not (lit or show_all):
            continue
        # value-stack readout: logit-lens top-1 at a few late layers — what the model "holds" at this position
        traj = []
        for L in sorted(set([res] + late[-3:])):
            tid = int(lens(hs[L][0, p]).argmax())
            traj.append(f"L{L}:{tok.decode([tid]).strip()!r}")
        mark = "▶" if lit else " "
        print(f" {mark}{p:>3}  {repr(t):<{name_w}}  {dep:>5}  {res:>5}/{nL}  {deferred:>4.2f}  {reach:>4}  {conc:>4.2f}   "
              f"{back:>3}:{repr(toks[back]):<8}  {' '.join(traj)}")
    print(f"  → {len(lit_rows)} of {len(rows)} positions flagged as recursive computation")
    return {"text": text, "n_lit": len(lit_rows), "n_pos": len(rows)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--text", default=None, help="explain a single input (else run the demo suite)")
    p.add_argument("--defer", type=float, default=0.6, help="resolve-depth gate (fraction of layers)")
    p.add_argument("--reach-min", type=int, default=3, help="min back-attention reach to count as nested binding")
    p.add_argument("--conc-min", type=float, default=0.20, help="min attention concentration on the bound antecedent")
    p.add_argument("--show-all", action="store_true")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager").eval()
    print(f"[{args.model}] recursion-explain (defer≥{args.defer}, reach≥{args.reach_min})")

    if args.text is not None:
        explain(model, tok, args.text, args.show_all, args.defer, args.reach_min, args.conc_min)
    else:
        for name, text in DEMOS:
            print(f"\n##### {name} #####")
            explain(model, tok, text, args.show_all, args.defer, args.reach_min, args.conc_min)


if __name__ == "__main__":
    main()
