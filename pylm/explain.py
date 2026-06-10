"""Unified 'explain this prediction' — fuse the two halves of the decompilation for one next-token step.

Every pylm prediction has two readings, and this prints both for a given context:

  RETRIEVAL (the symbolic program, `lm.py`) — which named idiom fired: induction / n-gram backoff / knowledge
    lookup / grammar skeleton. The flat-file half of the model, attributable by construction.
  COMPOSITION (the real forward pass, `numpy_lm.py`) — which named circuits and features were live at the
    predicting position: attention-head idioms (previous-token / duplicate / induction / sink, the
    `idiom_library.py` signatures) read straight off the attention matrix, plus the top-activating MLP features
    (the dense composition's units). The computed half — the forge tax — made legible.

So a single call answers both "what rule produced this token" (retrieval) and "what was the network actually
doing" (composition), and whether the two agree. This is the explain capability the API will eventually serve;
it runs on the pure-numpy kernel (no torch) over the same flat weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import PyLM            # noqa: E402
from numpy_lm import NumpyGPT2  # noqa: E402
from numpy_rope import NumpyRoPE  # noqa: E402
from numpy_gemma import NumpyGemma  # noqa: E402
from numpy_moe import NumpyMoE  # noqa: E402


def load_kernel(weights, route_frac=0.0):
    """Pick the right pure-numpy kernel from the flat weights: GPT-2 ('config'), Qwen3-MoE ('moe_flags'), Gemma-2
    (4-norm sandwich) or the plain RoPE family (Llama/Qwen). MoE and Gemma share the RoPE 'cfg_i' layout, so dispatch
    on their distinct keys (moe_flags / the gemma input_layernorm name)."""
    keys = np.load(weights).files
    if "config" in keys:
        return NumpyGPT2(weights, route_frac)
    if "moe_flags" in keys:
        return NumpyMoE(weights, route_frac)
    return NumpyGemma(weights, route_frac) if "l0.input_layernorm" in keys else NumpyRoPE(weights, route_frac)


def classify_head(row, ctx):
    """Name an attention head's behaviour at the predicting position from its attention row (length seq).

    Uses the `idiom_library.py` signatures: where the last token attends tells us the circuit it is running."""
    seq = len(ctx); j = int(row.argmax()); mass = float(row[j]); cur = ctx[-1]
    if j == 0 and seq > 1:
        role = "sink"                                                # attend-to-first = the attention-sink NO-OP
    elif j == seq - 2:
        role = "previous-token"                                      # the induction feeder (attends to t-1)
    elif 0 < j < seq and ctx[j - 1] == cur:
        role = "induction"                                          # attends to the token AFTER a prev occurrence of cur
    elif ctx[j] == cur:
        role = "duplicate-token"                                    # attends to an earlier copy of the same token
    else:
        role = "diffuse"
    return role, j, mass


def explain(lm_sym, lm_net, tok, ctx, top_heads=6, top_feats=6, head_thr=0.15, sink_thr=0.5):
    """Return a structured explanation fusing the retrieval idiom and the live composition circuits/features."""
    cap = {}
    logits = lm_net.logits(ctx, capture=cap)
    model_id = int(logits[-1].argmax())
    sym_id, idiom = lm_sym.predict_explain(ctx)

    heads = []; n_sink = 0                                            # live attention-head circuits at the predict step
    for L, att in enumerate(cap["att_last"]):
        for hh in range(att.shape[0]):
            role, j, mass = classify_head(att[hh], ctx)
            if role == "sink" and mass >= sink_thr:
                n_sink += 1; continue                                # sinks are NO-OPs — count them, don't list them
            if role in ("induction", "duplicate-token", "previous-token") and mass >= head_thr:
                heads.append({"layer": L, "head": hh, "role": role, "attends_to": j,
                              "attends_tok": _tokstr(tok, ctx[j]), "mass": mass})
    order = {"induction": 0, "duplicate-token": 1, "previous-token": 2}  # content circuits first, keystone on top
    # sort + truncate on the FULL-precision mass, then round for display: rounding first creates spurious ties that a
    # stable sort breaks by layer order, which would diverge from a full-precision implementation (e.g. fieldrun) at the
    # top-k boundary (see scripts/explain_agreement.py).
    heads.sort(key=lambda r: (order[r["role"]], -r["mass"])); heads = heads[:top_heads]
    for r in heads:
        r["mass"] = round(r["mass"], 3)

    feats = []                                                       # top-activating MLP features (the composition units)
    for L, h in enumerate(cap["mlp_h"]):
        n = int(np.abs(h).argmax()); act = float(h[n])
        feats.append({"layer": L, "neuron": n, "act": act,
                      "promotes": _neuron_label(lm_net, tok, L, n, act)})
    feats.sort(key=lambda r: -abs(r["act"])); feats = feats[:top_feats]  # full-precision sort (see note above)
    for r in feats:
        r["act"] = round(r["act"], 2)

    return {
        "context_tail": [_tokstr(tok, t) for t in ctx[-8:]],
        "model_predicts": _tokstr(tok, model_id),
        "retrieval": {"idiom": idiom, "predicts": _tokstr(tok, sym_id),
                      "evidence": _evidence(idiom, ctx, tok),
                      "agrees_with_model": sym_id == model_id,
                      "grammar": _grammar_readout(lm_sym, tok, ctx, model_id)},
        "composition": {"head_circuits": heads, "sink_heads": n_sink, "mlp_features": feats},
    }


def _grammar_readout(lm_sym, tok, ctx, model_id):
    """The grammatical-skeleton signal in parallel — what the content-free closed-class scaffold predicts here,
    shown on every token (not only when it wins arbitration, where the lexical n-gram usually shadows it)."""
    g = getattr(lm_sym, "grammar", None)
    if g is None:
        return None
    ids, tag = g.lookup(ctx)
    if not ids:
        return None
    n = int(tag.split("-")[1])
    skel = "/".join((repr(tok.decode([t]).strip()) if t in g.closed else "O") for t in ctx[-n:])
    return {"tag": tag, "skeleton": skel, "predicts": _tokstr(tok, ids[0]),
            "agrees_with_model": ids[0] == model_id}


def _tokstr(tok, tid):
    return repr(tok.decode([tid]))


def _neuron_label(lm_net, tok, L, n, act, top=5):
    """Name an MLP neuron by the tokens it promotes — its write weight projected to the vocabulary (direct logit
    effect), signed by its activation. This is the neuron's 'feature': the vocabulary it pushes when it fires."""
    w_out = lm_net.write_mat(L)[n]                                    # (d,) the neuron's write direction (kernel-agnostic)
    eff = np.sign(act) * (w_out @ lm_net.unembed.T)                 # (V,) signed direct-logit contribution
    ids = np.argsort(-eff)[:top * 4]                                  # over-fetch, then dedup decoded strings
    out = []
    for i in ids:
        s = tok.decode([int(i)]).strip() or _tokstr(tok, int(i))
        if s not in out:
            out.append(s)
        if len(out) >= top:
            break
    return out


def _evidence(idiom, ctx, tok):
    if idiom.startswith("induction-"):
        n = int(idiom.split("-")[1]); tail = ctx[-n:]
        for i in range(len(ctx) - n - 1, -1, -1):
            if ctx[i:i + n] == tail:
                return f"in-context copy: the last {n} tokens recurred at position {i}; predict what followed"
        return f"in-context copy of the last {n} tokens"
    if idiom in ("quad", "trigram", "bigram", "unigram"):
        return f"flat n-gram store ({idiom}) successor lookup"
    if idiom.startswith("knowledge:"):
        return f"flat fact-table relational lookup ({idiom.split(':', 1)[1]})"
    if idiom.startswith("grammar") or idiom.startswith("skel"):
        return "content-free grammatical-skeleton successor"
    return idiom


def render(ex):
    L = [f"context …{' '.join(ex['context_tail'])}",
         f"model predicts {ex['model_predicts']}",
         f"  RETRIEVAL  idiom={ex['retrieval']['idiom']}  → {ex['retrieval']['predicts']}"
         f"  ({'agrees' if ex['retrieval']['agrees_with_model'] else 'differs'})",
         f"             {ex['retrieval']['evidence']}"]
    gr = ex["retrieval"].get("grammar")
    if gr:
        L.append(f"  GRAMMAR    {gr['tag']}: skeleton {gr['skeleton']} → {gr['predicts']}"
                 f"  ({'agrees' if gr['agrees_with_model'] else 'differs'})  [parallel closed-class scaffold]")
    L += [
         f"  COMPOSITION  content head circuits ({ex['composition']['sink_heads']} other heads idle on sink/NO-OP):"]
    for h in ex["composition"]["head_circuits"]:
        L.append(f"    L{h['layer']}.H{h['head']:<2} {h['role']:<15} → {h['attends_tok']} (mass {h['mass']})")
    if not ex["composition"]["head_circuits"]:
        L.append("    (none above threshold — prediction carried by MLP features below)")
    L.append("  COMPOSITION  top MLP features (neuron → tokens it promotes):")
    for f in ex["composition"]["mlp_features"]:
        L.append(f"    L{f['layer']} n{f['neuron']:<5} act {f['act']:<6} → {{{', '.join(f['promotes'])}}}")
    return "\n".join(L)


def _bucket(idiom, agrees):
    """Coarse provenance bucket for a token: which half of the model produced it."""
    if idiom.startswith("induction"):
        return "induction (in-context copy)"
    if idiom.startswith("knowledge"):
        return "knowledge (fact lookup)"
    if idiom.startswith(("grammar", "skel")):
        return "grammar (closed-class scaffold)" if agrees else "composition-carried (MLP)"
    if idiom in ("quad", "trigram", "bigram", "unigram"):
        return "n-gram (flat store)" if agrees else "composition-carried (MLP)"
    return "composition-carried (MLP)"


def explain_sequence(lm_sym, lm_net, tok, ids, ctx_window=48):
    """Walk a passage and decompose every next-token step: per-token provenance + an aggregate forge-tax breakdown
    (what fraction of tokens the flat store reproduces vs the dense composition carries, and which circuits/features
    do the carrying). This is the API-shaped explanation for a whole text."""
    from collections import Counter
    prov = Counter(); head_use, feat_use = Counter(), Counter(); n_agree = 0; n_gram_hit = n_gram_agree = 0
    rows = []
    for i in range(1, len(ids)):
        ctx = ids[max(0, i - ctx_window):i]
        ex = explain(lm_sym, lm_net, tok, ctx, top_heads=3, top_feats=3)
        agrees = ex["retrieval"]["agrees_with_model"]; n_agree += agrees
        gr = ex["retrieval"].get("grammar")                          # the parallel scaffold signal (even when shadowed)
        if gr:
            n_gram_hit += 1; n_gram_agree += gr["agrees_with_model"]
        b = _bucket(ex["retrieval"]["idiom"], agrees); prov[b] += 1
        for h in ex["composition"]["head_circuits"]:
            head_use[f"L{h['layer']}.H{h['head']} {h['role']}"] += 1
        for f in ex["composition"]["mlp_features"]:
            feat_use[f"L{f['layer']} n{f['neuron']} {{{','.join(f['promotes'][:3])}}}"] += 1
        rows.append({"token": _tokstr(tok, ids[i]), "predicts": ex["model_predicts"], "bucket": b,
                     "idiom": ex["retrieval"]["idiom"], "agrees": agrees})
    n = len(rows)
    return {"n_tokens": n, "retrieval_agreement": round(n_agree / max(n, 1), 3),
            "grammar_scaffold_coverage": round(n_gram_hit / max(n, 1), 3),
            "grammar_scaffold_agreement": round(n_gram_agree / max(n_gram_hit, 1), 3),
            "provenance": dict(prov.most_common()),
            "top_circuits": dict(head_use.most_common(8)),
            "top_features": dict(feat_use.most_common(8)),
            "trace": rows}


def render_sequence(sq):
    L = [f"=== passage forge-tax breakdown ({sq['n_tokens']} tokens) ===",
         f"retrieval (flat store) reproduces the model on {sq['retrieval_agreement']:.0%} of tokens",
         f"grammar scaffold: matches a skeleton on {sq['grammar_scaffold_coverage']:.0%} of tokens, "
         f"predicts the model on {sq['grammar_scaffold_agreement']:.0%} of those (parallel signal)",
         "provenance:"]
    for b, c in sq["provenance"].items():
        L.append(f"    {c:>4} ({c / max(sq['n_tokens'], 1):>4.0%})  {b}")
    L.append("most-used live circuits across the passage:")
    for h, c in list(sq["top_circuits"].items())[:6]:
        L.append(f"    {c:>4}×  {h}")
    L.append("most-used MLP features across the passage:")
    for f, c in list(sq["top_features"].items())[:6]:
        L.append(f"    {c:>4}×  {f}")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default="pylm/store_gpt2.json")
    p.add_argument("--weights", type=Path, default=Path("pylm/weights_gpt2.npz"))
    p.add_argument("--knowledge", default=None)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--text", default="The Eiffel Tower is located in the city of")
    p.add_argument("--sequence", action="store_true", help="explain a whole passage with a forge-tax aggregate")
    p.add_argument("--out", type=Path, default=Path("runs/pylm/explain.json"))
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lm_sym = PyLM(args.store, knowledge_path=args.knowledge)
    lm_net = load_kernel(args.weights)
    ids = tok(args.text)["input_ids"]
    if args.sequence:
        out = explain_sequence(lm_sym, lm_net, tok, ids)
        print(render_sequence(out))
    else:
        out = explain(lm_sym, lm_net, tok, ids)
        print(render(out))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    main()
