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
                              "attends_tok": _tokstr(tok, ctx[j]), "mass": round(mass, 3)})
    order = {"induction": 0, "duplicate-token": 1, "previous-token": 2}  # content circuits first, keystone on top
    heads.sort(key=lambda r: (order[r["role"]], -r["mass"])); heads = heads[:top_heads]

    feats = []                                                       # top-activating MLP features (the composition units)
    for L, h in enumerate(cap["mlp_h"]):
        n = int(np.abs(h).argmax()); act = float(h[n])
        feats.append({"layer": L, "neuron": n, "act": round(act, 2),
                      "promotes": _neuron_label(lm_net, tok, L, n, act)})
    feats.sort(key=lambda r: -abs(r["act"])); feats = feats[:top_feats]

    return {
        "context_tail": [_tokstr(tok, t) for t in ctx[-8:]],
        "model_predicts": _tokstr(tok, model_id),
        "retrieval": {"idiom": idiom, "predicts": _tokstr(tok, sym_id),
                      "evidence": _evidence(idiom, ctx, tok),
                      "agrees_with_model": sym_id == model_id},
        "composition": {"head_circuits": heads, "sink_heads": n_sink, "mlp_features": feats},
    }


def _tokstr(tok, tid):
    return repr(tok.decode([tid]))


def _neuron_label(lm_net, tok, L, n, act, top=5):
    """Name an MLP neuron by the tokens it promotes — its write weight projected to the vocabulary (direct logit
    effect), signed by its activation. This is the neuron's 'feature': the vocabulary it pushes when it fires."""
    w_out = lm_net.W[f"h{L}.mlp.c_proj.weight"][n]                    # (d,) the neuron's write direction
    eff = np.sign(act) * (w_out @ lm_net.W["wte"].T)                 # (V,) signed direct-logit contribution
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
         f"             {ex['retrieval']['evidence']}",
         f"  COMPOSITION  content head circuits ({ex['composition']['sink_heads']} other heads idle on sink/NO-OP):"]
    for h in ex["composition"]["head_circuits"]:
        L.append(f"    L{h['layer']}.H{h['head']:<2} {h['role']:<15} → {h['attends_tok']} (mass {h['mass']})")
    if not ex["composition"]["head_circuits"]:
        L.append("    (none above threshold — prediction carried by MLP features below)")
    L.append("  COMPOSITION  top MLP features (neuron → tokens it promotes):")
    for f in ex["composition"]["mlp_features"]:
        L.append(f"    L{f['layer']} n{f['neuron']:<5} act {f['act']:<6} → {{{', '.join(f['promotes'])}}}")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default="pylm/store_gpt2.json")
    p.add_argument("--weights", type=Path, default=Path("pylm/weights_gpt2.npz"))
    p.add_argument("--knowledge", default=None)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--text", default="The Eiffel Tower is located in the city of")
    p.add_argument("--out", type=Path, default=Path("runs/pylm/explain.json"))
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lm_sym = PyLM(args.store, knowledge_path=args.knowledge)
    lm_net = NumpyGPT2(args.weights)
    ctx = tok(args.text)["input_ids"]
    ex = explain(lm_sym, lm_net, tok, ctx)
    print(render(ex))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ex, indent=2, default=float))
    return ex


if __name__ == "__main__":
    main()
