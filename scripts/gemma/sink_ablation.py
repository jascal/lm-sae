"""Sink ablation across models — is the attention-sink load-bearing plumbing or an inert no-op?

Blocks attention to **key position 0** (the sink) at every layer/head and measures the damage to
next-token NLL. Intervention: a forward_pre_hook on every attention module rewrites the 4D additive
causal mask, setting key-0 to -inf for all query positions >= 1 (query 0 keeps its self-attention so its
row is not fully masked). The model's own attention math (GQA, logit-softcap, scaling, RoPE positions) is
otherwise untouched — we only remove the *option* to park budget on the sink, forcing each head to
redistribute onto content. Reports baseline vs ablated mean NLL (+ the measured sink fraction before/after,
which must drop to ~0 — the intervention's own validation), and **resolves ΔNLL by query position** (early
vs late in the window) to localize where the damage lands — early-concentrated would mean "the sink is the
rest-state heads fall back on when there's little context to redistribute onto."

Prediction (from the 4-model disassembly): large ΔNLL for the heavy-sink models (GPT-2 / Llama / Qwen,
sink 44-55%) and small ΔNLL for Gemma-2 (sink ~4%), whose sandwich-norm + attn_logit_softcap give an
output-side gain knob instead of a value-null BOS sink. Arch-generic: runs any HF causal LM via --model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _attn_modules(model):
    """(list of attention submodules, label) across GPT-2 (transformer.h[*].attn) and the
    model.model.layers[*].self_attn family (Gemma/Llama/Qwen)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [lyr.self_attn for lyr in model.model.layers]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [blk.attn for blk in model.transformer.h]
    raise SystemExit("unknown architecture: no model.model.layers or transformer.h")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--corpus", default="wikitext")
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/gemma/sink_ablation_summary.json"))
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    attns = _attn_modules(model)
    import urllib.request
    CORPORA = {"shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
               "wikitext": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"}
    txt = urllib.request.urlopen(urllib.request.Request(CORPORA.get(args.corpus, args.corpus),
                                 headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:400000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    print(f"{args.model}: {len(attns)} layers, {len(chunks)} chunks")

    P1 = args.ctx - 1

    def measure():
        """Per-query-position next-token NLL: returns (sum[P1], count[P1])."""
        s = np.zeros(P1); cnt = np.zeros(P1)
        with torch.no_grad():
            for c in chunks:
                t = torch.tensor([c], device=dev)
                lp = F.log_softmax(model(input_ids=t).logits[0, :-1].float(), -1)
                tgt = t[0, 1:]
                nl = (-lp[torch.arange(len(tgt), device=dev), tgt]).cpu().numpy()
                k = min(len(nl), P1); s[:k] += nl[:k]; cnt[:k] += 1
        return s, cnt

    def total(s, cnt):
        return float(s.sum() / max(cnt.sum(), 1))

    def sink_frac():
        num = den = 0.0
        with torch.no_grad():
            for c in chunks:
                atts = model(input_ids=torch.tensor([c], device=dev), output_attentions=True).attentions
                for a in atts:
                    a = a[0].float()                       # (H, Lc, Lc), rows sum to 1
                    num += float(a[:, 1:, 0].sum())        # mass to key-0 from queries >= 1
                    den += float(a[:, 1:, :].sum())        # = H * (Lc - 1)
        return num / max(den, 1e-9)

    # ---- intervention: block key-0 in every 4D float mask the attention modules receive ----
    def pre_hook(module, args_, kwargs):
        changed = False
        new_kwargs = dict(kwargs)
        for k, v in kwargs.items():
            if torch.is_tensor(v) and v.dim() == 4 and v.is_floating_point():
                v = v.clone(); v[:, :, 1:, 0] = torch.finfo(v.dtype).min
                new_kwargs[k] = v; changed = True
        new_args = list(args_)
        for i, v in enumerate(new_args):
            if torch.is_tensor(v) and v.dim() == 4 and v.is_floating_point():
                v = v.clone(); v[:, :, 1:, 0] = torch.finfo(v.dtype).min
                new_args[i] = v; changed = True
        return (tuple(new_args), new_kwargs) if changed else None

    print("[baseline] measuring NLL (by position) + sink fraction ...")
    sb, cb = measure(); base_nll = total(sb, cb); base_sink = sink_frac()
    print(f"  baseline NLL {base_nll:.4f}  | sink fraction {base_sink:.3f}")

    handles = [m.register_forward_pre_hook(pre_hook, with_kwargs=True) for m in attns]
    try:
        abl_sink = sink_frac()                              # validation: should drop to ~0
        sa, ca = measure()
    finally:
        for h in handles:
            h.remove()
    abl_nll = total(sa, ca)
    print(f"  ablated  NLL {abl_nll:.4f}  | sink fraction {abl_sink:.3f}  (intervention {'OK' if abl_sink < base_sink * 0.5 + 1e-3 else 'DID NOT BITE — check mask plumbing'})")

    # ---- position-resolved: does the damage concentrate at early query positions? ----
    base_pp = sb / np.maximum(cb, 1); abl_pp = sa / np.maximum(ca, 1); dpp = abl_pp - base_pp
    d_early = float(np.mean(dpp[1:9]))          # query positions 1..8 (little context to redistribute onto)
    d_late = float(np.mean(dpp[32:]))           # positions 32+ (plenty of content available)
    probe = [p for p in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80) if p < P1]
    print("  ΔNLL by query position: " + "  ".join(f"p{p}:{dpp[p]:+.2f}" for p in probe))
    shape = "EARLY-concentrated (decays with position)" if d_early > 2 * max(d_late, 1e-6) else "flat across positions"
    print(f"  early (pos1-8) ΔNLL {d_early:+.3f}  vs  late (pos32+) ΔNLL {d_late:+.3f}  ->  {shape}")

    d_nll = abl_nll - base_nll
    out = {"experiment": f"sink ablation (block key-0): {args.model}", "model": args.model,
           "corpus": args.corpus, "n_chunks": len(chunks), "ctx": args.ctx,
           "baseline_nll": base_nll, "ablated_nll": abl_nll,
           "delta_nll": d_nll, "delta_nll_frac_of_baseline": d_nll / base_nll,
           "sink_frac_baseline": base_sink, "sink_frac_ablated": abl_sink,
           "intervention_ok": bool(abl_sink < base_sink * 0.5 + 1e-3),
           "delta_nll_early_pos1_8": d_early, "delta_nll_late_pos32plus": d_late,
           "delta_nll_by_position": [float(x) for x in dpp],
           "baseline_nll_by_position": [float(x) for x in base_pp],
           "ablated_nll_by_position": [float(x) for x in abl_pp]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[verdict] sink {base_sink:.1%} -> ΔNLL {d_nll:+.3f} ({d_nll / base_nll:+.0%} of baseline); "
          f"early {d_early:+.2f} / late {d_late:+.2f}.")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
