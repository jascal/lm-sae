"""Cross-model test: is the prev-token head's position carried in the KEY CONTENT (GPT-2, absolute) or the
QK ROTATION (RoPE)? — i.e. is the positional-broadcast circuit GPT-2-specific?

`validate_new_edges.py` showed (causally, on GPT-2) that the prev-token head reads a positional signal piped in
from early SINK heads: remove them from its key and prev-token attention collapses. That should be an ABSOLUTE-
positional-embedding artifact — GPT-2 adds `wpe` to the residual, so its keys must ENCODE position as content for
the prev-token head to match s=q−1; and GPT-2 is the only family member that depends on its sink. RoPE models
(Gemma-2 / Llama-3 / Qwen-2.5) inject position as a ROTATION of q,k at attention time, so their keys should carry
TOKEN content, not position — needing no upstream positional broadcast.

This measures it directly and arch-generically: for each model find the top prev-token head, capture its **pre-
rotation key** vectors over the corpus, and decompose the key variance into the fraction explained by POSITION
(absolute index) vs TOKEN identity. Prediction: GPT-2's prev-token head is POSITION-dominated (keys encode where,
the broadcast's job); the RoPE models are TOKEN-dominated (keys encode what; position lives in the rotation). The
contrast is the cross-model explanation of why only GPT-2 needs the sink→prev-token broadcast.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _attn_modules(model):
    """(attention modules per layer, head_dim, n_query_heads, n_kv_heads, is_gpt2) — arch-generic."""
    cfg = model.config
    H = cfg.num_attention_heads
    nkv = getattr(cfg, "num_key_value_heads", None) or H
    if hasattr(model, "model") and hasattr(model.model, "layers"):                 # Gemma / Llama / Qwen
        hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
        return [ly.self_attn for ly in model.model.layers], hd, H, nkv, False
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):          # GPT-2 (MHA, no RoPE)
        return [b.attn for b in model.transformer.h], cfg.n_embd // H, H, H, True
    raise SystemExit("unknown architecture")


def run_model(model_id, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    cfg = model.config
    nL = cfg.num_hidden_layers
    attns, hd, H, nkv, is_gpt2 = _attn_modules(model)
    d = cfg.hidden_size
    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8][: args.chunks]

    def prevtok_mask(Lc):
        qi = np.arange(Lc)
        return (qi[None, :] == (qi[:, None] - 1)) & (qi[:, None] >= 1)

    # ---- pass 1: top prev-token head ----
    pt = np.zeros(nL * H); ntok = 0
    with torch.no_grad():
        for c in chunks:
            o = model(input_ids=torch.tensor([c], device=dev), output_attentions=True)
            Lc = len(c); ntok += Lc; PT = prevtok_mask(Lc)
            for L in range(nL):
                a = o.attentions[L][0].float().cpu().numpy()
                pt[L * H:(L + 1) * H] += (a * PT[None]).sum((1, 2))
    pt /= max(ntok, 1)
    B = int(np.argmax(pt)); LB, hB = B // H, B % H
    kvB = hB // (H // nkv)                                                          # prev-token head's KV group
    print(f"  prev-token head {LB}.{hB} (Δ=1 mass {pt[B]:.3f}); KV head {kvB}/{nkv}")

    # ---- pass 2: capture this layer's PRE-ROTATION keys for the prev-token head's KV group ----
    freq = [t for t, _ in Counter(j for c in chunks for j in c).most_common(args.n_tokens)]
    tix = {t: i for i, t in enumerate(freq)}; nv = len(freq)
    cap = {}

    def key_hook(m, inp, out):                                                      # capture key projection output (pre-RoPE)
        if is_gpt2:
            cap["k"] = out[0, :, d:2 * d].detach().float().cpu().numpy()            # c_attn -> q|k|v; take k
        else:
            cap["k"] = out[0].detach().float().cpu().numpy()                        # k_proj -> (seq, nkv*hd)
    mod = attns[LB].c_attn if is_gpt2 else attns[LB].k_proj
    handle = mod.register_forward_hook(key_hook)

    pos_sum = np.zeros((args.ctx, hd)); pos_cnt = np.zeros(args.ctx)
    tok_sum = np.zeros((nv, hd)); tok_cnt = np.zeros(nv)
    gsum = np.zeros(hd); gsq = 0.0; N = 0
    with torch.no_grad():
        for c in chunks:
            cap.clear()
            model(input_ids=torch.tensor([c], device=dev))
            K = cap["k"].reshape(len(c), -1, hd)[:, kvB, :]                         # (seq, hd) prev-tok head's key
            Lc = len(c)
            pos_sum[:Lc] += K; pos_cnt[:Lc] += 1
            for s in range(Lc):
                if c[s] in tix:
                    j = tix[c[s]]; tok_sum[j] += K[s]; tok_cnt[j] += 1
            gsum += K.sum(0); gsq += float((K ** 2).sum()); N += Lc
    handle.remove()

    gmean = gsum / N
    total_var = gsq / N - float(gmean @ gmean)                                      # trace of key covariance
    pm = pos_cnt > 0; tm = tok_cnt > 0
    pos_mean = pos_sum[pm] / pos_cnt[pm][:, None]
    tok_mean = tok_sum[tm] / tok_cnt[tm][:, None]
    between_pos = float((pos_cnt[pm][:, None] * (pos_mean - gmean) ** 2).sum() / N)
    between_tok = float((tok_cnt[tm][:, None] * (tok_mean - gmean) ** 2).sum() / N)
    pos_frac = between_pos / max(total_var, 1e-9); tok_frac = between_tok / max(total_var, 1e-9)
    print(f"  prev-token KEY variance: POSITION {pos_frac:.0%} vs TOKEN {tok_frac:.0%}  "
          f"(ratio pos/tok {pos_frac / max(tok_frac, 1e-9):.2f})")
    return {"model": model_id, "rope": not is_gpt2, "prevtok_head": f"{LB}.{hB}", "prevtok_mass": float(pt[B]),
            "key_position_fraction": pos_frac, "key_token_fraction": tok_frac,
            "pos_over_tok": pos_frac / max(tok_frac, 1e-9)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=40)
    p.add_argument("--n-tokens", type=int, default=80, help="# frequent tokens for the token-variance term")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/gemma/cross_model_positional_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/cross_model_positional.png"))
    args = p.parse_args(argv)

    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            results.append(run_model(mid, args))
        except Exception as e:  # pragma: no cover - gated/missing model, OOM
            print(f"  [skip] {e}")
            results.append({"model": mid, "error": str(e)})

    out = {"experiment": "cross-model: is prev-token position in the KEY content (absolute) or the rotation (RoPE)?",
           "ctx": args.ctx, "chunks": args.chunks, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    ok = [r for r in results if "key_position_fraction" in r]
    print("\n[cross-model] prev-token KEY: is position in the content or the rotation?")
    for r in ok:
        kind = "POSITION-dominated (absolute, needs broadcast)" if r["pos_over_tok"] > 1 else "TOKEN-dominated (position in rotation)"
        print(f"  {r['model']:>22} {'RoPE' if r['rope'] else 'abs '}: prev-tok {r['prevtok_head']:>5}  "
              f"key pos {r['key_position_fraction']:.0%} / tok {r['key_token_fraction']:.0%}  => {kind}")
    gpt2 = next((r for r in ok if not r["rope"]), None)
    rope = [r for r in ok if r["rope"]]
    if gpt2 and rope:
        gpt2_pos = gpt2["pos_over_tok"] > 1.0
        rope_tok = all(r["pos_over_tok"] < 1.0 for r in rope)
        rope_ratios = ", ".join(f"{r['pos_over_tok']:.1f}" for r in rope)
        if gpt2_pos and rope_tok:
            verdict = (f"CONFIRMED — the positional-broadcast circuit is GPT-2-SPECIFIC: GPT-2's prev-token key is "
                       f"POSITION-dominated (pos/tok {gpt2['pos_over_tok']:.1f}) — it must encode absolute position as "
                       f"content, which is exactly what the sink heads broadcast (validate_new_edges) — while every RoPE "
                       f"model's prev-token key is TOKEN-dominated (pos/tok {rope_ratios}), carrying position in the QK "
                       f"rotation and needing no upstream broadcast.")
        else:
            verdict = f"MIXED: GPT-2 position-dominated={gpt2_pos}, all-RoPE token-dominated={rope_tok} — see table"
        print(f"\n[verdict] {verdict}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        names = [r["model"].split("/")[-1] for r in ok]; x = np.arange(len(ok)); w = 0.38
        ax.bar(x - w / 2, [r["key_position_fraction"] for r in ok], w, color="#d62728", edgecolor="k", label="position")
        ax.bar(x + w / 2, [r["key_token_fraction"] for r in ok], w, color="#1f77b4", edgecolor="k", label="token")
        ax.set_xticks(x); ax.set_xticklabels([f"{n}\n({'abs' if not r['rope'] else 'RoPE'})" for n, r in zip(names, ok)], fontsize=8)
        ax.set_ylabel("fraction of prev-token KEY variance"); ax.legend()
        ax.set_title("prev-token key: position-encoded (GPT-2, absolute) vs token-encoded (RoPE)\n"
                     "— why only GPT-2 needs the sink→prev-token positional broadcast", fontsize=10)
        fig.tight_layout(); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
