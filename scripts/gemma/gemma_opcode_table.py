"""Gemma-2 QK opcode table with Gemma Scope SAE operands (RoPE/GQA-aware) — on the RTX 5050.

Ports the SAE-feature opcode table (sae_opcode_table.py, GPT-2) to Gemma-2-2B. Three architecture changes:
  - separate q_proj/k_proj + GQA: query head h uses KV head h // (n_head // n_kv); the content bilinear
      M_h = W_Q^h^T @ W_K^{kv(h)} / sqrt(query_pre_attn_scalar).
  - RMSNorm (not LayerNorm): fold the block's input_layernorm gain (Gemma uses 1 + weight) into the
      operand directions; NO mean-subtraction (easier than GPT-2).
  - RoPE: the runtime score is q·R_Δ·k (position-dependent). We read the CONTENT binding = the UNROTATED
      W_Q W_K^T (R_0). This is the standard content-circuit reading for RoPE models — RoPE's low-frequency
      rotary dims barely rotate over typical Δ, so the unrotated binding is the position-invariant content
      preference; the positional modulation is a separate axis (the behavioral/addressing channels). We
      validate B_h against empirical attention over dominant-SAE-feature pairs (vs a label-permuted null),
      exactly as the GPT-2 table — the permuted null controls for the content match specifically.

Operands = Gemma Scope (gemma-scope-2b-pt-res, JumpReLU) feature decoder directions for the analyzed layer,
content-weighted (structural-dominant features excluded), RMSNorm-folded + unit. Runs on cuda (bf16 model;
fp32 for the small B_h math). GPT-2's content opcode was legible (z>2 for most heads); the question is
whether Gemma-2's QK content-binding is equally legible in SAE-feature coords.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

SCOPE_GLOB = ("/home/allans/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res/"
              "**/layer_{L}/width_16k/average_l0_*/params.npz")


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def _is_struct(g0: str) -> bool:
    s = g0.replace("Ġ", "").replace("▁", "").replace("Ċ", "\n").strip()
    return s == "" or s in {"<0x0A>"} or all(not ch.isalnum() for ch in s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--sae-layer", type=int, default=12, help="Gemma Scope layer (only 12 cached)")
    p.add_argument("--head-layer", type=int, default=12, help="analyze this block's QK heads")
    p.add_argument("--hidden-index", type=int, default=12,
                   help="hidden_states index for operands (= resid entering --head-layer)")
    p.add_argument("--n-operands", type=int, default=48)
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--min-count", type=int, default=8)
    p.add_argument("--n-perm", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--corpus", default="wikitext")
    p.add_argument("--output", type=Path, default=Path("runs/gemma/gemma_opcode_table_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device if torch.cuda.is_available() else "cpu"
    print(f"[load] {args.model} on {dev} (bf16)")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    cfg = model.config
    H, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    qscale = float(getattr(cfg, "query_pre_attn_scalar", hd)) ** 0.5
    L = args.head_layer
    blk = model.model.layers[L]

    # ---- Gemma Scope JumpReLU SAE (operand dictionary) ----
    sae_f = glob.glob(SCOPE_GLOB.format(L=args.sae_layer), recursive=True)[0]
    sae = np.load(sae_f)
    Wenc = torch.tensor(sae["W_enc"], device=dev, dtype=torch.float32)        # (d, F)
    benc = torch.tensor(sae["b_enc"], device=dev, dtype=torch.float32)
    bdec = torch.tensor(sae["b_dec"], device=dev, dtype=torch.float32)
    thr = torch.tensor(sae["threshold"], device=dev, dtype=torch.float32)
    Wdec = sae["W_dec"].astype(np.float64)                                    # (F, d)
    d = Wdec.shape[1]
    print(f"  gemma-2: H={H} n_kv={n_kv} head_dim={hd} qscale={qscale:.1f}; SAE F={Wenc.shape[1]} d={d}")

    CORPORA = {"shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
               "wikitext": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"}
    import urllib.request
    url = CORPORA.get(args.corpus, args.corpus)
    txt = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                 timeout=15).read().decode("utf-8", "ignore")[:400000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    off = np.cumsum([0] + [len(c) for c in chunks])
    all_ids = np.array([j for c in chunks for j in c]); P = len(all_ids)

    # ---- pass 1: resid at hidden-index; collect residual + JumpReLU activation MASS per feature.
    # argmax-dominant is hyper-concentrated on Gemma, so operands = top features by total activation
    # (content-weighted), and each position is labelled by its strongest OPERAND feature. ----
    F = Wenc.shape[1]
    Xh = np.zeros((P, d), np.float32)
    act_mass = torch.zeros(F, device=dev)
    with torch.no_grad():
        for ci, c in enumerate(chunks):
            hs = model(input_ids=torch.tensor([c], device=dev),
                       output_hidden_states=True).hidden_states[args.hidden_index][0].float()
            Xh[off[ci]:off[ci + 1]] = hs.cpu().numpy()
            pre = (hs - bdec) @ Wenc + benc
            act_mass += torch.where(pre > thr, pre, torch.zeros_like(pre)).sum(0)
    order = act_mass.argsort(descending=True).cpu().numpy()

    # content-weighted operand selection: top features by activation mass, structural-dominant excluded.
    ln_gain = (1.0 + blk.input_layernorm.weight.detach().float().cpu().numpy().astype(np.float64))  # Gemma RMSNorm
    # exclude function-word / high-frequency features and DEDUPE by top-token so operands span DISTINCT
    # content (the top-activation-mass features are otherwise all " the"/","/space — no legible structure).
    STOP = {"the", "of", "and", "a", "to", "in", "that", "is", "was", "for", "it", "as", "with", "on",
            "by", "at", "an", "be", "or", "are", "from", "this", "his", "her", "he", "she", "they",
            "i", "you", "we", "but", "not", "have", "had", "s", "t", "the\n", ""}
    Xc = torch.tensor(Xh, device=dev) - bdec
    feats, gloss, seen_top = [], [], set()
    for f in order:
        if len(feats) >= args.n_operands:
            break
        active = ((Xc @ Wenc[:, f] + benc[f]) > thr[f]).cpu().numpy()
        if active.sum() < args.min_count:
            continue
        top = Counter(all_ids[active].tolist()).most_common(3)
        g = [tok.convert_ids_to_tokens(t).replace("Ġ", "_").replace("▁", "_") for t, _ in top]
        key = g[0].lstrip("_").lower()
        if _is_struct(g[0]) or key in STOP or g[0] in seen_top:
            continue
        seen_top.add(g[0]); feats.append(int(f)); gloss.append(g)
    Xt = torch.tensor(Xh, device=dev)
    nt = len(feats)
    D = Wdec[feats] * ln_gain
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)                 # (nt, d) RMSNorm-folded operands
    # label each position by its strongest OPERAND feature (>threshold), else -1
    preO = (Xt - bdec) @ Wenc[:, feats] + benc[feats]                        # (P, nt)
    actO = torch.where(preO > thr[feats], preO, torch.zeros_like(preO))
    mx, am = actO.max(1)
    pos_op = torch.where(mx > 0, am, torch.full_like(am, -1)).cpu().numpy()
    print(f"  operands: {nt} content features (e.g. f{feats[0]}~{gloss[0]}, f{feats[1]}~{gloss[1]}); "
          f"labelled positions {int((pos_op >= 0).sum())}/{P}")

    # ---- pass 2: empirical attention over dominant-feature pairs, per head of layer L ----
    asum = np.zeros((H, nt * nt)); acnt = np.zeros(nt * nt)
    with torch.no_grad():
        for ci, c in enumerate(chunks):
            aL = model(input_ids=torch.tensor([c], device=dev),
                       output_attentions=True).attentions[L][0].float().cpu().numpy()  # (H, Lc, Lc)
            poc = pos_op[off[ci]:off[ci + 1]]
            ti, si = np.tril_indices(len(c), k=-1)
            m = (poc[ti] >= 0) & (poc[si] >= 0)
            flat = poc[ti[m]] * nt + poc[si[m]]
            np.add.at(acnt, flat, 1.0)
            for h in range(H):
                np.add.at(asum[h], flat, aL[h][ti[m], si[m]])

    # ---- per-head content opcode B_h (GQA + unrotated QK) + legibility vs permuted null ----
    Wq = blk.self_attn.q_proj.weight.detach().float().cpu().numpy().astype(np.float64)   # (H*hd, d)
    Wk = blk.self_attn.k_proj.weight.detach().float().cpu().numpy().astype(np.float64)   # (n_kv*hd, d)
    cnt2 = acnt.reshape(nt, nt); offmask = ~np.eye(nt, dtype=bool)
    supp = (cnt2 >= args.min_count) & offmask
    rng = np.random.default_rng(0); perms = [rng.permutation(nt) for _ in range(args.n_perm)]
    rows = []
    for h in range(H):
        kv = h // (H // n_kv)
        WQh = Wq[h * hd:(h + 1) * hd, :]            # (hd, d)
        WKkv = Wk[kv * hd:(kv + 1) * hd, :]         # (hd, d)
        Mh = (WQh.T @ WKkv) / qscale                # (d, d) content bilinear, unrotated
        B = D @ Mh @ D.T                            # (nt, nt)
        A = asum[h].reshape(nt, nt) / np.maximum(cnt2, 1)
        leg = _spearman(B[supp], A[supp]) if supp.sum() >= 4 else float("nan")
        null = [_spearman(B[np.ix_(pp, pp)][supp], A[supp]) for pp in perms] if supp.sum() >= 4 else [0]
        z = (leg - np.nanmean(null)) / (np.nanstd(null) + 1e-9) if np.isfinite(leg) else float("nan")
        Boff = B.copy(); np.fill_diagonal(Boff, -np.inf)
        qi, ki = np.unravel_index(int(np.argmax(Boff)), Boff.shape)
        rows.append({"head": f"{L}.{h}", "kv_head": kv, "offdiag_legibility": leg, "leg_z": float(z),
                     "top_bind": {"q": gloss[qi], "k": gloss[ki], "value": float(B[qi, ki])}})

    legible = sum(1 for r in rows if np.isfinite(r["leg_z"]) and r["leg_z"] > 2)
    out = {"experiment": "Gemma-2 QK opcode table (Gemma Scope operands, RoPE/GQA)", "model": args.model,
           "head_layer": L, "sae_layer": args.sae_layer, "n_operands": nt, "supported_cells": int(supp.sum()),
           "n_legible_z2": legible, "heads": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\nlayer-{L} heads (content opcode B_h vs empirical attn over {int(supp.sum())} supported cells):")
    print(f"  {'head':>6} {'kv':>3} {'offdiag_leg':>11} {'z':>6}  top content binding (q->k)")
    for r in sorted(rows, key=lambda r: -(r["leg_z"] if np.isfinite(r["leg_z"]) else -9)):
        q = "/".join(r["top_bind"]["q"]); k = "/".join(r["top_bind"]["k"])
        print(f"  {r['head']:>6} {r['kv_head']:>3} {r['offdiag_legibility']:>11.3f} {r['leg_z']:>6.2f}  {q!r}->{k!r}")
    print(f"\n[verdict] {legible}/{H} layer-{L} heads have a behaviorally-legible content binding (z>2) in "
          f"Gemma Scope feature coords -> {'the QK content-opcode reading PORTS to Gemma-2 (RoPE/GQA)' if legible >= H // 2 else 'weak legibility — try --head-layer/--hidden-index alignment or more operands/tokens'}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
