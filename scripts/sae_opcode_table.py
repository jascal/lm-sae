"""SAE-feature opcode table: read B_h with MONOSEMANTIC SAE FEATURES as operands, not token centroids.

The coverage scorecard found ~7% of GPT-2's long-range (content) attention DARK — concentrated in
layer-1 heads (1.10/1.2/1.4) whose token-centroid B_h legibility is ~0 or negative. Token IDENTITY is the
wrong operand basis for them; they likely bind on syntactic / positional / semantic CONCEPTS. This swaps
the operand basis to a published SAELens GPT-2 SAE (jbloom/GPT2-Small-SAEs-Reformatted, resid_pre, 24576
feats/layer): operand_i = the feature's decoder direction d_i = unit(ln_1 * W_dec[i]).

  B_h[i,j] = d_i . M_h . d_j         (M_h = W_Q^h W_K^h.T / sqrt(head_dim))  -- feature i binds feature j

Each position is labelled by its DOMINANT SAE feature (argmax activation); we keep the top-N features as
operands. The first SAE-operand run found the dominant features are STRUCTURAL-heavy (Ċ/./,) and so
underperform token-id on already-legible heads. This adds a CONTENT-WEIGHTED selection: features whose
dominant token is structural (newline / punctuation) are EXCLUDED, so the operand basis spans content
features and the legibility validation focuses on content-feature attention. We run BOTH selections and
compare 3-way against the token-operand baseline. Features glossed by top corpus tokens at their dominant
positions (offline; no neuronpedia). Two forward passes. GPT-2 + downloaded per-layer SAEs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

MODES = ["dominant", "content"]


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def _is_struct(g0: str) -> bool:
    """A glossed top-token is structural if it is newline/whitespace or pure punctuation."""
    s = g0.lstrip("_")
    return s in ("", "Ċ", "ĊĊ") or all(not ch.isalnum() for ch in s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--layers", default="1,4,9", help="layers whose heads to read with that layer's resid_pre SAE")
    p.add_argument("--n-operands", type=int, default=60, help="# operands per layer per selection mode")
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--min-count", type=int, default=10, help="min ordered (i->j) pairs to score a cell")
    p.add_argument("--n-perm", type=int, default=25)
    p.add_argument("--sae-repo", default="jbloom/GPT2-Small-SAEs-Reformatted")
    p.add_argument("--token-summary", type=Path, default=Path("runs/qk_opcode_table_summary.json"))
    p.add_argument("--output", type=Path, default=Path("runs/sae_opcode_table_summary.json"))
    args = p.parse_args(argv)

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    analyzed = [int(x) for x in args.layers.split(",")]
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    off = np.cumsum([0] + [len(c) for c in chunks])
    all_ids = np.array([j for c in chunks for j in c])
    P = len(all_ids)
    N = args.n_operands
    print(f"{args.pretrained}: layers {analyzed}  operands/layer/mode {N}  positions {P}")

    # ---- pass 1: residual (resid_pre = hidden_states[L]) for analyzed layers ----
    Hs = {L: [] for L in analyzed}
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True)
            for L in analyzed:
                Hs[L].append(o.hidden_states[L][0].float().numpy().astype(np.float64))
    Hs = {L: np.concatenate(v, 0) for L, v in Hs.items()}

    # ---- per layer: load SAE, encode -> dominant feature; select operands for BOTH modes ----
    # operands[mode][L] = (N,d) directions; pos_op[mode][L] = (P,) operand index or -1; gloss[mode][L]
    operands = {m: {} for m in MODES}; pos_op = {m: {} for m in MODES}; gloss = {m: {} for m in MODES}
    for L in analyzed:
        st = load_file(hf_hub_download(args.sae_repo, f"blocks.{L}.hook_resid_pre/sae_weights.safetensors"))
        Wenc = st["W_enc"].astype(np.float64); benc = st["b_enc"].astype(np.float64)
        Wdec = st["W_dec"].astype(np.float64); bdec = st["b_dec"].astype(np.float64)
        F = Wenc.shape[1]
        dom = np.full(P, -1)
        for i in range(0, P, 512):
            a = (Hs[L][i:i + 512] - bdec) @ Wenc + benc
            np.maximum(a, 0, out=a)
            mx = a.max(1); am = a.argmax(1)
            dom[i:i + 512] = np.where(mx > 0, am, -1)
        counts = np.bincount(dom[dom >= 0], minlength=F)
        ln_w = tr.h[L].ln_1.weight.detach().numpy().astype(np.float64)
        # walk features in descending dominance, glossing lazily, filling each mode's operand list
        sel = {m: [] for m in MODES}; gl = {m: [] for m in MODES}
        for f in np.argsort(-counts):
            if counts[f] == 0 or all(len(sel[m]) >= N for m in MODES):
                break
            top = Counter(all_ids[dom == f].tolist()).most_common(3)
            toks_g = [tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t, _ in top]
            for m in MODES:
                if len(sel[m]) >= N:
                    continue
                if m == "content" and _is_struct(toks_g[0]):
                    continue
                sel[m].append(int(f)); gl[m].append(toks_g)
        for m in MODES:
            feats = sel[m]
            D = Wdec[feats] * ln_w
            operands[m][L] = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
            f2i = {f: i for i, f in enumerate(feats)}
            po = np.full(P, -1)
            for f, i in f2i.items():
                po[dom == f] = i
            pos_op[m][L] = po; gloss[m][L] = gl[m]
        print(f"  layer {L}: F={F}  dominant[0]={gl['dominant'][0]}  content[0]={gl['content'][0]}  "
              f"content[1]={gl['content'][1]}")
        del st, Wenc, Wdec

    # ---- pass 2: empirical attention over operand pairs, per mode ----
    att_sum = {m: {L: np.zeros((H, N * N)) for L in analyzed} for m in MODES}
    att_cnt = {m: {L: np.zeros(N * N) for L in analyzed} for m in MODES}
    with torch.no_grad():
        for ci, c in enumerate(chunks):
            o = tr(input_ids=torch.tensor([c]), output_attentions=True)
            Lc = len(c)
            ti, si = np.tril_indices(Lc, k=-1)
            for L in analyzed:
                aL = o.attentions[L][0].float().numpy()
                for m in MODES:
                    po = pos_op[m][L][off[ci]:off[ci] + Lc]
                    msk = (po[ti] >= 0) & (po[si] >= 0)
                    tt, ss = ti[msk], si[msk]
                    flat = po[tt] * N + po[ss]
                    np.add.at(att_cnt[m][L], flat, 1.0)
                    for h in range(H):
                        np.add.at(att_sum[m][L][h], flat, aL[h][tt, ss])

    # ---- B_h + legibility (vs label-permuted null) for both modes ----
    tokleg = {}
    if args.token_summary.exists():
        tj = json.loads(args.token_summary.read_text())
        tokleg = {f"{h['layer']}.{h['head']}": float(h.get("leg_z", float("nan"))) for h in tj.get("heads", [])}
    rng = np.random.default_rng(0)
    perms = [rng.permutation(N) for _ in range(args.n_perm)]
    offmask = ~np.eye(N, dtype=bool)

    def legz_and_bind(B, A, cnt2):
        supp = (cnt2 >= args.min_count) & offmask
        if supp.sum() < 4:
            return float("nan"), float("nan"), int(supp.sum()), (0, 0)
        bv, av = B[supp], A[supp]
        leg = _spearman(bv, av)
        null = [_spearman(B[np.ix_(pp, pp)][supp], av) for pp in perms]
        z = (leg - np.nanmean(null)) / (np.nanstd(null) + 1e-9) if np.isfinite(leg) else float("nan")
        Boff = B.copy(); np.fill_diagonal(Boff, -np.inf)
        qi, ki = np.unravel_index(int(np.argmax(Boff)), Boff.shape)
        return leg, float(z), int(supp.sum()), (qi, ki)

    out_heads = []
    for L in analyzed:
        W = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wq, Wk = W[:, :d], W[:, d:2 * d]
        for h in range(H):
            Mh = Wq[:, h * hd:(h + 1) * hd] @ Wk[:, h * hd:(h + 1) * hd].T / np.sqrt(hd)
            rec = {"head": f"{L}.{h}", "layer": L, "z_token": tokleg.get(f"{L}.{h}", float("nan"))}
            for m in MODES:
                Dl = operands[m][L]
                cnt2 = att_cnt[m][L].reshape(N, N)
                B = Dl @ Mh @ Dl.T
                A = att_sum[m][L][h].reshape(N, N) / np.maximum(cnt2, 1)
                leg, z, ns, (qi, ki) = legz_and_bind(B, A, cnt2)
                rec[f"z_{m}"] = z
                rec[f"supp_{m}"] = ns
                rec[f"bind_{m}"] = {"q": gloss[m][L][qi], "k": gloss[m][L][ki]} if ns >= 4 else None
            out_heads.append(rec)

    out = {"experiment": "SAE-feature opcode table (dominant vs content-weighted operands)",
           "model": args.pretrained, "layers": analyzed, "n_operands": N, "sae_repo": args.sae_repo,
           "heads": out_heads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    # ---- report: 3-way legibility + content/dark focus ----
    def b(rec, m):
        bd = rec.get(f"bind_{m}")
        return f"{'/'.join(bd['q'])}->{'/'.join(bd['k'])}" if bd else "(insufficient support)"
    print(f"\n{'head':>5} {'z_token':>8} {'z_domSAE':>9} {'z_cntSAE':>9}  content-mode top binding (q->k)")
    for x in sorted(out_heads, key=lambda r: -(r["z_content"] if np.isfinite(r["z_content"]) else -9)):
        print(f"{x['head']:>5} {x['z_token']:>8.2f} {x['z_dominant']:>9.2f} {x['z_content']:>9.2f}  {b(x, 'content')!r}")

    content_heads = {"9.6", "9.9", "9.0", "9.8", "4.4"}      # name-movers + the resolved dark head + an inductor
    print("\n[content-head focus] does content-weighted selection beat dominant on content heads?")
    for x in out_heads:
        if x["head"] in content_heads:
            print(f"  {x['head']:>5}: token {x['z_token']:+.2f} | dom-SAE {x['z_dominant']:+.2f} | "
                  f"content-SAE {x['z_content']:+.2f}  binds {b(x, 'content')!r}")
    dark = {"1.10", "1.2", "1.4", "9.8"}
    print("\n[dark-head check] scorecard's dark heads under each operand basis (z>2 = legible):")
    for x in out_heads:
        if x["head"] in dark:
            print(f"  {x['head']:>5}: token {x['z_token']:+.2f} | dom-SAE {x['z_dominant']:+.2f} | "
                  f"content-SAE {x['z_content']:+.2f}")
    for m in MODES:
        legm = sum(1 for x in out_heads if np.isfinite(x[f"z_{m}"]) and x[f"z_{m}"] > 2)
        print(f"[legible z>2] {m:9} SAE operands: {legm}/{len(out_heads)}")
    legt = sum(1 for x in out_heads if np.isfinite(x["z_token"]) and x["z_token"] > 2)
    print(f"[legible z>2] token     operands: {legt}/{len(out_heads)}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
