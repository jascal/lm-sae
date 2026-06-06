"""SAE-feature opcode table: read B_h with MONOSEMANTIC SAE FEATURES as operands, not token centroids.

The coverage scorecard found ~7% of GPT-2's long-range (content) attention DARK — concentrated in
layer-1 heads (1.10/1.2/1.4) whose token-centroid B_h legibility is ~0 or negative. Token IDENTITY is the
wrong operand basis for them; they likely bind on syntactic / positional / semantic CONCEPTS. This swaps
the operand basis to a published SAELens GPT-2 SAE (jbloom/GPT2-Small-SAEs-Reformatted, resid_pre, 24576
feats/layer): operand_i = the feature's decoder direction d_i = unit(ln_1 * W_dec[i]).

  B_h[i,j] = d_i . M_h . d_j         (M_h = W_Q^h W_K^h.T / sqrt(head_dim))  -- feature i binds feature j

Each position is labelled by its DOMINANT SAE feature (argmax activation); we keep the top-N most-dominant
features as operands, so the empirical-attention validation is identical to the token table (positions
whose dominant feature is i query positions whose dominant feature is j) and leg_z is directly comparable.
Features are glossed by the top corpus tokens at their dominant positions (offline; no neuronpedia). Two
forward passes (hidden states -> encode -> operands; then attentions). GPT-2 + downloaded per-layer SAEs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--layers", default="1,4,9", help="layers whose heads to read with that layer's resid_pre SAE")
    p.add_argument("--n-operands", type=int, default=60, help="# top dominant SAE features as operands")
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--min-count", type=int, default=15, help="min ordered (i->j) pairs to score a cell")
    p.add_argument("--n-perm", type=int, default=25)
    p.add_argument("--sae-repo", default="jbloom/GPT2-Small-SAEs-Reformatted")
    p.add_argument("--token-summary", type=Path, default=Path("runs/qk_opcode_table_summary.json"),
                   help="token-operand opcode table, for the leg_z comparison")
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
    print(f"{args.pretrained}: layers {analyzed}  operands/layer {args.n_operands}  positions {P}")

    # ---- pass 1: residual (resid_pre = hidden_states[L]) for analyzed layers ----
    Hs = {L: [] for L in analyzed}
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True)
            for L in analyzed:
                Hs[L].append(o.hidden_states[L][0].float().numpy().astype(np.float64))
    Hs = {L: np.concatenate(v, 0) for L, v in Hs.items()}

    # ---- per layer: load SAE, encode -> dominant feature per position, pick operands + gloss ----
    operands = {}; pos_op = {}; gloss = {}
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
            seg = np.where(mx > 0, am, -1)
            dom[i:i + 512] = seg
        counts = np.bincount(dom[dom >= 0], minlength=F)
        feats = [int(f) for f in np.argsort(-counts) if counts[f] > 0][:args.n_operands]
        ln_w = tr.h[L].ln_1.weight.detach().numpy().astype(np.float64)
        D = Wdec[feats] * ln_w
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
        operands[L] = D
        po = np.full(P, -1)
        g = []
        for oi, f in enumerate(feats):
            sel = dom == f
            po[sel] = oi
            top = Counter(all_ids[sel].tolist()).most_common(3)
            g.append([tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t, _ in top])
        pos_op[L] = po; gloss[L] = g
        print(f"  layer {L}: F={F}  operands={len(feats)}  (e.g. f{feats[0]}~{g[0]}, f{feats[1]}~{g[1]})")
        del st, Wenc, Wdec

    # ---- pass 2: empirical attention over dominant-feature operand pairs ----
    N = args.n_operands
    att_sum = {L: np.zeros((H, N * N)) for L in analyzed}
    att_cnt = {L: np.zeros(N * N) for L in analyzed}
    with torch.no_grad():
        for ci, c in enumerate(chunks):
            o = tr(input_ids=torch.tensor([c]), output_attentions=True)
            Lc = len(c)
            ti, si = np.tril_indices(Lc, k=-1)                  # t > s
            for L in analyzed:
                po = pos_op[L][off[ci]:off[ci] + Lc]
                m = (po[ti] >= 0) & (po[si] >= 0)
                tt, ss = ti[m], si[m]
                flat = po[tt] * N + po[ss]
                np.add.at(att_cnt[L], flat, 1.0)
                aL = o.attentions[L][0].float().numpy()
                for h in range(H):
                    np.add.at(att_sum[L][h], flat, aL[h][tt, ss])

    # ---- B_h + legibility (vs label-permuted null), comparable to the token table ----
    tokleg = {}
    if args.token_summary.exists():
        tj = json.loads(args.token_summary.read_text())
        tokleg = {f"{h['layer']}.{h['head']}": float(h.get("leg_z", float("nan"))) for h in tj.get("heads", [])}
    rng = np.random.default_rng(0)
    perms = [rng.permutation(N) for _ in range(args.n_perm)]
    offmask = ~np.eye(N, dtype=bool)
    out_heads = []
    for L in analyzed:
        cnt2 = att_cnt[L].reshape(N, N)
        supp = (cnt2 >= args.min_count) & offmask
        blk = tr.h[L]
        W = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wq, Wk = W[:, :d], W[:, d:2 * d]
        Dl = operands[L]
        for h in range(H):
            Mh = Wq[:, h * hd:(h + 1) * hd] @ Wk[:, h * hd:(h + 1) * hd].T / np.sqrt(hd)
            B = Dl @ Mh @ Dl.T
            A = att_sum[L][h].reshape(N, N) / np.maximum(cnt2, 1)
            bv, av = B[supp], A[supp]
            leg = _spearman(bv, av)
            null = [_spearman(B[np.ix_(pp, pp)][supp], av) for pp in perms]
            z = (leg - np.nanmean(null)) / (np.nanstd(null) + 1e-9) if np.isfinite(leg) else float("nan")
            Boff = B.copy(); np.fill_diagonal(Boff, -np.inf)
            qi, ki = np.unravel_index(int(np.argmax(Boff)), Boff.shape)
            hn = f"{L}.{h}"
            out_heads.append({"head": hn, "layer": L, "leg_sae": leg, "z_sae": float(z),
                              "z_token": tokleg.get(hn, float("nan")), "n_supp": int(supp.sum()),
                              "top_bind": {"q_feat_gloss": gloss[L][qi], "k_feat_gloss": gloss[L][ki],
                                           "value": float(B[qi, ki])}})

    out = {"experiment": "SAE-feature opcode table", "model": args.pretrained, "layers": analyzed,
           "n_operands": N, "sae_repo": args.sae_repo, "heads": out_heads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    # ---- report: SAE vs token legibility, dark-head focus ----
    print(f"\n{'head':>5} {'z_token':>8} {'z_sae':>7} {'Δz':>6} {'supp':>5}  top SAE binding (query-feat -> key-feat)")
    for x in sorted(out_heads, key=lambda r: -(r["z_sae"] if np.isfinite(r["z_sae"]) else -9)):
        dz = x["z_sae"] - x["z_token"] if np.isfinite(x["z_token"]) else float("nan")
        q = "/".join(x["top_bind"]["q_feat_gloss"]); k = "/".join(x["top_bind"]["k_feat_gloss"])
        print(f"{x['head']:>5} {x['z_token']:>8.2f} {x['z_sae']:>7.2f} {dz:>6.2f} {x['n_supp']:>5}  {q!r}->{k!r}")
    dark = {"1.10", "1.2", "1.4", "9.8"}
    dk = [x for x in out_heads if x["head"] in dark]
    if dk:
        print("\n[dark-head check] did SAE operands make the scorecard's dark heads legible (z>2)?")
        for x in dk:
            verdict = "NOW LEGIBLE" if x["z_sae"] > 2 else "still dark"
            print(f"  {x['head']:>5}: z_token {x['z_token']:+.2f} -> z_sae {x['z_sae']:+.2f}  [{verdict}]  "
                  f"binds {'/'.join(x['top_bind']['q_feat_gloss'])!r}->{'/'.join(x['top_bind']['k_feat_gloss'])!r}")
    leg_sae = sum(1 for x in out_heads if x["z_sae"] > 2)
    leg_tok = sum(1 for x in out_heads if np.isfinite(x["z_token"]) and x["z_token"] > 2)
    print(f"\n[legible heads z>2] token-operand {leg_tok}/{len(out_heads)}  ->  SAE-operand {leg_sae}/{len(out_heads)}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
