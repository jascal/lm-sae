"""Composition graph — the model's call graph (which component's write feeds which read).

The disassembly is a flat listing; this adds control flow. Canonical Q/K/V-composition scores
(Elhage et al.), computed on RAW WEIGHTS (operand-basis-free — sidesteps the centroid-basis confound that
faked the 'narrow write bus'): head A writes via OV_A = W_V^A W_O^A; head B reads its query/key/value via
W_{Q,K,V}^B. The edge
  comp(A->B, port) = ||OV_A @ W_port^B||_F / (||OV_A||_F ||W_port^B||_F)
is how much of A's output lands in B's query/key/value subspace. We project out the shared mean-write
direction first (the write_bus_check artifact), build the causal adjacency (L_A < L_B), and VALIDATE:
do the prev-token heads K-compose into the induction heads (induction = K-composition)? Then list the
strongest edges + the attention->MLP and MLP->attention edges. GPT-2; weights + one forward for labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-prevtok", type=int, default=6)
    p.add_argument("--n-induct", type=int, default=6)
    p.add_argument("--output", type=Path, default=Path("runs/composition_graph_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/composition_graph.png"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]

    # ---- behavioural labels: prev-token (Δ=1) + induction per head ----
    pt = np.zeros((nL, H)); ptn = 0; ind = np.zeros((nL, H)); indn = 0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_attentions=True)
            Lc = len(c); ca = np.array(c); ptn += Lc - 1
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]; qi = np.arange(Lc)
            IM = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
            indn += int(IM.sum())
            for L in range(nL):
                aL = o.attentions[L][0].float().numpy()
                pt[L] += np.diagonal(aL, offset=-1, axis1=1, axis2=2).sum(1)
                ind[L] += (aL * IM[None]).sum((1, 2))
    prevtok = (pt / max(ptn, 1)).reshape(-1)
    induct = (ind / max(indn, 1)).reshape(-1)
    heads = [(L, h) for L in range(nL) for h in range(H)]
    layer_of = np.array([L for L, _ in heads])

    # ---- per-head weights: OV (write), Q/K/V (read) ----
    OV = np.zeros((nL * H, d, d), np.float32)
    WQ = np.zeros((nL * H, d, hd), np.float32); WK = np.zeros_like(WQ); WV = np.zeros_like(WQ)
    Win = []; Wout = []
    for L in range(nL):
        blk = tr.h[L]
        Wc = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wo = blk.attn.c_proj.weight.detach().numpy().astype(np.float64)
        Wq, Wk, Wv = Wc[:, :d], Wc[:, d:2 * d], Wc[:, 2 * d:3 * d]
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd); i = L * H + h
            OV[i] = (Wv[:, sl] @ Wo[sl, :]).astype(np.float32)
            WQ[i] = Wq[:, sl].astype(np.float32); WK[i] = Wk[:, sl].astype(np.float32)
            WV[i] = Wv[:, sl].astype(np.float32)
        Win.append(blk.mlp.c_fc.weight.detach().numpy().astype(np.float32))    # (d, dff)
        Wout.append(blk.mlp.c_proj.weight.detach().numpy().astype(np.float32))  # (dff, d)

    # remove the shared mean-write direction (write_bus_check artifact) from OV outputs
    outdirs = OV.reshape(nL * H, d * d)  # cheap proxy: top PC of stacked OV output columns
    u = np.linalg.svd((OV.transpose(0, 2, 1).reshape(-1, d)), full_matrices=False)[2][0]  # (d,)
    P = np.eye(d, dtype=np.float32) - np.outer(u, u).astype(np.float32)
    OV = np.einsum("nij,jk->nik", OV, P)                                       # project output onto complement
    ovn = np.linalg.norm(OV.reshape(nL * H, -1), axis=1) + 1e-9

    def comp(Wport):
        portn = np.linalg.norm(Wport.reshape(nL * H, -1), axis=1) + 1e-9
        S = np.zeros((nL * H, nL * H), np.float32)
        for a in range(nL * H):
            La = layer_of[a]
            for b in range(nL * H):
                if layer_of[b] <= La:
                    continue
                S[a, b] = np.linalg.norm(OV[a] @ Wport[b]) / (ovn[a] * portn[b])
        return S

    Kc = comp(WK); Qc = comp(WQ)
    print(f"{args.pretrained}: {nL*H} heads; composition scores computed (mean-write removed)")

    A = [i for i in np.argsort(-prevtok)[:args.n_prevtok]]
    B = [i for i in np.argsort(-induct)[:args.n_induct]]
    causal = layer_of[:, None] < layer_of[None, :]
    base_k = float(Kc[causal].mean())
    pt2ind = float(np.mean([Kc[a, b] for a in A for b in B if layer_of[a] < layer_of[b]]))
    rand_ind = float(np.mean([Kc[a, b] for a in np.random.default_rng(0).integers(0, nL * H, len(A) * 4)
                              for b in B if layer_of[a] < layer_of[b]] or [0]))
    print(f"\n[validation] K-composition prev-token->induction {pt2ind:.3f}  vs  causal baseline {base_k:.3f}  "
          f"vs random-writer->induction {rand_ind:.3f}  -> "
          f"{'RECOVERS the induction wiring (prev-tok K-composes into inductors)' if pt2ind > 1.3 * base_k else 'no clear induction wiring'}")

    def name(i):
        return f"{heads[i][0]}.{heads[i][1]}"
    print(f"\nprev-token writers: {[name(a) for a in A]}   induction heads: {[name(b) for b in B]}")
    print("\ntop K-composition edges (writer -> reader.key):")
    flat = [(a, b, Kc[a, b]) for a in range(nL * H) for b in range(nL * H) if Kc[a, b] > 0]
    for a, b, s in sorted(flat, key=lambda r: -r[2])[:12]:
        tag = "  [prev-tok->induction]" if (a in A and b in B) else ""
        print(f"  {name(a):>5} -> {name(b):>5}  {s:.3f}{tag}")
    print("\ntop Q-composition edges (writer -> reader.query):")
    flatq = [(a, b, Qc[a, b]) for a in range(nL * H) for b in range(nL * H) if Qc[a, b] > 0]
    for a, b, s in sorted(flatq, key=lambda r: -r[2])[:8]:
        print(f"  {name(a):>5} -> {name(b):>5}  {s:.3f}")

    # attention -> MLP and MLP -> attention (layer-level, mean-removed write already on OV)
    a2m, m2a = {}, {}
    for L in range(nL):
        # how much each earlier head feeds this layer's MLP read (W_in)
        wi = Win[L]; win_n = np.linalg.norm(wi) + 1e-9
        sc = [np.linalg.norm(OV[a] @ wi) / (ovn[a] * win_n) for a in range(nL * H) if layer_of[a] < L]
        a2m[L] = float(np.mean(sc)) if sc else 0.0
    out = {"experiment": "composition graph", "model": args.pretrained, "n_heads": nL * H,
           "K_prevtok_to_induction": pt2ind, "K_causal_baseline": base_k, "K_random_to_induction": rand_ind,
           "prevtok_writers": [name(a) for a in A], "induction_heads": [name(b) for b in B],
           "top_K_edges": [[name(a), name(b), float(s)] for a, b, s in sorted(flat, key=lambda r: -r[2])[:20]],
           "top_Q_edges": [[name(a), name(b), float(s)] for a, b, s in sorted(flatq, key=lambda r: -r[2])[:12]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axK, axS) = plt.subplots(1, 2, figsize=(12.5, 5.2))
        im = axK.imshow(Kc, cmap="magma", aspect="auto")
        axK.set_title("K-composition adjacency (writer row -> reader-key col)", fontsize=10)
        axK.set_xlabel("reader head (by layer)"); axK.set_ylabel("writer head (by layer)")
        for b in B:
            axK.axvline(b, color="cyan", lw=0.4, alpha=0.5)
        for a in A:
            axK.axhline(a, color="lime", lw=0.4, alpha=0.5)
        fig.colorbar(im, ax=axK, fraction=0.046)
        axS.bar([0, 1, 2], [pt2ind, base_k, rand_ind], color=["#d62728", "#999999", "#1f77b4"], edgecolor="k")
        axS.set_xticks([0, 1, 2]); axS.set_xticklabels(["prev-tok\n→induction", "causal\nbaseline", "random\n→induction"])
        axS.set_ylabel("mean K-composition"); axS.set_title("induction wiring recovered?", fontsize=10)
        fig.suptitle("GPT-2 composition graph: which write feeds which read (mean-write removed)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
