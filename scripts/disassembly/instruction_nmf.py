"""NMF head-clustering of the QK instruction tensor — a parts-based opcode vocabulary.

SVD gave an orthogonal (hence diffuse-by-construction) basis. NMF is parts-based: V ≈ W H with W, H ≥ 0,
so each component H_k is an ADDITIVE template that heads COMPOSE (W = non-negative head loadings), which
is the right model for "heads built from a few reused opcodes". B_h is signed, so we feed NMF the
non-negative split [relu(B), relu(-B)] per head and recombine each component into a signed template
T_k = pos_k - neg_k. For each component we name the shape (copy/broadcast/bind), its top token bindings,
and the heads that load on it (argmax = hard cluster). GPT-2.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from build_lm_bundle import COMMON  # noqa: E402


def _nmf(V, k, steps=400, seed=0):
    """Frobenius NMF via multiplicative updates. V (n,m) >= 0 -> W (n,k), H (k,m)."""
    rng = np.random.default_rng(seed)
    n, m = V.shape
    scale = np.sqrt(V.mean() / k) + 1e-9
    W = np.abs(rng.standard_normal((n, k))) * scale
    H = np.abs(rng.standard_normal((k, m))) * scale
    for _ in range(steps):
        H *= (W.T @ V) / (W.T @ W @ H + 1e-9)
        W *= (V @ H.T) / (W @ (H @ H.T) + 1e-9)
    return W, H


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--ref-layer", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=40)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--k", type=int, default=8, help="# NMF opcodes")
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/instruction_nmf_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained).eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx)]
    all_ids = [j for c in chunks for j in c]
    cnt = Counter(all_ids)
    cand = [tok(c, add_special_tokens=False)["input_ids"][0] for c in COMMON
            if len(tok(c, add_special_tokens=False)["input_ids"]) == 1]
    cand += [t for t, _ in cnt.most_common(200)]
    seen, toks = set(), []
    for t in cand:
        if t not in seen and cnt[t] >= args.min_pos:
            seen.add(t); toks.append(t)
        if len(toks) >= args.n_tokens:
            break
    nt = len(toks); tok2i = {t: i for i, t in enumerate(toks)}
    nm = [tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t in toks]

    csum = np.zeros((nt, d)); ccnt = np.zeros(nt); gsum = np.zeros(d); gn = 0
    with torch.no_grad():
        for c in chunks:
            hs = tr(input_ids=torch.tensor([c]), output_hidden_states=True).hidden_states[args.ref_layer][0]
            hs = hs.float().numpy()
            pid = np.array([tok2i.get(t, -1) for t in c]); m = pid >= 0
            np.add.at(csum, pid[m], hs[m]); np.add.at(ccnt, pid[m], 1)
            gsum += hs.sum(0); gn += len(c)
    D = csum / np.maximum(ccnt, 1)[:, None] - gsum / gn
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)

    heads, Bs = [], []
    for L in range(nL):
        Wc = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wq, Wk = Wc[:, :d], Wc[:, d:2 * d]
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            B = D @ (Wq[:, sl] @ Wk[:, sl].T / np.sqrt(hd)) @ D.T
            Bs.append(B / (np.linalg.norm(B) + 1e-12)); heads.append((L, h))
    B = np.stack(Bs, 0)                                   # (144, nt, nt) shape-normalised
    flat = B.reshape(B.shape[0], -1)
    V = np.concatenate([np.maximum(flat, 0), np.maximum(-flat, 0)], 1)   # (144, 2nt^2) non-negative
    Wld, Hc = _nmf(V, args.k, steps=500, seed=0)
    recon = 1 - np.linalg.norm(V - Wld @ Hc) / (np.linalg.norm(V) + 1e-9)
    assign = Wld.argmax(1)
    print(f"{args.pretrained}: NMF k={args.k}, reconstruction {recon:.2f}; cluster sizes "
          f"{dict(sorted(Counter(assign.tolist()).items()))}")

    off = ~np.eye(nt, dtype=bool)
    out = {"experiment": "instruction NMF head-clustering", "model": args.pretrained, "k": args.k,
           "nmf_reconstruction": float(recon), "opcodes": []}
    order = np.argsort(-np.array([(assign == i).sum() for i in range(args.k)]))
    for i in order:
        Tm = Hc[i, :nt * nt].reshape(nt, nt) - Hc[i, nt * nt:].reshape(nt, nt)   # signed template
        nrm = np.linalg.norm(Tm) + 1e-12
        Tm = Tm / nrm
        diag = float(np.diag(Tm).mean() - Tm[off].mean())
        colmass = np.abs(Tm).sum(0); broadcast = float(colmass.max() / (colmass.sum() + 1e-9))
        rowpk = float((np.abs(Tm) / (np.abs(Tm).sum(1, keepdims=True) + 1e-9)).max(1).mean())
        shape = ("COPY" if diag > 0.4 * np.abs(Tm).max() else
                 "BROADCAST" if broadcast > 3.5 / nt else
                 "BIND" if rowpk > 0.4 else "DIFFUSE")
        Toff = Tm.copy(); np.fill_diagonal(Toff, 0.0)
        binds = [[nm[qi], nm[ki], round(float(Toff[qi, ki]), 3)]
                 for qi, ki in zip(*np.unravel_index(np.argsort(-np.abs(Toff), axis=None)[:6], Toff.shape))]
        members = [heads[x] for x in np.where(assign == i)[0]]
        layers = sorted(Counter(L for L, _ in members).items())
        top_load = [f"{heads[x][0]}.{heads[x][1]}" for x in np.argsort(-Wld[:, i])[:6]]
        out["opcodes"].append({"id": int(i), "shape": shape, "n_heads": len(members),
                               "diag_score": diag, "broadcast": broadcast, "row_peak": rowpk,
                               "top_binds": binds, "top_heads": top_load,
                               "layer_hist": {str(k_): v for k_, v in layers}})
        col = ", ".join(f"L{L}:{c}" for L, c in layers)
        print(f"\n[opcode {i}] {shape}  {len(members)} heads  diag {diag:+.2f} bcast {broadcast:.2f} rowpk {rowpk:.2f}")
        print(f"   top heads: {', '.join(top_load)}   layers: {col}")
        print("   binds (query->key): " + "; ".join(f"{q!r}->{k!r} {v:+.2f}" for q, k, v in binds))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    shapes = Counter(o["shape"] for o in out["opcodes"])
    clean = sum(1 for o in out["opcodes"] if o["shape"] != "DIFFUSE")
    print(f"\n[vocabulary] {args.k} NMF opcodes: {dict(shapes)}; {clean}/{args.k} are clean (non-DIFFUSE) "
          f"vs SVD's 0/6 -> {'NMF gives a cleaner parts-based opcode set' if clean >= 3 else 'still mostly diffuse'}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
