"""lm-sae cov95 + N1 mechanism ablation on GPT-2 (Recipe A, frozen-LLM regime).

The first cov95 forge-tax *regime* check on a real text language model. Trains a
TopK SAE on GPT-2 residual activations (stand-in for a SAELens production
dictionary), scores per-tier cov95/mAUC against the exact-lexical oracle, then runs
bio-sae's N1 host-side probes:
  - rank : project host onto top-r decoder-atom subspace, sweep r
  - LN   : one LayerNorm on the host activation
  - TopK : re-score at varying encode-k
The headline question: is GPT-2's GT signal RANK-ROBUST (like the frozen ESM-2 ->
emergent tax -> *preserve* regime) or rank-sensitive (like the trainable econ host
-> *concentrate*)? The cross-substrate prediction is rank-robust.

Self-contained: torch + numpy only. (The forged-vs-host tax + preserve hybrid need
the sae-forge GPT2Adapter -- the immediate follow-up; this is the host-side regime
probe, exactly bio-sae's N1 core.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _column_ranks(Z):
    n, d = Z.shape
    order = Z.argsort(axis=0)
    ranks = np.empty((n, d), dtype=np.float64)
    ranks[order, np.arange(d)[None, :]] = np.arange(1, n + 1, dtype=np.float64)[:, None]
    return ranks


def _best_auc_per_label(Z, Y, chunk=512):
    """Per-label best symmetric AUC over latents. Z (N,F), Y (N,M) -> (M,)."""
    n, F = Z.shape
    Yf = Y.astype(np.float64)
    npos = Yf.sum(0); nneg = n - npos
    valid = (npos > 0) & (nneg > 0)
    denom = np.where(valid, npos * nneg, 1.0)
    best = np.full(Y.shape[1], -np.inf)
    for s in range(0, F, chunk):
        R = _column_ranks(Z[:, s:s + chunk])              # (N, k)
        spos = Yf.T @ R                                   # (M, k)
        auc = (spos - (npos * (npos + 1) / 2.0)[:, None]) / denom[:, None]
        sym = np.maximum(auc, 1.0 - auc)
        best = np.maximum(best, sym.max(axis=1))
    return np.where(valid, best, np.nan)


def _per_tier(best, tiers):
    src = np.array(tiers)
    out = {"all": {"cov95": float(np.nanmean(best >= 0.95)), "mauc": float(np.nanmean(best))}}
    for t in sorted(set(tiers)):
        b = best[src == t]
        out[t] = {"n": int((src == t).sum()), "cov95": float(np.nanmean(b >= 0.95)),
                  "mauc": float(np.nanmean(b))}
    return out


def _train_topk_sae(X, width, k, steps, lr, seed):
    import torch

    torch.manual_seed(seed)
    n, d = X.shape
    Xt = torch.from_numpy(X)
    W_dec = torch.randn(d, width)
    W_dec /= W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
    W_dec = torch.nn.Parameter(W_dec)
    W_enc = torch.nn.Parameter(W_dec.detach().t().clone())
    b_dec = torch.nn.Parameter(Xt.mean(0).clone())
    b_enc = torch.nn.Parameter(torch.zeros(width))
    opt = torch.optim.Adam([W_enc, W_dec, b_enc, b_dec], lr=lr)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, n, (2048,), generator=g)
        x = Xt[idx]
        pre = torch.relu((x - b_dec) @ W_enc.t() + b_enc)
        topv, topi = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre).scatter(-1, topi, topv)
        xhat = z @ W_dec.t() + b_dec
        loss = ((xhat - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            W_dec /= W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return (W_enc.detach(), W_dec.detach(), b_enc.detach(), b_dec.detach())


def _encode(X, params, k):
    import torch

    W_enc, W_dec, b_enc, b_dec = params
    with torch.no_grad():
        pre = torch.relu((torch.from_numpy(X) - b_dec) @ W_enc.t() + b_enc)
        topv, topi = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre).scatter(-1, topi, topv)
    return z.numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=Path, default=Path("data/lm_bundle_gpt2.npz"))
    p.add_argument("--labels", type=Path, default=Path("data/lm_labels.json"))
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("runs/substrate/cov_mechanism_summary.json"))
    args = p.parse_args(argv)

    lab = json.loads(args.labels.read_text())
    tiers = lab["tiers"]
    data = np.load(args.npz)
    Xraw = data["X"].astype(np.float32)
    Y = data["Y"]
    mu, sd = Xraw.mean(0, keepdims=True), Xraw.std(0, keepdims=True) + 1e-6
    X = ((Xraw - mu) / sd).astype(np.float32)
    d_model = X.shape[1]
    oc = args.width / d_model
    print(f"[1] {lab['model']} layer {lab['layer']}  X={X.shape}  Y={Y.shape}  "
          f"SAE width={args.width} ({oc:.1f}x over-complete) k={args.k}")

    print(f"[2] train TopK SAE ({args.steps} steps)")
    params = _train_topk_sae(X, args.width, args.k, args.steps, args.lr, args.seed)

    host = _per_tier(_best_auc_per_label(_encode(X, params, args.k), Y), tiers)
    print(f"[host] cov95={host['all']['cov95']:.3f} mAUC={host['all']['mauc']:.3f}  "
          + "  ".join(f"{t}={host[t]['cov95']:.2f}" for t in host if t != "all"))
    out = {"model": lab["model"], "layer": lab["layer"], "d_model": d_model,
           "n_samples": int(X.shape[0]), "sae_width": args.width, "over_complete": round(oc, 2),
           "note": "self-trained TopK SAE (SAELens dictionary stand-in)", "host": host}

    # N1-rank
    print("[N1-rank] project host onto top-r decoder-atom subspace")
    W_dec = params[1].numpy().astype(np.float64)            # (d, width); atoms = columns
    order = np.argsort(-np.linalg.norm(W_dec, axis=0))
    rank_rows = []
    for r in [4, 8, 16, 32, 64, 128, 256, d_model]:
        Q, _ = np.linalg.qr(W_dec[:, order[:r]])
        Xp = (X @ (Q @ Q.T)).astype(np.float32)
        st = _per_tier(_best_auc_per_label(_encode(Xp, params, args.k), Y), tiers)
        rank_rows.append({"r": r, "rank": int(Q.shape[1]), **st})
        print(f"    r={r:>4} (rank {Q.shape[1]:>3})  cov95={st['all']['cov95']:.3f}  "
              + "  ".join(f"{t}={st[t]['cov95']:.2f}" for t in st if t != "all"))
    out["N1_rank"] = rank_rows

    # N1-LN
    ln = X / (np.linalg.norm(X, axis=1, keepdims=True) / np.sqrt(d_model) + 1e-6)
    out["N1_layernorm"] = _per_tier(_best_auc_per_label(_encode(ln.astype(np.float32), params, args.k), Y), tiers)
    print(f"[N1-LN]   cov95={out['N1_layernorm']['all']['cov95']:.3f}")

    # N1-TopK
    print("[N1-TopK] re-score at varying encode-k")
    topk_rows = []
    for k in sorted({8, 16, args.k, 64, 128, args.width}):
        st = _per_tier(_best_auc_per_label(_encode(X, params, k), Y), tiers)
        topk_rows.append({"k": int(k), **st})
        print(f"    k={k:>4}  cov95={st['all']['cov95']:.3f}")
    out["N1_topk"] = topk_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
