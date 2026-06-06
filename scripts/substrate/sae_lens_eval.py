"""SAELens-dictionary cov95 + N1 on GPT-2 (replaces the self-trained-SAE stand-in).

Loads a published SAELens GPT-2 SAE (jbloom/GPT2-Small-SAEs-Reformatted,
blocks.8.hook_resid_pre, 24576 features) and scores per-tier cov95/mAUC against the
exact-lexical oracle on the matching layer-8 activations — then N1-rank/LN. The
self-trained SAE was token-dominated (lexical tier 0.00); a real production
dictionary should recover more. Streams the encode over feature chunks (24k feats
× 16k tokens won't fit dense). ReLU SAE: z = relu((x - b_dec) @ W_enc + b_enc).
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


def _best_auc_streaming(X, Wenc, benc, bdec, Y, feat_chunk=1024):
    """Per-label best symmetric AUC over latents, encoding feature-chunks on the fly."""
    n = X.shape[0]
    Yf = Y.astype(np.float64)
    npos = Yf.sum(0); nneg = n - npos
    valid = (npos > 0) & (nneg > 0)
    denom = np.where(valid, npos * nneg, 1.0)
    best = np.full(Y.shape[1], -np.inf)
    Xc = X - bdec
    F = Wenc.shape[1]
    for s in range(0, F, feat_chunk):
        z = Xc @ Wenc[:, s:s + feat_chunk] + benc[s:s + feat_chunk]
        np.maximum(z, 0, out=z)                          # ReLU
        keep = z.std(0) > 0
        if not keep.any():
            continue
        R = _column_ranks(z[:, keep])
        spos = Yf.T @ R
        auc = (spos - (npos * (npos + 1) / 2.0)[:, None]) / denom[:, None]
        best = np.maximum(best, np.maximum(auc, 1.0 - auc).max(axis=1))
    return np.where(valid, best, np.nan)


def _per_tier(best, tiers):
    src = np.array(tiers)
    out = {"all": {"cov95": float(np.nanmean(best >= 0.95)), "mauc": float(np.nanmean(best))}}
    for t in sorted(set(tiers)):
        b = best[src == t]
        out[t] = {"n": int((src == t).sum()), "cov95": float(np.nanmean(b >= 0.95)),
                  "mauc": float(np.nanmean(b))}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=Path("data/lm_bundle_gpt2_l8.npz"))
    p.add_argument("--labels", type=Path, default=Path("data/lm_labels_l8.json"))
    p.add_argument("--sae-repo", default="jbloom/GPT2-Small-SAEs-Reformatted")
    p.add_argument("--sae-file", default="blocks.8.hook_resid_pre/sae_weights.safetensors")
    p.add_argument("--output", type=Path, default=Path("runs/substrate/sae_lens_eval_summary.json"))
    args = p.parse_args(argv)

    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    lab = json.loads(args.labels.read_text())
    tiers = lab["tiers"]
    data = np.load(args.npz)
    X = data["X"].astype(np.float64)
    Y = data["Y"]
    d_model = X.shape[1]

    st = load_file(hf_hub_download(repo_id=args.sae_repo, filename=args.sae_file))
    Wenc = st["W_enc"].astype(np.float64)        # (d, F)
    Wdec = st["W_dec"].astype(np.float64)        # (F, d)
    benc = st["b_enc"].astype(np.float64)
    bdec = st["b_dec"].astype(np.float64)
    F = Wenc.shape[1]
    print(f"[1] SAELens SAE {args.sae_file}  d={d_model} F={F} ({F/d_model:.0f}x)  "
          f"X={X.shape} Y={Y.shape}")

    host = _per_tier(_best_auc_streaming(X, Wenc, benc, bdec, Y), tiers)
    print(f"[host] cov95={host['all']['cov95']:.3f} mAUC={host['all']['mauc']:.3f}  "
          + "  ".join(f"{t}={host[t]['cov95']:.2f}(m{host[t]['mauc']:.2f})" for t in host if t != "all"))
    out = {"sae_repo": args.sae_repo, "sae_file": args.sae_file, "d_model": d_model,
           "n_features": int(F), "n_samples": int(X.shape[0]), "host": host}

    # N1-rank: project host onto top-r decoder-atom subspace (atoms = W_dec rows)
    print("[N1-rank] project host onto top-r decoder-atom subspace")
    order = np.argsort(-np.linalg.norm(Wdec, axis=1))
    rows = []
    for r in [8, 32, 128, 384, d_model]:
        Q, _ = np.linalg.qr(Wdec[order[:r]].T)       # (d, <=d)
        Xp = X @ (Q @ Q.T)
        stt = _per_tier(_best_auc_streaming(Xp, Wenc, benc, bdec, Y), tiers)
        rows.append({"r": r, "rank": int(Q.shape[1]), **stt})
        print(f"    r={r:>4} (rank {Q.shape[1]:>3})  cov95={stt['all']['cov95']:.3f}  "
              + "  ".join(f"{t}={stt[t]['cov95']:.2f}" for t in stt if t != "all"))
    out["N1_rank"] = rows

    ln = X / (np.linalg.norm(X, axis=1, keepdims=True) / np.sqrt(d_model) + 1e-6)
    out["N1_layernorm"] = _per_tier(_best_auc_streaming(ln, Wenc, benc, bdec, Y), tiers)
    print(f"[N1-LN]   cov95={out['N1_layernorm']['all']['cov95']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
