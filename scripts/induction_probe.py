"""Induction pair-probe: is the entangled core RELATIONAL (inference rules) or opaque?

The bigram pair-cov95 test failed because common bigrams are COMPILED into unary features.
Induction (in-context copy) can't be compiled: "is this token a repeat of one earlier in
THIS context" is variable per-sequence, so no fixed feature encodes it -- it must be read
by binding. The decisive labels:
  - unary control:      cur==X (token identity)                          arity-1
  - generic relational: is_repeat[t] (token appeared earlier in chunk)   arity-2 (maybe compiled by induction heads -> a "repeat flag")
  - per-token relational: cur==X AND is_repeat (un-compilable: too many X)  arity-2, NOT compilable
Compare single-cov95 (unary detector) vs pair-cov95 (AND z_i*z_j, XOR |z_i-z_j|). If
per-token-repeat is invisible to singles but read by the PAIR (cur=X latent x repeat-flag
latent), the core's induction is relational -- read by binding, not a compiled feature.
Run on GPT-2 (has induction heads), mid-late layer.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lm_bundle import COMMON  # noqa: E402
from forge_cov_mechanism import _column_ranks, _encode, _train_topk_sae  # noqa: E402
from preserve_hybrid_tiny import _auc_matrix  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--layer", type=int, default=6, help="residual layer (induction live mid-late)")
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--sae-steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--topk-pairs", type=int, default=48)
    p.add_argument("--output", type=Path, default=Path("runs/induction_probe_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = GPT2LMHeadModel(GPT2Config(**ck["config"])); model.load_state_dict(ck["state_dict"])
    else:
        model = GPT2LMHeadModel.from_pretrained(args.pretrained)
    model.eval(); tr = model.transformer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx)]
    all_ids = [j for c in chunks for j in c]
    N = len(all_ids)
    with torch.no_grad():
        X = np.concatenate([tr(input_ids=torch.tensor([c]), output_hidden_states=True)
                            .hidden_states[args.layer][0].float().numpy() for c in chunks], 0).astype(np.float32)
    print(f"model={'ckpt' if args.ckpt else args.pretrained} layer {args.layer}  X={X.shape}")

    # is_repeat[t]: token appeared earlier IN THIS CHUNK (in-context, un-compilable)
    is_rep = np.zeros(N, np.uint8); off = 0
    for c in chunks:
        seen = set()
        for i, tid in enumerate(c):
            if tid in seen:
                is_rep[off + i] = 1
            seen.add(tid)
        off += len(c)
    print(f"is_repeat base rate = {is_rep.mean():.2f}")

    common_ids = {c: tok(c, add_special_tokens=False)["input_ids"][0]
                  for c in COMMON if len(tok(c, add_special_tokens=False)["input_ids"]) == 1}
    cols, tiers = [], []
    for c, cid in common_ids.items():                       # arity-1 control
        cols.append(np.array([1 if j == cid else 0 for j in all_ids], np.uint8)); tiers.append("unary_cur")
    cols.append(is_rep.copy()); tiers.append("relational_isrepeat")  # generic repeat flag
    # repeat at fixed distance k (induction at offset k) -- clean relational, not per-token-degenerate
    for kk in (1, 2, 3, 4):
        rk = np.zeros(N, np.uint8); off = 0
        for c in chunks:
            for i in range(kk, len(c)):
                if c[i] == c[i - kk]:
                    rk[off + i] = 1
            off += len(c)
        cols.append(rk); tiers.append("relational_distk")
    # per-token-repeat for BALANCED tokens (cur=X occurs both as first AND repeat): un-compilable
    cur_cnt = Counter(all_ids)
    rep_by_tok = Counter(all_ids[t] for t in range(N) if is_rep[t])
    balanced = [tid for tid, cnt in cur_cnt.most_common(2000)
                if cnt >= 4 * args.min_pos and 0.3 <= rep_by_tok[tid] / cnt <= 0.7][:12]
    for tid in balanced:
        cols.append(np.array([1 if (all_ids[t] == tid and is_rep[t]) else 0 for t in range(N)], np.uint8))
        tiers.append("relational_pertoken")
        cols.append(np.array([1 if all_ids[t] == tid else 0 for t in range(N)], np.uint8))
        tiers.append("unary_balanced")
    print(f"balanced tokens for per-token-repeat: {len(balanced)}")

    Y = np.stack(cols, 1); npos = Y.sum(0)
    keep = (npos >= args.min_pos) & (npos <= N - args.min_pos)
    Y = Y[:, keep]; tiers = np.array([t for t, k in zip(tiers, keep) if k])
    print(f"labels={Y.shape}  tiers={dict(Counter(tiers))}")

    params = _train_topk_sae(X, args.width, args.k, args.sae_steps, 1e-3, 0)
    z = _encode(X, params, args.k)
    A_s, ok = _auc_matrix(_column_ranks(z), Y)
    single = np.where(ok, A_s.max(1), np.nan)

    K = args.topk_pairs
    topi = np.argsort(-z.std(0))[:K]
    zt = z[:, topi]
    feats = []
    for i in range(K):
        for j in range(i + 1, K):
            feats.append(zt[:, i] * zt[:, j]); feats.append(np.abs(zt[:, i] - zt[:, j]))
    P = np.stack(feats, 1).astype(np.float32)
    A_p, _ = _auc_matrix(_column_ranks(P), Y)
    pair = np.where(ok, np.maximum(A_s.max(1), A_p.max(1)), np.nan)
    print(f"pairs scored: {P.shape[1]} (AND+XOR over top-{K})")

    out = {"experiment": "induction pair-probe", "model": (str(args.ckpt) if args.ckpt else args.pretrained),
           "layer": args.layer, "is_repeat_rate": float(is_rep.mean()), "per_tier": {}}
    print(f"\n{'tier':22} {'n':>3} {'single_cov95':>12} {'pair_cov95':>11} {'delta':>7} "
          f"{'single_mAUC':>11} {'pair_mAUC':>9}")
    for t in ["unary_cur", "unary_balanced", "relational_isrepeat", "relational_distk",
              "relational_pertoken"]:
        msk = tiers == t
        if not msk.any():
            continue
        s, pr = single[msk], pair[msk]
        st = {"n": int(msk.sum()), "single_cov95": float(np.nanmean(s >= 0.95)),
              "pair_cov95": float(np.nanmean(pr >= 0.95)), "single_mauc": float(np.nanmean(s)),
              "pair_mauc": float(np.nanmean(pr))}
        out["per_tier"][t] = st
        print(f"{t:22} {st['n']:>3} {st['single_cov95']:>12.3f} {st['pair_cov95']:>11.3f} "
              f"{st['pair_cov95']-st['single_cov95']:>+7.3f} {st['single_mauc']:>11.3f} {st['pair_mauc']:>9.3f}")

    pt = out["per_tier"].get("relational_pertoken", {})
    un = out["per_tier"].get("unary_cur", {})
    if pt and un:
        d_rel = pt["pair_cov95"] - pt["single_cov95"]; d_un = un["pair_cov95"] - un["single_cov95"]
        out["pertoken_pair_gain"] = d_rel; out["unary_pair_gain"] = d_un
        print(f"\n[verdict] per-token-repeat pair-gain {d_rel:+.3f} (mAUC {pt['single_mauc']:.2f}->{pt['pair_mauc']:.2f}) "
              f"vs unary {d_un:+.3f} -> "
              f"{'RELATIONAL: induction read by BINDING (core is relational, not opaque)' if d_rel > d_un + 0.08 else 'no clear relational binding signal'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
