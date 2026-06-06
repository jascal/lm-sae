"""Cross-position pair-probe: is induction readable by binding features ACROSS positions?

The within-position pair-probe failed because induction binds token t to t-k (it lives in
attention, across positions). This builds the right probe: a CROSS-position pair
z_i[t] * z_j[t-k] -- the current-token latent times a latent k positions back. For
repeat-at-distance-k, the diagonal z_X[t]*z_X[t-k] fires exactly when token X is at BOTH
positions -> reads the repeat. Compare, per distance k, on label repeat_at_k[t]=(tok[t]==tok[t-k]):
  - single        best z_i[t]                  (within-position; should FAIL -- can't see t-k)
  - within-pair   best z_i[t]*z_j[t]           (within-position; should FAIL)
  - CROSS-pair    best z_i[t]*z_j[t-k]          (cross-position; should SUCCEED)
If cross >> single/within, the induction relation IS pair-readable -- just across positions
(the within-residual shadow of the attention binding). GPT-2, mid layer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from forge_cov_mechanism import _column_ranks, _encode, _train_topk_sae  # noqa: E402
from preserve_hybrid_tiny import _auc_matrix  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--sae-steps", type=int, default=600)
    p.add_argument("--topk", type=int, default=48, help="# latents to pair")
    p.add_argument("--dists", default="1,2,3")
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/cross_position_probe_summary.json"))
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

    params = _train_topk_sae(X, args.width, args.k, args.sae_steps, 1e-3, 0)
    z = _encode(X, params, args.k)
    topi = np.argsort(-z.std(0))[:args.topk]
    zt = z[:, topi]                                          # (N, K)
    chunk_id = np.concatenate([[ci] * len(c) for ci, c in enumerate(chunks)])

    def best_auc(F, lab, valid):
        A, ok = _auc_matrix(_column_ranks(F[valid]), lab[valid].reshape(-1, 1))
        return float(A.max()) if ok[0] else float("nan")

    def cov95(F, lab, valid):
        A, ok = _auc_matrix(_column_ranks(F[valid]), lab[valid].reshape(-1, 1))
        return float((A[0] >= 0.95).any()) if ok[0] else 0.0

    out = {"experiment": "cross-position pair-probe", "model": (str(args.ckpt) if args.ckpt else args.pretrained),
           "layer": args.layer, "per_dist": {}}
    # within-position pair features (distance 0)
    K = args.topk
    within = np.stack([zt[:, i] * zt[:, j] for i in range(K) for j in range(i + 1, K)], 1).astype(np.float32)

    print(f"\n{'dist k':>6} {'n_pos':>6} {'single':>8} {'within_pair':>12} {'CROSS_pair':>11} "
          f"{'single_c95':>10} {'cross_c95':>9}")
    for kk in [int(x) for x in args.dists.split(",")]:
        # repeat_at_k label + cross-position shifted latents (within chunk)
        lab = np.zeros(N, np.uint8); ztp = np.zeros_like(zt); valid = np.zeros(N, bool)
        for t in range(N):
            if t - kk >= 0 and chunk_id[t] == chunk_id[t - kk]:
                valid[t] = True
                ztp[t] = zt[t - kk]
                if all_ids[t] == all_ids[t - kk]:
                    lab[t] = 1
        if lab[valid].sum() < args.min_pos:
            continue
        cross = (zt[:, :, None] * ztp[:, None, :]).reshape(N, K * K).astype(np.float32)
        s = best_auc(z, lab, valid)               # single latent (full width)
        w = best_auc(within, lab, valid)          # within-position pair
        c = best_auc(cross, lab, valid)           # cross-position pair
        sc = cov95(z, lab, valid); cc = cov95(cross, lab, valid)
        out["per_dist"][f"k{kk}"] = {"n_pos": int(lab[valid].sum()), "single_mauc": s,
                                     "within_pair_mauc": w, "cross_pair_mauc": c,
                                     "single_cov95": sc, "cross_cov95": cc}
        print(f"{kk:>6} {int(lab[valid].sum()):>6} {s:>8.3f} {w:>12.3f} {c:>11.3f} {sc:>10.0f} {cc:>9.0f}")

    any_k = next(iter(out["per_dist"].values()), {})
    gain = (any_k.get("cross_pair_mauc", 0) - max(any_k.get("single_mauc", 0), any_k.get("within_pair_mauc", 0)))
    print(f"\n[verdict] cross-position pair gain over within-position: {gain:+.3f} -> "
          f"{'RELATIONAL across positions: induction IS pair-readable (core relational, read by binding t<->t-k)' if gain > 0.1 else 'cross-position pairs do not read it either'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
