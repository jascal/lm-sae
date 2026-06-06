"""Relational oracle + pair-cov95: is the entangled core OPAQUE, or just RELATIONAL?

The "features = assertions, entangled core = inference rules" framing predicts that the
high-chi content is invisible to single-latent cov95 (a UNARY detector) but readable by a
RELATIONAL detector (a bilinear pair z_i*z_j = a chi=2 / AND detector). Test it:
  - unary oracle:     cur-token identity, lexical, prev-token identity (arity-1)
  - relational oracle: bigrams (prev==A AND cur==B)                    (arity-2)
Compare single-cov95 (best single latent) vs pair-cov95 (best over singles + bilinear
pairs). The CLEAN signal is the delta-of-deltas: pairs should help MUCH more on the
relational tier than the unary tier (which cancels the multiple-comparison inflation from
searching ~1k pairs). If so, the core is interpretable RELATIONALLY -> chi = logical arity.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from build_lm_bundle import COMMON, _lexical_features  # noqa: E402
from forge_cov_mechanism import _column_ranks, _encode, _train_topk_sae  # noqa: E402
from preserve_hybrid_tiny import _auc_matrix  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/tiny_gpt.pt"))
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--layer", type=int, default=2, help="mid layer (relational info live, not consumed)")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--sae-steps", type=int, default=400)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--topk-pairs", type=int, default=48, help="# latents to pair (K -> K*(K-1)/2 pairs)")
    p.add_argument("--n-bigrams", type=int, default=24)
    p.add_argument("--output", type=Path, default=Path("runs/cov95_forge_tax/pair_cov95_tiny_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m = GPT2LMHeadModel(GPT2Config(**ck["config"])); m.load_state_dict(ck["state_dict"]); m.eval()
    tr = m.transformer
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
                            .hidden_states[args.layer][0].float().numpy()
                            for c in chunks], 0).astype(np.float32)
    print(f"using layer {args.layer}/{m.config.n_layer}")

    # ---- oracles ----
    cols, names, tiers = [], [], []
    common_ids = {c: tok(c, add_special_tokens=False)["input_ids"][0]
                  for c in COMMON if len(tok(c, add_special_tokens=False)["input_ids"]) == 1}
    # arity-1: cur-token identity
    for c, cid in common_ids.items():
        cols.append(np.array([1 if j == cid else 0 for j in all_ids], np.uint8))
        names.append(f"cur={c!r}"); tiers.append("unary_cur")
    # arity-1: lexical
    lex = [_lexical_features(s) for s in tok.convert_ids_to_tokens(all_ids)]
    for name in lex[0]:
        cols.append(np.array([r[name] for r in lex], np.uint8)); names.append(name); tiers.append("unary_lex")
    # arity-1 diagnostic: prev-token identity (does the final residual encode prev?)
    for c, cid in list(common_ids.items())[:12]:
        cols.append(np.array([1 if (t >= 1 and all_ids[t - 1] == cid) else 0 for t in range(N)], np.uint8))
        names.append(f"prev={c!r}"); tiers.append("unary_prev")
    # arity-2: GENUINELY relational bigrams -- cur==B appears in many contexts, so cur=B
    # alone is a poor detector of (prev==A AND cur==B); the conjunction needs prev too.
    cur_cnt = Counter(all_ids)
    bg = Counter((all_ids[t - 1], all_ids[t]) for t in range(1, N))
    cands = [(ab, cnt) for ab, cnt in bg.items() if cnt >= args.min_pos and cur_cnt[ab[1]] >= 5 * cnt]
    # rank by relationality = fraction of cur=B occurrences NOT preceded by A (high = relational)
    cands.sort(key=lambda x: -(1 - x[1] / cur_cnt[x[0][1]]))
    chosen = [ab for ab, _ in cands][: args.n_bigrams]
    for (A, B) in chosen:
        cols.append(np.array([1 if (t >= 1 and all_ids[t - 1] == A and all_ids[t] == B) else 0
                              for t in range(N)], np.uint8))
        names.append(f"({tok.decode([A])!r},{tok.decode([B])!r})"); tiers.append("relational_bigram")

    Y = np.stack(cols, 1)
    npos = Y.sum(0); keep = (npos >= args.min_pos) & (npos <= N - args.min_pos)
    Y = Y[:, keep]; names = [n for n, k in zip(names, keep) if k]; tiers = np.array([t for t, k in zip(tiers, keep) if k])
    print(f"X={X.shape}  labels={Y.shape}  tiers={dict(Counter(tiers))}")

    # ---- SAE latents ----
    params = _train_topk_sae(X, args.width, args.k, args.sae_steps, 1e-3, 0)
    z = _encode(X, params, args.k)

    # ---- single-cov95 (unary detector): best single latent ----
    A_s, ok = _auc_matrix(_column_ranks(z), Y)             # (M, width)
    single = np.where(ok, A_s.max(1), np.nan)

    # ---- pair-cov95 (relational detector): best over singles + bilinear pairs z_i*z_j ----
    K = args.topk_pairs
    topi = np.argsort(-z.std(0))[:K]
    zt = z[:, topi]
    pidx = [(i, j) for i in range(K) for j in range(i + 1, K)]
    feats = []
    for (i, j) in pidx:
        feats.append(zt[:, i] * zt[:, j])                  # AND / conjunction
        feats.append(np.abs(zt[:, i] - zt[:, j]))          # XOR / difference
    P = np.stack(feats, 1).astype(np.float32)
    A_p, _ = _auc_matrix(_column_ranks(P), Y)              # (M, 2*n_pairs)
    pair = np.where(ok, np.maximum(A_s.max(1), A_p.max(1)), np.nan)
    print(f"pairs scored: {P.shape[1]} (AND+XOR over top-{K} latents)")

    def tier_stats(mask):
        s, pr = single[mask], pair[mask]
        return {"n": int(mask.sum()),
                "single_cov95": float(np.nanmean(s >= 0.95)), "pair_cov95": float(np.nanmean(pr >= 0.95)),
                "single_mauc": float(np.nanmean(s)), "pair_mauc": float(np.nanmean(pr))}

    out = {"experiment": "relational oracle + pair-cov95", "n_pairs": len(pidx), "per_tier": {}}
    print(f"\n{'tier':18} {'n':>3} {'single_cov95':>12} {'pair_cov95':>11} {'delta':>7} "
          f"{'single_mAUC':>11} {'pair_mAUC':>9}")
    for t in ["unary_cur", "unary_lex", "unary_prev", "relational_bigram"]:
        msk = tiers == t
        if not msk.any():
            continue
        st = tier_stats(msk); out["per_tier"][t] = st
        print(f"{t:18} {st['n']:>3} {st['single_cov95']:>12.3f} {st['pair_cov95']:>11.3f} "
              f"{st['pair_cov95']-st['single_cov95']:>+7.3f} {st['single_mauc']:>11.3f} {st['pair_mauc']:>9.3f}")

    # delta-of-deltas: relational pair-gain vs unary pair-gain (cancels pair-search inflation)
    un = tiers != "relational_bigram"; rel = tiers == "relational_bigram"
    d_un = float(np.nanmean(pair[un] >= 0.95) - np.nanmean(single[un] >= 0.95))
    d_rel = float(np.nanmean(pair[rel] >= 0.95) - np.nanmean(single[rel] >= 0.95))
    out["unary_pair_gain"] = d_un; out["relational_pair_gain"] = d_rel
    print(f"\n[delta-of-deltas] pair-gain: unary {d_un:+.3f}  relational {d_rel:+.3f}  "
          f"=> {'RELATIONS are pair-readable (core is relational, not opaque)' if d_rel > d_un + 0.05 else 'no clear relational pair signal'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
