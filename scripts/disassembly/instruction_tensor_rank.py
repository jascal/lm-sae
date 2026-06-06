"""Is the instruction set low-rank? — TN/Tucker rank of the 144-head {B_h, V_h} tensor.

The opcode table SUGGESTED a "small reused instruction set" (4 shapes, typed bindings). This turns
that into a NUMBER. In a shared token-feature operand basis D (per-token residual centroids at one
reference layer), read every head's instruction as a matrix:
  QK  B_h[X,Y] = d_X . M_h . d_Y      (M_h = W_Q^h W_K^h.T / sqrt(hd))   — which query-feat binds which key-feat
  OV  V_h[Y,Z] = d_Y . (W_V^h W_O^h) . d_Z                              — which read-feat writes which write-feat
Stack across all 144 heads -> tensor T (144, nt, nt). Measure:
  - head-mode effective rank (90/99% singular energy, participation ratio) = # distinct instruction
    templates. Small rank => a few shared factors reconstruct all 144 heads = SMALL REUSED instruction set.
  - vs a norm-matched RANDOM null (Gaussian per-head Frobenius-matched) — real << random => genuine reuse.
  - Tucker/HOSVD ranks per mode (head, query-feat, key-feat); the query/key factor subspaces are a JOINT
    operand/composition subspace candidate (a cross-head upgrade to two-basis U_C).
GPT-2; weights-only modulo one forward pass for the shared centroid basis.
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


def _energy_rank(s, frac):
    e = s ** 2
    return int(np.searchsorted(np.cumsum(e) / e.sum(), frac) + 1)


def _participation(s):
    e = s ** 2
    return float(e.sum() ** 2 / (e ** 2).sum())


def _head_unfold_svd(T):
    """SVD of the head-mode unfolding (n_head_total, nt*nt)."""
    H = T.reshape(T.shape[0], -1)
    return np.linalg.svd(H, full_matrices=False)[1]


def _recon_err_at_rank(T, k):
    H = T.reshape(T.shape[0], -1)
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    Hk = (U[:, :k] * s[:k]) @ Vt[:k]
    return float(np.linalg.norm(H - Hk) / (np.linalg.norm(H) + 1e-12))


def _tucker_ranks(T, frac):
    ranks = []
    for mode in range(T.ndim):
        M = np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)
        ranks.append(_energy_rank(np.linalg.svd(M, full_matrices=False)[1], frac))
    return ranks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--ref-layer", type=int, default=6, help="layer whose centroids form the shared basis")
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=40)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--frac", type=float, default=0.90)
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/instruction_tensor_rank_summary.json"))
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

    # shared operand basis: per-token centroids at the reference layer (one forward pass)
    csum = np.zeros((nt, d)); ccnt = np.zeros(nt); gsum = np.zeros(d); gn = 0
    with torch.no_grad():
        for c in chunks:
            hs = tr(input_ids=torch.tensor([c]), output_hidden_states=True).hidden_states[args.ref_layer][0]
            hs = hs.float().numpy()
            pid = np.array([tok2i.get(t, -1) for t in c]); m = pid >= 0
            np.add.at(csum, pid[m], hs[m]); np.add.at(ccnt, pid[m], 1)
            gsum += hs.sum(0); gn += len(c)
    D = csum / np.maximum(ccnt, 1)[:, None] - gsum / gn
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)        # (nt, d) shared operand directions
    print(f"{args.pretrained}: {nL}x{H}={nL*H} heads, shared basis nt={nt} @ layer {args.ref_layer}")

    # build the instruction tensors over all heads
    Bs, Vs = [], []
    for L in range(nL):
        Wc = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wo = tr.h[L].attn.c_proj.weight.detach().numpy().astype(np.float64)
        Wq, Wk, Wv = Wc[:, :d], Wc[:, d:2 * d], Wc[:, 2 * d:3 * d]
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            Mh = Wq[:, sl] @ Wk[:, sl].T / np.sqrt(hd)
            OVh = Wv[:, sl] @ Wo[sl, :]
            Bs.append(D @ Mh @ D.T)
            Vs.append(D @ OVh @ D.T)
    Tqk = np.stack(Bs, 0); Tov = np.stack(Vs, 0)                     # (144, nt, nt)

    def _norm_slices(T):
        # unit-Frobenius per head -> rank measures shared SHAPE, not per-head magnitude
        f = np.linalg.norm(T.reshape(T.shape[0], -1), axis=1)
        return T / (f[:, None, None] + 1e-12)
    Tqk = _norm_slices(Tqk); Tov = _norm_slices(Tov)

    rng = np.random.default_rng(0)

    def analyze(T, name):
        s = _head_unfold_svd(T)
        # norm-matched random null: Gaussian per-head slice scaled to the real head Frobenius norms
        fro = np.linalg.norm(T.reshape(T.shape[0], -1), axis=1)
        R = rng.standard_normal(T.shape)
        R = R / (np.linalg.norm(R.reshape(T.shape[0], -1), axis=1)[:, None, None] + 1e-12) * fro[:, None, None]
        s_rand = _head_unfold_svd(R)
        out = {
            "head_rank_90": _energy_rank(s, 0.90), "head_rank_99": _energy_rank(s, 0.99),
            "head_participation": _participation(s), "n_heads": int(T.shape[0]),
            "rand_head_rank_90": _energy_rank(s_rand, 0.90),
            "rand_head_participation": _participation(s_rand),
            "tucker_ranks_90": _tucker_ranks(T, 0.90),
            "recon_err_at_5": _recon_err_at_rank(T, 5),
            "recon_err_at_10": _recon_err_at_rank(T, 10),
        }
        print(f"\n[{name}] head-mode effective rank (90% energy): REAL {out['head_rank_90']}  "
              f"vs RANDOM {out['rand_head_rank_90']}  (of {out['n_heads']})")
        print(f"    participation ratio: real {out['head_participation']:.1f} vs random {out['rand_head_participation']:.1f}; "
              f"99%-rank {out['head_rank_99']}")
        print(f"    Tucker ranks (head, query-feat, key-feat) @90%: {out['tucker_ranks_90']}")
        print(f"    head-rank-5 reconstructs {1-out['recon_err_at_5']:.0%} of energy; rank-10 {1-out['recon_err_at_10']:.0%}")
        return out

    res = {"experiment": "instruction-tensor low-rank", "model": args.pretrained, "n_tokens": nt,
           "ref_layer": args.ref_layer, "QK": analyze(Tqk, "QK / B_h"), "OV": analyze(Tov, "OV / V_h")}
    small = res["QK"]["head_rank_90"] < 0.5 * res["QK"]["rand_head_rank_90"]
    print(f"\n[verdict] QK instruction set: {res['QK']['head_rank_90']}/{nL*H} templates capture 90% "
          f"(random needs {res['QK']['rand_head_rank_90']}) -> "
          f"{'LOW-RANK / SMALL REUSED instruction set (the program compresses; query/key Tucker factors = a joint operand subspace)' if small else 'not clearly low-rank vs the norm-matched null'}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2, default=float))
    print(f"[done] {args.output}")
    return res


if __name__ == "__main__":
    main()
