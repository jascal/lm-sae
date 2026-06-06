"""Norm-preserving substrate ablation on GPT-2 — is the interpretable substrate essential?

The capable-host test used core = h - tower (zeroing the substrate), which lowers the
norm and is OOD for lm_head -> the substrate-removal cost was confounded. This redoes it
cleanly with MEAN-ABLATION: remove the substrate's information by replacing its per-token
component with the dataset MEAN (keeps norm + on-distribution), and measure the real LM
loss increase. Controls:
  - mean-ablate the SUBSTRATE subspace (span of harvested tower atoms)
  - mean-ablate a RANDOM subspace of the same dimension (baseline: removing ANY dirs)
  - mean-ablate the COMPLEMENT (keep only the substrate)
If substrate-ablation >> random-ablation, the interpretable substrate is genuinely
predictively essential (not a norm artifact).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from build_lm_bundle import COMMON, _lexical_features  # noqa: E402
from forge_cov_mechanism import _column_ranks, _encode, _train_topk_sae  # noqa: E402
from preserve_hybrid_tiny import _auc_matrix  # noqa: E402


def _oracle(tok, all_ids, tok_strs, min_pos, n):
    cols = []
    for c in COMMON:
        cid = tok(c, add_special_tokens=False)["input_ids"]
        if len(cid) == 1:
            cols.append(np.array([1 if j == cid[0] else 0 for j in all_ids], np.uint8))
    lex = [_lexical_features(s) for s in tok_strs]
    for name in lex[0]:
        cols.append(np.array([rr[name] for rr in lex], np.uint8))
    Y = np.stack(cols, 1); npos = Y.sum(0)
    keep = (npos >= min_pos) & (npos <= n - min_pos)
    return Y[:, keep]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--harvest", type=int, default=24)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--sae-steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--output", type=Path, default=Path("runs/cov95_forge_tax/substrate_ablation_gpt2_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained).eval()
    tr, lm_head = model.transformer, model.lm_head
    d = model.config.n_embd
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx)]
    all_ids = [j for c in chunks for j in c]
    Y = _oracle(tok, all_ids, tok.convert_ids_to_tokens(all_ids), args.min_pos, len(all_ids))

    H = []
    with torch.no_grad():
        for c in chunks:
            H.append(tr(input_ids=torch.tensor([c])).last_hidden_state[0].float().numpy())
    X = np.concatenate(H, 0).astype(np.float32)
    print(f"{args.pretrained}: d={d}  X={X.shape}  Y={Y.shape}")

    # ---- build the tower; collect harvested atom DIRECTIONS (the substrate subspace) ----
    R = X.copy(); dirs = []
    for rd in range(args.rounds):
        We, Wd, be, bd = _train_topk_sae(R, args.width, args.k, args.sae_steps, 1e-3, rd)
        z = _encode(R, (We, Wd, be, bd), args.k)
        valid = z.std(0) > 0
        A, _ok = _auc_matrix(_column_ranks(z), Y)
        mono = np.where(valid, A.max(0), -np.inf)
        sel = np.array([s for s in np.argsort(-mono)[:args.harvest] if valid[s]])
        Wd_np = Wd.numpy().astype(np.float32)
        dirs.append(Wd_np[:, sel])                         # (d, m)
        R = (R - z[:, sel] @ Wd_np[:, sel].T).astype(np.float32)
        print(f"  M{rd}: atoms={len(sel)} mono={float(np.mean(mono[sel])):.3f}")
    U, _ = np.linalg.qr(np.concatenate(dirs, 1))           # (d, r) orthonormal substrate basis
    r = U.shape[1]
    print(f"substrate subspace rank={r}")

    def mean_ablate(P):
        """Remove span(P)'s per-token info, replace with its mean (norm-preserving)."""
        Pb = P @ P.T
        comp = X @ Pb
        return (X - comp + comp.mean(0, keepdims=True)).astype(np.float32)

    Xt = torch.from_numpy(X)
    off = np.cumsum([0] + [len(c) for c in chunks])

    def lm_loss(arr):
        A = torch.from_numpy(arr) if not torch.is_tensor(arr) else arr
        tot, n = 0.0, 0
        with torch.no_grad():
            for ci, c in enumerate(chunks):
                Pc = A[off[ci]:off[ci + 1]]
                lp = torch.log_softmax(lm_head(Pc).float()[:-1], -1)
                tgt = torch.tensor(c[1:])
                tot += float(-lp[range(len(c) - 1), tgt].sum()); n += len(c) - 1
        return tot / n

    rng = np.random.default_rng(0)
    Ur, _ = np.linalg.qr(rng.standard_normal((d, r)))       # random subspace, same dim
    Ucomp, _ = np.linalg.qr(rng.standard_normal((d, d - r)))  # complement-ish (random of comp dim)
    # complement of the substrate (keep only substrate when ablated)
    Pc_comp = np.eye(d) - U @ U.T
    Ucmp, _ = np.linalg.qr(Pc_comp[:, :d - r] if d - r > 0 else Pc_comp)

    res = {
        "full": lm_loss(Xt),
        "mean_ablate_SUBSTRATE": lm_loss(mean_ablate(U)),
        "mean_ablate_RANDOM_samedim": lm_loss(mean_ablate(Ur)),
        "mean_ablate_COMPLEMENT(keep substrate only)": lm_loss(mean_ablate(Ucmp)),
    }
    out = {"experiment": "norm-preserving substrate ablation", "model": args.pretrained,
           "d_model": d, "substrate_rank": int(r), "losses": res,
           "substrate_cost": res["mean_ablate_SUBSTRATE"] - res["full"],
           "random_cost": res["mean_ablate_RANDOM_samedim"] - res["full"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print("\n[norm-preserving mean-ablation LM loss]")
    for kname, v in res.items():
        print(f"  {kname:42} {v:.3f}  (+{v-res['full']:+.3f})")
    print(f"\n[verdict] substrate-removal cost {out['substrate_cost']:+.3f} vs random-subspace "
          f"{out['random_cost']:+.3f}  ({r} dims each) -> "
          f"{'substrate ESSENTIAL (real, not norm artifact)' if out['substrate_cost'] > 2*max(out['random_cost'],0.05) else 'no clear substrate-specific cost'}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
