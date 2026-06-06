"""Phase 2: SERVE the M0..Mn tower as additive chi-banded experts — truncation = the dial.

v0 decomposed the residual into chi-ordered levels and showed cov95/fidelity on fixed
activations. Phase 2 makes it a runnable model: the final residual is reconstructed as
an additive cascade of experts  h ~= M0(h) + M1(h) + ... + Mn(h) + core(h)  (core = the
un-harvested entangled remainder, so the full sum is exact), and at inference you keep
M0..M_{T-1} and feed that truncated residual to lm_head. We measure the model's REAL LM
loss (capability) AND cov95 (interpretability) AND active-atom count (compute) vs the
truncation level T -> the dial, on actual next-token prediction.

Each expert Mk is SAEk (trained on the cascade residual) restricted to its harvested
atoms; applied sequentially (Mk operates on what M0..M_{k-1} left).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
    p.add_argument("--ckpt", type=Path, default=Path("runs/tiny_gpt.pt"))
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--harvest", type=int, default=24)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--sae-steps", type=int, default=400)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--output", type=Path, default=Path("runs/serve_tower_tiny_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT2LMHeadModel(GPT2Config(**ck["config"])); model.load_state_dict(ck["state_dict"]); model.eval()
    tr, lm_head = model.transformer, model.lm_head
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx)]
    all_ids = [j for c in chunks for j in c]
    Y = _oracle(tok, all_ids, tok.convert_ids_to_tokens(all_ids), args.min_pos, len(all_ids))

    # per-chunk final residual (post-ln_f) — what lm_head consumes
    H = []
    with torch.no_grad():
        for c in chunks:
            H.append(tr(input_ids=torch.tensor([c])).last_hidden_state[0].float().numpy())
    X = np.concatenate(H, 0).astype(np.float32)

    # ---- build the tower: experts = SAEk restricted to harvested atoms ----
    print(f"[build] {args.rounds} expert levels")
    experts = []; harv_Z = []
    R = X.copy()
    for rd in range(args.rounds):
        We, Wd, be, bd = _train_topk_sae(R, args.width, args.k, args.sae_steps, 1e-3, rd)
        z = _encode(R, (We, Wd, be, bd), args.k)
        valid = z.std(0) > 0
        A, _ok = _auc_matrix(_column_ranks(z), Y)
        mono = np.where(valid, A.max(0), -np.inf)
        sel = np.array([s for s in np.argsort(-mono)[:args.harvest] if valid[s]])
        Wd_np = Wd.numpy().astype(np.float32)
        Mk = z[:, sel] @ Wd_np[:, sel].T
        R = (R - Mk).astype(np.float32)
        experts.append({"We": We, "Wd": Wd, "be": be, "bd": bd, "sel": sel})
        harv_Z.append(z[:, sel])
        print(f"  M{rd}: atoms={len(sel)} mono={float(np.mean(mono[sel])):.3f}")

    def expert_apply(level, Rk):
        """Mk(Rk): SAE level restricted to its harvested atoms, applied to residual Rk."""
        e = experts[level]
        z = _encode(Rk, (e["We"], e["Wd"], e["be"], e["bd"]), args.k)
        return (z[:, e["sel"]] @ e["Wd"].numpy().astype(np.float32)[:, e["sel"]].T).astype(np.float32)

    # ---- serve: LM loss of lm_head(P_T) per truncation T, on real next-token prediction ----
    print("\n[serve] truncation dial: keep M0..M{T-1} -> lm_head -> real LM loss")
    Z_all = np.concatenate(harv_Z, 1)
    cuts = np.cumsum([h.shape[1] for h in harv_Z])
    Aacc, okv = _auc_matrix(_column_ranks(Z_all), Y)

    def lm_loss(P_of_chunk):
        tot, n = 0.0, 0
        with torch.no_grad():
            off = 0
            for c in chunks:
                L = len(c); Pc = P_of_chunk[off:off + L]; off += L
                logits = lm_head(torch.from_numpy(Pc))            # (L, vocab)
                lp = torch.log_softmax(logits[:-1].float(), -1)
                tgt = torch.tensor(c[1:])
                tot += float(-lp[range(L - 1), tgt].sum()); n += L - 1
        return tot / n

    # precompute the cascade reconstruction P_T for every T (additive)
    dial = []
    # floor (T=0, predict from the mean residual) and ceiling (exact residual)
    loss_floor = lm_loss(np.tile(X.mean(0, keepdims=True), (X.shape[0], 1)).astype(np.float32))
    loss_full = lm_loss(X)
    for T in range(1, args.rounds + 1):
        # cascade the additive reconstruction P_T = sum_{lv<T} M_lv (recompute each T; cheap)
        Pacc = np.zeros_like(X); r = X.copy()
        for lv in range(T):
            m = expert_apply(lv, r); Pacc = Pacc + m; r = (r - m).astype(np.float32)
        loss = lm_loss(Pacc.astype(np.float32))
        cov95 = float(np.mean(Aacc[:, :cuts[T - 1]].max(1)[okv] >= 0.95))
        dial.append({"levels_kept": T, "n_atoms": int(cuts[T - 1]), "lm_loss": loss,
                     "cov95": cov95, "fidelity": 1.0 - float((r ** 2).sum()) / float((X ** 2).sum())})
        print(f"  M0..M{T-1} ({cuts[T-1]:>3} atoms)  lm_loss={loss:.3f}  cov95={cov95:.3f}  "
              f"fidelity={dial[-1]['fidelity']:.3f}")
    print(f"  [+core / exact h]            lm_loss={loss_full:.3f}  (floor, b_dec-only: {loss_floor:.3f})")

    # the user's question: is the entangled CORE useful WITHOUT the low-chi features?
    Pfull = np.zeros_like(X); r = X.copy()
    for lv in range(args.rounds):
        m = expert_apply(lv, r); Pfull = Pfull + m; r = (r - m).astype(np.float32)
    loss_core_alone = lm_loss(r.astype(np.float32))            # core = X - tower
    print(f"  [CORE ALONE (X - tower)]     lm_loss={loss_core_alone:.3f}  "
          f"(tower-alone {dial[-1]['lm_loss']:.3f}, full {loss_full:.3f})")

    out = {"experiment": "Phase 2: served chi-banded tower", "lm_loss_floor": loss_floor,
           "lm_loss_full": loss_full, "lm_loss_tower_alone": dial[-1]["lm_loss"],
           "lm_loss_core_alone": loss_core_alone, "dial": dial}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
