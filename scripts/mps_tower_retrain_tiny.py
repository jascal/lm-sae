"""Phase 1b: the M0..Mn tower WITH retrain-between-rounds (the full reverse algorithm).

v0 (mps_tower_tiny.py) decomposed a FIXED model -> entangled core ~0.24. The real idea
RETRAINS between rounds: after harvesting a level's clean atoms, FREEZE that subspace
and fine-tune the model so it re-expresses its computation in the freed (complement)
capacity, then re-harvest. Mechanism: a forward_pre_hook on lm_head splits the final
residual h = detach(h @ P_banked) + h @ P_complement, so the LM-loss gradient flows
ONLY through the complement -> the model is pushed to route prediction into the
not-yet-harvested directions, regenerating clean structure to harvest next round.

Test: does retraining drive the entangled core BELOW the fixed-model 0.24, i.e. does the
model adapt toward forgeability? (Caveat: fine-tunes the existing tiny GPT, not from
scratch.)
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
    feat, cols = [], []
    for c in COMMON:
        cid = tok(c, add_special_tokens=False)["input_ids"]
        if len(cid) == 1:
            cols.append(np.array([1 if j == cid[0] else 0 for j in all_ids], np.uint8)); feat.append(c)
    lex = [_lexical_features(s) for s in tok_strs]
    for name in lex[0]:
        cols.append(np.array([r[name] for r in lex], np.uint8)); feat.append(name)
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
    p.add_argument("--sae-steps", type=int, default=300)
    p.add_argument("--ft-steps", type=int, default=120)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--output", type=Path, default=Path("runs/mps_tower_retrain_tiny_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT2LMHeadModel(GPT2Config(**ck["config"])); model.load_state_dict(ck["state_dict"])
    tr = model.transformer
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

    def residual():
        model.eval()
        with torch.no_grad():
            return np.concatenate([tr(input_ids=torch.tensor([c])).last_hidden_state[0].float().numpy()
                                   for c in chunks], 0).astype(np.float32)

    # projection state (banked subspace), updated each round; used by the lm_head pre-hook
    proj = {"Pb": torch.zeros(d, d), "Pc": torch.eye(d)}

    def pre(mod, inp):
        h = inp[0]
        return ((h @ proj["Pc"]) + (h @ proj["Pb"]).detach(),)
    model.lm_head.register_forward_pre_hook(pre)

    banked = np.zeros((d, 0), dtype=np.float64)   # orthonormal columns
    X0 = residual()
    tot0 = float((X0 ** 2).sum())
    rounds = []
    for rd in range(args.rounds):
        X = residual()
        Pc = (np.eye(d) - banked @ banked.T) if banked.shape[1] else np.eye(d)
        Rc = (X @ Pc).astype(np.float32)                       # complement residual
        # harvest cleanest atoms from the complement (chi-meter = mono vs oracle)
        params = _train_topk_sae(Rc, args.width, args.k, args.sae_steps, 1e-3, rd)
        z = _encode(Rc, params, args.k)
        Wd = params[1].numpy().astype(np.float32)              # (d, width)
        valid = z.std(0) > 0
        A, _ok = _auc_matrix(_column_ranks(z), Y)
        atom_mono = np.where(valid, A.max(0), -np.inf)
        sel = [s for s in np.argsort(-atom_mono)[:args.harvest] if valid[s]]
        Mk = z[:, sel] @ Wd[:, sel].T
        core = float(((Rc - Mk) ** 2).sum()) / tot0            # un-harvested complement var (vs round-0 total)
        mono = float(np.mean([atom_mono[s] for s in sel])) if sel else float("nan")
        # bank the harvested directions (raw residual space), re-orthonormalize
        newdirs = Wd[:, sel].astype(np.float64)
        stacked = np.concatenate([banked, newdirs], axis=1)
        banked, _ = np.linalg.qr(stacked)
        proj["Pb"] = torch.from_numpy((banked @ banked.T).astype(np.float32))
        proj["Pc"] = torch.from_numpy((np.eye(d) - banked @ banked.T).astype(np.float32))

        # retrain: fine-tune with grad restricted to the complement (banked frozen)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        g = torch.Generator().manual_seed(rd)
        ftloss = None
        for _ in range(args.ft_steps):
            c = chunks[int(torch.randint(0, len(chunks), (1,), generator=g))]
            t = torch.tensor([c])
            out = model(input_ids=t, labels=t)
            opt.zero_grad(); out.loss.backward(); opt.step()
            ftloss = float(out.loss.item())
        rounds.append({"round": rd, "n_banked": int(banked.shape[1]), "mean_monosemanticity": mono,
                       "entangled_core_frac": core, "ft_loss": ftloss})
        print(f"  M{rd}: banked={banked.shape[1]:>3}  mono={mono:.3f}  core={core:.3f}  ft_loss={ftloss:.3f}",
              flush=True)

    # ---- Phase B: fixed v0-style decomposition of the ADAPTED model (fair core comparison) ----
    print("\n[Phase B] fixed decomposition of the adapted model (vs original core 0.239)", flush=True)
    proj["Pb"] = torch.zeros(d, d); proj["Pc"] = torch.eye(d)   # drop the constraint
    R = residual(); totB = float((R ** 2).sum()); fixed = []
    for rd in range(8):
        params = _train_topk_sae(R, args.width, args.k, args.sae_steps, 1e-3, 100 + rd)
        z = _encode(R, params, args.k); Wd = params[1].numpy().astype(np.float32)
        valid = z.std(0) > 0
        A, _ok = _auc_matrix(_column_ranks(z), Y)
        atom_mono = np.where(valid, A.max(0), -np.inf)
        sel = [s for s in np.argsort(-atom_mono)[:args.harvest] if valid[s]]
        R = (R - z[:, sel] @ Wd[:, sel].T).astype(np.float32)
        fixed.append({"round": rd, "mono": float(np.mean([atom_mono[s] for s in sel])),
                      "core": float((R ** 2).sum()) / totB})
        print(f"  fixed M{rd}: mono={fixed[-1]['mono']:.3f}  core={fixed[-1]['core']:.3f}", flush=True)
    adapted_core = fixed[-1]["core"]

    out = {"experiment": "MPS tower + retrain (Phase 1b)", "fixed_v0_core": 0.239,
           "adapt_rounds": rounds, "fixed_decomp_of_adapted": fixed, "adapted_model_core": adapted_core}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[verdict] entangled core (fixed decomposition): original 0.239  ->  "
          f"adapted {adapted_core:.3f}  "
          f"({'LOWER — retrain made it more forgeable' if adapted_core < 0.239 else 'NOT lower'})")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
