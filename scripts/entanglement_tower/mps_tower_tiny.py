"""Phase 1 prototype: the M0..Mn entanglement tower on the tiny LM.

The reverse algorithm: harvest the cleanest (lowest-chi, most monosemantic) features
first, subtract them, repeat on the residual -> an ADDITIVE tower
X ~= M0 + M1 + ... where early levels are clean/interpretable and later ones are
progressively entangled. Tests three things:
  - TAPER       : do later rounds harvest fewer + more-entangled (less monosemantic) atoms?
  - DIAL        : keeping M0..Mk, does cov95 (interpretability) vs reconstruction
                  fidelity (capability proxy) trace a smooth graceful curve = MPS truncation?
  - CONVERGENCE : does per-round harvested variance shrink toward a fixed point?

chi-meter = monosemanticity against the oracle labels (the "hold onto labels"
instinct): harvest the atoms that most cleanly read a ground-truth feature first.
v0 = fixed model (no retrain between rounds); the from-scratch retrain loop is next.
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
    feat, cols, tiers = [], [], []
    for c in COMMON:
        cid = tok(c, add_special_tokens=False)["input_ids"]
        if len(cid) == 1:
            cols.append(np.array([1 if j == cid[0] else 0 for j in all_ids], np.uint8))
            feat.append(c); tiers.append("token")
    lex = [_lexical_features(s) for s in tok_strs]
    for name in lex[0]:
        cols.append(np.array([r[name] for r in lex], np.uint8)); feat.append(name)
        tiers.append("lexical")
    Y = np.stack(cols, 1); npos = Y.sum(0)
    keep = (npos >= min_pos) & (npos <= n - min_pos)
    return Y[:, keep], [t for t, k in zip(tiers, keep) if k]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/tiny_gpt.pt"))
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--harvest", type=int, default=24, help="atoms harvested per round")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--sae-steps", type=int, default=400)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--output", type=Path, default=Path("runs/entanglement_tower/mps_tower_tiny_summary.json"))
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

    with torch.no_grad():
        X = np.concatenate([tr(input_ids=torch.tensor([c])).last_hidden_state[0].float().numpy()
                            for c in chunks], 0).astype(np.float32)
    all_ids = [j for c in chunks for j in c]
    Y, tiers = _oracle(tok, all_ids, tok.convert_ids_to_tokens(all_ids), args.min_pos, X.shape[0])
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xz = ((X - mu) / sd).astype(np.float32)
    totvar = float((Xz ** 2).sum())
    print(f"X={Xz.shape}  Y={Y.shape}  rounds={args.rounds} harvest/round={args.harvest}")

    R = Xz.copy()
    harv_Z = []                  # accumulated harvested atom activations (on original X)
    rounds = []
    for rd in range(args.rounds):
        params = _train_topk_sae(R, args.width, args.k, args.sae_steps, 1e-3, rd)
        z = _encode(R, params, args.k)               # (N, width) on current residual
        Wd = params[1].numpy().astype(np.float32)    # (d, width)
        valid = z.std(0) > 0
        # chi-meter: per-atom best AUC against the oracle (monosemanticity)
        A, ok = _auc_matrix(_column_ranks(z), Y)     # (M, width)
        atom_mono = np.where(valid, A.max(0), -np.inf)   # each atom's best label AUC
        sel = np.argsort(-atom_mono)[:args.harvest]
        sel = [s for s in sel if valid[s]]
        Mk = z[:, sel] @ Wd[:, sel].T                # (N, d) reconstruction of this level
        var_before = float((R ** 2).sum())
        R = (R - Mk).astype(np.float32)
        var_after = float((R ** 2).sum())
        # atoms' activations measured on the ORIGINAL residual (as detectors of X's features)
        harv_Z.append(z[:, sel])
        mono = float(np.mean([atom_mono[s] for s in sel]))
        rounds.append({"round": rd, "n_harvested": len(sel),
                       "mean_monosemanticity": mono,
                       "var_captured_frac": (var_before - var_after) / totvar,
                       "residual_var_frac": var_after / totvar})
        print(f"  M{rd}: harvested {len(sel):>2}  mono(meanAUC)={mono:.3f}  "
              f"var_captured={rounds[-1]['var_captured_frac']:.3f}  "
              f"resid_var={rounds[-1]['residual_var_frac']:.3f}")

    # ---- truncation dial: keep M0..Mk, measure cov95 (labels) + reconstruction fidelity ----
    print("\n[dial] keep M0..Mk -> cov95 (interpretability) vs fidelity (capability proxy)")
    Z_all = np.concatenate(harv_Z, 1)                # (N, total atoms) accumulated dictionary
    cuts = np.cumsum([h.shape[1] for h in harv_Z])
    Avals, okv = _auc_matrix(_column_ranks(Z_all), Y)   # (M, total)
    src = np.array(tiers)
    dial = []
    for k, cut in enumerate(cuts):
        sub = Avals[:, :cut]
        best = sub.max(1)
        cov95 = float(np.mean(best[okv] >= 0.95))
        fidelity = 1.0 - rounds[k]["residual_var_frac"]
        tok_cov = float(np.mean((sub[(src == "token")].max(1)) >= 0.95))
        dial.append({"keep_levels": k + 1, "n_atoms": int(cut), "cov95": cov95,
                     "token_cov95": tok_cov, "fidelity": fidelity})
        print(f"  M0..M{k} ({cut:>3} atoms)  cov95={cov95:.3f}  token={tok_cov:.2f}  fidelity={fidelity:.3f}")

    out = {"experiment": "MPS tower (Phase 1, fixed-model v0)", "rounds": rounds, "dial": dial,
           "taper_monosemanticity": [r["mean_monosemanticity"] for r in rounds],
           "taper_var_captured": [r["var_captured_frac"] for r in rounds]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
