"""P1 on the tiny LM: does PRESERVE-VERBATIM recover the cov95 forge tax?

The width sweep showed the LM tax is EMERGENT (not over-completeness) -> the lever is
preserve-verbatim, not concentrate. This tests it: keep the top-K oracle-reading SAE
atoms VERBATIM (host readout) and take the rest from the FORGED model, sweep K. If
combined cov95 climbs from forged (~0.12) back toward host (~0.65), preserve is the
lever for the LM tax — the lm-sae analog of bio's P1 knee.

  combined_best(label) = max( best_AUC_host(label | atom in top-K),
                              best_AUC_forged(label | atom not in top-K) )
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lm_bundle import COMMON, _lexical_features  # noqa: E402
from forge_cov_mechanism import _column_ranks, _encode, _per_tier, _train_topk_sae  # noqa: E402


def _build_oracle(tok, all_ids, tok_strs, min_pos, n):
    feat, cols, tiers = [], [], []
    for c in COMMON:
        cid = tok(c, add_special_tokens=False)["input_ids"]
        if len(cid) == 1:
            feat.append(f"token:{c!r}"); tiers.append("token")
            cols.append(np.array([1 if j == cid[0] else 0 for j in all_ids], dtype=np.uint8))
    lex = [_lexical_features(s) for s in tok_strs]
    for name in lex[0]:
        feat.append(name); tiers.append("struct" if name.startswith("struct") else "lexical")
        cols.append(np.array([r[name] for r in lex], dtype=np.uint8))
    Y = np.stack(cols, 1)
    npos = Y.sum(0)
    keep = (npos >= min_pos) & (npos <= n - min_pos)
    return Y[:, keep], [t for t, k in zip(tiers, keep) if k]


def _auc_matrix(R, Y):
    """Per-(label, atom) symmetric AUC. R (N,F) ranks, Y (N,M) binary -> (M,F), ok (M,)."""
    n = R.shape[0]
    Yf = Y.astype(np.float64)
    npos = Yf.sum(0); nneg = n - npos
    ok = (npos > 0) & (nneg > 0)
    denom = np.where(ok, npos * nneg, 1.0)
    auc = (Yf.T @ R - (npos * (npos + 1) / 2.0)[:, None]) / denom[:, None]
    return np.maximum(auc, 1.0 - auc), ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/tiny_gpt.pt"))
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--ks", default="0,8,16,32,64,128,256,512")
    p.add_argument("--output", type=Path, default=Path("runs/preserve_hybrid_tiny_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    from saeforge import SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.basis import FeatureBasis
    from saeforge.model import NativeModel

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT2LMHeadModel(GPT2Config(**ck["config"])); model.load_state_dict(ck["state_dict"]); model.eval()
    tr = model.transformer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]

    print("[1] host final-layer acts + oracle")
    acts, all_ids = [], []
    with torch.no_grad():
        for i in range(0, len(ids), args.ctx):
            ch = ids[i:i + args.ctx]
            acts.append(tr(input_ids=torch.tensor([ch])).last_hidden_state[0].float().numpy())
            all_ids.extend(ch)
    Xraw = np.concatenate(acts, 0).astype(np.float32)
    Y, tiers = _build_oracle(tok, all_ids, tok.convert_ids_to_tokens(all_ids), args.min_pos, Xraw.shape[0])
    mu, sd = Xraw.mean(0, keepdims=True), Xraw.std(0, keepdims=True) + 1e-6
    X = ((Xraw - mu) / sd).astype(np.float32)
    d = X.shape[1]

    print(f"[2] train SAE (width {args.width}) + forge")
    params = _train_topk_sae(X, args.width, args.k, args.steps, 1e-3, 0)
    Wdec = params[1].numpy().astype(np.float64)
    basis = FeatureBasis(kept_ids=np.arange(args.width, dtype=np.int64), W_dec=Wdec.T,
                         merged_norms=np.linalg.norm(Wdec.T, axis=1),
                         original_norms=np.linalg.norm(Wdec.T, axis=1), metadata={"src": "tiny"})
    projector = SubspaceProjector(basis, scale_boost="auto")
    weights = projector.project_module(tr, attention_width="host")
    cfg = adapter_for(tr).build_native_config(tr, args.width); cfg.forward_mode = "native_in_basis"
    forged = NativeModel.from_projected_weights(cfg, weights).torch_module; forged.eval()

    Wdec_t = torch.from_numpy(Wdec.astype(np.float32))
    cap = {}
    h = forged.lm_head.register_forward_pre_hook(lambda m, i: cap.__setitem__("h", i[0].detach()))
    facts = []
    with torch.no_grad():
        for i in range(0, len(ids), args.ctx):
            forged(torch.tensor([ids[i:i + args.ctx]]))
            facts.append((cap["h"][0].float() @ Wdec_t.t()).numpy())
    h.remove()
    Xf = ((np.concatenate(facts, 0).astype(np.float32) - mu) / sd).astype(np.float32)

    print("[3] preserve sweep: host-sharp (verbatim) + forged-diffuse")
    z_host = _encode(X, params, args.k)
    z_forged = _encode(Xf, params, args.k)
    valid_h = z_host.std(0) > 0
    valid_f = z_forged.std(0) > 0
    Ah, okh = _auc_matrix(_column_ranks(z_host), Y)        # (M, width)
    Af, okf = _auc_matrix(_column_ranks(z_forged), Y)
    Ah = np.where(valid_h[None, :], Ah, -np.inf)
    Af = np.where(valid_f[None, :], Af, -np.inf)
    ok = okh & okf
    strength = np.where(valid_h, Ah.max(0), -np.inf)       # rank atoms by host oracle strength
    order = np.argsort(-strength)
    src = np.array(tiers)

    def tiers_at(best):
        out = {"all": float(np.mean(best[ok] >= 0.95))}
        for t in sorted(set(tiers)):
            m = (src == t) & ok
            out[t] = float(np.mean(best[m] >= 0.95)) if m.any() else 0.0
        return out

    ks = [int(x) for x in args.ks.split(",")]
    rows = []
    for K in ks:
        S = np.zeros(args.width, dtype=bool); S[order[:K]] = True
        hb = Ah[:, S].max(1) if K else np.full(Ah.shape[0], -np.inf)
        fb = Af[:, ~S].max(1) if K < args.width else np.full(Af.shape[0], -np.inf)
        best = np.maximum(hb, fb)
        rows.append({"K": K, **tiers_at(best)})
        print(f"    K={K:>4}  cov95={rows[-1]['all']:.3f}  "
              + "  ".join(f"{t}={rows[-1][t]:.2f}" for t in rows[-1] if t not in ("K", "all")))
    out = {"model": "tiny-gpt", "width": args.width, "d_model": d, "ks": ks, "sweep": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
