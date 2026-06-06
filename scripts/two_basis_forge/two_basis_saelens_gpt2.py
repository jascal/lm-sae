"""Two-basis forge on REAL GPT-2 + a SAELens SAE — the clean scale test for U_C.

The tiny-GPT experiment said composition-preserve (U_C) is regime-dependent and modest, but the tiny
forge wrecks the model globally (global_kl 5-7) so the circuit signal is swamped. This runs the same
comparison on real GPT-2 small + a real SAELens SAE, where the forge can be much less degraded — the
fair test of whether preserving U_C lowers the induction-predictable KL.

Pipeline (all on one box, CPU or MPS): GPT-2 + SAELens SAE -> FeatureBasis (optionally top-N by
activation to control over-completeness / memory) -> forge single / U_C / two-basis via sae-forge 0.13.0
-> stream circuit_kl (induction-predictable) + assertion cov95 vs the lm-sae lexical oracle.

UNTESTED ON HARDWARE (written on a box without sae_lens / GPU). Designed for a 24 GB M4 Mac:
  python -m pip install 'sae-forge @ git+https://github.com/jascal/sae-forge@v0.13.0' sae-lens transformers torch numpy
  PYTHONPATH=. python scripts/two_basis_saelens_gpt2.py --device cpu --max-features 2048
Memory: forged residual width = #features; ~ (features/768) x GPT-2 weights. 2048 feats ~ 3 GB; the full
~24k SAE ~ 10-15 GB (use --max-features to stay under 24 GB). Start small (2048), then scale up.
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


def _build_oracle(tok, all_ids, tok_strs, min_pos, n):
    cols, tiers = [], []
    for c in COMMON:
        cid = tok(c, add_special_tokens=False)["input_ids"]
        if len(cid) == 1:
            cols.append(np.array([1 if j == cid[0] else 0 for j in all_ids], np.uint8)); tiers.append("token")
    lex = [_lexical_features(s) for s in tok_strs]
    for name in lex[0]:
        cols.append(np.array([r[name] for r in lex], np.uint8))
        tiers.append("struct" if name.startswith("struct") else "lexical")
    Y = np.stack(cols, 1); npos = Y.sum(0)
    keep = (npos >= min_pos) & (npos <= n - min_pos)
    return Y[:, keep], [t for t, k in zip(tiers, keep) if k]


def _best_auc(F, Y):
    """best single-feature AUC per label column; F (N,K), Y (N,L) -> (L,)."""
    ranks = np.argsort(np.argsort(F, axis=0), axis=0).astype(np.float64) + 1
    out = np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        pos = Y[:, j].astype(bool); npos = int(pos.sum()); nneg = len(pos) - npos
        if npos == 0 or nneg == 0:
            out[j] = np.nan; continue
        auc = (ranks[pos].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)
        out[j] = np.nanmax(np.abs(auc - 0.5)) + 0.5
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", help="cpu or mps (M4)")
    p.add_argument("--sae-release", default="gpt2-small-res-jb")
    p.add_argument("--sae-id", default="blocks.8.hook_resid_pre")
    p.add_argument("--layer", type=int, default=8, help="residual layer the SAE reads (for cov95)")
    p.add_argument("--max-features", type=int, default=2048, help="top-N SAE feats by activation (0=all)")
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--comp-rank", type=int, default=16)
    p.add_argument("--assert-k", type=int, default=128)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--scale-boost", default="auto")
    p.add_argument("--output", type=Path, default=Path("runs/two_basis_forge/two_basis_saelens_gpt2_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    from saeforge import SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.augmented_basis import AugmentedBasis
    from saeforge.basis import FeatureBasis
    from saeforge.composition_subspace import extract_composition_subspace
    from saeforge.eval.circuit_faithfulness import circuit_kl, induction_predictable
    from saeforge.model import NativeModel

    dev = args.device
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(dev)
    tr = model.transformer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    # ---- SAELens SAE -> decoder (n_feat, d_model) ----
    from sae_lens import SAE
    loaded = SAE.from_pretrained(args.sae_release, args.sae_id, device=dev)
    sae = loaded[0] if isinstance(loaded, (tuple, list)) else loaded
    W_dec_full = sae.W_dec.detach().cpu().numpy().astype(np.float64)        # (n_feat, d_model)
    print(f"SAE {args.sae_release}/{args.sae_id}: W_dec {W_dec_full.shape}")

    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]

    # ---- host residual @ layer (cov95 + feature selection), W_U, induction mask ----
    acts, all_ids, ind_mask = [], [], []
    with torch.no_grad():
        for ch in chunks:
            o = model(input_ids=torch.tensor([ch], device=dev), output_hidden_states=True)
            acts.append(o.hidden_states[args.layer][0].float().cpu().numpy())
            ind_mask.append(induction_predictable(ch)[1:].astype(bool))
            all_ids.extend(ch)
    Xh = np.concatenate(acts, 0).astype(np.float32)
    W_U = model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
    tok_strs = tok.convert_ids_to_tokens(all_ids); N = len(all_ids)
    Y, tiers = _build_oracle(tok, all_ids, tok_strs, args.min_pos, N)

    def sae_encode(x_np):
        with torch.no_grad():
            z = sae.encode(torch.tensor(x_np, device=dev, dtype=sae.W_dec.dtype))
        return z.float().cpu().numpy()

    Zh = sae_encode(Xh)
    # select top-N features by mean activation (control over-completeness / memory)
    if args.max_features and args.max_features < W_dec_full.shape[0]:
        sel = np.argsort(-np.abs(Zh).mean(0))[: args.max_features]
        W_dec = W_dec_full[sel]
        def enc(x):
            return sae_encode(x)[:, sel]
    else:
        W_dec = W_dec_full
        def enc(x):
            return sae_encode(x)
    n_feat = W_dec.shape[0]
    host_cov = float(np.nanmean(_best_auc(enc(Xh), Y) >= 0.95))
    print(f"basis: {n_feat} feats ({n_feat/768:.1f}x)  host cov95 {host_cov:.3f}  induction-rate {np.concatenate(ind_mask).mean():.3f}")

    norms = np.linalg.norm(W_dec, axis=1)
    basis = FeatureBasis(kept_ids=np.arange(n_feat, dtype=np.int64), W_dec=W_dec,
                         merged_norms=norms, original_norms=norms, metadata={"src": "saelens"})
    Wdec_t = torch.tensor(W_dec.T.astype(np.float32), device=dev)
    sb = args.scale_boost if args.scale_boost == "auto" else float(args.scale_boost)
    n_layer = model.config.n_layer
    comp = extract_composition_subspace(model, layers=list(range(n_layer)), rank=args.comp_rank)
    # U_A: sharpest atoms by best oracle AUC (lm-sae oracle = the right, label-driven selector)
    Za = enc(Xh)
    atom_auc = np.array([np.nanmax(_best_auc(Za[:, [j]], Y)) for j in range(n_feat)])
    U_A = W_dec[np.argsort(-atom_auc)[: args.assert_k]]

    def forge(aug, name):
        proj = SubspaceProjector(basis, scale_boost=sb)
        weights = proj.project_module(tr, attention_width="host", augmented=aug)
        cfg = adapter_for(tr).build_native_config(tr, basis.n_features)
        cfg.forward_mode = "native_in_basis"
        fm = NativeModel.from_projected_weights(cfg, weights).torch_module.to(dev).eval()
        cap = {}

        def _pre(mod, inp):
            cap["h"] = (inp[0] if isinstance(inp, (tuple, list)) else inp).detach()
        handle = fm.lm_head.register_forward_pre_hook(_pre)
        msum = mn = csum = cn = gsum = gn = 0.0
        fres = []
        with torch.no_grad():
            for i, ch in enumerate(chunks):
                lg = (lambda o: o.logits if hasattr(o, "logits") else o)(fm(torch.tensor([ch], device=dev)))
                lg = lg[0, :-1].float().cpu().numpy()
                hl = acts[i][:-1] @ W_U.T
                ck = circuit_kl(hl, lg, mask=ind_mask[i])
                nm, tot = ck["n_masked"], lg.shape[0]
                msum += ck["masked_kl"] * nm; mn += nm
                csum += ck["complement_kl"] * (tot - nm); cn += (tot - nm)
                gsum += ck["global_kl"] * tot; gn += tot
                fres.append((cap["h"][0].float() @ Wdec_t.t()).cpu().numpy())
        handle.remove()
        Xf = np.concatenate(fres, 0).astype(np.float32)
        cov = float(np.nanmean(_best_auc(enc(Xf), Y) >= 0.95))
        r = {"induction_kl": msum / max(mn, 1), "complement_kl": csum / max(cn, 1),
             "global_kl": gsum / max(gn, 1), "cov95": cov}
        r["excess"] = r["induction_kl"] - r["complement_kl"]
        print(f"  {name:>12}: induction_kl {r['induction_kl']:.3f} excess {r['excess']:+.3f} "
              f"global_kl {r['global_kl']:.3f} cov95 {r['cov95']:.3f}")
        return r

    print("[forge] single / U_C / two-basis")
    res = {"single": forge(None, "single"),
           "uc": forge(AugmentedBasis(basis, composition=comp), "uc"),
           "two_basis": forge(AugmentedBasis(basis, assertion_atoms=U_A, composition=comp), "two_basis")}
    s, c, tb = res["single"], res["uc"], res["two_basis"]
    out = {"experiment": "two-basis SAELens GPT-2", "sae": f"{args.sae_release}/{args.sae_id}",
           "n_features": n_feat, "over_complete": round(n_feat / 768, 2), "host_cov95": host_cov,
           "scale_boost": sb, "configs": res}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    red = s["excess"] - c["excess"]
    print(f"\n[U_C] circuit-specific excess single {s['excess']:+.3f} -> U_C {c['excess']:+.3f} "
          f"(Δ {red:+.3f}, {red/max(s['excess'],1e-9):.0%}) -> "
          f"{'U_C PROTECTS the circuit at scale' if red > 0.1 * abs(s['excess']) else 'no clear protection'}")
    print(f"[cov95] host {host_cov:.3f} -> single {s['cov95']:.3f} -> two_basis {tb['cov95']:.3f} "
          f"(Δ {tb['cov95']-s['cov95']:+.3f})")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
