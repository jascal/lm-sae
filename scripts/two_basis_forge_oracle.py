"""Two-basis forge vs the lm-sae oracle — does composition-preserve save the circuit?

The GPU-scale-ish test of the laptop result that motivated sae-forge's two-basis
forge (v0.13.0). Forge the tiny GPT three ways and score against lm-sae's EXACT
oracle (which a bare LM has not):

  single        project_module (one residual/feature basis) — the forge tax
  composition   + preserve U_C (host attention QK/OV geometry) per layer
  two_basis     + preserve U_A (sharp atoms) AND U_C

Metrics: induction-predictable KL (the circuit-faithfulness target — does the
forged model still do induction?), global KL, and assertion cov95 on the forged
final residual (the monosemantic-assertion target). The claim under test:
composition-preserve LOWERS induction KL vs single-basis (protects the circuit
the single basis smears), and two-basis additionally recovers assertion cov95.

Uses saeforge 0.13.0 (composition_subspace / augmented_basis / eval.circuit_kl).
Run with the local 0.13.0 on the path (see lm-sae/requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lm_bundle import COMMON, _lexical_features  # noqa: E402
from forge_cov_mechanism import _best_auc_per_label, _encode, _per_tier, _train_topk_sae  # noqa: E402


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


def _joint_uc(model, rank):
    """ONE shared composition subspace pooled over ALL heads AND layers (the instruction-tensor
    low-rank result: the QK/OV geometry is shared, so a single subspace covers every layer)."""
    tr = model.transformer; cfg = model.config
    d, Hn, nL = cfg.n_embd, cfg.n_head, cfg.n_layer
    hd = d // Hn
    reads, writes = [], []
    for L in range(nL):
        blk = tr.h[L]
        ln = blk.ln_1.weight.detach().numpy().astype(np.float64)
        Wc = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wo = blk.attn.c_proj.weight.detach().numpy().astype(np.float64)
        Wq, Wk, Wv = Wc[:, :d], Wc[:, d:2 * d], Wc[:, 2 * d:3 * d]
        reads.append(Wq * ln[:, None]); reads.append(Wk * ln[:, None])      # ln-folded read dirs
        for h in range(Hn):
            sl = slice(h * hd, (h + 1) * hd)
            writes.append(Wv[:, sl] @ Wo[sl, :])                            # write dirs
    Ur = np.linalg.svd(np.concatenate(reads, 1), full_matrices=False)[0][:, :rank]
    Uw = np.linalg.svd(np.concatenate(writes, 1), full_matrices=False)[0][:, :rank]
    Q, _ = np.linalg.qr(np.concatenate([Ur, Uw], 1))
    return Q[:, :min(Q.shape[1], d)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/tiny_gpt.pt"))
    p.add_argument("--gpt2", action="store_true", help="forge REAL gpt2 with a self-trained final-layer SAE (not the tiny ckpt / not SAELens)")
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--comp-rank", type=int, default=12)
    p.add_argument("--assert-k", type=int, default=48)
    p.add_argument("--output", type=Path, default=Path("runs/two_basis_forge_oracle_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    from saeforge import SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.augmented_basis import AugmentedBasis
    from saeforge.basis import FeatureBasis
    from saeforge.composition_subspace import CompositionSubspace, extract_composition_subspace
    from saeforge.eval.circuit_faithfulness import circuit_kl, induction_predictable
    from saeforge.model import NativeModel

    if args.gpt2:
        model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        cfg0 = model.config
        print("forging REAL gpt2 (124M) with a self-trained final-layer TopK SAE")
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg0 = GPT2Config(**ck["config"])
        model = GPT2LMHeadModel(cfg0)
        model.load_state_dict(ck["state_dict"]); model.eval()
    transformer = model.transformer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]

    # ---- host: final residual (SAE + cov95) + induction mask. Logits are recomputed
    # per chunk from the final hidden (vocab is GPT-2's 50k → never materialise them all). ----
    print("[1] host final hidden + induction mask")
    acts, all_ids, ind_mask = [], [], []
    with torch.no_grad():
        for ch in chunks:
            out = model(input_ids=torch.tensor([ch]), output_hidden_states=True)
            acts.append(out.hidden_states[-1][0].float().numpy())     # (seq, d), post-ln_f
            ind_mask.append(induction_predictable(ch)[1:].astype(bool))
            all_ids.extend(ch)
    Xraw = np.concatenate(acts, 0).astype(np.float32)
    W_U = model.lm_head.weight.detach().numpy().astype(np.float32)     # (vocab, d), tied, no bias
    M = np.concatenate(ind_mask)
    tok_strs = tok.convert_ids_to_tokens(all_ids)
    N, d = Xraw.shape
    Y, tiers = _build_oracle(tok, all_ids, tok_strs, args.min_pos, N)
    mu, sd = Xraw.mean(0, keepdims=True), Xraw.std(0, keepdims=True) + 1e-6
    X = ((Xraw - mu) / sd).astype(np.float32)
    print(f"    X={X.shape} Y={Y.shape} induction-predictable={M.mean():.3f}")

    print(f"[2] TopK SAE (width {args.width}, {args.width/d:.1f}x) + host cov95")
    params = _train_topk_sae(X, args.width, args.k, args.steps, 1e-3, 0)
    host_cov = _per_tier(_best_auc_per_label(_encode(X, params, args.k), Y), tiers)
    print(f"    host cov95={host_cov['all']['cov95']:.3f} mAUC={host_cov['all']['mauc']:.3f}")

    Wdec = params[1].numpy().astype(np.float64)           # (d, width)
    Wdec_rows = Wdec.T
    norms = np.linalg.norm(Wdec_rows, axis=1)
    basis = FeatureBasis(kept_ids=np.arange(args.width, dtype=np.int64), W_dec=Wdec_rows,
                         merged_norms=norms, original_norms=norms, metadata={"src": "tiny-gpt sae"})
    Wdec_t = torch.from_numpy(Wdec.astype(np.float32))

    n_layer = cfg0.n_layer
    comp = extract_composition_subspace(model, layers=list(range(n_layer)), rank=args.comp_rank)
    # joint cross-head/layer U_C: one shared subspace at every layer (instruction-tensor low-rank → shared)
    Uj = _joint_uc(model, args.comp_rank)
    comp_joint = {L: CompositionSubspace(U=Uj, layer=L, rank=Uj.shape[1], source_heads="all", d_model=d)
                  for L in range(n_layer)}
    print(f"    per-layer U_C ~{comp[0].rank}d/layer  vs  joint U_C {Uj.shape[1]}d shared across {n_layer} layers")
    # U_A: sharpest atoms by best oracle AUC (lm-sae HAS the labels — the right selector,
    # unlike sae-forge's label-free proxy; this is the oracle-driven U_A the spec deferred)
    Zall = _encode(X, params, args.k)
    per_atom = np.array([_best_auc_per_label(Zall[:, [j]], Y).max() for j in range(args.width)])
    U_A = Wdec_rows[np.argsort(-per_atom)[: args.assert_k]]

    def forge(augmented):
        proj = SubspaceProjector(basis, scale_boost="auto")
        weights = proj.project_module(transformer, attention_width="host", augmented=augmented)
        cfg = adapter_for(transformer).build_native_config(transformer, basis.n_features)
        cfg.forward_mode = "native_in_basis"
        fm = NativeModel.from_projected_weights(cfg, weights).torch_module
        fm.eval()
        cap = {}

        def _pre(mod, inp):
            cap["h"] = (inp[0] if isinstance(inp, (tuple, list)) else inp).detach()
        handle = fm.lm_head.register_forward_pre_hook(_pre)
        msum = mn = csum = cn = gsum = gn = 0.0
        fresid = []
        with torch.no_grad():
            for i, ch in enumerate(chunks):
                o = fm(torch.tensor([ch]))
                lg = (o.logits if hasattr(o, "logits") else o)[0, :-1].float().numpy()
                hl = acts[i][:-1] @ W_U.T                              # host logits (seq-1, vocab)
                ck_ = circuit_kl(hl, lg, mask=ind_mask[i])             # streamed — one chunk at a time
                nm, tot = ck_["n_masked"], lg.shape[0]
                msum += ck_["masked_kl"] * nm; mn += nm
                csum += ck_["complement_kl"] * (tot - nm); cn += (tot - nm)
                gsum += ck_["global_kl"] * tot; gn += tot
                fresid.append((cap["h"][0].float() @ Wdec_t.t()).numpy())
        handle.remove()
        Xf = ((np.concatenate(fresid, 0).astype(np.float32) - mu) / sd).astype(np.float32)
        cov = _per_tier(_best_auc_per_label(_encode(Xf, params, args.k), Y), tiers)
        return {"global_kl": gsum / max(gn, 1), "induction_kl": msum / max(mn, 1),
                "complement_kl": csum / max(cn, 1), "cov95": cov["all"]["cov95"],
                "token_cov95": cov.get("token", {}).get("cov95", float("nan"))}

    print("[3] forge: single / U_C per-layer / U_C joint / two-basis(joint)")
    configs = {
        "single": None,
        "uc_perlayer": AugmentedBasis(basis, composition=comp),
        "uc_joint": AugmentedBasis(basis, composition=comp_joint),
        "two_basis_joint": AugmentedBasis(basis, assertion_atoms=U_A, composition=comp_joint),
    }
    res = {}
    for name, aug in configs.items():
        res[name] = forge(aug)
        r = res[name]
        r["excess"] = r["induction_kl"] - r["complement_kl"]
        print(f"  {name:>16}: induction_kl {r['induction_kl']:.3f}  excess {r['excess']:+.3f}  "
              f"global_kl {r['global_kl']:.3f}  cov95 {r['cov95']:.3f}")

    s, pl, j, tb = res["single"], res["uc_perlayer"], res["uc_joint"], res["two_basis_joint"]
    out = {"model": f"tiny-gpt (d={d}, {n_layer} layers)", "d_model": d, "sae_width": args.width,
           "over_complete": round(args.width / d, 2), "comp_rank": args.comp_rank,
           "assert_k": args.assert_k, "induction_pred_rate": float(M.mean()),
           "host_cov95": host_cov["all"]["cov95"], "configs": res}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    red_pl = s["excess"] - pl["excess"]
    red_j = s["excess"] - j["excess"]
    print(f"\n[a: joint vs per-layer U_C] circuit-specific excess single {s['excess']:+.3f} -> "
          f"per-layer {pl['excess']:+.3f} (Δ {red_pl:+.3f}, {red_pl/s['excess']:.0%}) | "
          f"joint {j['excess']:+.3f} (Δ {red_j:+.3f}, {red_j/s['excess']:.0%})")
    print(f"    budget/layer: per-layer {comp[0].rank}d  vs  joint {Uj.shape[1]}d shared -> "
          f"{'JOINT WINS (>= per-layer)' if red_j >= red_pl - 1e-9 else 'per-layer wins'}")
    d_cov = tb["cov95"] - s["cov95"]
    print(f"[claim 2: two-basis(joint) recovers assertions] cov95 single {s['cov95']:.3f} -> "
          f"two_basis_joint {tb['cov95']:.3f}  (Δ {d_cov:+.3f}, {'RECOVERED' if d_cov > 0.02 else 'no gain'})")
    print(f"[context] global_kl single {s['global_kl']:.3f} / per-layer {pl['global_kl']:.3f} / "
          f"joint {j['global_kl']:.3f} / two_basis {tb['global_kl']:.3f}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
