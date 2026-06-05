"""Prototype: is the preserve set selectable LABEL-FREE from the post-training residual?

P2 (bio) showed STATIC label-free proxies (pre-training fragility, selectivity, norm)
fail to pick the preserve set. The algorithm claim: after forge + FINE-TUNE, the atoms
training CANNOT recover (high host-vs-forged-trained disagreement) are the ones to
preserve — a label-free, dynamic signal. This tests it on the tiny LM by comparing
each selector's preserve-hybrid cov95-recovery curve to the oracle (label-based) ceiling.

Selectors (all label-FREE except 'oracle'):
  oracle      host oracle-strength (max AUC over labels)        -- ceiling (P1)
  random                                                          -- floor
  norm        decoder column norm                                -- static baseline
  frag_proj   1 - corr(z_host, z_forged_projection)              -- P2's static signal
  frag_train  1 - corr(z_host, z_forged_FINETUNED)               -- THE NEW claim
Grading uses the exact-lexical oracle (the preserve hybrid keeps host atoms verbatim,
diffuse from the fine-tuned forged model).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forge_cov_mechanism import _column_ranks, _encode, _train_topk_sae  # noqa: E402
from preserve_hybrid_tiny import _auc_matrix, _build_oracle  # noqa: E402


def _col_corr(A, B):
    A0 = A - A.mean(0); B0 = B - B.mean(0)
    den = np.sqrt((A0 ** 2).sum(0) * (B0 ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (A0 * B0).sum(0) / den


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/tiny_gpt.pt"))
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--sae-steps", type=int, default=600)
    p.add_argument("--ft-steps", type=int, default=300)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--ks", default="0,16,32,64,128,256,512")
    p.add_argument("--output", type=Path, default=Path("runs/residual_selector_tiny_summary.json"))
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
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx)]

    print("[1] host acts + oracle + SAE")
    with torch.no_grad():
        Xraw = np.concatenate([tr(input_ids=torch.tensor([c])).last_hidden_state[0].float().numpy()
                               for c in chunks], 0).astype(np.float32)
    all_ids = [j for c in chunks for j in c]
    Y, tiers = _build_oracle(tok, all_ids, tok.convert_ids_to_tokens(all_ids), args.min_pos, Xraw.shape[0])
    mu, sd = Xraw.mean(0, keepdims=True), Xraw.std(0, keepdims=True) + 1e-6
    X = ((Xraw - mu) / sd).astype(np.float32)
    d = X.shape[1]
    params = _train_topk_sae(X, args.width, args.k, args.sae_steps, 1e-3, 0)
    Wdec = params[1].numpy().astype(np.float64)

    print("[2] forge (projection)")
    _nrm = np.linalg.norm(Wdec.T, axis=1)
    basis = FeatureBasis(kept_ids=np.arange(args.width, dtype=np.int64), W_dec=Wdec.T,
                         merged_norms=_nrm, original_norms=_nrm, metadata={})
    projector = SubspaceProjector(basis, scale_boost="auto")
    weights = projector.project_module(tr, attention_width="host")
    cfg = adapter_for(tr).build_native_config(tr, args.width); cfg.forward_mode = "native_in_basis"
    forged = NativeModel.from_projected_weights(cfg, weights).torch_module

    Wdec_t = torch.from_numpy(Wdec.astype(np.float32))
    Xh = torch.from_numpy(Xraw)                                  # host targets (raw coords)
    cap = {}
    forged.lm_head.register_forward_pre_hook(lambda m, i: cap.__setitem__("h", i[0]))

    def forged_acts():
        outs = []
        with torch.no_grad():
            for c in chunks:
                forged(torch.tensor([c]))
                outs.append((cap["h"][0] @ Wdec_t.t()).numpy())
        return ((np.concatenate(outs, 0).astype(np.float32) - mu) / sd).astype(np.float32)

    z_host = _encode(X, params, args.k)
    z_proj = _encode(forged_acts(), params, args.k)

    print(f"[3] fine-tune forged ({args.ft_steps} steps, distill to host residual)")
    opt = torch.optim.AdamW(forged.parameters(), lr=5e-4)
    g = torch.Generator().manual_seed(0)
    offs = np.cumsum([0] + [len(c) for c in chunks])
    for step in range(args.ft_steps):
        bi = int(torch.randint(0, len(chunks), (1,), generator=g))
        c = chunks[bi]
        forged(torch.tensor([c]))
        dec = cap["h"][0] @ Wdec_t.t()                          # (L, d) host coords (grad)
        tgt = Xh[offs[bi]:offs[bi] + len(c)]
        loss = ((dec - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"    ft step {step}  loss {loss.item():.3f}")
    forged.eval()
    z_train = _encode(forged_acts(), params, args.k)

    # ---- score helpers ----
    Ah, okh = _auc_matrix(_column_ranks(z_host), Y)
    valid_h = z_host.std(0) > 0
    Ah = np.where(valid_h[None, :], Ah, -np.inf)
    src = np.array(tiers)

    def hybrid_cov95(order, z_diffuse):
        Af, okf = _auc_matrix(_column_ranks(z_diffuse), Y)
        Af = np.where((z_diffuse.std(0) > 0)[None, :], Af, -np.inf)
        ok = okh & okf
        rows = []
        for K in [int(x) for x in args.ks.split(",")]:
            S = np.zeros(args.width, dtype=bool); S[order[:K]] = True
            hb = Ah[:, S].max(1) if K else np.full(Ah.shape[0], -np.inf)
            fb = Af[:, ~S].max(1) if K < args.width else np.full(Af.shape[0], -np.inf)
            best = np.maximum(hb, fb)
            rows.append({"K": K, "cov95": float(np.mean(best[ok] >= 0.95))})
        return rows

    # ---- selectors (label-free except oracle) ----
    rng = np.random.default_rng(0)
    strength = np.where(valid_h, Ah.max(0), -np.inf)            # ORACLE (labels)
    selectors = {
        "oracle": strength,
        "random": np.where(valid_h, rng.random(args.width), -np.inf),
        "norm": np.where(valid_h, np.linalg.norm(Wdec, axis=0), -np.inf),
        "frag_proj": np.where(valid_h, 1.0 - np.nan_to_num(_col_corr(z_host, z_proj), nan=1.0), -np.inf),
        "frag_train": np.where(valid_h, 1.0 - np.nan_to_num(_col_corr(z_host, z_train), nan=1.0), -np.inf),
    }
    orders = {n: np.argsort(-s) for n, s in selectors.items()}
    oracle_top = set(orders["oracle"][:64])

    # K=0 cov95 of each forged variant (shows whether fine-tune over-recovered the tax)
    def _cov0(z):
        Af, okf = _auc_matrix(_column_ranks(z), Y)
        Af = np.where((z.std(0) > 0)[None, :], Af, -np.inf)
        ok = okh & okf
        return float(np.mean(np.maximum(Af.max(1), -np.inf)[ok] >= 0.95))
    out = {"width": args.width, "ks": [int(x) for x in args.ks.split(",")],
           "mauc_proj": float(np.nanmean(_auc_matrix(_column_ranks(z_proj), Y)[0].max(1))),
           "mauc_train": float(np.nanmean(_auc_matrix(_column_ranks(z_train), Y)[0].max(1))),
           "cov95_forged_proj": _cov0(z_proj), "cov95_forged_train": _cov0(z_train),
           "diffuse": "projection (real tax — frozen-LLM analog)", "selectors": {}}
    print(f"\n    mAUC: proj={out['mauc_proj']:.3f} -> finetuned={out['mauc_train']:.3f}    "
          f"cov95(K=0): proj={out['cov95_forged_proj']:.3f} finetuned={out['cov95_forged_train']:.3f}\n")
    print(f"{'selector':12} {'cov95@32':>9} {'cov95@64':>9} {'overlap@64':>11}")
    for n, order in orders.items():
        rows = hybrid_cov95(order, z_proj)  # diffuse = PROJECTION forge: real tax, preserve must do the work
        c32 = next(r["cov95"] for r in rows if r["K"] == 32)
        c64 = next(r["cov95"] for r in rows if r["K"] == 64)
        ov = len(set(order[:64]) & oracle_top) / 64.0
        out["selectors"][n] = {"curve": rows, "cov95_at_32": c32, "cov95_at_64": c64,
                               "overlap64_vs_oracle": ov}
        print(f"{n:12} {c32:>9.3f} {c64:>9.3f} {ov:>11.2f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
