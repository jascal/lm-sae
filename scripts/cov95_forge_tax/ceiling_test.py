"""The decompilation ceiling test (v2 of reconstruction coverage) — forge-basis recompilation.

residual_vm.py (milestone 1) measured op-SELECTION coverage by keeping heads at *full fidelity* — so it
reaches 1.0 by construction. This is the real ceiling test: recompile the model by forcing its computation
**through the SAE feature basis** (sae-forge `NativeModel`, `native_in_basis`) and ask how much of the
forward pass survives. We measure BOTH axes on the same forged host:

  - **capability / reconstruction** = 1 - KL(host || forged) / KL(host || unigram)  (does the forged model
    reproduce the host's next-token distribution? = "runs a circuit")
  - **legibility / monosemanticity** = forged cov95 against the exact lexical oracle ("is it factorable")

The unifying claim in DECOMPILATION.md predicts reconstruction-coverage plateaus at the tower's entangled-core
fraction. The project's own forge-tax thesis predicts the opposite for these two axes: capability SURVIVES
forging (mAUC robust) while cov95 COLLAPSES — so KL-coverage and cov95 should **decouple**. This script settles
it on the tiny GPT (where the whole forge is alive), sweeping SAE width, and reports both curves + the gap.

Reuses whole_loop_tiny's loader/oracle/SAE/forge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
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
    Y = np.stack(cols, 1); npos = Y.sum(0)
    keep = (npos >= min_pos) & (npos <= n - min_pos)
    return Y[:, keep], [t for t, k in zip(tiers, keep) if k]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/tiny_gpt.pt"))
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--widths", default="1,2,4,8", help="SAE over-completeness multipliers to sweep")
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--tower-core", type=float, default=0.24, help="entanglement-tower irreducible core fraction (mps_tower_tiny)")
    p.add_argument("--output", type=Path, default=Path("runs/cov95_forge_tax/ceiling_test_summary.json"))
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    from saeforge import SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.basis import FeatureBasis
    from saeforge.model import NativeModel

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT2LMHeadModel(GPT2Config(**ck["config"])); model.load_state_dict(ck["state_dict"]); model.eval()
    transformer = model.transformer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx)]

    # ---- host: final-layer acts (for SAE/oracle) + next-token logits (for KL) ----
    print("[1] host activations + logits")
    acts, all_ids, host_lp = [], [], []
    with torch.no_grad():
        for ch in chunks:
            o = model(input_ids=torch.tensor([ch]), output_hidden_states=True)
            acts.append(o.hidden_states[-1][0].float().numpy()); all_ids.extend(ch)
            host_lp.append(F.log_softmax(o.logits[0].float(), -1))     # (seq, V)
    Xraw = np.concatenate(acts, 0).astype(np.float32); N, d = Xraw.shape
    tok_strs = tok.convert_ids_to_tokens(all_ids)
    Y, tiers = _build_oracle(tok, all_ids, tok_strs, args.min_pos, N)
    mu, sd = Xraw.mean(0, keepdims=True), Xraw.std(0, keepdims=True) + 1e-6
    X = ((Xraw - mu) / sd).astype(np.float32)

    # unigram floor: KL(host || unigram) — the context-free predictor
    V = model.config.vocab_size
    cnt = np.bincount(np.array(all_ids), minlength=V).astype(np.float64) + 1.0
    uni = torch.from_numpy((cnt / cnt.sum()).astype(np.float32)); uni_lp = uni.log()
    kl_floor = ntok = 0.0
    with torch.no_grad():
        for hlp in host_lp:
            ph = hlp.exp()
            kl_floor += float((ph * (hlp - uni_lp[None, :])).sum(-1).sum()); ntok += hlp.shape[0]
    kl_floor /= ntok
    print(f"    N={N} d={d}  KL(host||unigram) floor = {kl_floor:.4f}")

    def kl_host_forged(forged):
        cap = {}
        h = forged.lm_head.register_forward_pre_hook(lambda m, inp: cap.__setitem__("h", (inp[0] if isinstance(inp, (tuple, list)) else inp).detach()))
        tot = 0.0; n = 0
        try:
            with torch.no_grad():
                for ci, ch in enumerate(chunks):
                    out = forged(torch.tensor([ch]))
                    logits = (out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, (tuple, list)) else out))[0].float()
                    flp = F.log_softmax(logits, -1)
                    hlp = host_lp[ci]; ph = hlp.exp()
                    tot += float((ph * (hlp - flp)).sum(-1).sum()); n += hlp.shape[0]
        finally:
            h.remove()
        return tot / n

    rows = []
    for w in [int(x) for x in args.widths.split(",")]:
        width = w * d
        print(f"[forge] width {width} ({w}x): train SAE -> forge -> KL + cov95")
        params = _train_topk_sae(X, width, args.k, args.steps, 1e-3, 0)
        host = _per_tier(_best_auc_per_label(_encode(X, params, args.k), Y), tiers)
        Wdec = params[1].numpy().astype(np.float64); Wdec_rows = Wdec.T
        norms = np.linalg.norm(Wdec_rows, axis=1)
        basis = FeatureBasis(kept_ids=np.arange(width, dtype=np.int64), W_dec=Wdec_rows,
                             merged_norms=norms, original_norms=norms, metadata={"src": "tiny-gpt sae"})
        projector = SubspaceProjector(basis, scale_boost="auto")
        adapter = adapter_for(transformer)
        weights = projector.project_module(transformer, attention_width="host")
        cfg = adapter.build_native_config(transformer, basis.n_features); cfg.forward_mode = "native_in_basis"
        forged = NativeModel.from_projected_weights(cfg, weights).torch_module; forged.eval()

        kl = kl_host_forged(forged)
        recon_cov = 1.0 - kl / kl_floor
        # forged cov95 (legibility): decode forged final residual -> re-score
        Wdec_t = torch.from_numpy(Wdec.astype(np.float32)); cap = {}
        hh = forged.lm_head.register_forward_pre_hook(lambda m, inp: cap.__setitem__("h", (inp[0] if isinstance(inp, (tuple, list)) else inp).detach()))
        facts = []
        with torch.no_grad():
            for ch in chunks:
                forged(torch.tensor([ch])); facts.append((cap["h"][0].float() @ Wdec_t.t()).numpy())
        hh.remove()
        Xf = ((np.concatenate(facts, 0).astype(np.float32) - mu) / sd).astype(np.float32)
        forged_t = _per_tier(_best_auc_per_label(_encode(Xf, params, args.k), Y), tiers)
        rows.append({"over_complete": w, "width": width, "kl_host_forged": kl, "recon_coverage": recon_cov,
                     "host_cov95": host["all"]["cov95"], "forged_cov95": forged_t["all"]["cov95"],
                     "host_mauc": host["all"]["mauc"], "forged_mauc": forged_t["all"]["mauc"]})
        print(f"    {w}x: recon-coverage(KL) {recon_cov:+.3f} (KL {kl:.3f}) | cov95 {host['all']['cov95']:.2f}->{forged_t['all']['cov95']:.2f} "
              f"| mAUC {host['all']['mauc']:.2f}->{forged_t['all']['mauc']:.2f}")

    # ---- verdict ----
    # Two distinct "ceilings" surface, on different axes:
    #  (1) feature-content (mAUC) vs monosemantic-factorization (cov95) — the forge-tax axis, robust at scale;
    #  (2) forged-MODEL output faithfulness (KL) — confounded at this scale (the whole-model tiny forge is
    #      globally broken, KL > unigram), so it bounds capability-reconstruction quality, not the core ceiling.
    best_cov = max(r["recon_coverage"] for r in rows)
    cov95_ret = float(np.mean([r["forged_cov95"] / max(r["host_cov95"], 1e-9) for r in rows]))
    mauc_ret = float(np.mean([r["forged_mauc"] / max(r["host_mauc"], 1e-9) for r in rows]))
    forge_alive = best_cov > 0.0                       # did any width give a forge better than unigram?
    decoupled = (mauc_ret > 0.7) and (cov95_ret < 0.4)  # feature content survives, monosemantic structure doesn't
    out = {"experiment": "decompilation ceiling test (forge-basis reconstruction coverage)", "model": "tiny-gpt",
           "d_model": d, "n_samples": int(N), "kl_unigram_floor": kl_floor, "tower_core_fraction": args.tower_core,
           "sweep": rows, "best_recon_coverage_kl": best_cov, "mean_cov95_retention": cov95_ret,
           "mean_mauc_retention": mauc_ret, "forge_capability_alive": bool(forge_alive),
           "feature_content_vs_monosemanticity_decoupled": bool(decoupled)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[verdict] mAUC retention {mauc_ret:.0%}  vs  cov95 retention {cov95_ret:.0%}  |  "
          f"best forged-model KL-coverage {best_cov:+.2f} ({'alive' if forge_alive else 'globally broken < unigram'}).")
    if decoupled:
        print("  => AXES DECOUPLE: forging retains feature CONTENT (mAUC) but destroys MONOSEMANTIC FACTORIZATION")
        print("     (cov95). The forge tax is a loss of interpretable STRUCTURE, on a different axis than capability —")
        print("     the unifying single-ceiling claim is REFINED into two axes (content vs factorability).")
    if not forge_alive:
        print("  CAVEAT: the tiny whole-model forge is globally broken (KL > unigram) — a known scale artifact, NOT")
        print("     the entangled-core ceiling. Isolating the capability-reconstruction plateau needs a high-quality")
        print("     (GPU-scale SAELens/polygram) forge; this run bounds the metric + settles the content/structure axis.")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
