"""Cross-substrate recoverability model on a STANDARD LM (GPT-2 + SAELens).

The third substrate for the recoverability law (after econ-sae's macro-regime and
bio-sae's ESM-2). For every exact-lexical ground-truth feature on GPT-2 layer-8
residuals we measure four quantities and test whether two cheap, SAE-free
predictors forecast the two expensive measurements:

  var_share = p(1-p)||Δμ||² / tr(Σ)     ALLOCATION predictor  (rate-distortion)
  fisher    = Δμᵀ(Σ_w+λI)⁻¹Δμ           PRESENCE predictor    (detection theory)
  probe_auc = in-sample ridge-LDA AUC    PRESENCE measurement  (linear-readable?)
  sae_auc   = best-latent recovery AUC   ALLOCATION measurement (SAELens 24k-feature
                                          production dictionary, blocks.8.hook_resid_pre)

The substrate has a built-in present-vs-allocated contrast: the `token` tier
(sharp one-token detectors — high variance, recovered) vs the `lexical` tier
(diffuse properties: is_capitalized, has_digit, len-bucket — probe-readable but
variance-cheap, dropped). The prediction: lexical features carry high Fisher /
probe yet low var_share / SAE — present yet dropped, on a real LM.

  partial Spearman(var_share -> sae_auc | fisher)   should DOMINATE
  partial Spearman(fisher    -> probe   | var_share) should DOMINATE

Pure cached path: the layer-8 bundle + the SAELens dictionary (HF cache). No GPT-2
forward, no SAE training.

Run:  .venv/bin/python scripts/recoverability_theory.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FISHER_LAM = 1e-2


def spearman(a, b) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(a, b, z) -> float:
    r_ab, r_az, r_bz = spearman(a, b), spearman(a, z), spearman(b, z)
    denom = np.sqrt(max((1 - r_az**2) * (1 - r_bz**2), 1e-12))
    return float((r_ab - r_az * r_bz) / denom)


def recoverability_axes(Xz, Y, lam=FISHER_LAM) -> dict:
    """Per-feature var_share, fisher (Sherman-Morrison downdate), probe_auc.

    Same vectorised core as bio-sae's recoverability_theory.py — see there for the
    Sherman-Morrison derivation (fisher = q/(1-c·q)). Y is small here (≈28 cols) so
    no chunking needed.
    """
    N, d = Xz.shape
    Yf = Y.astype(np.float64)
    npos = Yf.sum(0); nneg = N - npos
    valid = (npos >= 2) & (nneg >= 2)
    p = npos / N
    totvar = float(Xz.var(0).sum())

    sumX = Xz.sum(0)
    Mpos = Yf.T @ Xz                                          # (V, d)
    mu_pos = Mpos / np.where(npos > 0, npos, 1.0)[:, None]
    mu_neg = (sumX[None, :] - Mpos) / np.where(nneg > 0, nneg, 1.0)[:, None]
    dmu = mu_pos - mu_neg
    dmu2 = (dmu * dmu).sum(1)
    var_share = p * (1 - p) * dmu2 / totvar

    Xc = Xz - Xz.mean(0, keepdims=True)
    A = (Xc.T @ Xc) / max(N - 2, 1) + lam * np.eye(d)
    Ainv = np.linalg.inv(A)
    AinvDmu = dmu @ Ainv
    q = (AinvDmu * dmu).sum(1)
    c = npos * nneg / (N * max(N - 2, 1))
    denom = 1.0 - c * q
    fisher = np.where(denom > 1e-9, q / denom, np.nan)

    # ridge-LDA probe (presence measurement): score = Xz · A⁻¹Δμ, symmetric AUC
    W = Ainv @ dmu.T                                          # (d, V)
    S = Xz @ W                                                # (N, V)
    order = S.argsort(axis=0)
    ranks = np.empty_like(S, dtype=np.float64)
    ranks[order, np.arange(Y.shape[1])[None, :]] = np.arange(1, N + 1, dtype=np.float64)[:, None]
    spos = (Yf * ranks).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (spos - npos * (npos + 1) / 2.0) / (npos * nneg)
    probe = np.where(valid, np.maximum(auc, 1.0 - auc), np.nan)

    return {"prevalence": p, "var_share": np.where(valid, var_share, np.nan),
            "fisher": fisher, "probe_auc": probe, "valid": valid}


def _column_ranks(Z):
    n, d = Z.shape
    order = Z.argsort(axis=0)
    ranks = np.empty((n, d), dtype=np.float64)
    ranks[order, np.arange(d)[None, :]] = np.arange(1, n + 1, dtype=np.float64)[:, None]
    return ranks


def sae_per_feature_auc(X_raw, Wenc, benc, bdec, Y, feat_chunk=2048):
    """SAELens ReLU SAE per-feature best-latent symmetric AUC (allocation measure)."""
    n = X_raw.shape[0]
    Yf = Y.astype(np.float64)
    npos = Yf.sum(0); nneg = n - npos
    valid = (npos > 0) & (nneg > 0)
    denom = np.where(valid, npos * nneg, 1.0)
    best = np.full(Y.shape[1], -np.inf)
    Xc = X_raw - bdec
    F = Wenc.shape[1]
    for s in range(0, F, feat_chunk):
        z = Xc @ Wenc[:, s:s + feat_chunk] + benc[s:s + feat_chunk]
        np.maximum(z, 0, out=z)
        keep = z.std(0) > 0
        if not keep.any():
            continue
        R = _column_ranks(z[:, keep])
        spos = Yf.T @ R
        auc = (spos - (npos * (npos + 1) / 2.0)[:, None]) / denom[:, None]
        best = np.maximum(best, np.maximum(auc, 1.0 - auc).max(axis=1))
    return np.where(valid, best, np.nan)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=REPO / "data/lm_bundle_gpt2_l8.npz")
    ap.add_argument("--labels", type=Path, default=REPO / "data/lm_labels_l8.json")
    ap.add_argument("--sae-repo", default="jbloom/GPT2-Small-SAEs-Reformatted")
    ap.add_argument("--sae-file", default="blocks.8.hook_resid_pre/sae_weights.safetensors")
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    lab = json.loads(args.labels.read_text())
    tiers = np.array(lab["tiers"])
    names = np.array(lab["feature_vocab"])
    data = np.load(args.npz)
    Xraw = data["X"].astype(np.float64)
    Y = data["Y"]
    N, d = Xraw.shape
    Xz = (Xraw - Xraw.mean(0)) / (Xraw.std(0) + 1e-8)
    print(f"[{lab['model']} layer {lab['layer']}]  X={Xraw.shape}  Y={Y.shape}  "
          f"tiers={sorted(set(lab['tiers']))}")

    st = load_file(hf_hub_download(repo_id=args.sae_repo, filename=args.sae_file))
    Wenc = st["W_enc"].astype(np.float64); benc = st["b_enc"].astype(np.float64)
    bdec = st["b_dec"].astype(np.float64)
    print(f"  SAELens dict {args.sae_file}  F={Wenc.shape[1]} ({Wenc.shape[1]/d:.0f}x)")

    sa = sae_per_feature_auc(Xraw, Wenc, benc, bdec, Y)
    ax = recoverability_axes(Xz, Y)
    vs, fi, pr = ax["var_share"], ax["fisher"], ax["probe_auc"]
    valid = ax["valid"] & ~np.isnan(sa)
    vsV, fiV, prV, saV = vs[valid], fi[valid], pr[valid], sa[valid]
    tiersV, namesV, prevV = tiers[valid], names[valid], ax["prevalence"][valid]

    print("\n  --- the predictive model (partial Spearman, confound-free) ---")
    print(f"  {'':<26}{'-> SAE_AUC (allocation)':>24}{'-> probe_AUC (presence)':>26}")
    p_vs_sae = partial_spearman(vsV, saV, fiV); p_vs_pr = partial_spearman(vsV, prV, fiV)
    p_fi_sae = partial_spearman(fiV, saV, vsV); p_fi_pr = partial_spearman(fiV, prV, vsV)
    print(f"  {'partial var_share | fisher':<26}{p_vs_sae:>+24.3f}{p_vs_pr:>+26.3f}")
    print(f"  {'partial fisher | var_share':<26}{p_fi_sae:>+24.3f}{p_fi_pr:>+26.3f}")
    print(f"\n  raw Spearman:  var_share->SAE {spearman(vsV, saV):+.3f}   "
          f"fisher->SAE {spearman(fiV, saV):+.3f}   fisher->probe {spearman(fiV, prV):+.3f}")

    print("\n  per-tier means (the built-in present-vs-allocated contrast):")
    print(f"  {'tier':<10}{'n':>4}{'prev':>8}{'var_share':>11}{'fisher':>9}{'probe':>8}{'SAE':>8}{'cov95':>8}")
    by_tier = {}
    for t in sorted(set(tiersV)):
        i = tiersV == t
        row = dict(n=int(i.sum()), prevalence=float(prevV[i].mean()),
                   var_share=float(np.nanmean(vsV[i])), fisher=float(np.nanmean(fiV[i])),
                   probe=float(np.nanmean(prV[i])), sae=float(np.nanmean(saV[i])),
                   cov95=float(np.nanmean(saV[i] >= 0.95)))
        by_tier[t] = row
        print(f"  {t:<10}{row['n']:>4}{row['prevalence']:>8.4f}{row['var_share']:>11.5f}"
              f"{row['fisher']:>9.1f}{row['probe']:>8.3f}{row['sae']:>8.3f}{row['cov95']:>8.1%}")

    print("\n  present-yet-dropped exemplars (sorted by lowest var_share):")
    mask = (prV >= 0.85) & (saV < 0.85)
    print(f"  {int(mask.sum())} / {int(valid.sum())} valid features are present (probe>=0.85) "
          f"yet dropped (SAE<0.85)")
    exemplars = []
    if mask.any():
        idx = np.where(mask)[0]; idx = idx[np.argsort(vsV[idx])][:8]
        print(f"  {'feature':<22}{'tier':<10}{'prev':>8}{'var_share':>11}{'fisher':>9}{'probe':>8}{'SAE':>8}")
        for i in idx:
            print(f"  {str(namesV[i])[:21]:<22}{tiersV[i]:<10}{prevV[i]:>8.4f}{vsV[i]:>11.6f}"
                  f"{fiV[i]:>9.1f}{prV[i]:>8.3f}{saV[i]:>8.3f}")
            exemplars.append(dict(feature=str(namesV[i]), tier=str(tiersV[i]),
                                  prevalence=float(prevV[i]), var_share=float(vsV[i]),
                                  fisher=float(fiV[i]), probe=float(prV[i]), sae=float(saV[i])))

    out = REPO / "runs" / "substrate" / "recoverability_theory_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "substrate": f"lm-sae / {lab['model']} layer {lab['layer']} + SAELens {args.sae_file}",
        "n_features_valid": int(valid.sum()), "lambda": FISHER_LAM,
        "partial": {"var_share_vs_sae_given_fisher": p_vs_sae,
                    "fisher_vs_sae_given_var_share": p_fi_sae,
                    "var_share_vs_probe_given_fisher": p_vs_pr,
                    "fisher_vs_probe_given_var_share": p_fi_pr},
        "spearman_raw": {"var_share_vs_sae": spearman(vsV, saV), "fisher_vs_sae": spearman(fiV, saV),
                         "fisher_vs_probe": spearman(fiV, prV), "var_share_vs_probe": spearman(vsV, prV)},
        "per_tier": by_tier, "exemplars": exemplars}, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
