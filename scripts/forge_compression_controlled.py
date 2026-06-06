"""Compression-controlled re-validation of the two-basis writer-output U_C.

The forge-tax claim used excess = induction_kl - complement_kl, which is GAMEABLE: preserving a subspace
changes the overall compression (complement_kl), and a subspace can lower excess just by damaging the
complement. PR #7's mechanism controls showed the writer-preserve barely moved ABSOLUTE induction_kl
(3.901->3.830) while making complement 20% worse -> the headline -111% "tax removal" was mostly the
complement rising, not induction preserved.

This re-validates at MATCHED COMPRESSION. The forge is forged = recon + (X-recon) @ P_r where P_r projects
onto a rank-r preserved subspace; as r grows, BOTH induction_kl and complement_kl fall toward 0 (full
preservation = not forging). For each subspace TYPE we sweep r and trace the (complement_kl, induction_kl)
curve. The non-gameable question:

  at a MATCHED complement_kl (compression budget), does the WRITER/OV subspace give a lower induction_kl
  (smaller induction tax ind-comp) than random subspaces?

  - if writer_OV's curve sits below random at matched complement_kl -> genuine circuit preservation (RESCUE)
  - if all types coincide -> the tax just tracks compression; no circuit-specific preservation (RETIRE)

Subspace types: writer_OV (detected prev-token writers' OV-output), random_OV (random heads' OV),
top_pc (residual PCs = max-variance), random_orth (variance-agnostic). Alive single-layer GPT-2 forge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forge_cov_mechanism import _encode, _train_topk_sae  # noqa: E402


def _induction_predictable(c):
    n = len(c); pred = np.zeros(n, bool)
    for t in range(2, n):
        ps = [p for p in range(t - 1) if c[p] == c[t - 1]]
        if ps and c[ps[-1] + 1] == c[t]:
            pred[t] = True
    return pred


def _kl(hl, fl, mask):
    def lsm(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))
    lp = lsm(hl); lq = lsm(fl); p = np.exp(lp)
    kl = (p * (lp - lq)).sum(-1)
    m = mask.astype(bool)
    return (float(kl[m].mean()) if m.any() else 0.0, float(kl[~m].mean()) if (~m).any() else 0.0,
            int(m.sum()), int((~m).sum()))


def _interp(xs, ys, x0):
    """linear-interpolate y at x0 from (xs, ys) sorted by x ascending; nan if out of range."""
    o = np.argsort(xs); xs, ys = np.array(xs)[o], np.array(ys)[o]
    if x0 < xs[0] or x0 > xs[-1]:
        return float("nan")
    return float(np.interp(x0, xs, ys))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--ranks", default="4,8,16,32,64,128")
    p.add_argument("--top-k", type=int, default=4, help="# detected/random heads per OV subspace")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("runs/forge_compression_controlled_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/forge_compression_controlled.png"))
    args = p.parse_args(argv)

    import torch
    import saeforge
    from saeforge.circuit_heads import prev_token_heads
    from saeforge.composition_subspace import extract_writer_subspace
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    L = args.layer
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    tr = model.transformer
    nL = model.config.n_layer; Hn = model.config.n_head
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    ranks = [int(x) for x in args.ranks.split(",")]

    print(f"[1] host pass: layer-{L} residual + induction mask")
    X5, ind_mask, Hf = [], [], []
    with torch.no_grad():
        for c in chunks:
            o = model(input_ids=torch.tensor([c]), output_hidden_states=True)
            X5.append(o.hidden_states[L][0].float().numpy())
            Hf.append(o.hidden_states[-1][0].float().numpy())
            ind_mask.append(_induction_predictable(c)[1:].astype(bool))
    X5 = np.concatenate(X5, 0).astype(np.float32)
    W_U = model.lm_head.weight.detach().numpy().astype(np.float32)
    N, d = X5.shape
    mu, sd = X5.mean(0, keepdims=True), X5.std(0, keepdims=True) + 1e-6
    Xz = ((X5 - mu) / sd).astype(np.float32)
    print(f"    X={X5.shape}  induction-rate {np.concatenate(ind_mask).mean():.3f}")

    print(f"[2] train TopK SAE (width {args.width}) + recon")
    params = _train_topk_sae(Xz, args.width, args.k, args.steps, 1e-3, 0)
    Wd = params[1].numpy().astype(np.float64)
    r_recon = ((_encode(Xz, params, args.k) @ Wd.T) * sd + mu).astype(np.float32)
    off = np.cumsum([0] + [len(c) for c in chunks])
    rng = np.random.default_rng(args.seed)

    det = [(Li, h) for (Li, h, _s) in prev_token_heads(model, ids, top_k=nL * Hn, ctx=args.ctx, min_attention=0.0)
           if Li < L][:args.top_k]
    rnd_heads = [(int(rng.integers(0, L)), int(rng.integers(0, Hn))) for _ in range(args.top_k)]
    Xc = X5 - X5.mean(0)
    Vpc = np.linalg.svd(Xc, full_matrices=False)[2]                     # (min(N,d), d) residual PC dirs
    print(f"    detected writers <{L}: {[f'{a}.{b}' for a, b in det]}; random heads: {[f'{a}.{b}' for a, b in rnd_heads]}")

    def subspace(kind, r):
        if kind == "writer_OV":
            return extract_writer_subspace(model, writer_heads=det, rank=r).U
        if kind == "random_OV":
            return extract_writer_subspace(model, writer_heads=rnd_heads, rank=r).U
        if kind == "top_pc":
            return Vpc[:r].T
        if kind == "random_orth":
            return np.linalg.qr(rng.standard_normal((d, r)))[0][:, :r]
        raise ValueError(kind)

    def forge_kl(U):
        P = (U @ U.T).astype(np.float32)
        Rf = (r_recon + (X5 - r_recon) @ P).astype(np.float32)
        inj = {}

        def pre(mod, a, kw):
            return ((inj["t"],) + a[1:], kw) if len(a) else (a, {**kw, "hidden_states": inj["t"]})
        h = tr.h[L].register_forward_pre_hook(pre, with_kwargs=True)
        ms = mn = cs = cn = 0.0
        with torch.no_grad():
            for i, c in enumerate(chunks):
                inj["t"] = torch.tensor(Rf[off[i]:off[i + 1]][None])
                lg = model(input_ids=torch.tensor([c])).logits[0, :-1].float().numpy()
                mk, ck, nm, nc = _kl(Hf[i][:-1] @ W_U.T, lg, ind_mask[i])
                ms += mk * nm; mn += nm; cs += ck * nc; cn += nc
        h.remove()
        return ms / max(mn, 1), cs / max(cn, 1)

    # baseline: SAE recon only (no preserve)
    base_ik, base_ck = forge_kl(np.zeros((d, 1)))   # P~0 -> recon only (rank-1 of zeros => projector 0)
    print(f"\n[3] sweep rank x subspace-type (alive layer-{L} forge); baseline recon-only "
          f"ind {base_ik:.3f} comp {base_ck:.3f} (tax {base_ik-base_ck:+.3f})")
    kinds = ["writer_OV", "random_OV", "top_pc", "random_orth"]
    curves = {kind: [] for kind in kinds}
    print(f"  {'type':>12} {'rank':>5} {'comp_kl':>8} {'ind_kl':>8} {'tax(ind-comp)':>13}")
    for kind in kinds:
        for r in ranks:
            U = subspace(kind, r)
            ik, ck = forge_kl(U)
            curves[kind].append({"rank": r, "actual_rank": int(U.shape[1]), "comp_kl": ck,
                                 "ind_kl": ik, "tax": ik - ck})
            print(f"  {kind:>12} {r:>5} {ck:>8.3f} {ik:>8.3f} {ik-ck:>+13.3f}")

    # ---- matched-compression comparison: interpolate ind_kl + tax at common complement_kl targets ----
    lo = max(min(p["comp_kl"] for p in curves[k]) for k in kinds)
    hi = min(max(p["comp_kl"] for p in curves[k]) for k in kinds)
    targets = [round(t, 2) for t in np.linspace(lo, hi, 4)] if hi > lo else []
    print(f"\n[4] MATCHED-COMPRESSION comparison (induction tax = ind-comp at equal complement_kl):")
    print(f"  {'comp_kl':>8} " + " ".join(f"{k:>12}" for k in kinds))
    matched = []
    for t in targets:
        taxes = {k: _interp([p["comp_kl"] for p in curves[k]], [p["tax"] for p in curves[k]], t) for k in kinds}
        matched.append({"comp_kl": t, "tax": taxes})
        print(f"  {t:>8.2f} " + " ".join(f"{taxes[k]:>+12.3f}" for k in kinds))

    out = {"experiment": "compression-controlled two-basis re-validation", "layer": L,
           "sae_forge_version": saeforge.__version__, "ranks": ranks, "detected_writers": [list(w) for w in det],
           "baseline_recon_only": {"ind_kl": base_ik, "comp_kl": base_ck},
           "curves": curves, "matched_compression": matched}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    # verdict: at matched complement_kl, is writer_OV's tax meaningfully below the random subspaces?
    if matched:
        deltas = []
        for m in matched:
            w = m["tax"]["writer_OV"]; rnd = np.nanmean([m["tax"]["random_OV"], m["tax"]["random_orth"]])
            if np.isfinite(w) and np.isfinite(rnd):
                deltas.append(rnd - w)   # >0 => writer_OV has LOWER tax than random at matched compression
        md = float(np.nanmean(deltas)) if deltas else float("nan")
        rescue = np.isfinite(md) and md > 0.05
        print(f"\n[verdict] mean(random_tax - writer_OV_tax) at matched complement_kl = {md:+.3f}")
        print(f"  {'RESCUE: writer/OV subspace preserves induction beyond random at matched compression' if rescue else 'RETIRE: at matched compression the induction tax is ~the same across subspace types => the writer-output U_C is NOT circuit-specific; the -111% excess was a compression/complement-damage artifact'}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5.5))
        col = {"writer_OV": "#d62728", "random_OV": "#1f77b4", "top_pc": "#2ca02c", "random_orth": "#999999"}
        for k in kinds:
            cc = sorted(curves[k], key=lambda p: p["comp_kl"])
            ax.plot([p["comp_kl"] for p in cc], [p["ind_kl"] for p in cc], "-o", color=col[k], label=k, ms=4)
        lim = [0, max(base_ik, base_ck) * 1.1]
        ax.plot(lim, lim, "k:", lw=0.8, label="ind=comp (no tax)")
        ax.scatter([base_ck], [base_ik], c="k", marker="x", s=60, label="recon-only")
        ax.set_xlabel("complement_kl  (compression budget; left = less compression)")
        ax.set_ylabel("induction_kl  (circuit damage)")
        ax.set_title("Compression-controlled: induction vs complement KL by preserved subspace")
        ax.legend(fontsize=8)
        fig.tight_layout(); args.fig.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.fig, dpi=130)
        print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
