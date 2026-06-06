"""Forge writer-specificity — do the CAUSALLY-VALIDATED writers improve circuit preservation, or would
any rank-r subspace do? Closes the disassembly -> forge loop.

The two-basis forge preserves the writers' OV-output U_C and that removes the induction tax (-111%, see
two_basis_single_layer.py). But is it the SPECIFIC writers, or just "preserve some rank-r subspace"? The
disassembly says prev-token 4.11 is the writer that feeds induction (causally validated: ablating it raises
induction-NLL z=2.5; corpus-robust: prev-token head identity Spearman 0.99 verse<->prose). This forges the
layer-L residual (alive: blocks L..11 host) and preserves U_C from competing writer sets:

  single          recon only (the tax)
  detected        prev-token writers <L (released circuit_heads detector) — the validated default
  head_4.11       JUST the canonical, causally-validated + corpus-robust prev-token head
  random x R      matched-count random heads <L — the SPECIFICITY CONTROL (any rank-r subspace?)

Metric: induction-predictable circuit excess (induction_kl - complement_kl) and its removal vs `single`.
If detected/4.11 remove the tax and random does NOT, circuit preservation needs the SPECIFIC validated
writers — the disassembly is load-bearing for the forge. Released sae-forge 0.14.0 API. GPT-2, CPU.
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
    mk = float(kl[m].mean()) if m.any() else 0.0
    ck = float(kl[~m].mean()) if (~m).any() else 0.0
    return mk, ck, int(m.sum()), int((~m).sum())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=5, help="forge the residual entering this block (induction-feed)")
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--comp-rank", type=int, default=10, help="U_C rank (matched across all writer sets)")
    p.add_argument("--top-k", type=int, default=4, help="# detected prev-token writers (and the random set size)")
    p.add_argument("--n-random", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("runs/forge_writer_specificity_summary.json"))
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

    # ---- pass 1: layer-L residual + host final hidden + induction mask ----
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

    # ---- writer sets (rank-matched U_C) ----
    rng = np.random.default_rng(args.seed)
    det = [(Li, h) for (Li, h, _s) in prev_token_heads(model, ids, top_k=nL * Hn, ctx=args.ctx, min_attention=0.0)
           if Li < L][:args.top_k]
    sets = {"detected": det, "head_4.11": [(4, 11)]}
    for i in range(args.n_random):
        sets[f"random_{i}"] = [(int(rng.integers(0, L)), int(rng.integers(0, Hn))) for _ in range(len(det))]
    print(f"    detected prev-token writers <{L}: {[f'{a}.{b}' for a, b in det]}")

    def proj(heads):
        U = extract_writer_subspace(model, writer_heads=heads, rank=args.comp_rank).U
        return (U @ U.T).astype(np.float32)
    forged = {"single": r_recon}
    for name, heads in sets.items():
        P = proj(heads)
        forged[name] = (r_recon + (X5 - r_recon) @ P).astype(np.float32)
    # mechanism controls: top residual PCs (pure VARIANCE recapture, no head geometry) +
    # random orthonormal subspace (truly ANY rank-r, variance-agnostic)
    Upc = np.linalg.svd(X5 - X5.mean(0), full_matrices=False)[2][:args.comp_rank].T
    forged["top_pc"] = (r_recon + (X5 - r_recon) @ (Upc @ Upc.T)).astype(np.float32)
    Q = np.linalg.qr(rng.standard_normal((d, args.comp_rank)))[0]
    forged["random_orth"] = (r_recon + (X5 - r_recon) @ (Q @ Q.T)).astype(np.float32)

    off = np.cumsum([0] + [len(c) for c in chunks])

    def run(name, Rf):
        inj = {}

        def pre(mod, a, kw):
            t = inj["t"]
            return ((t,) + a[1:], kw) if len(a) else (a, {**kw, "hidden_states": t})
        h = tr.h[L].register_forward_pre_hook(pre, with_kwargs=True)
        msum = mn = csum = cn = 0.0
        with torch.no_grad():
            for i, c in enumerate(chunks):
                inj["t"] = torch.tensor(Rf[off[i]:off[i + 1]][None])
                lg = model(input_ids=torch.tensor([c])).logits[0, :-1].float().numpy()
                hl = Hf[i][:-1] @ W_U.T
                mk, ck, nm, nc = _kl(hl, lg, ind_mask[i])
                msum += mk * nm; mn += nm; csum += ck * nc; cn += nc
        h.remove()
        ik = msum / max(mn, 1); ck = csum / max(cn, 1)
        return {"induction_kl": ik, "complement_kl": ck, "excess": ik - ck}

    print(f"[3] forge layer-{L} residual (alive: blocks {L}-11 host)")
    res = {name: run(name, Rf) for name, Rf in forged.items()}
    base_ik = res["single"]["induction_kl"]; base_ex = res["single"]["excess"]
    # PRIMARY metric = induction_kl (lower = circuit better preserved). excess (ind-comp) is reported but is
    # GAMEABLE: a subspace that damages the complement lowers excess without preserving induction.
    for name in forged:
        r = res[name]
        tag = "" if name == "single" else f"  Δind_kl {r['induction_kl']-base_ik:+.3f}"
        print(f"  {name:>10}: induction_kl {r['induction_kl']:.3f}  excess {r['excess']:+.3f}{tag}")

    rand_ik = np.array([res[n]["induction_kl"] for n in sets if n.startswith("random")])
    rmu = float(rand_ik.mean()); rsd = float(rand_ik.std() + 1e-9)
    det_ik = res["detected"]["induction_kl"]; h411_ik = res["head_4.11"]["induction_kl"]
    # specificity on the non-gameable metric: is detected's induction_kl BELOW the random distribution?
    spec_z = (rmu - det_ik) / rsd                         # >2 => detected preserves induction beyond random
    out = {"experiment": "forge writer-specificity (causal writers vs matched-rank random control)",
           "layer": L, "comp_rank": args.comp_rank, "sae_forge_version": saeforge.__version__,
           "single_induction_kl": base_ik, "single_excess": base_ex,
           "detected_writers": [list(w) for w in det],
           "induction_kl": {n: res[n]["induction_kl"] for n in forged},
           "detected_vs_random_z": spec_z, "random_induction_kl_mean": rmu, "random_induction_kl_sd": rsd,
           "results": res}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    specific = base_ex > 0 and (det_ik < rmu) and spec_z > 2
    print(f"\n[specificity, on induction_kl] detected {det_ik:.3f} | 4.11-alone {h411_ik:.3f} | "
          f"random {rmu:.3f}±{rsd:.3f}  (single {base_ik:.3f}; lower=better-preserved)")
    print(f"  detected-vs-random z = {spec_z:+.1f}")
    print(f"[verdict] {'SPECIFIC: the causally-validated writers preserve induction beyond a rank-matched '
                       'random subspace' if specific else 'NOT writer-specific: a rank-matched RANDOM OV-output '
                       'subspace preserves induction comparably => the forge needs WRITE (OV-output) geometry, not '
                       'these particular heads. (The earlier win was write-vs-read geometry, not writer-identity.)'}")
    pc_ik = res["top_pc"]["induction_kl"]; ro_ik = res["random_orth"]["induction_kl"]
    print(f"[mechanism] top-residual-PC {pc_ik:.3f} | random-orthonormal {ro_ik:.3f}  (single {base_ik:.3f}) -> "
          f"{'VARIANCE-RECAPTURE: top-PC preserves induction, random-orthonormal does not => any subspace ALIGNED '
            'WITH THE RESIDUALS HIGH-VARIANCE directions works; OV-output works only because it spans them' if (pc_ik < base_ik - 0.3 and ro_ik > base_ik - 0.2) else 'mixed; see numbers'}")
    out["mechanism"] = {"top_pc_induction_kl": pc_ik, "random_orth_induction_kl": ro_ik}
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
