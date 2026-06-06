"""Single-layer U_C test — alive forge of the induction-feeding residual on real GPT-2.

The whole-model forge dies on 12-layer GPT-2 (one basis can't carry 12 layers). This tests the U_C
MECHANISM without that: forge ONLY the residual entering the induction heads (layer L) with a self-trained
SAE, keep the other 11 layers HOST (so the model stays alive), and ask whether smearing that residual breaks
induction — and which U_C protects it. Compares five recoveries of the smeared residual:

  single          r -> SAE_decode(SAE_encode(r))                  (lossy recon smears the predecessor signal)
  uc_readers      recon + (r - recon) onto the readers' geometry  (the U_C that FAILED, -6%)
  uc_writers      recon + (r - recon) onto the writers' OV-output (the VALIDATED U_C, -111%)
  uc_attribution  recon + (r - recon) onto top d(loss)/d(residual) (label-free control, +14% worse)
  two_basis_wr    writers' OV-output + the sharp assertion atoms (U_A)

The writer detection and the writers' OV-output U_C now come from the RELEASED sae-forge 0.14.0 API
(saeforge.circuit_heads.prev_token_heads + composition_subspace.extract_writer_subspace), so this is the
consumer-side validation that the shipped library reproduces the -111% excess removal.

Metric: induction-predictable circuit KL (excess over complement) — the circuit-specific damage. The model
stays alive because only layer L's residual is perturbed; blocks L..11 run host weights. Real GPT-2, CPU.
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


def _oracle(tok, all_ids, tok_strs, min_pos, n):
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


def _induction_predictable(c):
    n = len(c); pred = np.zeros(n, bool)
    for t in range(2, n):
        ps = [p for p in range(t - 1) if c[p] == c[t - 1]]
        if ps and c[ps[-1] + 1] == c[t]:
            pred[t] = True
    return pred


def _kl(hl, fl, mask):
    def lsm(x):
        x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
    lp = lsm(hl); lq = lsm(fl); p = np.exp(lp)
    kl = (p * (lp - lq)).sum(-1)
    m = mask.astype(bool)
    mk = float(kl[m].mean()) if m.any() else 0.0
    ck = float(kl[~m].mean()) if (~m).any() else 0.0
    return mk, ck, int(m.sum()), int((~m).sum())


def _uc(model, layers, rank):
    """attention read+write geometry (residual directions) over `layers`, ln_1-folded read."""
    tr = model.transformer; cfg = model.config
    d, Hn = cfg.n_embd, cfg.n_head; hd = d // Hn
    reads, writes = [], []
    for L in layers:
        blk = tr.h[L]
        ln = blk.ln_1.weight.detach().numpy().astype(np.float64)
        Wc = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wo = blk.attn.c_proj.weight.detach().numpy().astype(np.float64)
        Wq, Wk, Wv = Wc[:, :d], Wc[:, d:2 * d], Wc[:, 2 * d:3 * d]
        reads.append(Wq * ln[:, None]); reads.append(Wk * ln[:, None])
        for h in range(Hn):
            sl = slice(h * hd, (h + 1) * hd); writes.append(Wv[:, sl] @ Wo[sl, :])
    Ur = np.linalg.svd(np.concatenate(reads, 1), full_matrices=False)[0][:, :rank]
    Uw = np.linalg.svd(np.concatenate(writes, 1), full_matrices=False)[0][:, :rank]
    return np.linalg.qr(np.concatenate([Ur, Uw], 1))[0]


def _attribution_uc(model, chunks, L, X5, ind_mask, rank, dev):
    """Label-FREE circuit subspace: top principal directions of ∂(induction-predictable NLL)/∂(layer-L
    residual). Backprop the circuit loss to a leaf residual injected at block L; the gradient directions
    are what the circuit is sensitive to — no idiom labels needed."""
    import torch
    tr = model.transformer
    off = np.cumsum([0] + [len(c) for c in chunks])
    grads = []
    for i, c in enumerate(chunks):
        r_leaf = torch.tensor(X5[off[i]:off[i + 1]][None], device=dev, requires_grad=True)
        inj = {"t": r_leaf}

        def pre(mod, a, kw):
            return ((inj["t"],) + a[1:], kw) if len(a) else (a, {**kw, "hidden_states": inj["t"]})
        h = tr.h[L].register_forward_pre_hook(pre, with_kwargs=True)
        logits = model(input_ids=torch.tensor([c], device=dev)).logits[0]
        h.remove()
        lp = torch.log_softmax(logits[:-1].float(), -1)
        tgt = torch.tensor(c[1:], device=dev)
        nll = -lp[torch.arange(len(c) - 1), tgt]
        m = torch.tensor(ind_mask[i].astype(np.float32), device=dev)
        loss = (nll * m).sum()
        if float(m.sum()) > 0:
            loss.backward()
            grads.append(r_leaf.grad[0].detach().cpu().numpy())
        model.zero_grad(set_to_none=True)
    G = np.concatenate(grads, 0)
    return np.linalg.svd(G, full_matrices=False)[2][:rank].T   # (d, rank) top gradient directions


def _overlap(A, B):
    """fraction of subspace A captured by subspace B (both orthonormal-column d x r)."""
    return float(np.linalg.norm(B.T @ A) ** 2 / A.shape[1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--layer", type=int, default=5, help="residual entering this block is forged (induction-feed)")
    p.add_argument("--uc-layers", default="5,6,7", help="layers whose attention geometry U_C preserves")
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--comp-rank", type=int, default=10)
    p.add_argument("--assert-k", type=int, default=96)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--output", type=Path, default=Path("runs/two_basis_single_layer_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    import saeforge
    from saeforge.circuit_heads import prev_token_heads
    from saeforge.composition_subspace import extract_writer_subspace

    dev = args.device
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval().to(dev)
    tr = model.transformer
    nL = model.config.n_layer; Hn = model.config.n_head
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    L = args.layer

    # ---- pass 1: layer-L residual (SAE + cov95), host final hidden, induction mask, oracle ----
    print("[1] host pass: layer-%d residual + induction mask" % L)
    X5, all_ids, ind_mask, Hf = [], [], [], []
    with torch.no_grad():
        for c in chunks:
            o = model(input_ids=torch.tensor([c], device=dev), output_hidden_states=True)
            X5.append(o.hidden_states[L][0].float().cpu().numpy())
            Hf.append(o.hidden_states[-1][0].float().cpu().numpy())
            ind_mask.append(_induction_predictable(c)[1:].astype(bool)); all_ids.extend(c)
    X5 = np.concatenate(X5, 0).astype(np.float32)
    W_U = model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
    N, d = X5.shape
    Y, tiers = _oracle(tok, all_ids, tok.convert_ids_to_tokens(all_ids), args.min_pos, N)
    mu, sd = X5.mean(0, keepdims=True), X5.std(0, keepdims=True) + 1e-6
    Xz = ((X5 - mu) / sd).astype(np.float32)
    M = np.concatenate(ind_mask)
    print(f"    X={X5.shape}  Y={Y.shape}  induction-rate {M.mean():.3f}")

    print(f"[2] train TopK SAE (width {args.width}, {args.width/d:.1f}x) + host cov95")
    params = _train_topk_sae(Xz, args.width, args.k, args.steps, 1e-3, 0)
    host_cov = _per_tier(_best_auc_per_label(_encode(Xz, params, args.k), Y), tiers)
    Wd = params[1].numpy().astype(np.float64)            # (d, width)
    Wdr = Wd.T                                           # (width, d) atoms-as-rows

    uc_layers = [int(x) for x in args.uc_layers.split(",")]
    Uc = _uc(model, uc_layers, args.comp_rank)           # readers' geometry (the U_C that failed)
    # prev-token WRITER heads (Δ=1 movers) BELOW the forged layer, via the RELEASED sae-forge 0.14.0
    # behavioral detector; their OV output is the predecessor-write the SAE smears.
    det = [t for t in prev_token_heads(model, ids, top_k=nL * Hn, ctx=args.ctx, min_attention=0.0)
           if t[0] < L][:4]
    writers = [(Li, h) for (Li, h, _s) in det]
    writer_scores = [float(s) for (_Li, _h, s) in det]
    # released OV-output U_C; match the readers' dim for a fair head-to-head
    Uw = extract_writer_subspace(model, writer_heads=writers, rank=Uc.shape[1]).U
    Za = _encode(Xz, params, args.k)
    atom_auc = np.array([np.nanmax(_best_auc_per_label(Za[:, [j]], Y)) for j in range(args.width)])
    Ua = np.linalg.qr(Wdr[np.argsort(-atom_auc)[: args.assert_k]].T)[0]
    print("    computing attribution subspace (∂induction-loss/∂residual)...")
    Uatt = _attribution_uc(model, chunks, L, X5, ind_mask, Uc.shape[1], dev)   # label-free
    ov_aw = _overlap(Uatt, Uw)                                                  # does attribution = writers?
    Preaders = Uc @ Uc.T
    Pwriters = Uw @ Uw.T
    Patt = Uatt @ Uatt.T
    Uwa = np.linalg.qr(np.concatenate([Uw, Ua], 1))[0]; Pwa = Uwa @ Uwa.T
    print(f"    host cov95 {host_cov['all']['cov95']:.3f}; readers-U_C {Uc.shape[1]}d (layers {uc_layers}); "
          f"writers-U_C {Uw.shape[1]}d (prev-tok {[f'{a}.{b}' for a, b in writers]}); attribution-U_C {Uatt.shape[1]}d")
    print(f"    [science check] subspace overlap(attribution, writers) = {ov_aw:.2f} "
          f"(high => the gradient REDISCOVERS the prev-token writers, no labels needed)")

    def recon(rz):
        return _encode(rz, params, args.k) @ Wd.T        # (n, d) in z-space

    # precompute per-config forged residual at layer L (original space)
    rec_z = recon(Xz)
    r_recon = (rec_z * sd + mu).astype(np.float32)
    forged = {
        "single": r_recon,
        "uc_readers": (r_recon + (X5 - r_recon) @ Preaders).astype(np.float32),
        "uc_writers": (r_recon + (X5 - r_recon) @ Pwriters).astype(np.float32),
        "uc_attribution": (r_recon + (X5 - r_recon) @ Patt).astype(np.float32),
        "two_basis_wr": (r_recon + (X5 - r_recon) @ Pwa).astype(np.float32),
    }
    off = np.cumsum([0] + [len(c) for c in chunks])

    def run(name, Rf):
        inj = {}

        def pre(mod, a, kw):
            t = inj["t"]
            return ((t,) + a[1:], kw) if len(a) else (a, {**kw, "hidden_states": t})
        h = tr.h[L].register_forward_pre_hook(pre, with_kwargs=True)
        msum = mn = csum = cn = gsum = gn = 0.0; fres = []
        with torch.no_grad():
            for i, c in enumerate(chunks):
                inj["t"] = torch.tensor(Rf[off[i]:off[i + 1]][None], device=dev)
                lg = model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float().cpu().numpy()
                hl = Hf[i][:-1] @ W_U.T
                mk, ck, nm, nc = _kl(hl, lg, ind_mask[i]); tot = lg.shape[0]
                msum += mk * nm; mn += nm; csum += ck * nc; cn += nc
                gsum += (mk * nm + ck * nc); gn += tot
                fres.append(Rf[off[i]:off[i + 1]])
        h.remove()
        Xf = ((np.concatenate(fres, 0) - mu) / sd).astype(np.float32)
        cov = _per_tier(_best_auc_per_label(_encode(Xf, params, args.k), Y), tiers)
        r = {"induction_kl": msum / max(mn, 1), "complement_kl": csum / max(cn, 1),
             "global_kl": gsum / max(gn, 1), "cov95": cov["all"]["cov95"]}
        r["excess"] = r["induction_kl"] - r["complement_kl"]
        print(f"  {name:>10}: induction_kl {r['induction_kl']:.3f} excess {r['excess']:+.3f} "
              f"global_kl {r['global_kl']:.3f} cov95 {r['cov95']:.3f}")
        return r

    print(f"[3] forge layer-{L} residual (alive: blocks {L}-11 host)")
    res = {name: run(name, Rf) for name, Rf in forged.items()}
    s, ur, uw, ua, tb = (res["single"], res["uc_readers"], res["uc_writers"],
                         res["uc_attribution"], res["two_basis_wr"])
    out = {"experiment": "single-layer U_C: readers vs writers vs attribution", "layer": L,
           "uc_layers": uc_layers, "writers": [list(w) for w in writers], "writer_scores": writer_scores,
           "n_features": args.width, "sae_forge_version": saeforge.__version__,
           "overlap_attribution_writers": ov_aw, "host_cov95": host_cov["all"]["cov95"], "configs": res}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    def pct(r):
        return (s["excess"] - r["excess"]) / max(s["excess"], 1e-9)
    print(f"\n[circuit-specific excess] single {s['excess']:+.3f}  ->  readers {ur['excess']:+.3f} ({pct(ur):.0%})  |  "
          f"writers {uw['excess']:+.3f} ({pct(uw):.0%})  |  ATTRIBUTION {ua['excess']:+.3f} ({pct(ua):.0%})")
    attr_works = pct(ua) > 0.5 and s["excess"] > 0
    print(f"[verdict] {'ATTRIBUTION (label-free) PROTECTS induction too — overlap with the idiom-identified writers '+f'{ov_aw:.2f}'+'; the gradient rediscovers the circuit-critical writer subspace WITHOUT labels, so the fix generalises to any circuit via ∂loss/∂residual' if attr_works else 'attribution subspace does NOT protect (label-free version fails; writer identification still needed)'}")
    print(f"[alive?] global_kl single {s['global_kl']:.3f} / attribution {ua['global_kl']:.3f}  host cov95 {host_cov['all']['cov95']:.3f}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
