"""Broadened compression-controlled re-validation — does the RETIRE verdict hold across layers/seeds?

PR #8 retired the writer-output U_C claim at layer-5/seed-0 (compression-controlled): the writer-OV
subspace never reduces induction_kl below the recon-only baseline and is indistinguishable from a
random-OV subspace at matched compression. Before recommending a caveat/deprecation of a RELEASED feature
(sae-forge v0.14.0), confirm generality: sweep the forged layer L in {induction-feeding band} and compare
writer_OV against a DISTRIBUTION of random_OV subspaces (multiple seeds) at matched rank.

Per layer L: train a TopK SAE on the layer-L residual, recon, then for each rank compare
  writer_OV tax (ind-comp)   vs   random_OV tax distribution (mean +/- sd over seeds)
and check whether writer_OV ever (a) reduces induction_kl below the recon-only baseline or (b) beats the
random_OV distribution (tax below mean - sd). If neither across all (layer, rank) -> RETIRE confirmed broadly.
Alive single-layer forge; consumes released sae-forge 0.14.0 API. GPT-2, CPU.
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", default="5,6,7", help="forged layers (residual entering each; induction band)")
    p.add_argument("--ranks", default="8,32")
    p.add_argument("--n-random-seeds", type=int, default=4)
    p.add_argument("--top-k", type=int, default=4,
                   help="# prev-token writer heads (the induction predecessor-write is carried by a few — "
                        "4.11 dominant + a small tail; top-4 covers them and matches the random-set size)")
    p.add_argument("--seed", type=int, default=0, help="offsets the SAE-train seed + the random-head seeds")
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--output", type=Path, default=Path("runs/forge_revalidate_broad_summary.json"))
    args = p.parse_args(argv)

    import torch
    import saeforge
    from saeforge.circuit_heads import prev_token_heads
    from saeforge.composition_subspace import extract_writer_subspace
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    tr = model.transformer
    nL = model.config.n_layer; Hn = model.config.n_head
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    layers = [int(x) for x in args.layers.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    off = np.cumsum([0] + [len(c) for c in chunks])

    print(f"[1] host pass: residuals for layers {layers} + final hidden + induction mask")
    Xs = {L: [] for L in layers}; Hf = []; ind_mask = []
    with torch.no_grad():
        for c in chunks:
            o = model(input_ids=torch.tensor([c]), output_hidden_states=True)
            for L in layers:
                Xs[L].append(o.hidden_states[L][0].float().numpy())
            Hf.append(o.hidden_states[-1][0].float().numpy())
            ind_mask.append(_induction_predictable(c)[1:].astype(bool))
    Xs = {L: np.concatenate(v, 0).astype(np.float32) for L, v in Xs.items()}
    W_U = model.lm_head.weight.detach().numpy().astype(np.float32)

    def forge_kl(L, Rf):
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

    report = {}
    wins = 0; tests = 0
    for L in layers:
        X5 = Xs[L]
        mu, sd = X5.mean(0, keepdims=True), X5.std(0, keepdims=True) + 1e-6
        Xz = ((X5 - mu) / sd).astype(np.float32)
        params = _train_topk_sae(Xz, args.width, args.k, args.steps, 1e-3, args.seed)
        Wd = params[1].numpy().astype(np.float64)
        r_recon = ((_encode(Xz, params, args.k) @ Wd.T) * sd + mu).astype(np.float32)
        base_ik, base_ck = forge_kl(L, r_recon)
        det = [(Li, h) for (Li, h, _s) in prev_token_heads(model, ids, top_k=nL * Hn, ctx=args.ctx,
                                                           min_attention=0.0) if Li < L][:args.top_k]

        def forge_sub(heads, rank):
            U = extract_writer_subspace(model, writer_heads=heads, rank=rank).U
            Rf = (r_recon + (X5 - r_recon) @ (U @ U.T)).astype(np.float32)
            ik, ck = forge_kl(L, Rf)
            return ik, ck, ik - ck

        print(f"\n[L={L}] recon-only baseline ind {base_ik:.3f} comp {base_ck:.3f} (tax {base_ik-base_ck:+.3f}); "
              f"writers {[f'{a}.{b}' for a, b in det]}")
        layer_rows = []
        for r in ranks:
            wik, wck, wtax = forge_sub(det, r)
            rnds = []
            for s in range(args.n_random_seeds):
                rg = np.random.default_rng(1_000_000 * args.seed + 1000 * L + 10 * r + s)
                rheads = [(int(rg.integers(0, L)), int(rg.integers(0, Hn))) for _ in range(args.top_k)]
                rnds.append(forge_sub(rheads, r))
            rtax = np.array([t for _ik, _ck, t in rnds]); rmu = float(rtax.mean()); rsd = float(rtax.std() + 1e-9)
            below_base = wik < base_ik - 0.05
            beats_random = wtax < rmu - rsd
            wins += int(below_base or beats_random); tests += 1
            layer_rows.append({"rank": r, "writer_ind": wik, "writer_comp": wck, "writer_tax": wtax,
                               "random_tax_mean": rmu, "random_tax_sd": rsd,
                               "writer_ind_below_baseline": bool(below_base), "writer_beats_random": bool(beats_random)})
            print(f"  rank {r:>3}: writer_OV ind {wik:.3f} comp {wck:.3f} tax {wtax:+.3f} | "
                  f"random_OV tax {rmu:+.3f}±{rsd:.3f} | below-base {below_base} beats-rand {beats_random}")
        report[str(L)] = {"baseline_ind": base_ik, "baseline_comp": base_ck, "writers": [list(w) for w in det],
                          "ranks": layer_rows}

    out = {"experiment": "broadened compression-controlled re-validation (writer-output U_C)",
           "model": "gpt2", "sae_forge_version": saeforge.__version__, "layers": layers, "ranks": ranks,
           "n_random_seeds": args.n_random_seeds, "seed": args.seed, "per_layer": report,
           "writer_wins": wins, "tests": tests}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[verdict] writer_OV beat baseline-or-random in {wins}/{tests} (layer,rank) configs")
    print(f"  {'RETIRE CONFIRMED across layers/seeds: writer-output U_C does not preserve induction beyond a random-OV subspace at matched rank, and never reduces induction below the recon-only baseline' if wins == 0 else 'partial: writer_OV wins in some configs — inspect before deprecating'}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
