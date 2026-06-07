"""Direct-monosemanticity aux terms vs the linear-recoverability proxy — what training pressure lifts cov95?

host_width_sweep.py showed an oracle-recovery aux loss lifts native cov95 for ~free, but it pressures only
*linear recoverability* (a dense linear head can spread a feature across many dims and still recover it — not
monosemantic). This compares aux **modes** at a fixed host width, training tiny GPTs from scratch:

  - none      : LM loss only (baseline).
  - linear    : + BCE from a dense linear head (residual -> oracle labels). The recoverability proxy.
  - decorr    : linear + an orthogonality penalty on the per-feature read directions (rows of the probe
                pushed toward mutually orthogonal) -> each feature reads from a DISTINCT direction.
  - dedicated : + BCE using the first F residual dims *directly* as feature detectors (one neuron per feature
                = the sparsest dictionary / direct axis-alignment). Needs width >= F.

Question: does a DIRECT monosemanticity objective (decorr / dedicated) lift cov95 *more* than linear
recoverability, and at what capability cost? Measures native cov95 + mAUC + held-out LM loss per mode.
Reuses the oracle (host_width_sweep) + the SAE/cov95 scorer (forge_cov_mechanism).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[1] / "common"))
sys.path.insert(0, str(_here.parent))
from forge_cov_mechanism import _best_auc_per_label, _encode, _per_tier, _train_topk_sae  # noqa: E402
from host_width_sweep import CORPUS_URL, build_oracle_table  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--modes", default="none,linear,decorr,dedicated,sparsedict")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--aux-lambda", type=float, default=1.0)
    p.add_argument("--decorr-beta", type=float, default=1.0, help="weight on the orthogonality penalty (decorr mode)")
    p.add_argument("--align-alpha", type=float, default=1.0, help="oracle-alignment vs reconstruction weight (sparsedict mode)")
    p.add_argument("--sae-over", type=int, default=4)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--sae-steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=400000)
    p.add_argument("--eval-tokens", type=int, default=6000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/cov95_forge_tax/monosemantic_aux_summary.json"))
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as F
    import urllib.request
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    text = urllib.request.urlopen(CORPUS_URL, timeout=8).read().decode("utf-8", "ignore")[: args.max_chars]
    ids = np.array(tok(text)["input_ids"], dtype=np.int64)
    n_train = int(len(ids) * 0.9)
    train_ids, eval_ids = ids[:n_train], ids[n_train:]
    Ttab, tiers = build_oracle_table(tok, ids, args.min_pos)
    Ttab_t = torch.from_numpy(Ttab).to(dev); Ffeat = Ttab.shape[1]
    print(f"corpus {len(ids)} tok; width {args.width}; oracle {Ffeat} features; modes {args.modes}")
    if args.width < Ffeat:
        print(f"[warn] width {args.width} < {Ffeat} features -> 'dedicated' mode will be skipped")

    def batch(g):
        starts = torch.randint(0, len(train_ids) - args.ctx - 1, (args.batch,), generator=g)
        x = torch.stack([torch.from_numpy(train_ids[s:s + args.ctx]) for s in starts])
        y = torch.stack([torch.from_numpy(train_ids[s + 1:s + 1 + args.ctx]) for s in starts])
        return x.to(dev), y.to(dev)

    m_sae = args.sae_over * args.width                        # in-loop SAE width (matches the eval SAE)

    class LoopSAE(torch.nn.Module):
        """A sparse dictionary trained jointly on the host residual; first F latents aligned to the oracle."""
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Linear(args.width, m_sae)
            self.dec = torch.nn.Linear(m_sae, args.width)
            self.scale = torch.nn.Parameter(torch.ones(Ffeat))
            self.bias = torch.nn.Parameter(torch.zeros(Ffeat))

        def forward(self, h):
            z = torch.relu(self.enc(h))
            if args.k < m_sae:                                # TopK sparsity per token (kept entries differentiable)
                thr = z.topk(args.k, dim=-1).values[..., -1:].detach()
                z = z * (z >= thr)
            recon = ((self.dec(z) - h) ** 2).mean()
            logit = self.scale * z[..., :Ffeat] + self.bias   # aligned latents 0..F-1 -> oracle-feature logits
            return recon, logit

    def make_aux(mode):
        if mode in ("linear", "decorr"):
            return torch.nn.Linear(args.width, Ffeat).to(dev)
        if mode == "sparsedict":
            return LoopSAE().to(dev)
        return None

    def aux_term(mode, h, lab, aux):
        if mode == "none":
            return h.new_zeros(())
        if mode == "dedicated":
            return F.binary_cross_entropy_with_logits(h[..., :Ffeat], lab)
        if mode == "sparsedict":
            recon, logit = aux(h)                             # reconstruct h sparsely + align latents to oracle
            return recon + args.align_alpha * F.binary_cross_entropy_with_logits(logit, lab)
        bce = F.binary_cross_entropy_with_logits(aux(h), lab)
        if mode == "decorr":
            W = aux.weight                                    # (F, width)
            Wn = W / (W.norm(dim=1, keepdim=True) + 1e-6)
            G = Wn @ Wn.t()
            off = G - torch.diag(torch.diagonal(G))
            bce = bce + args.decorr_beta * (off ** 2).sum() / (Ffeat * (Ffeat - 1))
        return bce

    def train(mode):
        cfg = GPT2Config(vocab_size=tok.vocab_size, n_positions=args.ctx, n_ctx=args.ctx,
                         n_embd=args.width, n_layer=args.n_layer, n_head=args.n_head)
        model = GPT2LMHeadModel(cfg).to(dev).train()
        aux = make_aux(mode)
        params = list(model.parameters()) + (list(aux.parameters()) if aux is not None else [])
        opt = torch.optim.AdamW(params, lr=args.lr)
        g = torch.Generator().manual_seed(0)
        for _ in range(args.steps):
            x, y = batch(g)
            out = model(input_ids=x, labels=y, output_hidden_states=True)
            loss = out.loss + (args.aux_lambda * aux_term(mode, out.hidden_states[-1], Ttab_t[x], aux) if mode != "none" else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        return model.eval()

    def evaluate(model):
        with torch.no_grad():
            ev = eval_ids[: args.eval_tokens]
            chunks = [ev[i:i + args.ctx] for i in range(0, len(ev) - 1, args.ctx) if len(ev[i:i + args.ctx]) >= 8]
            lm_tot = lm_n = 0.0; acts = []; ev_ids = []
            for ch in chunks:
                xx = torch.from_numpy(np.ascontiguousarray(ch[:-1]))[None].to(dev)
                yy = torch.from_numpy(np.ascontiguousarray(ch[1:]))[None].to(dev)
                o = model(input_ids=xx, labels=yy, output_hidden_states=True)
                lm_tot += float(o.loss) * yy.numel(); lm_n += yy.numel()
                acts.append(o.hidden_states[-1][0].float().cpu().numpy()); ev_ids.extend(ch[:-1])
        Xraw = np.concatenate(acts, 0).astype(np.float32); Y = Ttab[np.array(ev_ids)]
        mu, sd = Xraw.mean(0, keepdims=True), Xraw.std(0, keepdims=True) + 1e-6
        X = ((Xraw - mu) / sd).astype(np.float32)
        sae = _train_topk_sae(X, args.sae_over * args.width, args.k, args.sae_steps, 1e-3, 0)
        res = _per_tier(_best_auc_per_label(_encode(X, sae, args.k), Y), tiers)
        return lm_tot / lm_n, res

    rows = []
    for mode in [m.strip() for m in args.modes.split(",")]:
        if mode == "dedicated" and args.width < Ffeat:
            continue
        lm_loss, res = evaluate(train(mode))
        rows.append({"mode": mode, "lm_loss": lm_loss, "cov95": res["all"]["cov95"], "mauc": res["all"]["mauc"],
                     "cov95_token": res.get("token", {}).get("cov95"), "cov95_lexical": res.get("lexical", {}).get("cov95")})
        print(f"  {mode:>10}: LM-loss {lm_loss:.3f}  cov95 {res['all']['cov95']:.3f}  mAUC {res['all']['mauc']:.3f}")

    base = next((r for r in rows if r["mode"] == "none"), None)
    lin = next((r for r in rows if r["mode"] == "linear"), None)

    def lift(r):
        return (r["cov95"] - base["cov95"]) if base else None
    out = {"experiment": "monosemanticity aux modes vs linear-recoverability", "width": args.width,
           "n_oracle_features": Ffeat, "rows": rows,
           "cov95_lift_vs_none": {r["mode"]: lift(r) for r in rows if base},
           "best_mode": max(rows, key=lambda r: r["cov95"])["mode"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print("\n[verdict] cov95 by mode: " + "  ".join(f"{r['mode']} {r['cov95']:.2f}" for r in rows))
    if base and lin:
        direct = [r for r in rows if r["mode"] in ("decorr", "dedicated", "sparsedict")]
        beats = [r["mode"] for r in direct if r["cov95"] > lin["cov95"] + 0.02]
        print(f"  best mode: {out['best_mode']}; linear(recoverability) cov95 {lin['cov95']:.2f}; "
              f"direct-monosemanticity modes beating it: {beats or 'none'}")
        print("  => " + ("a DIRECT monosemanticity objective lifts cov95 beyond linear-recoverability "
                         f"(by {max((r['cov95'] - lin['cov95']) for r in direct):+.2f} best)" if beats
                         else "direct objectives do NOT beat the linear-recoverability proxy here"))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
