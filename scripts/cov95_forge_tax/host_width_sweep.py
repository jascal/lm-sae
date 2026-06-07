"""Host-width × oracle-supervision sweep — is the forge tax a capacity-scarcity artifact, and is
interpretability reachable by training pressure? (the existence-vs-reachability test)

Trains tiny GPTs FROM SCRATCH across host widths (n_embd), with and without an **auxiliary
oracle-feature-recovery loss** (a linear head from the final residual must predict the exact per-token
oracle labels, BCE, added to the LM loss). Then measures, on each trained host: native **cov95**
(monosemanticity — train a TopK SAE on the final residual, fraction of oracle features with a single
latent at AUC≥0.95), **mAUC** (feature content), and held-out **LM loss** (capability).

Two predictions it adjudicates:
  (a) SCARCITY: superposition is forced by a capacity shortage (Elhage), so **native cov95 should rise toward
      ceiling as host width grows, with little capability cost for the head of the feature distribution** — if
      so, the forge tax was a starvation artifact, not a law. (Our prior width sweep moved the *SAE/dictionary*
      width, not the *host* width — this fixes that.)
  (b) REACHABILITY: interpretable equicapable solutions exist (superposition is linear compression — decompress
      and capability is unchanged), so the open question is whether SGD *reaches* them. The oracle-as-aux-loss
      is a reachability lever: does supervision **lift cov95 at equal LM loss**?

CPU/GPU; reuses the oracle (build_lm_bundle) + the SAE/cov95 scorer (forge_cov_mechanism).
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

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def build_oracle_table(tok, corpus_ids, min_pos):
    """Per-vocab oracle-label table T[V,F] (token-identity + lexical + struct), filtered to features with
    >= min_pos positives in the corpus. Returns (T float32 [V,F], tiers)."""
    V = tok.vocab_size
    feats, tiers, cols = [], [], []
    for c in COMMON:                                   # one-token detectors
        cid = tok(c, add_special_tokens=False)["input_ids"]
        if len(cid) == 1:
            col = np.zeros(V, np.float32); col[cid[0]] = 1.0
            feats.append(f"token:{c!r}"); tiers.append("token"); cols.append(col)
    vocab_strs = tok.convert_ids_to_tokens(list(range(V)))
    lex = [_lexical_features(s) for s in vocab_strs]
    for name in lex[0]:                                # lexical / struct, per token-string
        col = np.array([r[name] for r in lex], np.float32)
        feats.append(name); tiers.append("struct" if name.startswith("struct") else "lexical"); cols.append(col)
    T = np.stack(cols, 1)                              # (V, F)
    # filter by corpus frequency
    corpus_pos = T[corpus_ids].sum(0); n = len(corpus_ids)
    keep = (corpus_pos >= min_pos) & (corpus_pos <= n - min_pos)
    return T[:, keep], [t for t, k in zip(tiers, keep) if k]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--widths", default="32,64,128,256")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--aux-lambda", type=float, default=1.0, help="weight on the oracle-recovery aux loss")
    p.add_argument("--sae-over", type=int, default=4, help="SAE width = sae_over x host width (cov95 probe)")
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--sae-steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=400000)
    p.add_argument("--eval-tokens", type=int, default=6000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/cov95_forge_tax/host_width_sweep_summary.json"))
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
    Ttab_t = torch.from_numpy(Ttab).to(dev)
    Ffeat = Ttab.shape[1]
    print(f"corpus {len(ids)} tok ({n_train} train); oracle {Ffeat} features {dict((t, tiers.count(t)) for t in set(tiers))}")

    def batch(src, bs, g):
        starts = torch.randint(0, len(src) - args.ctx - 1, (bs,), generator=g)
        x = torch.stack([torch.from_numpy(src[s:s + args.ctx]) for s in starts])
        y = torch.stack([torch.from_numpy(src[s + 1:s + 1 + args.ctx]) for s in starts])
        return x.to(dev), y.to(dev)

    def train_one(width, supervised):
        cfg = GPT2Config(vocab_size=tok.vocab_size, n_positions=args.ctx, n_ctx=args.ctx,
                         n_embd=width, n_layer=args.n_layer, n_head=args.n_head)
        model = GPT2LMHeadModel(cfg).to(dev).train()
        aux_head = torch.nn.Linear(width, Ffeat).to(dev)
        params = list(model.parameters()) + (list(aux_head.parameters()) if supervised else [])
        opt = torch.optim.AdamW(params, lr=args.lr)
        g = torch.Generator().manual_seed(0)
        for step in range(args.steps):
            x, y = batch(train_ids, args.batch, g)
            out = model(input_ids=x, labels=y, output_hidden_states=True)
            loss = out.loss
            if supervised:
                h = out.hidden_states[-1]                       # (B, ctx, width)
                lab = Ttab_t[x]                                 # (B, ctx, F)
                loss = loss + args.aux_lambda * F.binary_cross_entropy_with_logits(aux_head(h), lab)
            opt.zero_grad(); loss.backward(); opt.step()
        return model.eval()

    def evaluate(model, width):
        # capability: held-out LM loss
        model.eval()
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
        lm_loss = lm_tot / lm_n
        # native cov95: SAE on final residual vs oracle
        Xraw = np.concatenate(acts, 0).astype(np.float32)
        Y = Ttab[np.array(ev_ids)]                              # (M, F) labels for eval tokens
        mu, sd = Xraw.mean(0, keepdims=True), Xraw.std(0, keepdims=True) + 1e-6
        X = ((Xraw - mu) / sd).astype(np.float32)
        sae = _train_topk_sae(X, args.sae_over * width, args.k, args.sae_steps, 1e-3, 0)
        res = _per_tier(_best_auc_per_label(_encode(X, sae, args.k), Y), tiers)
        return lm_loss, res

    rows = []
    for w in [int(x) for x in args.widths.split(",")]:
        for sup in (False, True):
            model = train_one(w, sup)
            lm_loss, res = evaluate(model, w)
            np_ = sum(pp.numel() for pp in model.parameters())
            rows.append({"width": w, "supervised": sup, "n_params": int(np_), "lm_loss": lm_loss,
                         "cov95": res["all"]["cov95"], "mauc": res["all"]["mauc"],
                         "cov95_token": res.get("token", {}).get("cov95"),
                         "cov95_lexical": res.get("lexical", {}).get("cov95")})
            print(f"  width {w:>4} sup={int(sup)} ({np_/1e6:.1f}M): LM-loss {lm_loss:.3f}  cov95 {res['all']['cov95']:.3f}  mAUC {res['all']['mauc']:.3f}")

    # ---- tests ----
    def by(w, s):
        return next(r for r in rows if r["width"] == w and r["supervised"] == s)
    widths = sorted({r["width"] for r in rows})
    cov_off = [by(w, False)["cov95"] for w in widths]
    peak_w = widths[int(np.argmax(cov_off))]
    rise_to_peak = float(max(cov_off) - cov_off[0])                       # cov95 gain from narrowest to its peak
    # is the widest host undertrained? (worse LM loss than some narrower host -> budget artifact, not scarcity counter-evidence)
    widest_undertrained = bool(by(widths[-1], False)["lm_loss"] > min(by(w, False)["lm_loss"] for w in widths[:-1]))
    sup_lift = float(np.mean([by(w, True)["cov95"] - by(w, False)["cov95"] for w in widths]))
    cap_cost = float(np.mean([by(w, True)["lm_loss"] - by(w, False)["lm_loss"] for w in widths]))
    out = {"experiment": "host-width x oracle-supervision sweep", "widths": widths, "n_oracle_features": Ffeat,
           "rows": rows, "cov95_unsup_by_width": dict(zip(map(str, widths), cov_off)),
           "cov95_rise_narrowest_to_peak": rise_to_peak, "cov95_peak_width": peak_w,
           "widest_host_undertrained": widest_undertrained, "supervision_cov95_lift_mean": sup_lift,
           "supervision_capability_cost_mean_nats": cap_cost}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print("\n[scarcity]  unsup cov95 by width: " + " ".join(f"w{w}:{c:.2f}" for w, c in zip(widths, cov_off))
          + f"  -> rises {cov_off[0]:.2f}->{max(cov_off):.2f} to a peak at w{peak_w} "
          + (f"(widest w{widths[-1]} is UNDERTRAINED — worse LM loss — so its drop is a budget artifact)" if widest_undertrained else "")
          + (" => cov95 RISES with host width (forge tax partly a capacity-scarcity artifact)" if rise_to_peak > 0.1 else " => flat"))
    print(f"[reachability] oracle-supervision lifts cov95 by {sup_lift:+.3f} on average at {cap_cost:+.3f} nats LM-loss cost "
          f"-> {'REACHABLE (cov95 up, capability ~free or better)' if sup_lift > 0.05 and cap_cost < 0.3 else 'no clear free lift'}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
