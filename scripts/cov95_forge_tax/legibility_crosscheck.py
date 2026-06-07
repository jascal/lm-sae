"""Is the supervision cov95 lift real monosemanticity, or just the eval-SAE finding what we planted?

monosemantic_aux.py showed an oracle-recovery aux loss lifts SAE-measured cov95. But cov95 fits a TopK SAE to
the residual — so the worry: does the *linear* aux just make features recoverable in a way a *linear-ish SAE*
preferentially finds (circularity)? This cross-checks the lift in **three bases**, only one of which involves
an SAE:

  - **sae**    : best-latent AUC after fitting a TopK SAE (the existing cov95).
  - **neuron** : best single *raw residual dimension* AUC (NO dictionary fit at all — the natural basis).
  - **pca**    : best single PCA component AUC (an SAE-free *rotated* basis).

All three use the same symmetric-AUC scorer (`_best_auc_per_label`, sign-aware) on the same eval residual +
oracle labels. Trains none vs linear at fixed width, multi-seed, and asks: do the SAE-free metrics (neuron,
pca) corroborate the SAE-cov95 lift? If yes -> genuine monosemanticity, not circular. If only sae lifts ->
the gain is SAE-specific recoverability (weaker claim).
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
    p.add_argument("--modes", default="none,linear")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--aux-lambda", type=float, default=1.0)
    p.add_argument("--sae-over", type=int, default=4)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--sae-steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=400000)
    p.add_argument("--eval-tokens", type=int, default=6000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/cov95_forge_tax/legibility_crosscheck_summary.json"))
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
    seeds = [int(s) for s in args.seeds.split(",")]; modes = [m.strip() for m in args.modes.split(",")]
    print(f"corpus {len(ids)} tok; width {args.width}; oracle {Ffeat} features; modes {modes}; seeds {seeds}")

    def batch(g):
        starts = torch.randint(0, len(train_ids) - args.ctx - 1, (args.batch,), generator=g)
        x = torch.stack([torch.from_numpy(train_ids[s:s + args.ctx]) for s in starts])
        y = torch.stack([torch.from_numpy(train_ids[s + 1:s + 1 + args.ctx]) for s in starts])
        return x.to(dev), y.to(dev)

    def train(mode, seed):
        torch.manual_seed(seed)
        cfg = GPT2Config(vocab_size=tok.vocab_size, n_positions=args.ctx, n_ctx=args.ctx,
                         n_embd=args.width, n_layer=args.n_layer, n_head=args.n_head)
        model = GPT2LMHeadModel(cfg).to(dev).train()
        head = torch.nn.Linear(args.width, Ffeat).to(dev)
        params = list(model.parameters()) + (list(head.parameters()) if mode == "linear" else [])
        opt = torch.optim.AdamW(params, lr=args.lr); g = torch.Generator().manual_seed(seed)
        for _ in range(args.steps):
            x, y = batch(g)
            out = model(input_ids=x, labels=y, output_hidden_states=True)
            loss = out.loss + (args.aux_lambda * F.binary_cross_entropy_with_logits(head(out.hidden_states[-1]), Ttab_t[x]) if mode == "linear" else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        return model.eval()

    def legibility(model, seed):
        with torch.no_grad():
            ev = eval_ids[: args.eval_tokens]
            chunks = [ev[i:i + args.ctx] for i in range(0, len(ev) - 1, args.ctx) if len(ev[i:i + args.ctx]) >= 8]
            acts = []; ev_ids = []
            for ch in chunks:
                xx = torch.from_numpy(np.ascontiguousarray(ch))[None].to(dev)
                acts.append(model(input_ids=xx, output_hidden_states=True).hidden_states[-1][0].float().cpu().numpy())
                ev_ids.extend(ch)
        Xraw = np.concatenate(acts, 0).astype(np.float32); Y = Ttab[np.array(ev_ids)]
        mu, sd = Xraw.mean(0, keepdims=True), Xraw.std(0, keepdims=True) + 1e-6
        X = ((Xraw - mu) / sd).astype(np.float32)
        # three bases, same symmetric-AUC scorer:
        neuron = _per_tier(_best_auc_per_label(X, Y), tiers)                          # raw dims (no fit)
        Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)[2]
        pca = _per_tier(_best_auc_per_label((X - X.mean(0)) @ Vt.T, Y), tiers)        # rotated basis (no labels in fit)
        sae_p = _train_topk_sae(X, args.sae_over * args.width, args.k, args.sae_steps, 1e-3, seed)
        sae = _per_tier(_best_auc_per_label(_encode(X, sae_p, args.k), Y), tiers)     # fitted dictionary
        return {b: r["all"]["cov95"] for b, r in (("neuron", neuron), ("pca", pca), ("sae", sae))}

    rows = []
    for seed in seeds:
        for mode in modes:
            cov = legibility(train(mode, seed), seed)
            rows.append({"mode": mode, "seed": seed, **{f"cov95_{b}": v for b, v in cov.items()}})
            print(f"  seed {seed} {mode:>7}: sae {cov['sae']:.3f}  neuron {cov['neuron']:.3f}  pca {cov['pca']:.3f}")

    bases = ["sae", "neuron", "pca"]
    def get(mode, seed, b):
        return next(r[f"cov95_{b}"] for r in rows if r["mode"] == mode and r["seed"] == seed)
    agg = {b: {m: {"mean": float(np.mean([r[f"cov95_{b}"] for r in rows if r["mode"] == m])),
                   "std": float(np.std([r[f"cov95_{b}"] for r in rows if r["mode"] == m]))} for m in modes} for b in bases}
    lift = {b: [get("linear", s, b) - get("none", s, b) for s in seeds] for b in bases} if set(modes) >= {"none", "linear"} else {}
    out = {"experiment": "non-SAE legibility cross-check (circularity test)", "width": args.width,
           "seeds": seeds, "n_oracle_features": Ffeat, "rows": rows, "agg": agg,
           "linear_minus_none_by_basis": lift}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print("\n[verdict] cov95 none -> linear (mean over seeds), per basis:")
    for b in bases:
        d = lift.get(b, [])
        npos = sum(x > 0 for x in d)
        print(f"  {b:>7}: {agg[b]['none']['mean']:.3f} -> {agg[b]['linear']['mean']:.3f}  "
              f"(Δ {np.mean(d):+.3f} ± {np.std(d):.3f}; +{npos}/{len(d)} seeds)" if d else f"  {b}: {agg[b]}")
    if lift.get("neuron") and lift.get("pca") and lift.get("sae"):
        sae_up = np.mean(lift["sae"]) > 0.02
        nonsae_up = np.mean(lift["neuron"]) > 0.02 and np.mean(lift["pca"]) > 0.02
        print("  => " + ("CORROBORATED: the SAE-free bases (neuron, pca) also lift -> supervision yields GENUINE "
                         "monosemanticity, not just SAE-findable recoverability (not circular)." if sae_up and nonsae_up
                         else ("SAE-SPECIFIC: only the fitted-SAE cov95 lifts; the SAE-free bases do not -> the gain is "
                               "recoverability the eval-SAE prefers (circularity caveat stands)." if sae_up else
                               "no clear SAE lift to corroborate.")))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
