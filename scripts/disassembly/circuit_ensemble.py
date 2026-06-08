"""Is the distributing induction circuit a WEIGHTED ENSEMBLE of duplicates, or a distributed DECOMPOSITION?

The dossier shows the induction circuit's necessity + sufficiency decay across the GPT-2 ladder — it "distributes."
But "distributed" conflates ≥3 mechanisms that this script separates, over the FULL induction-head population
(not the dossier's top-4), built on the ResidualVM:

  effective-N  — the inverse-Simpson (Hill) number of the per-head induction-logit contributions: how many heads
                 *share the load* (1 = one dominant head; N = N equal contributors). Should GROW with scale either way.
  participation ratio (PR) — the dimensional effective rank of the [position x head] single-head-contribution
                 matrix's covariance. LOW = the heads' per-position effects are COLLINEAR (they help the *same*
                 tokens -> a weighted ensemble of near-duplicates); HIGH = they help *different* tokens (a
                 decomposition / committee of specialists).
  functional cosine — mean cosine between heads' position-effect vectors (do they help the SAME tokens?).
  structural OV cosine — mean cosine between heads' OV matrices (do they do the SAME operation, weight-wise?).
  layer span — how spread the population is across depth (a replicated ensemble clusters in one band).

Two axes disambiguate the THREE hypotheses (functional overlap is high in all "distributed" cases):
  - weighted ENSEMBLE of duplicates : functional-cos HIGH, **structural OV-cos HIGH**, layers TIGHT.
  - heterogeneous PARALLEL circuits  : functional-cos HIGH, **structural OV-cos LOW**, layers SPREAD (new circuits
                                       with different wiring but overlapping function "woven in" with scale).
  - distributed DECOMPOSITION        : functional-cos LOW (heads tile *different* tokens), PR HIGH.

effective-N (Hill number of contributions) should grow with scale in all "distributed" cases — it counts members,
not their kind. Per-head contribution c_h(pos) = logp_full(pos) - logp_(ablate h alone)(pos) at each induction
target position (positive = the head helps that prediction).

Output: runs/disassembly/circuits/ensemble_summary.json (merge-safe). Findings -> docs/FINDINGS.md (existing page).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def per_position_logp(vm, seqs, target, ablate=()):
    """Vector of logp at each (pos,target) across seqs, with `ablate` heads mean-ablated. Composes via ResidualVM."""
    torch = vm.torch; out = []
    with vm.ablate_heads(set(ablate)):
        with torch.no_grad():
            for s in seqs:
                lp = torch.log_softmax(vm.logits(s).float(), -1)
                for pos, t in target(s):
                    out.append(float(lp[pos, t]))
    return np.array(out)


def ensemble_one_model(vm, model_id, args):
    rng = np.random.default_rng(args.seed)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:160000]
    pids = vm.tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]
    vm.fit_means(chunks)
    if "gpt2" in model_id.lower():
        cnt = {}
        for t in pids:
            cnt[t] = cnt.get(t, 0) + 1
        vocab = [t for t, _ in sorted(cnt.items(), key=lambda r: -r[1])[:400]]
        rep = lambda L: [int(vocab[i]) for i in rng.integers(0, len(vocab), L)]   # noqa: E731
    else:
        V = vm.model.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
        rep = lambda L: [int(x) for x in rng.integers(lo, hi, L)]                 # noqa: E731
    rep_seqs = [(lambda s: s + s)(rep(args.rep_len)) for _ in range(args.probes)]

    def ind_target(s):
        L = len(s) // 2
        return [(p, s[p + 1]) for p in range(L, 2 * L - 1)]

    # ---- the FULL induction-head population (all heads with induction mass above threshold, capped) ----
    mass = vm.head_mass(rep_seqs[: args.id_probes], "induction")
    order = np.argsort(-mass)
    pop = [int(i) for i in order if mass[int(i)] > args.mass_thresh][: args.max_pop]
    if len(pop) < 2:
        pop = [int(i) for i in order[:2]]
    pop_hh = [(i // vm.H, i % vm.H) for i in pop]

    lp_full = per_position_logp(vm, rep_seqs, ind_target)                 # [P]
    P = len(lp_full)
    # single-head ablation contributions: c_h(pos) = logp_full - logp_(ablate h) ; positive = head helps
    C = np.zeros((P, len(pop_hh)))
    for j, hh in enumerate(pop_hh):
        C[:, j] = lp_full - per_position_logp(vm, rep_seqs, ind_target, ablate=[hh])
    mean_c = C.mean(0)                                                     # per-head mean contribution
    pos_c = np.clip(mean_c, 0, None)                                       # the helping mass

    # effective-N (inverse-Simpson / Hill number of the contribution magnitudes)
    eff_n = float((pos_c.sum() ** 2) / (np.square(pos_c).sum() + 1e-12)) if pos_c.sum() > 0 else 0.0
    # dimensional participation ratio of the [P x N] contribution covariance (across heads)
    Cc = C - C.mean(0, keepdims=True)
    cov = Cc.T @ Cc / max(P - 1, 1)                                        # [N x N]
    ev = np.linalg.eigvalsh(cov); ev = np.clip(ev, 0, None)
    pr = float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)) if ev.sum() > 0 else 0.0
    pr_frac = pr / max(len(pop_hh), 1)
    # FUNCTIONAL similarity: mean pairwise cosine between heads' position-effect vectors (do they help the same tokens?)
    iu = np.triu_indices(len(pop_hh), 1)
    U = C / (np.linalg.norm(C, axis=0) + 1e-12)
    func_cos = float((U.T @ U)[iu].mean()) if len(iu[0]) else 0.0

    # STRUCTURAL similarity: mean pairwise cosine of the heads' OV matrices (do they do the same operation?) +
    # layer spread (a replicated ensemble clusters in one layer-band; woven-in heterogeneous circuits spread out).
    ov = np.stack([vm.head_OV(L, h).astype(np.float32).ravel() for L, h in pop_hh])   # [N, d*d]
    ov /= (np.linalg.norm(ov, axis=1, keepdims=True) + 1e-12)
    ov_cos = float((ov @ ov.T)[iu].mean()) if len(iu[0]) else 0.0
    layers = np.array([L for L, _ in pop_hh])
    layer_span = float((layers.max() - layers.min()) / max(vm.nL - 1, 1))
    layer_std = float(layers.std() / max(vm.nL - 1, 1))
    n_layers_used = int(len(set(layers.tolist())))

    return {
        "model": model_id.split("/")[-1], "rope": not vm.is_gpt2, "n_layers": vm.nL, "n_heads": vm.H,
        "population": [f"{L}.{h}" for L, h in pop_hh], "pop_size": len(pop_hh),
        "effective_n": eff_n, "participation_ratio": pr, "participation_ratio_frac": pr_frac,
        "functional_cosine": func_cos, "structural_ov_cosine": ov_cos,
        "layer_span": layer_span, "layer_std": layer_std, "n_layers_used": n_layers_used, "n_positions": P,
        "top_share": float(pos_c.max() / (pos_c.sum() + 1e-12)) if pos_c.sum() > 0 else 0.0,
        "mean_contrib": {f"{L}.{h}": float(mean_c[j]) for j, (L, h) in enumerate(pop_hh)},
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,gpt2-xl,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--probes", type=int, default=24)
    p.add_argument("--id-probes", type=int, default=16)
    p.add_argument("--rep-len", type=int, default=22)
    p.add_argument("--mass-thresh", type=float, default=0.03, help="min induction attention mass to enter the population")
    p.add_argument("--max-pop", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/circuits"))
    args = p.parse_args(argv)

    import torch
    from residual_vm import ResidualVM
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            vm = ResidualVM(mid, device=dev)
            r = ensemble_one_model(vm, mid, args)
            del vm
            if dev == "cuda":
                torch.cuda.empty_cache()
            results.append(r)
            print(f"  pop {r['pop_size']} heads (layers {r['n_layers_used']}, span {r['layer_span']:.0%}) | "
                  f"effective-N {r['effective_n']:.1f} | PR {r['participation_ratio']:.1f} ({r['participation_ratio_frac']:.0%}) | "
                  f"func-cos {r['functional_cosine']:+.2f} | OV-struct-cos {r['structural_ov_cosine']:+.2f} | top-share {r['top_share']:.0%}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "ensemble_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "induction circuit: weighted-ensemble vs distributed-decomposition (effective-N / PR / cosine)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'effective_n' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
