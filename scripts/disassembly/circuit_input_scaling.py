"""Does the induction-head population scale with INPUT size? — "same function block over more possible inputs".

A hypothesis for *why* induction distributes: not more parameters recruiting duplicates, but the **same function
applied over a larger input domain** — as the input space grows (more token-types, longer context) the circuit
recruits more heads, each covering a slice of the inputs. The clean test holds the MODEL FIXED and scales the
**input** instead, isolating input-size from parameter-count.

Two input-size axes, model fixed:
  VOCABULARY — repeated-random probes drawn from a pool of V distinct token-types (V = 8 … 1024). More token-types =
               a larger input domain the induction must cover.
  CONTEXT    — the repeat length (sequence length) 12 … 96. A longer context = more positions to address.

For each setting we read the induction-head population from one forward pass (no ablation, so confound-free): per-head
induction **attention mass** (`ResidualVM.head_mass`), then the **effective number of active heads** (Hill / inverse-
Simpson of the mass), the count above threshold, the top head's share, the active head SET (do NEW heads join, or do
the same heads just work harder?), and the total induction mass.

Prediction (same-fb-over-more-inputs): effective-N of induction heads RISES with V (and/or context) at fixed model,
and new heads join the active set. Null: the active population is flat — only per-head load / total strength moves.

Output: runs/disassembly/circuits/input_scaling_summary.json (merge-safe). Findings -> docs/FINDINGS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def hill(mass):
    """Effective number of active heads: (Σm)² / Σm² over positive induction mass (inverse-Simpson / Hill)."""
    m = np.clip(mass, 0, None); s = m.sum()
    return float((s * s) / (np.square(m).sum() + 1e-12)) if s > 0 else 0.0


def population(vm, seqs, thresh):
    mass = vm.head_mass(seqs, "induction")
    active = [(int(i) // vm.H, int(i) % vm.H) for i in np.argsort(-mass) if mass[int(i)] > thresh]
    act_mass = mass[mass > thresh]                                                 # Hill over ACTIVE heads (noise-floor-free)
    return {"effective_n": hill(act_mass), "n_active": len(active), "total_mass": float(np.clip(mass, 0, None).sum()),
            "top_share": float(mass.max() / (np.clip(mass, 0, None).sum() + 1e-12)) if mass.max() > 0 else 0.0,
            "active_heads": [f"{L}.{h}" for L, h in active[:30]]}


def scaling_one_model(vm, model_id, args):
    rng = np.random.default_rng(args.seed)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
    pids = vm.tok(prose)["input_ids"]
    if "gpt2" in model_id.lower():
        cnt = {}
        for t in pids:
            cnt[t] = cnt.get(t, 0) + 1
        pool_all = [t for t, _ in sorted(cnt.items(), key=lambda r: -r[1])]            # frequent prose tokens
    else:
        V = vm.model.config.vocab_size
        pool_all = list(range(int(0.02 * V), int(0.5 * V)))                            # a mid-vocab band

    def rep_seqs(vocab_size, rep_len):
        pool = pool_all[:vocab_size]
        return [(lambda s: s + s)([int(pool[i]) for i in rng.integers(0, len(pool), rep_len)]) for _ in range(args.probes)]

    # VOCABULARY axis (context fixed)
    vocab_curve = []
    for V in [int(v) for v in args.vocabs.split(",")]:
        seqs = rep_seqs(V, args.rep_len)
        rec = population(vm, seqs, args.thresh); rec["vocab"] = V
        vocab_curve.append(rec)
    # CONTEXT axis (vocabulary fixed)
    ctx_curve = []
    for cl in [int(c) for c in args.ctxlens.split(",")]:
        seqs = rep_seqs(args.fixed_vocab, cl)
        rec = population(vm, seqs, args.thresh); rec["rep_len"] = cl
        ctx_curve.append(rec)
    # growth of the active SET across the vocabulary axis (cumulative unique heads; are NEW heads joining?)
    seen = set(); growth = []
    for rec in vocab_curve:
        new = [h for h in rec["active_heads"] if h not in seen]; seen.update(rec["active_heads"])
        growth.append({"vocab": rec["vocab"], "n_active": rec["n_active"], "n_new_heads": len(new), "cumulative_unique": len(seen)})
    return {"model": model_id.split("/")[-1], "rope": not vm.is_gpt2, "n_heads": vm.H, "n_layers": vm.nL,
            "vocab_axis": vocab_curve, "context_axis": ctx_curve, "active_set_growth": growth}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-large,google/gemma-2-2b")
    p.add_argument("--vocabs", default="8,16,32,64,128,256,512,1024")
    p.add_argument("--ctxlens", default="12,24,48,96")
    p.add_argument("--fixed-vocab", type=int, default=128, help="vocab size held fixed for the context-length axis")
    p.add_argument("--rep-len", type=int, default=24, help="repeat length held fixed for the vocabulary axis")
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--thresh", type=float, default=0.05, help="min induction attention mass to count a head active")
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
            r = scaling_one_model(vm, mid, args)
            del vm
            if dev == "cuda":
                torch.cuda.empty_cache()
            results.append(r)
            print("  vocab → effective-N (n_active, top-share):")
            for rec in r["vocab_axis"]:
                print(f"    V={rec['vocab']:>5}: eff-N {rec['effective_n']:5.1f}  n_active {rec['n_active']:>3}  "
                      f"top-share {rec['top_share']:.0%}  total-mass {rec['total_mass']:.2f}")
            print("  context (rep_len) → effective-N:")
            for rec in r["context_axis"]:
                print(f"    L={rec['rep_len']:>4}: eff-N {rec['effective_n']:5.1f}  n_active {rec['n_active']:>3}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "input_scaling_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "does the induction-head population scale with input size (vocabulary / context), model fixed?",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'vocab_axis' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
