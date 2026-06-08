"""Do the recruited induction heads TILE the input domain? — the input→head specialization map.

The input-scaling result: a larger input vocabulary recruits more induction heads (effective-N grows, top-share
flat). That predicts each head covers a *slice* of the input domain. This tests it directly: for every induction
position, find the head that dominates its induction attention, then ask whether different heads own different
*kinds* of tokens — i.e. whether the population partitions the input space rather than all doing the same thing.

For each induction query position q (second half of a repeated-random probe), the induction key is the earlier
position whose predecessor token equals the current token; each head's "vote" is its attention there. The dominant
head = argmax over the induction population. We then test specialization on two token properties of the matched
(current) token:
  FREQUENCY — does the dominant head depend on the token's corpus-frequency rank? (η²: fraction of log-freq-rank
              variance explained by the dominant-head label, vs a label-permutation null.)
  IDENTITY  — do heads own DISJOINT token-id sets? (mean pairwise Jaccard of heads' owned-token sets; low = partition.)

High η² (above null) + low Jaccard = the heads tile the input domain by token-kind (the "same function over more
inputs" mechanism made explicit). Low η² + high Jaccard = no input partition (heads redundant on the same inputs).

Output: runs/disassembly/circuits/domain_tiling_summary.json (merge-safe). Findings -> docs/FINDINGS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def eta_squared(labels, values):
    """Fraction of variance in `values` explained by the categorical `labels` (one-way η²)."""
    values = np.asarray(values, float); grand = values.mean(); sst = float(np.square(values - grand).sum())
    if sst < 1e-12:
        return 0.0
    ssb = 0.0
    for lab in set(labels):
        v = values[np.array([x == lab for x in labels])]           # labels are (L,h) tuples — mask elementwise
        ssb += len(v) * (v.mean() - grand) ** 2
    return float(ssb / sst)


def tiling_one_model(vm, model_id, args):
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
        pool = [t for t, _ in sorted(cnt.items(), key=lambda r: -r[1])][: args.vocab]
    else:
        V = vm.model.config.vocab_size; pool = list(range(int(0.02 * V), int(0.02 * V) + args.vocab))
    freq_rank = {t: i for i, t in enumerate(pool)}                       # 0 = most frequent (by construction)
    H = vm.H
    # the induction population (heads with induction mass above threshold)
    probe0 = [(lambda s: s + s)([int(pool[i]) for i in rng.integers(0, len(pool), args.rep_len)]) for _ in range(args.id_probes)]
    mass = vm.head_mass(probe0, "induction")
    pop = [int(i) for i in np.argsort(-mass) if mass[int(i)] > args.thresh][: args.max_pop]
    if len(pop) < 2:
        return {"model": model_id.split("/")[-1], "note": "induction population <2", "pop_size": len(pop)}
    pop_set = {(i // H, i % H): k for k, i in enumerate(pop)}

    # per induction position: dominant head + the matched (current) token's frequency rank
    dom_heads = []; ranks = []; owned = {hh: set() for hh in pop_set}
    seqs = [(lambda s: s + s)([int(pool[i]) for i in rng.integers(0, len(pool), args.rep_len)]) for _ in range(args.probes)]
    for s in seqs:
        att = vm.attn(s); Lc = len(s); half = Lc // 2
        ca = np.array(s); pv = np.full(Lc, -1); pv[1:] = ca[:-1]
        for q in range(half, Lc - 1):
            cur = ca[q]
            keys = [k for k in range(1, q) if pv[k] == cur]              # induction key candidates
            if not keys:
                continue
            best = None; bestv = -1.0
            for (L, h) in pop_set:
                a = att[L][h]
                v = float(a[q, keys].sum())
                if v > bestv:
                    bestv = v; best = (L, h)
            if best is None or bestv < args.min_vote:
                continue
            dom_heads.append(best); ranks.append(freq_rank.get(int(cur), len(pool)))
            owned[best].add(int(cur))
    if len(dom_heads) < 8:
        return {"model": model_id.split("/")[-1], "note": "too few induction positions", "n_positions": len(dom_heads)}

    # FREQUENCY specialization: η²(dominant head -> log freq-rank) vs a label-permutation null
    logr = np.log1p(np.array(ranks, float))
    eta = eta_squared(dom_heads, logr)
    null = []
    for _ in range(args.perms):
        perm = list(dom_heads); rng.shuffle(perm); null.append(eta_squared(perm, logr))
    null = np.array(null); eta_z = (eta - null.mean()) / (null.std() + 1e-9)
    # IDENTITY partition: mean pairwise Jaccard of heads' owned-token sets (low = disjoint = partition)
    active = [hh for hh in pop_set if len(owned[hh]) >= args.min_owned]
    jac = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a, b = owned[active[i]], owned[active[j]]
            u = len(a | b)
            jac.append(len(a & b) / u if u else 0.0)
    mean_jac = float(np.mean(jac)) if jac else None
    # per-head frequency-band signature (mean freq-rank of the tokens each head dominates)
    head_freq = {f"{L}.{h}": float(np.mean([ranks[i] for i in range(len(dom_heads)) if dom_heads[i] == (L, h)]))
                 for (L, h) in pop_set if (L, h) in dom_heads}
    return {"model": model_id.split("/")[-1], "rope": not vm.is_gpt2, "pop_size": len(pop_set),
            "n_positions": len(dom_heads), "n_active_owners": len(active),
            "freq_eta2": eta, "freq_eta2_null_mean": float(null.mean()), "freq_eta2_z": float(eta_z),
            "identity_mean_jaccard": mean_jac, "head_freq_rank": head_freq}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--vocab", type=int, default=256, help="input vocabulary size (the domain to tile)")
    p.add_argument("--rep-len", type=int, default=32)
    p.add_argument("--probes", type=int, default=24)
    p.add_argument("--id-probes", type=int, default=12)
    p.add_argument("--thresh", type=float, default=0.03)
    p.add_argument("--max-pop", type=int, default=16)
    p.add_argument("--min-vote", type=float, default=0.05, help="min dominant-head induction attention to count a position")
    p.add_argument("--min-owned", type=int, default=3, help="min owned tokens for a head to enter the Jaccard set")
    p.add_argument("--perms", type=int, default=200)
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
        vm = None
        try:
            vm = ResidualVM(mid, device=dev)
            r = tiling_one_model(vm, mid, args)
            results.append(r)
            if "freq_eta2" in r:
                print(f"  pop {r['pop_size']} | positions {r['n_positions']} | freq η² {r['freq_eta2']:.2f} "
                      f"(null {r['freq_eta2_null_mean']:.2f}, z {r['freq_eta2_z']:.1f}) | "
                      f"identity Jaccard {r['identity_mean_jaccard']:.2f} ({r['n_active_owners']} owners)")
            else:
                print(f"  {r.get('note')}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})
        finally:
            vm = None                                              # always free the model (skips would leak the GPU)
            if dev == "cuda":
                torch.cuda.empty_cache()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "domain_tiling_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "do induction heads tile the input domain? (dominant-head vs token frequency / identity)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'freq_eta2' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
