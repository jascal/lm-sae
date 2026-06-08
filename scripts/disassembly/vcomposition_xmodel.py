"""Cross-model V-composition — the value pathway, the last Elhage edge type still GPT-2-only (completes the triad).

K-composition (induction prev-token→key chain) and Q-composition (IOI S-inhibition→name-mover chain) are now
cross-model. V-composition — head A's OV output re-read as head B's **value** (a composed-OV "virtual head", a 2-hop
copy circuit) — was GPT-2-only (`vcomposition.py`: induction L5 → L6-values, e.g. 5.9→6.7). This measures it
weight-legibly across all models (the same static basis the catalog scores K/Q composition in):

  comp_V(A→B) = ‖ W_V^B · OV_A ‖_F / (‖OV_A‖_F · ‖W_V^B‖_F)   — how much of A's residual-space OV output lands in
                                                                B's value-read subspace (mean-write removed via the
                                                                head OV/W_V weights; arch-generic incl. GQA).

For each model: A = the top induction head (the content-mover whose output is the candidate composed value), scan
downstream B; report the top V-edges and — the key control — whether induction-A's V-composition into downstream
values **exceeds a random non-induction writer's** (specificity), i.e. whether induction content is *specifically*
re-read as a value. Dynamic ΔV-out confirmation stays GPT-2-validated (`vcomposition.py`, ρ(static,ΔV-out)=+0.36).

Output: runs/disassembly/circuits/vcomposition_xmodel_summary.json (merge-safe). Findings -> docs/circuits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def head_WV(vm, L, h):
    """B's value projection W_V^h as (d, hd): residual -> head-value. Arch-generic (GQA-aware), mirrors head_OV."""
    a = vm.a; H = vm.H; hd = vm.hd; d = vm.d; kvB = h // (H // vm.nkv)
    if a["is_gpt2"]:
        return a["cattn"][L].weight.detach().float().cpu().numpy().astype(np.float64)[:, 2 * d:3 * d][:, h * hd:(h + 1) * hd]
    return a["vproj"][L].weight.detach().float().cpu().numpy().astype(np.float64)[kvB * hd:(kvB + 1) * hd, :].T


def comp_v(ov_a, wv_b):
    """static V-composition: ‖W_V^B · OV_A‖_F / (‖OV_A‖‖W_V^B‖). wv_b is (d,hd) -> use its (hd,d) action on OV_A (d,d)."""
    num = np.linalg.norm(wv_b.T @ ov_a)                                    # (hd,d)@(d,d) = (hd,d)
    return float(num / (np.linalg.norm(ov_a) * np.linalg.norm(wv_b) + 1e-12))


def vcomp_one_model(vm, model_id, args):
    rng = np.random.default_rng(args.seed)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    pids = vm.tok(prose)["input_ids"]
    if "gpt2" in model_id.lower():
        cnt = {}
        for t in pids:
            cnt[t] = cnt.get(t, 0) + 1
        vocab = [t for t, _ in sorted(cnt.items(), key=lambda r: -r[1])[:400]]
        rep = lambda Ln: [int(vocab[i]) for i in rng.integers(0, len(vocab), Ln)]    # noqa: E731
    else:
        V = vm.model.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
        rep = lambda Ln: [int(x) for x in rng.integers(lo, hi, Ln)]                   # noqa: E731
    probes = [(lambda s: s + s)(rep(22)) for _ in range(args.id_probes)]
    ind_heads, _ = vm.find_heads(probes, "induction", top=args.top_writers)
    H = vm.H; nL = vm.nL
    ind_set = set(ind_heads)
    # downstream readers B: all heads strictly below... above each writer's layer (B reads A as a value -> L_B > L_A)
    results = {}
    for A in ind_heads:
        LA = A[0]
        ov_a = vm.head_OV(*A)
        edges = []
        for LB in range(LA + 1, nL):
            for hB in range(H):
                edges.append(((LB, hB), comp_v(ov_a, head_WV(vm, LB, hB))))
        edges.sort(key=lambda e: -e[1])
        # specificity null: random NON-induction writers A' at the same layer -> same downstream B set
        null = []
        for _ in range(args.null):
            hAp = int(rng.integers(0, H))
            if (LA, hAp) in ind_set:
                continue
            ovp = vm.head_OV(LA, hAp)
            null.append(np.mean([comp_v(ovp, head_WV(vm, LB, hB)) for (LB, hB), _ in edges[: args.topk]]))
        top_mean = float(np.mean([s for _, s in edges[: args.topk]]))
        null_mean = float(np.mean(null)) if null else 0.0
        results[f"{A[0]}.{A[1]}"] = {
            "writer": f"{A[0]}.{A[1]}", "top_edges": [{"reader": f"{b[0]}.{b[1]}", "comp_v": s} for b, s in edges[: args.topk]],
            "top_mean_comp_v": top_mean, "random_writer_null": null_mean, "specificity": top_mean - null_mean}
    best = max(results.values(), key=lambda r: r["specificity"]) if results else None
    return {"model": model_id.split("/")[-1], "rope": not vm.is_gpt2, "n_layers": nL, "n_heads": H,
            "induction_writers": [f"{L}.{h}" for L, h in ind_heads], "by_writer": results,
            "best_writer": best["writer"] if best else None, "best_specificity": best["specificity"] if best else None,
            "best_top_edges": best["top_edges"][:5] if best else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--id-probes", type=int, default=12)
    p.add_argument("--top-writers", type=int, default=3, help="top induction heads to test as composed-OV writers")
    p.add_argument("--topk", type=int, default=8, help="top downstream V-edges per writer")
    p.add_argument("--null", type=int, default=8, help="random non-induction writers for the specificity null")
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
            r = vcomp_one_model(vm, mid, args)
            results.append(r)
            be = ", ".join(f"{e['reader']}({e['comp_v']:.2f})" for e in (r["best_top_edges"] or [])[:4])
            print(f"  induction writers {r['induction_writers']} | best composed-OV writer {r['best_writer']} → "
                  f"values [{be}] | specificity vs random {r['best_specificity']:+.3f}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})
        finally:
            vm = None
            if dev == "cuda":
                torch.cuda.empty_cache()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "vcomposition_xmodel_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "cross-model V-composition (composed-OV virtual heads: induction writer -> downstream value)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'by_writer' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
