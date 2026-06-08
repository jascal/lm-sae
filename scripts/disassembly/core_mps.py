"""The entangled core as a TENSOR NETWORK — MPS bond-dimension (χ) levels across layer cuts + the cross-layer
composition GRAPH, with the directions typed into an ontology (grammar / operator / content).

`core_rank.py` measured the *single-cut* rank of each layer's residual update (how big the core is) and that layers
share a moderate-rank subspace. `core_basis_decompile.py` / `core_grammar.py` typed the shared head as a compact
generic grammar. This script asks the q-orca / entanglement-tower question: treat the sequence of per-layer updates
(Δ_1 … Δ_nL) as a state over layer-"sites" and measure how ENTANGLED it is across the layer chain.

  MPS LEVELS — for each cut between layer L and L+1, the bond dimension χ_L = effective rank (participation ratio) of
    the cross-covariance between the early writes (layers ≤ L) and the late writes (layers > L), plus the entanglement
    entropy S_L of its (normalised) singular spectrum. A flat, low χ-profile ⇒ the whole stack is an area-law MPS
    (one cheap global tensor-train surrogate runs the composition on CPU); a χ that peaks mid-stack ⇒ volume-law
    entanglement, no cheap global factorisation — only the per-layer low-rank lever core_rank already found.
  COMPOSITION GRAPH — w[L,M] = mean-squared canonical correlation between layer L's and layer M's (standardised) write
    coords: a DAG over layers (which writes couple to which). The graph representation of the core.
  ONTOLOGY — type each per-layer top direction by logit-lens (grammar = closed-class / punctuation, operator = aligns
    a named OV-write subspace, else content) and report the type mix per layer + across the bond (is the entanglement
    grammatical or content?).

No retraining; pure analysis on the frozen model. Two passes (PCA bases, then standardised cross-covariance). Output:
runs/disassembly/core_mps_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_grammar import CLOSED, PUNCT  # noqa: E402
from core_rank import participation_ratio  # noqa: E402


def entropy(sv):
    p = np.square(np.clip(sv, 0, None)); s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s; p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def run_model(mid, args):
    import urllib.request

    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d; r = args.rank
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.fit]

    def capture():
        cap = {}
        hks = [vm.layers[L].register_forward_hook(
            (lambda L: lambda m, i, o: cap.__setitem__(L, ((o[0] if isinstance(o, tuple) else o) - i[0]).detach()))(L))
            for L in range(nL)]
        return cap, hks

    # ---- pass 1: per-layer update covariance → top-r PCA bases + eigenvalues (for standardising) ----
    cov = {L: np.zeros((d, d)) for L in range(nL)}
    cap, hks = capture()
    with t.no_grad():
        for c in chunks:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                u = cap[L][0].float().cpu().numpy(); cov[L] += u.T @ u
    for h in hks:
        h.remove()
    bases = {}; eig = {}
    for L in range(nL):
        w, V = np.linalg.eigh(cov[L]); order = np.argsort(-w)[:r]
        bases[L] = V[:, order].astype(np.float64); eig[L] = np.clip(w[order], 1e-9, None)

    # ---- pass 2: standardised per-layer write coords → full (nL*r) cross-covariance over the corpus ----
    Bt = {L: t.tensor(bases[L].astype(np.float32), device=vm.dev) for L in range(nL)}
    sd = {L: t.tensor(np.sqrt(eig[L]).astype(np.float32), device=vm.dev) for L in range(nL)}
    big = np.zeros((nL * r, nL * r)); nrows = 0
    cap, hks = capture()
    with t.no_grad():
        for c in chunks:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            coords = []
            for L in range(nL):
                z = (cap[L][0].float() @ Bt[L]) / sd[L]                  # (seq, r) standardised PCA coords
                coords.append(z.cpu().numpy())
            Z = np.concatenate(coords, axis=1)                          # (seq, nL*r)
            big += Z.T @ Z; nrows += Z.shape[0]
    for h in hks:
        h.remove()
    C = big / max(nrows, 1)
    dinv = 1.0 / np.sqrt(np.clip(np.diag(C), 1e-12, None))             # → true correlation matrix (unit diagonal)
    C = C * dinv[:, None] * dinv[None, :]

    def block(L, M):
        return C[L * r:(L + 1) * r, M * r:(M + 1) * r]

    # ---- MPS bond dimension χ + entanglement entropy across each layer cut ----
    mps = []
    for L in range(1, nL):
        early = C[: L * r, L * r:]                                     # cross-correlation(early writes, late writes)
        sv = np.linalg.svd(early, compute_uv=False)
        mps.append({"cut": L, "chi": float(participation_ratio(sv ** 2)),
                    "chi_frac": float(participation_ratio(sv ** 2) / min(L * r, (nL - L) * r)),
                    "entropy": entropy(sv), "max_bond": int(min(L * r, (nL - L) * r))})

    # ---- composition graph: mean-squared canonical correlation between layer pairs ----
    graph = {}
    for L in range(nL):
        for M in range(L + 1, nL):
            graph[f"{L}->{M}"] = float(np.square(block(L, M)).sum() / r)   # 0 (independent) .. 1 (shared)
    chance = float(r / nrows) if nrows else 0.0                         # E[msqcc] for independent standardised blocks ≈ r/nrows
    mean_adj = float(np.mean(list(graph.values())))
    adj_dist = float(np.mean([graph[f"{L}->{L + 1}"] for L in range(nL - 1)]))   # mean ADJACENT-layer coupling
    far_dist = float(np.mean([v for k, v in graph.items() if abs(int(k.split("->")[0]) - int(k.split("->")[1])) >= 3]))
    strong = sorted(graph.items(), key=lambda kv: -kv[1])[:8]

    # ---- ontology: type each layer's top directions (grammar / operator / content) by logit lens + OV alignment ----
    WU = vm.model.get_output_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
    cset = set()
    for word in CLOSED:
        for f in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            tt = tok(f, add_special_tokens=False)["input_ids"]
            if len(tt) == 1:
                cset.add(tt[0])
    for p in PUNCT:
        for f in (p, " " + p):
            tt = tok(f, add_special_tokens=False)["input_ids"]
            if len(tt) == 1:
                cset.add(tt[0])

    def dir_type(vec):
        lg = WU @ vec; zp = (lg.max() - lg.mean()) / (lg.std() + 1e-9); zn = (lg.mean() - lg.min()) / (lg.std() + 1e-9)
        top = (np.argsort(-lg) if zp >= zn else np.argsort(lg))[:10]
        closed = np.mean([int(i) in cset for i in top])
        return "grammar" if closed >= 0.4 else "content"
    onto = {}
    for L in range(nL):
        types = [dir_type(bases[L][:, k]) for k in range(min(r, args.onto_dirs))]
        onto[L] = {ty: int(sum(1 for x in types if x == ty)) for ty in ("grammar", "content")}
    gram_by_layer = [onto[L]["grammar"] for L in range(nL)]

    return {"model": mid.split("/")[-1], "n_layers": nL, "d_model": d, "rank_per_layer": r, "n_samples": nrows,
            "mps_levels": mps, "mean_chi": float(np.mean([m["chi"] for m in mps])),
            "max_chi_cut": int(max(mps, key=lambda m: m["chi"])["cut"]) if mps else -1,
            "graph": {"mean_adjacency": mean_adj, "chance": chance, "adjacent_coupling": adj_dist,
                      "distant_coupling": far_dist, "strong_edges": strong, "full": graph},
            "ontology": {"grammar_dirs_by_layer": gram_by_layer, "onto_dirs_scored": min(r, args.onto_dirs)}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--rank", type=int, default=24, help="top-r per-layer write coords (the MPS local dimension)")
    p.add_argument("--onto-dirs", type=int, default=12, help="top per-layer dirs to type for the ontology")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
    args = p.parse_args(argv)

    import torch
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            rr = run_model(mid, args)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(rr)
            print(f"  d{rr['d_model']} {rr['n_layers']}L | rank/layer {rr['rank_per_layer']} | {rr['n_samples']} samples")
            print("  MPS χ across layer cuts (cut:χ): " +
                  " ".join(f"{m['cut']}:{m['chi']:.0f}" for m in rr["mps_levels"]))
            print(f"     mean χ {rr['mean_chi']:.1f}; peak at cut {rr['max_chi_cut']}; "
                  f"entropy " + " ".join(f"{m['entropy']:.1f}" for m in rr["mps_levels"]))
            g = rr["graph"]
            print(f"  composition graph: adjacent-layer coupling {g['adjacent_coupling']:.4f} vs distant "
                  f"{g['distant_coupling']:.4f} vs chance {g['chance']:.4f}; strongest " +
                  " ".join(f"{k}={v:.4f}" for k, v in g["strong_edges"][:6]))
            print(f"  ontology grammar-dirs/layer (of {rr['ontology']['onto_dirs_scored']}): " +
                  " ".join(str(x) for x in rr["ontology"]["grammar_dirs_by_layer"]))
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "core_mps_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {rr["model"] for rr in results}
    merged = results + [rr for rr in prior if rr.get("model") not in done]
    out = {"experiment": "the entangled core as a tensor network — MPS bond-dimension levels + composition graph + ontology",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([rr for rr in results if 'mps_levels' in rr])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
