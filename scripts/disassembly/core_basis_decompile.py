"""What IS the entangled core's shared subspace? — decompile the STRUCTURE of the cross-layer write-basis.

`core_rank.py` established that the layers write their residual updates into a *shared* moderate-rank subspace (the
union basis Ug, ~⅓–⅖·d, ~2× random pairwise overlap). That answers "how big" the core is. This script asks the
deeper question — **is that shared basis interpretable?** Three decompilation tests on the union basis Ug (top-K):

  (A) TOKEN / LOGIT alignment — does the shared core live in the model's high-logit-variance ("vocabulary") subspace?
      Project Ug onto the top-K right singular vectors of the unembedding W_U; compare to the chance fraction K/d.
      A core that is mostly *readout-aligned* points at tokens; a core orthogonal to W_U's top is internal compute.
  (B) OPERATOR alignment — is the shared core built from the CATALOG operators' OV-write subspaces? Behaviourally
      locate induction / prev-token / duplicate / sink heads (find_heads), take each head's OV write subspace, and
      measure the fraction captured by Ug vs the chance fraction K/d and vs random heads. If named ≈ random the core
      is the *aggregate* of all writers, not a few named ops; if named ≫ random the catalog ops preferentially live
      in the core.
  (C) LOGIT-LENS readability — decode the top shared directions through W_U (logit lens) and report the top tokens +
      a peakedness z-score vs random directions. Sharp, coherent token lists ⇒ the basis is human-readable.

No retraining; pure analysis on the frozen model. Reuses core_rank's covariance/PCA/union-basis construction so the
basis under the microscope is *exactly* the one core_rank measured. Output: runs/disassembly/core_basis_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_rank import participation_ratio  # noqa: E402


def captured_fraction(Ug, B):
    """Fraction of subspace B (d,kb orthonormal) captured by the shared core Ug (d,K orthonormal): ‖Ugᵀ B‖²_F / kb."""
    return float(np.square(Ug.T @ B).sum() / B.shape[1])


def orth(M):
    """Orthonormal basis for the column space of M (drops null directions)."""
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > 1e-6 * (s[0] if len(s) else 1.0)]


def run_model(mid, args):
    import urllib.request

    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8]
    fit = chunks[: args.fit]

    # ---- per-layer residual update Δ_L covariance → PCA bases (identical to core_rank) ----
    cov = {L: np.zeros((d, d)) for L in range(nL)}; nrows = 0; cap = {}
    hks = [vm.layers[L].register_forward_hook(
        (lambda L: lambda m, i, o: cap.__setitem__(L, ((o[0] if isinstance(o, tuple) else o) - i[0]).detach()))(L))
        for L in range(nL)]
    with t.no_grad():
        for c in fit:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                u = cap[L][0].float().cpu().numpy()
                cov[L] += u.T @ u
            nrows += cap[0].shape[1]
    for h in hks:
        h.remove()
    bases = {}
    for L in range(nL):
        w, V = np.linalg.eigh(cov[L]); order = np.argsort(-w)
        bases[L] = V[:, order].astype(np.float64)                       # columns = components, desc

    # ---- the SHARED union basis Ug (concat per-layer top-rs → SVD), exactly as core_rank ----
    rs = min(args.share_rank, d // 2)
    stacked = np.concatenate([bases[L][:, :rs] for L in range(nL)], axis=1)       # (d, nL*rs)
    Uall, sv, _ = np.linalg.svd(stacked, full_matrices=False)
    K = max(1, min(int(round(participation_ratio(sv ** 2))), d))                  # core dim = union effective rank
    Ug = Uall[:, :K]                                                              # (d, K) the shared write-basis
    rng = np.random.default_rng(0)
    Rrand = orth(rng.standard_normal((d, K)))                                     # random K-subspace control

    # ---- W_U: unembedding (V, d) ----
    WU = vm.model.get_output_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)

    # (A) token/logit alignment: top-K right singular vectors of W_U = principal logit directions
    _, _, VtU = np.linalg.svd(WU, full_matrices=False)
    Plogit = VtU[:K].T                                                           # (d, K) dominant logit subspace
    tokenA = {"core_in_logit_subspace": captured_fraction(Plogit, Ug),
              "random_in_logit_subspace": captured_fraction(Plogit, Rrand),
              "chance": K / d, "K": K, "d": d}

    # (B) operator alignment: locate named ops, capture OV-write subspaces, fraction inside the core
    seqs = [(lambda s: s + s)([int(v) for v in rng.integers(0, min(2000, len(tok)), 22)]) for _ in range(args.probe)]
    ops = ["induction", "prevtok", "duplicate"] + (["sink"] if vm.is_gpt2 else [])
    opA = {}
    all_heads = [(L, h) for L in range(nL) for h in range(vm.H)]
    rand_heads = [all_heads[int(i)] for i in rng.integers(0, len(all_heads), min(8, len(all_heads)))]

    def head_write_basis(L, h):
        ov = vm.head_OV(L, h)                                                    # (d,d), rank ≤ hd
        _, s, Vt = np.linalg.svd(ov, full_matrices=False)
        return Vt[s > 1e-6 * s[0]].T                                             # row space = residual write subspace

    def op_capture(heads):
        fr = [captured_fraction(Ug, head_write_basis(L, h)) for (L, h) in heads]
        return float(np.mean(fr)) if fr else 0.0
    for op in ops:
        heads, mass = vm.find_heads(seqs, op, top=4)
        opA[op] = {"heads": [f"{L}.{h}" for (L, h) in heads], "top_mass": float(mass.max()),
                   "core_capture": op_capture(heads)}
    opA["_random_heads"] = {"heads": [f"{L}.{h}" for (L, h) in rand_heads], "core_capture": op_capture(rand_heads)}
    opA["_chance"] = K / d

    # (C) logit-lens readability of the top shared directions
    def lens(vec, topn=8):
        lg = WU @ vec
        z = float((lg.max() - lg.mean()) / (lg.std() + 1e-9))                    # peakedness toward one token
        toks = [tok.convert_ids_to_tokens(int(i)).replace("Ġ", "_").replace("Ċ", "\\n")
                for i in np.argsort(-lg)[:topn]]
        return z, toks
    lens_rows = []
    for k in range(min(args.lens_dirs, K)):
        zp, tp = lens(Ug[:, k]); zn, tn = lens(-Ug[:, k])                        # PCA sign-ambiguous → both poles
        side = (zp, tp) if zp >= zn else (zn, tn)
        lens_rows.append({"dir": k, "peak_z": side[0], "top_tokens": side[1]})
    core_z = float(np.mean([r["peak_z"] for r in lens_rows]))
    rand_z = float(np.mean([lens(Rrand[:, k])[0] for k in range(min(args.lens_dirs, K))]))

    return {"model": mid.split("/")[-1], "n_layers": nL, "d_model": d, "core_dim_K": K, "share_rank": rs,
            "token_logit_alignment": tokenA, "operator_alignment": opA,
            "logit_lens": {"core_mean_peak_z": core_z, "random_mean_peak_z": rand_z, "directions": lens_rows}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--share-rank", type=int, default=64)
    p.add_argument("--probe", type=int, default=16, help="repeated-vocab seqs for behavioural head-finding")
    p.add_argument("--lens-dirs", type=int, default=16, help="top shared directions to logit-lens")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
    args = p.parse_args(argv)

    import torch
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = run_model(mid, args)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(r)
            ta = r["token_logit_alignment"]
            print(f"  d{r['d_model']} {r['n_layers']}L | core dim K={r['core_dim_K']}")
            print(f"  (A) token/logit: core {ta['core_in_logit_subspace']:.2f} of its mass in W_U's top-K logit "
                  f"subspace vs random-subspace {ta['random_in_logit_subspace']:.2f} (chance {ta['chance']:.2f})")
            oa = r["operator_alignment"]
            print(f"  (B) operators (fraction of OV-write subspace inside the core; chance {oa['_chance']:.2f}):")
            for op in [k for k in oa if not k.startswith("_")]:
                print(f"        {op:10s} [{','.join(oa[op]['heads'])}] mass {oa[op]['top_mass']:.2f} "
                      f"→ core-capture {oa[op]['core_capture']:.2f}")
            print(f"        random-heads → core-capture {oa['_random_heads']['core_capture']:.2f}")
            ll = r["logit_lens"]
            print(f"  (C) logit-lens peak-z: core {ll['core_mean_peak_z']:.1f} vs random {ll['random_mean_peak_z']:.1f}")
            for row in ll["directions"][:6]:
                print(f"        dir{row['dir']:2d} z{row['peak_z']:4.1f}: {' '.join(row['top_tokens'])}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "core_basis_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "decompile the structure of the entangled core's shared cross-layer write-basis", "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'core_dim_K' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
