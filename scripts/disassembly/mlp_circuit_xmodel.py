"""MLP nodes in the circuit DAG, cross-model — the COMPUTE class as first-class circuit nodes (closes a gap).

The cross-model circuit catalog is attention-only (head→head edges). `mlp_ops.py` added head↔MLP composition edges
but GPT-2-only; `induction_substrate.py` showed induction leans on the MLP substrate (MLP0 the detokenizer) but only
split MLP0-vs-rest. This makes MLPs first-class circuit **nodes** across all models, on the ResidualVM:

  IMPORTANCE — ablate each MLP layer alone -> Δ induction-NLL + Δ generic-NLL (the per-layer COMPUTE profile: which
               MLPs the induction circuit / general LM route through, not just MLP0); + the all-MLP-ablated necessity.
  READ EDGE (head→MLP) — weight-legible `‖W_in^L · OV_A‖ / (‖OV_A‖‖W_in^L‖)`: how much MLP L reads the top induction
               head A's output (does the load-bearing MLP read the induction head?).
  WRITE EDGE (MLP→head, key) — `‖W_K^B · W_out^L‖ / (‖W_K^B‖‖W_out^L‖)`: how much MLP L writes into the induction
               reader B's KEY addressing (does an MLP feed induction's match channel?).

So each model gets the induction circuit's MLP nodes + the head↔MLP edges that wire them in — the COMPUTE half of the
DAG, cross-model. Arch-generic (GPT-2 fused c_fc/c_proj + RoPE gate/down_proj, GQA-aware keys).

Output: runs/disassembly/circuits/mlp_circuit_xmodel_summary.json (merge-safe). Findings -> docs/circuits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def mlp_weights(vm, L):
    """(W_in (inter,d) residual->hidden, W_out (d,inter) hidden->residual) for MLP L, arch-generic."""
    m = vm.mlps[L]
    if vm.is_gpt2:
        win = m.c_fc.weight.detach().float().cpu().numpy().astype(np.float64).T      # c_fc (d,4d) -> (4d,d)
        wout = m.c_proj.weight.detach().float().cpu().numpy().astype(np.float64).T   # c_proj (4d,d) -> (d,4d)
    else:
        win = m.gate_proj.weight.detach().float().cpu().numpy().astype(np.float64)   # (inter,d)
        wout = m.down_proj.weight.detach().float().cpu().numpy().astype(np.float64)  # (d,inter)
    return win, wout


def head_WK(vm, L, h):
    """B's key projection W_K^h as (hd, d): residual -> head-key. Arch-generic (GQA-aware)."""
    a = vm.a; H = vm.H; hd = vm.hd; d = vm.d; kvB = h // (H // vm.nkv)
    if a["is_gpt2"]:
        return a["cattn"][L].weight.detach().float().cpu().numpy().astype(np.float64)[:, d:2 * d][:, h * hd:(h + 1) * hd].T
    return a["kproj"][L].weight.detach().float().cpu().numpy().astype(np.float64)[kvB * hd:(kvB + 1) * hd, :]


def fro_cos(X, Y):
    return float(np.linalg.norm(X) / (np.linalg.norm(Y) + 1e-12))


def mlp_circuit_one_model(vm, model_id, args):
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
        rep = lambda Ln: [int(vocab[i]) for i in rng.integers(0, len(vocab), Ln)]    # noqa: E731
    else:
        V = vm.model.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
        rep = lambda Ln: [int(x) for x in rng.integers(lo, hi, Ln)]                   # noqa: E731
    rep_seqs = [(lambda s: s + s)(rep(22)) for _ in range(args.probes)]
    nL = vm.nL

    def ind_target(s):
        Lh = len(s) // 2
        return [(p, s[p + 1]) for p in range(Lh, 2 * Lh - 1)]

    def gen_target(s):
        return [(i, s[i + 1]) for i in range(len(s) - 1)]

    base_ind = vm.nll(rep_seqs, ind_target); base_gen = vm.nll(chunks, gen_target)
    # IMPORTANCE: ablate each MLP layer alone
    prof = []
    for L in range(nL):
        with vm.ablate_mlps([L]):
            di = vm.nll(rep_seqs, ind_target) - base_ind
            dg = vm.nll(chunks, gen_target) - base_gen
        prof.append({"layer": L, "d_induction": di, "d_generic": dg})
    with vm.ablate_mlps(list(range(nL))):
        all_ind = vm.nll(rep_seqs, ind_target) - base_ind
        all_gen = vm.nll(chunks, gen_target) - base_gen
    top_ind = sorted(prof, key=lambda r: -r["d_induction"])[:5]
    top_gen = sorted(prof, key=lambda r: -r["d_generic"])[:5]

    # COMPOSITION EDGES on the top induction head
    ind_heads, _ = vm.find_heads(rep_seqs[: args.id_probes], "induction", top=2)
    A = ind_heads[0]; ov_a = vm.head_OV(*A); wk_b = head_WK(vm, *A)
    read_edges = []; write_edges = []
    for L in range(nL):
        win, wout = mlp_weights(vm, L)
        read_edges.append({"mlp": L, "read": fro_cos(win @ ov_a, win) / (np.linalg.norm(ov_a) + 1e-12)})   # ‖W_in·OV_A‖/(‖OV_A‖‖W_in‖)
        write_edges.append({"mlp": L, "write": fro_cos(wk_b @ wout, wk_b) / (np.linalg.norm(wout) + 1e-12)})  # ‖W_K·W_out‖/(…)
    read_edges.sort(key=lambda r: -r["read"]); write_edges.sort(key=lambda r: -r["write"])

    return {"model": model_id.split("/")[-1], "rope": not vm.is_gpt2, "n_layers": nL,
            "base_induction_nll": base_ind, "base_generic_nll": base_gen,
            "all_mlp_ablated_d_induction": all_ind, "all_mlp_ablated_d_generic": all_gen,
            "induction_head": f"{A[0]}.{A[1]}",
            "top_mlp_induction": [{"layer": r["layer"], "d_induction": r["d_induction"]} for r in top_ind],
            "top_mlp_generic": [{"layer": r["layer"], "d_generic": r["d_generic"]} for r in top_gen],
            "detokenizer_is_mlp0": bool(top_ind and top_ind[0]["layer"] == 0),
            "mlp0_d_induction": prof[0]["d_induction"], "mlp0_d_generic": prof[0]["d_generic"],
            "top_read_edges": read_edges[:4], "top_write_edges": write_edges[:4], "profile": prof}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=12)
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--id-probes", type=int, default=12)
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
            r = mlp_circuit_one_model(vm, mid, args)
            results.append(r)
            ti = ", ".join(f"L{e['layer']}({e['d_induction']:+.2f})" for e in r["top_mlp_induction"][:3])
            rd = ", ".join(f"L{e['mlp']}" for e in r["top_read_edges"][:3])
            print(f"  induction head {r['induction_head']} | all-MLP-ablated Δind {r['all_mlp_ablated_d_induction']:+.2f} | "
                  f"top-MLP(ind) {ti} | detok=MLP0 {r['detokenizer_is_mlp0']} | head→MLP read {rd}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})
        finally:
            vm = None
            if dev == "cuda":
                torch.cuda.empty_cache()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "mlp_circuit_xmodel_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "MLP nodes in the circuit DAG cross-model (per-layer COMPUTE importance + head↔MLP edges)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'profile' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
