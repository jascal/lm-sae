"""Does post-training threaten RoPE's shared early-writer induction front-end — or is that early node protected?

The writer-cluster result: RoPE models hang most of their induction population off ONE shared early
predecessor-writer (Llama L0 head 0.2 feeds 88% of induction readers; Gemma 0.0; Qwen ~3.2). Two competing
predictions for post-training (base -> instruct):

  FRAGILITY — that shared early writer is a single point of failure; if fine-tuning moves it, the whole downstream
              induction population breaks. RoPE would suffer more than GPT-2's distributed induction.
  BACKSTOP  — but fine-tuning may adjust LATE layers far more than early ones (early features are general/stable),
              so an early shared writer is *protected* precisely because it is early.

This measures both, on base/instruct pairs of the same model:
  (1) per-layer WEIGHT DRIFT vs depth — relative Frobenius ||W_inst − W_base|| / ||W_base|| for attn (q/k/v/o) and
      MLP (gate/up/down), read lazily from safetensors (two tensors in memory at a time, no full model). Tests
      "do early nodes get adjusted less?" — and reports the drift AT the induction writer's layer vs the model mean.
  (2) INDUCTION SURVIVAL — induction-NLL and the prev-token / induction head masses, base vs instruct (ResidualVM),
      did the in-context-copy capability and its heads survive post-training?

Output: runs/disassembly/posttrain_drift_summary.json (merge-safe). Findings -> docs/FINDINGS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# base -> instruct, with the induction shared-writer layer (from circuit_writer_cluster_summary, dominant writer).
PAIRS = [
    {"name": "Llama-3.2-1B", "base": "unsloth/Llama-3.2-1B", "instruct": "unsloth/Llama-3.2-1B-Instruct", "writer_layer": 0},
    {"name": "Qwen2.5-1.5B", "base": "Qwen/Qwen2.5-1.5B", "instruct": "Qwen/Qwen2.5-1.5B-Instruct", "writer_layer": 3},
    {"name": "gemma-2-2b", "base": "google/gemma-2-2b", "instruct": "google/gemma-2-2b-it", "writer_layer": 0},
]
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


def repo_reader(repo):
    """Lazy per-parameter tensor reader for a HF repo (handles sharded safetensors). Returns (get(name), keyset)."""
    import json as _json
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    handles = {}

    def open_file(fname):
        if fname not in handles:
            handles[fname] = safe_open(hf_hub_download(repo, fname), framework="pt")
        return handles[fname]
    try:
        idx = hf_hub_download(repo, "model.safetensors.index.json")
        weight_map = _json.loads(Path(idx).read_text())["weight_map"]
        return (lambda n: open_file(weight_map[n]).get_tensor(n)), set(weight_map)
    except Exception:
        h = open_file("model.safetensors")
        return (lambda n: h.get_tensor(n)), set(h.keys())


def weight_drift(base_id, inst_id, nL):
    gb, kb = repo_reader(base_id); gi, ki = repo_reader(inst_id)

    def rel(name):
        if name not in kb or name not in ki:
            return None
        wb = gb(name).float(); wi = gi(name).float()
        return float((wi - wb).norm() / (wb.norm() + 1e-9))
    curve = []
    for L in range(nL):
        pre = f"model.layers.{L}."
        a = [rel(pre + f"self_attn.{p}.weight") for p in ATTN]; a = [x for x in a if x is not None]
        m = [rel(pre + f"mlp.{p}.weight") for p in MLP]; m = [x for x in m if x is not None]
        row = {"layer": L, "attn_drift": float(np.mean(a)) if a else None, "mlp_drift": float(np.mean(m)) if m else None}
        for p in ATTN + MLP:
            comp = "self_attn." if p in ATTN else "mlp."
            row[p] = rel(pre + f"{comp}{p}.weight")
        curve.append(row)
    return curve


def induction_survival(model_id, args, dev):
    """induction-NLL + prev-token/induction head masses for one model (ResidualVM)."""
    from residual_vm import ResidualVM
    import urllib.request
    vm = ResidualVM(model_id, device=dev)
    rng = np.random.default_rng(args.seed)
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    pids = vm.tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]
    vm.fit_means(chunks)
    V = vm.model.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
    rep_seqs = [(lambda s: s + s)([int(x) for x in rng.integers(lo, hi, args.rep_len)]) for _ in range(args.probes)]

    def ind_target(s):
        L = len(s) // 2
        return [(p, s[p + 1]) for p in range(L, 2 * L - 1)]
    ind_heads, imass = vm.find_heads(rep_seqs[: args.id_probes], "induction", top=4)
    pv_heads, pmass = vm.find_heads(chunks[: args.id_probes], "prevtok", top=2)
    nll = vm.nll(rep_seqs, ind_target)
    out = {"induction_nll": nll, "induction_heads": [f"{L}.{h}" for L, h in ind_heads], "induction_top_mass": float(imass.max()),
           "prevtok_heads": [f"{L}.{h}" for L, h in pv_heads], "prevtok_top_mass": float(pmass.max()),
           "prevtok_top_layer": pv_heads[0][0] if pv_heads else None}
    del vm
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=12)
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--id-probes", type=int, default=12)
    p.add_argument("--rep-len", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-induction", action="store_true", help="weight-drift only (skip the GPU induction pass)")
    p.add_argument("--only", default=None, help="comma list of pair names to run")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoConfig
    dev = args.device if torch.cuda.is_available() else "cpu"
    pairs = [pr for pr in PAIRS if not args.only or pr["name"] in args.only.split(",")]
    results = []
    for pr in pairs:
        print(f"\n=== {pr['name']} (base {pr['base']} -> instruct {pr['instruct']}) ===")
        try:
            nL = AutoConfig.from_pretrained(pr["base"]).num_hidden_layers
            curve = weight_drift(pr["base"], pr["instruct"], nL)
            attn = np.array([r["attn_drift"] for r in curve if r["attn_drift"] is not None])
            mlp = np.array([r["mlp_drift"] for r in curve if r["mlp_drift"] is not None])
            early = attn[: max(nL // 4, 1)].mean(); late = attn[-max(nL // 4, 1):].mean()
            wl = pr["writer_layer"]; writer_attn = curve[wl]["attn_drift"]
            rec = {"model": pr["name"], "n_layers": nL, "drift_curve": curve,
                   "attn_drift_mean": float(attn.mean()), "mlp_drift_mean": float(mlp.mean()),
                   "attn_drift_early": float(early), "attn_drift_late": float(late),
                   "early_over_late": float(early / (late + 1e-9)),
                   "writer_layer": wl, "writer_attn_drift": writer_attn,
                   "writer_drift_vs_mean": float(writer_attn / (attn.mean() + 1e-9))}
            print(f"  attn drift: early(L0..) {early:.3f} vs late {late:.3f} (early/late {early/(late+1e-9):.2f}) | "
                  f"writer L{wl} drift {writer_attn:.3f} ({writer_attn/(attn.mean()+1e-9):.2f}x model mean)")
            if not args.no_induction:
                base_s = induction_survival(pr["base"], args, dev)
                if dev == "cuda":
                    torch.cuda.empty_cache()
                inst_s = induction_survival(pr["instruct"], args, dev)
                if dev == "cuda":
                    torch.cuda.empty_cache()
                rec["induction_base"] = base_s; rec["induction_instruct"] = inst_s
                rec["induction_nll_delta"] = inst_s["induction_nll"] - base_s["induction_nll"]
                print(f"  induction-NLL base {base_s['induction_nll']:.2f} -> instruct {inst_s['induction_nll']:.2f} "
                      f"(Δ {rec['induction_nll_delta']:+.2f}) | prevtok writer base {base_s['prevtok_heads'][0]} "
                      f"-> instruct {inst_s['prevtok_heads'][0]}")
            results.append(rec)
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": pr["name"], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "posttrain_drift_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "post-training weight drift vs depth + induction survival (RoPE base/instruct pairs)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'drift_curve' in r])} pairs → {sumpath}")
    return out


if __name__ == "__main__":
    main()
