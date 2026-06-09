"""Checkpoint test — does the per-token feature count k rise over TRAINING (at fixed width)?

The active-dim growth is carried by k ∝ d^1.15 across the Pythia ladder, entropy-independent. Two surviving mechanisms:
(2) feature economy — a wider model *structurally chooses* a larger per-token k from the start; (4) implicit bias /
optimization — training dynamics keep more features "warm" and k/m *rises over training*. Pythia ships intermediate
checkpoints (revisions `step{N}`), so at FIXED width we can watch k as a function of training step. If k/m rises through
training → mechanism 4; if it is set early and flat → mechanism 2 (a width/capacity choice, not a dynamics effect).

k = per-token participation ratio of the MLP hidden (|h|₁² / |h|₂²), averaged over tokens and layers. Output:
runs/disassembly/checkpoint_k_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_kv_sparsity import down_projs  # noqa: E402

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def k_at_step(mid, step, tok, chunks, dev):
    import torch
    from transformers import AutoModelForCausalLM
    t = torch
    model = AutoModelForCausalLM.from_pretrained(mid, revision=f"step{step}", dtype=torch.float32).eval().to(dev)
    downs = down_projs(model); nL = len(downs)
    m = downs[0].weight.shape[1]                                          # NeoX dense_4h_to_h: Linear(d_ff→d), weight (d,d_ff)
    cap = {}
    hs = [mod.register_forward_pre_hook((lambda L: lambda mod_, i: cap.__setitem__(L, i[0].detach()))(L))
          for L, mod in enumerate(downs)]
    ks = 0.0; n = 0
    with t.no_grad():
        for c in chunks:
            cap.clear(); model(input_ids=t.tensor([c], device=dev))
            for L in range(nL):
                h = cap[L][0].float()
                ks += float(((h.abs().sum(-1) ** 2) / (h.pow(2).sum(-1) + 1e-9)).sum()); n += h.shape[0]
    for h in hs:
        h.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ks / max(n, 1), int(m)        # n counts (layer×token), so ks/n is already the layer-averaged per-token PR


def run_model(mid, args):
    import torch
    from transformers import AutoTokenizer
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    text = urllib.request.urlopen(urllib.request.Request(CORPUS, headers={"User-Agent": "Mozilla/5.0"}),
                                  timeout=20).read().decode("utf-8", "ignore")[: args.chars]
    ids = tok(text)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16][: args.fit]
    steps = [int(x) for x in args.steps.split(",")]
    rows = []
    for s in steps:
        k, m = k_at_step(mid, s, tok, chunks, dev)
        rows.append({"step": s, "k": k, "k_over_m": k / m, "m": m})
        print(f"  step {s:7d}  k={k:6.0f}  k/m={k / m:.3f}")
    fin = rows[-1]["k_over_m"]
    return {"model": mid.split("/")[-1], "steps": rows, "k_over_m_final": fin,
            "k_over_m_first": rows[0]["k_over_m"], "ratio_final_over_first": fin / max(rows[0]["k_over_m"], 1e-9)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-160m")
    p.add_argument("--steps", default="0,128,1000,13000,143000", help="Pythia checkpoint steps (revisions step{N})")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--chars", type=int, default=140000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("runs/disassembly/checkpoint_k_summary.json"))
    args = p.parse_args(argv)

    import torch
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} (k per-token vs training step) ===")
        try:
            r = run_model(mid, args)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(r)
            print(f"  k/m: {r['k_over_m_first']:.3f} (init/early) → {r['k_over_m_final']:.3f} (final)  "
                  f"= ×{r['ratio_final_over_first']:.2f}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "k(training step) — does per-token feature recruitment rise over training?", "results": results}, indent=2, default=float))
    print(f"\n[done] → {args.out}")
    return results


if __name__ == "__main__":
    main()
