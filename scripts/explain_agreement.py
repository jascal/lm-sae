#!/usr/bin/env python3
"""Cross-repo `explain` agreement oracle (FABLE_DIRECTIONS D8).

`fieldrun`'s `explain` (Rust, the distribution form, `../fieldrun/src/explain.rs`) and `pylm`'s `explain.py` (numpy,
the research reference) are two implementations of the SAME per-token circuit + feature readout. Nothing checked that
they agree. This harness runs BOTH on the same model + the same contexts and measures agreement per token:

  - model_predicts  : the argmax next-token id (the forward pass itself).
  - head_circuits   : the set of (layer, head, role) attention idioms that fire above threshold (the firing idiom).
  - sink_heads      : the count of attend-to-first NO-OP heads.
  - mlp_features    : the set of (layer, neuron) top-activating MLP features (the dense composition units).

If they agree, `fieldrun explain` is a *validated* distribution-form of the lm-sae decompilation — every future arch's
explain inherits a correctness oracle. If they don't, a divergence is a bug (almost certainly in the less-exercised
Rust path); the harness itemises each one so it can be triaged and fixed.

Run with the torch venv from the lm-sae repo root:
    .venv/bin/python scripts/explain_agreement.py --model gpt2 --n 30 --out runs/pylm/explain_agreement_gpt2.json

The fieldrun binary is found at ../fieldrun/target/release/fieldrun (built `cargo build --release`).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PYLM = Path(__file__).resolve().parent.parent / "pylm"
sys.path.insert(0, str(PYLM))
import explain as px  # noqa: E402  the pylm reference implementation


# model presets: (fieldrun bundle stem, pylm flat weights, tokenizer, holdout token-id stream). The comparison is
# composition-only (heads / sink / features / predict), so no symbolic store is needed — both sides run the same
# pure-numpy/Rust forward pass over flat weights derived from the same HF checkpoint.
MODELS = {
    "gpt2": dict(bundle=PYLM / "gpt2", weights=PYLM / "weights_gpt2.npz",
                 tokenizer="gpt2", holdout=PYLM / "holdout_gpt2.json"),
    "qwen05b": dict(bundle=Path("/tmp/qwen05b_f32"), weights=PYLM / "weights_qwen05b_fp32.npz",
                    tokenizer="Qwen/Qwen2.5-0.5B", holdout=PYLM / "holdout_Qwen2.5-1.5B.json"),
    "gemma2": dict(bundle=Path("/tmp/gemma2_f16"), weights=PYLM / "weights_gemma2_2b_fp16.npz",
                   tokenizer="google/gemma-2-2b", holdout=PYLM / "holdout_gemma2.json"),
}


def py_composition(lm_net, ctx, top_heads=6, top_feats=6, head_thr=0.15, sink_thr=0.5):
    """The pylm reference composition readout for one context — heads named by `explain.classify_head` (the shared
    idiom signatures), sinks counted, top-|act| MLP features per layer. Mirrors `explain.explain`'s composition half
    (full-precision sort), without the symbolic-retrieval half (so no store is needed)."""
    cap = {}
    model_id = int(lm_net.logits(ctx, capture=cap)[-1].argmax())
    heads, n_sink = [], 0
    for L, att in enumerate(cap["att_last"]):
        for hh in range(att.shape[0]):
            role, j, mass = px.classify_head(att[hh], ctx)
            if role == "sink" and mass >= sink_thr:
                n_sink += 1
                continue
            if role in ("induction", "duplicate-token", "previous-token") and mass >= head_thr:
                heads.append({"layer": L, "head": hh, "role": role, "attends_to": j, "mass": float(mass)})
    order = {"induction": 0, "duplicate-token": 1, "previous-token": 2}
    heads.sort(key=lambda r: (order[r["role"]], -r["mass"]))
    heads = heads[:top_heads]
    feats = []
    for L, h in enumerate(cap["mlp_h"]):
        n = int(np.abs(h).argmax())
        feats.append({"layer": L, "neuron": n, "act": float(h[n])})
    feats.sort(key=lambda r: -abs(r["act"]))
    feats = feats[:top_feats]
    return model_id, {"head_circuits": heads, "sink_heads": n_sink, "mlp_features": feats}


def fieldrun_bin() -> Path:
    b = Path(__file__).resolve().parent.parent.parent / "fieldrun" / "target" / "release" / "fieldrun"
    if not b.exists():
        sys.exit(f"[explain_agreement] fieldrun binary not found at {b} — run `cargo build --release` in ../fieldrun")
    return b


def rust_explain(binary: Path, bundle: Path, ctx: list[int]) -> dict:
    """Run `fieldrun --explain --out-json` on one context (the prediction after the full ctx) and parse the JSON."""
    with tempfile.TemporaryDirectory() as td:
        ids_path = Path(td) / "ctx.json"
        out_path = Path(td) / "ex.json"
        ids_path.write_text(json.dumps({"holdout_ids": ctx}))
        subprocess.run(
            [str(binary), "--bundle", str(bundle), "--ids", str(ids_path),
             "--ctx", str(len(ctx)), "--explain", "--out-json", str(out_path)],
            capture_output=True, check=True,
        )
        return json.loads(out_path.read_text())


def head_set(circuits: list[dict]) -> set[tuple[int, int, str]]:
    return {(c["layer"], c["head"], c["role"]) for c in circuits}


def feat_set(feats: list[dict]) -> set[tuple[int, int]]:
    return {(f["layer"], f["neuron"]) for f in feats}


def compare_one(pc: dict, py_model_id: int, ru: dict) -> dict:
    """One token: per-field agreement + the itemised divergences. `pc` is the pylm composition dict, `ru` the Rust one."""
    rc = ru
    py_heads, ru_heads = head_set(pc["head_circuits"]), head_set(rc["head_circuits"])
    py_feats, ru_feats = feat_set(pc["mlp_features"]), feat_set(rc["mlp_features"])
    div = []
    if py_model_id != ru["model_predicts"]:
        div.append(f"predict py={py_model_id} rust={ru['model_predicts']}")
    if py_heads != ru_heads:
        div.append(f"heads py-only={sorted(py_heads - ru_heads)} rust-only={sorted(ru_heads - py_heads)}")
    if pc["sink_heads"] != rc["sink_heads"]:
        div.append(f"sink py={pc['sink_heads']} rust={rc['sink_heads']}")
    if py_feats != ru_feats:
        div.append(f"feats py-only={sorted(py_feats - ru_feats)} rust-only={sorted(ru_feats - py_feats)}")
    return {
        "predict_agree": py_model_id == ru["model_predicts"],
        "heads_agree": py_heads == ru_heads,
        "sink_agree": pc["sink_heads"] == rc["sink_heads"],
        "feats_agree": py_feats == ru_feats,
        "n_heads": len(py_heads),
        "divergences": div,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2", choices=list(MODELS))
    p.add_argument("--n", type=int, default=30, help="number of contexts to cross-check")
    p.add_argument("--ctx", type=int, default=48, help="context length per probe")
    p.add_argument("--stride", type=int, default=40, help="stride between probe windows in the holdout stream")
    p.add_argument("--out", type=Path, default=None,
                   help="summary JSON (default runs/pylm/explain_agreement_<model>_summary.json — tracked)")
    args = p.parse_args(argv)
    m = MODELS[args.model]
    if args.out is None:
        args.out = Path(f"runs/pylm/explain_agreement_{args.model}_summary.json")
    binary = fieldrun_bin()

    if not Path(str(m["bundle"]) + ".fieldrun.json").exists():
        sys.exit(f"[explain_agreement] no fieldrun bundle at {m['bundle']}.fieldrun.json — "
                 f"convert it first: ../fieldrun/.../fieldrun convert --model … --arch … -o {m['bundle']}")
    ids = json.loads(Path(m["holdout"]).read_text())["holdout_ids"]
    starts = list(range(args.ctx, min(len(ids), args.ctx + args.stride * args.n), args.stride))[: args.n]
    ctxs = [ids[end - args.ctx:end] for end in starts]

    # Two sequential phases (not interleaved) so the numpy kernel and the fieldrun process never hold the weights at the
    # same time — a multi-GB model on a small box would otherwise OOM. Phase 1: all Python forwards; free the kernel.
    import gc
    lm_net = px.load_kernel(str(m["weights"]))
    py_out = [py_composition(lm_net, ctx) for ctx in ctxs]
    del lm_net
    gc.collect()
    # Phase 2: all Rust forwards (each `fieldrun --explain` loads + frees its own bundle).
    ru_out = [rust_explain(binary, m["bundle"], ctx) for ctx in ctxs]

    rows = []
    for k, ((py_model_id, pc), ru) in enumerate(zip(py_out, ru_out)):
        r = compare_one(pc, py_model_id, ru)
        rows.append(r)
        flag = "" if not r["divergences"] else "  ⚠ " + " | ".join(r["divergences"])
        print(f"[{k + 1}/{len(ctxs)}] predict={'✓' if r['predict_agree'] else '✗'} "
              f"heads={'✓' if r['heads_agree'] else '✗'} sink={'✓' if r['sink_agree'] else '✗'} "
              f"feats={'✓' if r['feats_agree'] else '✗'}{flag}")

    n = len(rows)
    agg = {f: round(sum(r[f] for r in rows) / n, 4) for f in ("predict_agree", "heads_agree", "sink_agree", "feats_agree")}
    all_agree = sum(all(r[f] for f in ("predict_agree", "heads_agree", "sink_agree", "feats_agree")) for r in rows)
    summary = {
        "model": args.model, "n": n, "ctx": args.ctx,
        "agreement_rate": agg,
        "all_fields_agree_rate": round(all_agree / n, 4),
        "divergences": [{"i": i, "d": r["divergences"]} for i, r in enumerate(rows) if r["divergences"]],
    }
    print("\n=== explain agreement: fieldrun (Rust) vs pylm (numpy) ===")
    print(f"model={args.model}  n={n}  ctx={args.ctx}")
    for f, v in agg.items():
        print(f"  {f:16s} {100 * v:5.1f}%")
    print(f"  {'ALL fields':16s} {100 * summary['all_fields_agree_rate']:5.1f}%")
    if summary["divergences"]:
        print(f"  divergences on {len(summary['divergences'])}/{n} tokens (itemised in the JSON)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"  → {args.out}")
    return summary


if __name__ == "__main__":
    main()
