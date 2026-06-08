"""Scaling synthesis — assemble the scale-dependent quantities from the committed result JSONs into one table.

The recurring finding across the catalog is that several things people credit to *architecture* (absolute-position
vs RoPE) actually track **scale**. This stdlib-only generator (no GPU, no model) reads the committed summaries and
lines up the scale-varying quantities — induction key-sharpness, induction redundancy, induction-circuit
reconstruction coverage, MLP0 token-determinism, the knowledge causal-trace plateau, and succession's MLP depth —
across the GPT-2 size ladder (small → medium → large) and the RoPE family, so the "scale, not just architecture"
thesis is readable in one place. Regenerate after any of the underlying runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "Gemma-2-2B/gemma-2-2b", "Llama-3.2-1B", "Qwen2.5-1.5B"]
PARAMS = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1.5B", "gemma-2-2b": "2.6B",
          "Llama-3.2-1B": "1.2B", "Qwen2.5-1.5B": "1.5B"}


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else {}


def by_model(summary, key="results", name="model"):
    return {r.get(name): r for r in summary.get(key, []) if name in r}


def reconstruction_verdict(rd):
    if not rd:
        return "—"
    curve = rd.get("curve") or []; full = rd.get("full", 0.0); ms = rd.get("max_solo", 0.0)
    peak = max((c["effect"] for c in curve), default=full)
    if peak > 0.1 and full < 0.7 * peak:
        return "compensatory"
    if full <= 1.4 * ms and ms > 0.1:
        return "bottleneck"
    return "distributed"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/disassembly"))
    p.add_argument("--docs", type=Path, default=Path("docs"))
    args = p.parse_args(argv)
    R = args.root
    xd = by_model(load(R / "operators/xmodel_dossiers_summary.json"))
    rec = by_model(load(R / "circuits/circuit_reconstruction_summary.json"))
    det = by_model(load(R / "operators/mlp_detokenizer_summary.json"))
    succ = by_model(load(R / "operators/succession_summary.json"))
    trace = by_model(load(R / "circuits/causal_tracing_summary.json"))

    def model_key(name):
        return name.split("/")[-1]

    rows = []
    for spec in ORDER:
        name = model_key(spec)
        x = xd.get(name, {}); ops = x.get("ops", {})
        ind = ops.get("induction", {})
        ch = ind.get("channel", {})
        key_collapse = ch.get("key_top", {}).get("collapse") if "key_top" in ch else None
        red = reconstruction_verdict(ind.get("redundancy"))
        rr = rec.get(name, {})
        cov = rr.get("circuit_coverage"); rcov = rr.get("resample_circuit_coverage")
        dd = det.get(name, {})
        mlp0 = next((L["determinism"] for L in dd.get("layers", []) if L.get("layer") == 0), None)
        su = succ.get(name, {})
        succ_depth = su.get("top_mlp_layers", [{}])[0].get("depth") if su.get("top_mlp_layers") else None
        tr = trace.get(name, {})
        trace_depth = tr.get("peak_depth") if tr.get("n_facts_used", 0) >= 4 else None
        rows.append({"model": name, "params": PARAMS.get(name, "?"), "key_collapse": key_collapse, "redundancy": red,
                     "recon_cov": cov, "recon_resample": rcov, "mlp0_det": mlp0, "succ_depth": succ_depth, "trace_depth": trace_depth})

    def fpct(x):
        return f"{x:+.0%}" if isinstance(x, (int, float)) else "—"

    def fnum(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "—"

    L = ["---", "title: Scaling synthesis", "---", "", "# Scaling synthesis — what tracks scale, not just architecture", "",
         "The single clearest cross-cutting finding: several properties usually attributed to *architecture* "
         "(absolute-position vs RoPE) actually track **scale**. This table lines up the scale-varying quantities from "
         "across the catalog (assembled by `scaling_synthesis.py` from the committed result JSONs — no model run). "
         "Read the GPT-2 ladder (124M → 355M → 774M) top-to-bottom.", "",
         "| model | params | induction key-collapse | induction redundancy | recon. coverage (mean / resample) | MLP0 token-determinism | succession MLP depth | knowledge trace peak depth |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['params']} | {fpct(r['key_collapse'])} | {r['redundancy']} | "
                 f"{fpct(r['recon_cov'])} / {fpct(r['recon_resample'])} | {fnum(r['mlp0_det'])} | "
                 f"{fnum(r['succ_depth'])} | {fnum(r['trace_depth'])} |")
    L += ["", "## What the columns show", "",
          "- **Induction key-collapse** — how much removing the single top prev-token writer collapses the induction "
          "head's attention. The full GPT-2 ladder is monotone to zero: small (124M) **+39%** (one dominant writer) → "
          "medium +8% → large +1% → **xl (1.5B) +0%**. The single-writer circuit is a *small-model* trait; by 1.5B GPT-2 "
          "distributes the key entirely, like the RoPE models (~0–3%).",
          "- **Induction redundancy** — distributed (superadditive population) in the small models, **compensatory** "
          "(non-monotonic) in gpt2-large + Gemma: the population self-interferes once big enough.",
          "- **Reconstruction coverage** — how much the named 8-head induction circuit reconstructs in isolation. "
          "Decays with GPT-2 scale (small +17%/+30% → large +0%/+5%): bigger models spread induction across a wider "
          "supporting cast, so no compact circuit suffices ([reconstruction](circuits/reconstruction.md)).",
          "- **MLP0 token-determinism** — the early MLP is an [extended embedding](operators/mlp_detokenizer.md); the "
          "token-determined block *widens and strengthens* with GPT-2 scale (small 0.63 → large 0.75 → xl 0.80; the "
          "block also spreads from just L0 to L0–L2).",
          "- **Succession / knowledge depth** — the [succession](operators/succession.md) MLP and the "
          "[causal-trace](circuits/causal_tracing.md) knowledge site both sit deeper as the model grows "
          "(GPT-2-small ≈ L0–L2 / depth 0.1; gpt2-large ≈ L7–9 / depth 0.2 and a broad early-mid plateau).", "",
          "**The thesis:** as models scale, the *same* named circuits become **more distributed** — single dominant "
          "writers give way to populations, compact circuits stop being sufficient, and the load-bearing MLP sites "
          "broaden and deepen. Absolute-vs-RoPE is a real axis (the sink, positional broadcast), but much of what "
          "looks architectural is the small models being unusually *localized*. See [Cross-model findings](FINDINGS.md).", "",
          "_Assembled from the committed `runs/disassembly/**` summaries. Regenerate: "
          "[scaling_synthesis.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/scaling_synthesis.py)._"]
    args.docs.mkdir(parents=True, exist_ok=True)
    (args.docs / "scaling.md").write_text("\n".join(L))
    print(f"[done] {len(rows)} models → {args.docs / 'scaling.md'}")
    return rows


if __name__ == "__main__":
    main()
