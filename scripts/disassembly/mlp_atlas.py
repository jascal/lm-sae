"""The MLP / COMPUTE catalog (survey matrix) — the *other* instruction class, across models.

Attention MOVES operands; the MLP COMPUTES on them. The operator catalog (`operator_atlas.py`) is
attention-only — yet the ResidualVM discovery engine found **MLP0 is the single most load-bearing component for
every behaviour**. This catalogs the COMPUTE class across architectures: for each (model, layer) it mean-ablates
the whole MLP block and measures the causal ΔNLL on generic prose + on induction (repeated-random 2nd-copy), the
arch-generic COMPUTE-load-bearing profile and its depth localization.

The GPT-2 deep COMPUTE characterization already exists (`mlp_catalog.py` = neuron read→write vocabulary + the
low-rank test; `mlp_ops.py` = MLP-in-the-coverage-harness + head↔MLP edges + named MLP idioms) and is harvested
into the summary here. The new axis is **cross-model**: do MLPs carry COMPUTE the same way under RoPE/GQA/gated-MLP
as under GPT-2's GELU MLP? (Mamba/SSM has no separate MLP block — its mixer is the whole layer — so it is excluded,
the COMPUTE analog of "no attention heads".)

Output: [runs/disassembly/operators/mlp_compute_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/mlp_compute_summary.json) + `mlp_compute.png` + a generated
`docs/operators/mlp_compute.md` page.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mlp_blocks(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):           # GPT-2
        return [b.mlp for b in model.transformer.h]
    if hasattr(model, "model") and hasattr(model.model, "layers"):                  # RoPE (Gemma/Llama/Qwen)
        return [ly.mlp for ly in model.model.layers]
    raise SystemExit("no per-layer MLP blocks (SSM? — excluded)")


def run_model(model_id, args, dev):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval().to(dev)
    mlps = mlp_blocks(model); nL = len(mlps)
    is_gpt2 = hasattr(model, "transformer")
    V = model.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
    rng = np.random.default_rng(args.seed)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]
    rep = [(lambda s: s + s)([int(x) for x in rng.integers(lo, hi, args.rep_len)]) for _ in range(args.n_ind)]

    # per-layer mean MLP output (corpus)
    cap = {L: [] for L in range(nL)}
    hk = [mlps[L].register_forward_hook((lambda L: lambda m, i, o: cap[L].append((o[0] if isinstance(o, tuple) else o).detach().reshape(-1, (o[0] if isinstance(o, tuple) else o).shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in chunks:
            model(input_ids=torch.tensor([c], device=dev))
    for h in hk:
        h.remove()
    mean_mlp = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    def hooks(ablate):
        hs = []
        for L in ablate:
            def mk(L):
                def hook(m, i, o):
                    t = o[0] if isinstance(o, tuple) else o
                    rep_t = mean_mlp[L].to(t.dtype).expand_as(t)
                    return (rep_t,) + tuple(o[1:]) if isinstance(o, tuple) else rep_t
                return hook
            hs.append(mlps[L].register_forward_hook(mk(L)))
        return hs

    def generic_nll(ablate=()):
        hs = hooks(ablate); tot = 0.0; n = 0
        try:
            with torch.no_grad():
                for c in chunks:
                    lp = F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                    y = torch.tensor(c[1:], device=dev); tot += float(-lp[torch.arange(len(y)), y].sum()); n += len(y)
        finally:
            for h in hs:
                h.remove()
        return tot / max(n, 1)

    def induction_nll(ablate=()):
        hs = hooks(ablate); tot = 0.0; n = 0
        try:
            with torch.no_grad():
                for s in rep:
                    lp = F.log_softmax(model(input_ids=torch.tensor([s], device=dev)).logits[0].float(), -1)
                    L = len(s) // 2
                    for p in range(L, 2 * L - 1):
                        tot += float(-lp[p, s[p + 1]]); n += 1
        finally:
            for h in hs:
                h.remove()
        return tot / max(n, 1)

    g0 = generic_nll(); i0 = induction_nll()
    layers = []
    for L in range(nL):
        layers.append({"layer": L, "depth": round(L / max(nL - 1, 1), 2),
                       "generic_dNLL": generic_nll([L]) - g0, "induction_dNLL": induction_nll([L]) - i0})
    all_d = generic_nll(list(range(nL))) - g0
    top_g = sorted(layers, key=lambda r: -r["generic_dNLL"])[:3]
    top_i = sorted(layers, key=lambda r: -r["induction_dNLL"])[:3]
    return {"model": model_id.split("/")[-1], "arch": "GPT-2/absolute" if is_gpt2 else "RoPE", "n_layers": nL,
            "generic_base": g0, "induction_base": i0, "all_mlp_ablated_generic_dNLL": all_d,
            "layers": layers, "top_generic": top_g, "top_induction": top_i}


def write_doc(out, gpt2_harvest, docs):
    models = [r for r in out["results"] if "layers" in r]
    lines = ["# Operator class `MLP` / COMPUTE", "",
             "Attention **MOVES** operands; the MLP **COMPUTES** on them. The [operator catalog](README.md) is "
             "attention-only — but in the ResidualVM discovery sweeps **MLP0 had the largest single-component causal "
             "effect of anything measured** (induction / IOI / generic). A working catalog of the COMPUTE class "
             "across architectures — provisional and descriptive. For the *mechanism* of that load-bearing early "
             "MLP, see [**is MLP0 an extended embedding / detokenizer?**](mlp_detokenizer.md) — a token-determinism "
             "test (MLP0's output is largely fixed by the current token in 5/6 models; Llama-3.2-1B is the outlier).", "",
             "## Cross-model — per-layer MLP causal ΔNLL (mean-ablate the whole MLP block)", "",
             "Top MLP layers by causal damage when ablated (generic prose NLL; depth = layer/(L−1)):", "",
             "| model | arch | L | all-MLP ΔNLL (generic) | top generic-MLP (depth, ΔNLL) | top induction-MLP (depth, ΔNLL) |",
             "|---|---|---|---|---|---|"]
    for r in models:
        tg = r["top_generic"][0]; ti = r["top_induction"][0]
        lines.append(f"| {r['model']} | {r['arch']} | {r['n_layers']} | {r['all_mlp_ablated_generic_dNLL']:+.2f} | "
                     f"L{tg['layer']} (d{tg['depth']}, {tg['generic_dNLL']:+.2f}) | L{ti['layer']} (d{ti['depth']}, {ti['induction_dNLL']:+.2f}) |")
    lines += ["", "**Reading it:** COMPUTE is **depth-organized** — an early MLP (commonly called the *detokenizer*, "
              "low depth — a label, not a verified mechanism) is the biggest single COMPUTE op for induction in the "
              "GPT-2 family, and late MLPs carry generic-LM output. The whole-MLP-stack ablation ΔNLL is large in "
              "every model (COMPUTE is load-bearing everywhere, unlike any single attention-op class).", ""]
    mc, mo = gpt2_harvest if gpt2_harvest else ({}, {})
    gpt2 = next((r for r in models if r["model"] == "gpt2"), None)
    if gpt2 and mo:
        byL = {ly["layer"]: ly for ly in gpt2["layers"]}
        imp = mo.get("mlp_importance", {}); named = mo.get("named_mlp_idioms", {})

        def idiom_str(L):
            ns = named.get(f"L{L}", [])
            return "; ".join(f"`{'+'.join(n['reads'][:2])}→{'+'.join(n['writes'][:2])}`" for n in ns[:2]) or "(not catalogued)"
        order = sorted({0, 1, 2, 11} | {int(k[1:]) for k in named}, key=lambda L: -imp.get(f"L{L}", 0))
        lines += ["## Named MLP operators (GPT-2) — listed even where the mechanism is unverified", "",
                  "In the natural-history spirit we **list the load-bearing MLP specimens with what we measured**, "
                  "even where the mechanism is not established. (\"Detokenizer\" is the common *label* for L0; we "
                  "record the behaviour and operands, not a mechanism claim.)", "",
                  "| MLP | depth | causal ΔNLL (generic / induction) | recon-importance | top read→write neuron idioms | mechanism |",
                  "|---|---|---|---|---|---|"]
        for L in order:
            ly = byL.get(L)
            if not ly:
                continue
            mech = ('partial — the **"detokenizer"** label: writes sentence-initial / common tokens' if L == 0 else
                    "partial — punctuation / output writes" if L == 11 else "**unverified** (listed by measured effect)")
            tag = "**MLP0**" if L == 0 else f"MLP{L}"
            lines.append(f"| {tag} | {ly['depth']} | {ly['generic_dNLL']:+.2f} / {ly['induction_dNLL']:+.2f} | "
                         f"{imp.get(f'L{L}', 0):.2f} | {idiom_str(L)} | {mech} |")
        lines += ["", "_MLP0 is GPT-2's single most load-bearing component (recon-importance 0.77 of all MLPs; "
                  "ablating it costs induction +11.7 NLL) — listed here as a catalog entry even though *how* it works "
                  "is only partially characterized. The other models' early load-bearing MLP (the detokenizer-analog) "
                  "is in the cross-model table above; its per-neuron idioms are not yet run (no per-neuron basis off "
                  "GPT-2 — a documented gap)._", ""]
    if gpt2_harvest:
        cat = ", ".join(f"`{'+'.join(n['reads'][:2])}→{'+'.join(n['writes'][:2])}`" for n in mc.get("catalog", [])[:6])
        lines += ["## GPT-2 deep characterization (harvested)", "",
                  f"- **COMPUTE vocabulary is low-rank** (`mlp_catalog.py`): transform participation "
                  f"{mc.get('transform_participation', 0):.0f} vs random {mc.get('transform_participation_random', 0):.0f} "
                  f"— a small reused set of compute templates (heavier-tailed than attention's ~5: rank-90 ≈ {mc.get('transform_rank90')}).",
                  f"- **top neuron read→write idioms:** {cat}",
                  f"- **MLPs carry the reconstruction coverage** (`mlp_ops.py`): MLP-only coverage "
                  f"**{mo.get('coverage_mlp_only', 0):+.2f}** vs attention-only {mo.get('coverage_attention_only', 0):+.2f} "
                  f"(they interact — neither alone reaches the full pass); load-bearing MLPs concentrate in "
                  f"{', '.join(list(mo.get('named_mlp_idioms', {}))[:4])} (L0 = the detokenizer).",
                  f"- **head↔MLP composition edges** exist in weight space (top head→MLP {mo.get('top_head_to_mlp', [['?','?',0]])[0][:2]}, "
                  f"MLP→head {mo.get('top_mlp_to_head', [['?','?',0]])[0][:2]}) — the COMPUTE nodes the attention-only DAG missed.", ""]
    lines += ["## Gaps", "",
              "- **Mamba / SSM** has **no separate MLP block** (the state-space mixer is the whole layer) — excluded, "
              "the COMPUTE analog of \"no attention heads\".",
              "- Per-**neuron** read→write idioms are catalogued for **GPT-2 only** (the cheap token-unembedding basis); "
              "the cross-model rows are per-**layer** causal profiles. RoPE neuron-idioms need the per-layer SAE / "
              "token-centroid basis (the `disassemble_gemma.py` route).", "",
              "_Data: [runs/disassembly/operators/mlp_compute_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/mlp_compute_summary.json). Regenerate: [mlp_atlas.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/mlp_atlas.py)._"]
    (docs / "mlp_compute.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=24)
    p.add_argument("--n-ind", type=int, default=32)
    p.add_argument("--rep-len", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/operators"))
    p.add_argument("--disasm", type=Path, default=Path("runs/disassembly"))
    p.add_argument("--docs", type=Path, default=Path("docs/operators"))
    args = p.parse_args(argv)

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = run_model(mid, args, dev)
            if dev == "cuda":
                torch.cuda.empty_cache()
            results.append(r)
            tg = r["top_generic"][0]; ti = r["top_induction"][0]
            print(f"  {r['arch']} {r['n_layers']}L  all-MLP ablated ΔNLL(generic) {r['all_mlp_ablated_generic_dNLL']:+.2f}  "
                  f"| top generic L{tg['layer']}(d{tg['depth']},{tg['generic_dNLL']:+.2f}) "
                  f"| top induction L{ti['layer']}(d{ti['depth']},{ti['induction_dNLL']:+.2f})")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid, "error": str(e)})

    def load(name):
        f = args.disasm / f"{name}_summary.json"
        return json.loads(f.read_text()) if f.exists() else {}
    gpt2_harvest = (load("mlp_catalog"), load("mlp_ops"))
    out = {"experiment": "MLP/COMPUTE catalog — per-layer MLP causal profile across models", "results": results,
           "gpt2_harvest": {"mlp_catalog": gpt2_harvest[0], "mlp_ops": gpt2_harvest[1]},
           "note_gaps": "Mamba/SSM has no separate MLP block (mixer = whole layer); per-neuron idioms GPT-2-only."}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "mlp_compute_summary.json").write_text(json.dumps(out, indent=2, default=float))
    args.docs.mkdir(parents=True, exist_ok=True)
    write_doc(out, gpt2_harvest, args.docs)

    ok = [r for r in results if "layers" in r]
    print("\n[MLP/COMPUTE catalog] per-layer generic-ΔNLL profile (mean-ablate each MLP block), by model:")
    for r in ok:
        prof = " ".join(f"{ly['generic_dNLL']:+.1f}" for ly in r["layers"])
        print(f"  {r['model']:>14} ({r['n_layers']:>2}L): {prof}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (aG, aI) = plt.subplots(1, 2, figsize=(13.5, 5.0))
        for r in ok:
            xs = [ly["depth"] for ly in r["layers"]]; c = "#d62728" if r["arch"].startswith("GPT") else "#1f77b4"
            aG.plot(xs, [ly["generic_dNLL"] for ly in r["layers"]], "-o", ms=3, alpha=0.8, label=r["model"], color=c if r["arch"].startswith("GPT") else None)
            aI.plot(xs, [ly["induction_dNLL"] for ly in r["layers"]], "-o", ms=3, alpha=0.8, label=r["model"])
        for ax, ttl in ((aG, "generic-LM ΔNLL when MLP[L] ablated"), (aI, "induction ΔNLL when MLP[L] ablated")):
            ax.set_xlabel("relative depth"); ax.set_ylabel("ΔNLL"); ax.set_title(ttl, fontsize=10); ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=6, ncol=2)
        fig.suptitle("MLP / COMPUTE catalog — per-layer causal profile across architectures", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(args.outdir / "mlp_compute.png", dpi=130)
        print(f"[fig] {args.outdir / 'mlp_compute.png'}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.outdir / 'mlp_compute_summary.json'} + {args.docs / 'mlp_compute.md'}")
    return out


if __name__ == "__main__":
    main()
