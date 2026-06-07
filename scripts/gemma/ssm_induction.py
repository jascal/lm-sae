"""Does the in-context-copy (induction) operation appear without attention? — a SWEEP over SSM and attention models.

The disassembly reads sequence-mixing as ATTENTION (QK/OV heads). Induction — the in-context-copy op that is the
one genuinely reused instruction (#33/#35) — is a CONTENT operation. Does anything like it appear when the mixer
is NOT attention? Mamba is a pure state-space model: no attention, no heads, no QK match; mixing is a learned
linear recurrence (a scan).

This is run as an EXHAUSTIVE SWEEP, not a single comparison, and it reports the raw measurements rather than
declaring a verdict. Two families across sizes (Mamba 130m/370m/790m; GPT-2 small/medium/large), each measured:
  - induction GAIN over seeds × context lengths = (1st-copy NLL − 2nd-copy NLL) on repeated random sequences
    (how much the model uses the in-context repeat; mean ± std, and the length trend);
  - per-layer LOCALIZATION = mean-ablate each layer's sequence-mixer (Mamba `mixer.out_proj` / GPT-2 `attn.c_proj`)
    and record the induction-NLL increase, the full depth profile.
The output is the table + profiles; interpretation is deliberately deferred to the caveats (NLL gain ≠ "the same
mechanism"; whole-layer mean-ablation is coarse; the two families have different baselines and are not directly
comparable in magnitude).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _mixers(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "layers"):            # MambaForCausalLM
        return [ly.mixer.out_proj for ly in model.backbone.layers], len(model.backbone.layers), "ssm"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):           # GPT-2
        return [b.attn.c_proj for b in model.transformer.h], len(model.transformer.h), "attention"
    raise SystemExit("unknown architecture")


def run_model(model_id, seeds, lengths, n_seq, ref_len, device):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = device if torch.cuda.is_available() else "cpu"
    AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval().to(dev)
    mixers, nL, kind = _mixers(model)
    V = model.config.vocab_size; hidden = model.config.hidden_size
    lo, hi = int(0.02 * V), int(0.4 * V)
    mean_out = {}

    def make_seqs(seed, length):
        rng = np.random.default_rng(seed)
        return [[int(x) for x in rng.integers(lo, hi, length)] * 2 for _ in range(n_seq)]

    def induction_nll(seqs, ablate=()):
        hs = []
        for L in ablate:
            def mk(L):
                def hook(m, inp, out):
                    return mean_out[L].to(out.dtype).expand_as(out)
                return hook
            hs.append(mixers[L].register_forward_hook(mk(L)))
        t2 = t1 = n2 = n1 = 0.0
        try:
            with torch.no_grad():
                for s in seqs:
                    lp = F.log_softmax(model(input_ids=torch.tensor([s], device=dev)).logits[0].float(), -1)
                    L = len(s) // 2
                    for pos in range(1, L - 1):
                        t1 += float(-lp[pos, s[pos + 1]]); n1 += 1
                    for pos in range(L, 2 * L - 1):
                        t2 += float(-lp[pos, s[pos + 1]]); n2 += 1
        finally:
            for h in hs:
                h.remove()
        return t2 / max(n2, 1), t1 / max(n1, 1)

    # ---- induction gain over seeds x lengths ----
    grid = []
    for seed in seeds:
        for length in lengths:
            nll2, nll1 = induction_nll(make_seqs(seed, length))
            grid.append({"seed": seed, "length": length, "nll_2nd": nll2, "nll_1st": nll1, "gain": nll1 - nll2})
    gains = [g["gain"] for g in grid]
    by_len = {ln: float(np.mean([g["gain"] for g in grid if g["length"] == ln])) for ln in lengths}
    print(f"  {kind} {nL}L hidden{hidden}: gain {np.mean(gains):+.3f}±{np.std(gains):.3f}  "
          f"(by length " + ", ".join(f"{ln}:{by_len[ln]:+.2f}" for ln in lengths) + ")")

    # ---- per-layer localization at a reference config (seed 0, ref_len) ----
    ref = make_seqs(seeds[0], ref_len)
    cap = {L: [] for L in range(nL)}
    caps = [mixers[L].register_forward_hook((lambda L: lambda m, i, o: cap[L].append(o.detach().reshape(-1, o.shape[-1])))(L))
            for L in range(nL)]
    with torch.no_grad():
        for s in ref[: min(len(ref), 16)]:
            model(input_ids=torch.tensor([s], device=dev))
    for h in caps:
        h.remove()
    for L in range(nL):
        mean_out[L] = torch.cat(cap[L], 0).mean(0)
    base2, _ = induction_nll(ref)
    deltas = [induction_nll(ref, ablate=[L])[0] - base2 for L in range(nL)]
    order = sorted(range(nL), key=lambda L: -deltas[L])
    topk = [{"layer": L, "depth_frac": round(L / max(nL - 1, 1), 2), "delta": deltas[L]} for L in order[:5]]
    conc = float(max(deltas) / (np.sum([d for d in deltas if d > 0]) + 1e-9))       # concentration of the top layer
    print("    localization top layers (ΔNLL): " + ", ".join(f"L{t['layer']}({t['delta']:+.2f})" for t in topk)
          + f"  concentration {conc:.2f}")
    return {"model": model_id, "kind": kind, "n_layers": nL, "hidden": hidden,
            "gain_mean": float(np.mean(gains)), "gain_std": float(np.std(gains)), "gain_by_length": by_len,
            "grid": grid, "ref_len": ref_len, "loc_base_nll2": base2, "layer_deltas": deltas,
            "top_layers": topk, "loc_concentration": conc}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="state-spaces/mamba-130m-hf,state-spaces/mamba-370m-hf,state-spaces/mamba-790m-hf,"
                                       "gpt2,gpt2-medium,gpt2-large")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--lengths", default="12,24,48")
    p.add_argument("--n-seq", type=int, default=24)
    p.add_argument("--ref-len", type=int, default=24)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/gemma/ssm_induction_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/ssm_induction.png"))
    args = p.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]; lengths = [int(x) for x in args.lengths.split(",")]

    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            results.append(run_model(mid, seeds, lengths, args.n_seq, args.ref_len, args.device))
        except Exception as e:  # pragma: no cover - gated/missing model, OOM
            print(f"  [skip] {e}"); results.append({"model": mid, "error": str(e)})

    out = {"experiment": "does in-context copy (induction) appear without attention? — SSM vs attention sweep",
           "seeds": seeds, "lengths": lengths, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    ok = [r for r in results if "gain_mean" in r]
    print("\n[data] induction gain (1st−2nd copy NLL) + localization, by model (NOT a verdict — see caveats):")
    print(f"  {'model':>26} {'kind':>9} {'L':>3} {'hidden':>6} | {'gain(μ±σ)':>13} | {'gain@lengths':>20} | top-layer (depth, ΔNLL, conc)")
    for r in sorted(ok, key=lambda r: (r["kind"], r["n_layers"])):
        gl = " ".join(f"{r['gain_by_length'][ln]:+.1f}" for ln in lengths)
        t = r["top_layers"][0]
        print(f"  {r['model'].split('/')[-1]:>26} {r['kind']:>9} {r['n_layers']:>3} {r['hidden']:>6} | "
              f"{r['gain_mean']:+.2f}±{r['gain_std']:.2f}    | {gl:>20} | L{t['layer']} (d{t['depth_frac']}, {t['delta']:+.2f}, c{r['loc_concentration']:.2f})")

    ssm = [r for r in ok if r["kind"] == "ssm"]; att = [r for r in ok if r["kind"] == "attention"]
    print("\n[summary, descriptive] " + (
        f"{len(ssm)} SSM (Mamba) + {len(att)} attention (GPT-2) models swept over {len(seeds)} seeds × {len(lengths)} "
        f"lengths. SSM induction gain ranges {min(r['gain_mean'] for r in ssm):+.2f}..{max(r['gain_mean'] for r in ssm):+.2f}; "
        f"attention {min(r['gain_mean'] for r in att):+.2f}..{max(r['gain_mean'] for r in att):+.2f}. "
        f"Both families localize to specific layers (top-layer concentration ssm "
        f"{np.mean([r['loc_concentration'] for r in ssm]):.2f} / att {np.mean([r['loc_concentration'] for r in att]):.2f}). "
        f"NOT concluding 'same instruction' from this: the gain is a behavioural NLL effect, whole-layer mean-ablation "
        f"is coarse, and the two families' baselines/magnitudes are not directly comparable — see caveats." if ssm and att
        else "incomplete sweep — some models skipped; see table."))
    print("[caveats] (1) induction GAIN = the model uses the in-context repeat; it does NOT prove the same MECHANISM as "
          "attention-induction. (2) mean-ablating a whole mixer layer removes everything that layer does, not just "
          "induction (an over-estimate of 'induction localization'). (3) SSM vs attention gains are on different baselines "
          "— compare WITHIN family across size, not across families by magnitude. (4) single seed for the localization "
          "profile. (5) no head-level resolution in the SSM (no heads) — only layer-level.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axG, axL) = plt.subplots(1, 2, figsize=(13.5, 5.2))
        srt = sorted(ok, key=lambda r: (r["kind"], r["n_layers"]))
        x = np.arange(len(srt)); cols = ["#2ca02c" if r["kind"] == "ssm" else "#d62728" for r in srt]
        axG.bar(x, [r["gain_mean"] for r in srt], yerr=[r["gain_std"] for r in srt], capsize=3, color=cols, edgecolor="k")
        axG.set_xticks(x); axG.set_xticklabels([f"{r['model'].split('/')[-1]}\n{r['n_layers']}L" for r in srt], fontsize=7, rotation=15, ha="right")
        axG.set_ylabel("induction gain (1st−2nd copy NLL)")
        axG.set_title("in-context copy: SSM (green) vs attention (red), by size — raw measurements", fontsize=10)
        for r in srt:
            d = r["layer_deltas"]; xs = np.linspace(0, 1, len(d))
            axL.plot(xs, d, "-", lw=1.2, color=("#2ca02c" if r["kind"] == "ssm" else "#d62728"), alpha=0.8,
                     label=f"{r['model'].split('/')[-1]}")
        axL.set_xlabel("relative depth"); axL.set_ylabel("induction-NLL ΔL when layer mixer ablated")
        axL.set_title("localization profile (coarse: whole-layer mean-ablation)", fontsize=10); axL.legend(fontsize=6, ncol=2)
        fig.suptitle("Induction across mixers — SSM vs attention sweep (data, not a verdict)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
