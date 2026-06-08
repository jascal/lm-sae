"""The flagship unifying test — does the SAE feature basis tax COMPOSITION more than READOUT? (A ↔ B, real LM)

The program's throughline: *a language model is legible in the right basis even where it is not legible as single
SAE features* — composition doesn't factor through the SAE features for the same reason cov95 collapses under
forging. This connects the two contributions on a real LM, with the unified ResidualVM + loaded SAEs (GPT-2 jbloom,
Gemma Scope): **force the residual through the SAE feature basis** (decode∘encode = the forge bottleneck) at a
layer, and measure the damage to

  COMPOSITION  — induction-NLL (the in-context-copy that needs prev-token → induction *composition*), and
  READOUT      — generic next-token NLL (the prediction the SAE features are trained to carry),

each as a *relative* increase over its own clean baseline. The forge tax = **composition is taxed more than
readout** (rel_induction > rel_generic): the feature basis preserves what it reads out but not what it composes —
the decompilation ceiling and the cov95 forge tax, the same phenomenon, measured from the disassembly side.

Per SAE layer + the full stack. GPT-2 (all 12 resid_pre SAEs) + Gemma-2-2B (the Gemma-Scope layers).

Output: runs/disassembly/sae_forge_tax_summary.json (merge-safe). Findings -> docs/FINDINGS.md + DECOMPILATION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SAE_LAYERS = {"gpt2": None,                                                       # None → all layers (jbloom resid_pre)
              "google/gemma-2-2b": [0, 3, 6, 9, 12, 18, 21, 24]}                  # Gemma-Scope resid layers


def run_model(mid, args):
    import urllib.request
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); tok = vm.tok; nL = vm.nL
    rng = np.random.default_rng(args.seed)
    layers = SAE_LAYERS.get(mid)
    if layers is None:
        layers = list(range(nL))
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + 64] for i in range(0, len(pids), 64) if len(pids[i:i + 64]) >= 8][: args.chunks]
    if vm.is_gpt2:
        from collections import Counter
        vocab = [t for t, _ in Counter(t for c in chunks for t in c).most_common(400)]
        rep = lambda Ln: [int(vocab[i]) for i in rng.integers(0, len(vocab), Ln)]   # noqa: E731
    else:
        V = vm.model.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
        rep = lambda Ln: [int(x) for x in rng.integers(lo, hi, Ln)]                  # noqa: E731
    rep_seqs = [(lambda s: s + s)(rep(22)) for _ in range(args.probes)]

    def ind_target(s):
        Lh = len(s) // 2
        return [(p, s[p + 1]) for p in range(Lh, 2 * Lh - 1)]

    def gen_target(s):
        return [(i, s[i + 1]) for i in range(len(s) - 1)]

    base_ind = vm.nll(rep_seqs, ind_target); base_gen = vm.nll(chunks, gen_target)
    rows = []
    for L in layers:
        try:
            with vm.sae_bottleneck([L]):
                ind = vm.nll(rep_seqs, ind_target); gen = vm.nll(chunks, gen_target)
        except Exception as e:  # pragma: no cover
            print(f"    [layer {L} skip] {e}"); continue
        rel_ind = (ind - base_ind) / (base_ind + 1e-9); rel_gen = (gen - base_gen) / (base_gen + 1e-9)
        rows.append({"layer": L, "d_induction": ind - base_ind, "d_generic": gen - base_gen,
                     "rel_induction": rel_ind, "rel_generic": rel_gen, "tax": rel_ind - rel_gen})
    # the full forge: all SAE layers bottlenecked at once
    try:
        with vm.sae_bottleneck(layers):
            f_ind = vm.nll(rep_seqs, ind_target); f_gen = vm.nll(chunks, gen_target)
        full = {"rel_induction": (f_ind - base_ind) / (base_ind + 1e-9), "rel_generic": (f_gen - base_gen) / (base_gen + 1e-9)}
        full["tax"] = full["rel_induction"] - full["rel_generic"]
    except Exception as e:  # pragma: no cover
        full = {"error": str(e)}
    mt = float(np.mean([r["tax"] for r in rows])) if rows else None
    comp_more = sum(r["tax"] > 0 for r in rows)
    return {"model": mid.split("/")[-1], "rope": not vm.is_gpt2, "n_layers": nL, "n_sae_layers": len(rows),
            "base_induction_nll": base_ind, "base_generic_nll": base_gen,
            "mean_layer_tax": mt, "layers_composition_taxed_more": f"{comp_more}/{len(rows)}",
            "full_forge": full, "per_layer": rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,google/gemma-2-2b")
    p.add_argument("--chunks", type=int, default=12)
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
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
            ff = r["full_forge"]
            print(f"  base induction-NLL {r['base_induction_nll']:.2f}, generic-NLL {r['base_generic_nll']:.2f} | "
                  f"mean per-layer tax (rel_ind − rel_gen) {r['mean_layer_tax']:+.2f} | "
                  f"composition taxed more in {r['layers_composition_taxed_more']} layers")
            if "rel_induction" in ff:
                print(f"  FULL forge (all SAE layers): induction +{ff['rel_induction']:.0%} vs generic "
                      f"+{ff['rel_generic']:.0%} → composition-tax {ff['tax']:+.0%}")
        except Exception as e:  # pragma: no cover
            import traceback; traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "sae_forge_tax_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "SAE-feature-basis forge tax via reconstruction — does the feature bottleneck tax COMPOSITION more than READOUT? (A↔B)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'per_layer' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
