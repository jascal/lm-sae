"""Is there a fact-ADDRESSABLE site? — sweep edit-efficacy vs entity-leakage by layer (decompiler #2, follow-up).

`fact_edit_xmodel.py` found the early-MLP subject store is an *entity* address, not a *fact* address: editing the
capital there also flips the language 56–100% (entity-leakage). The open question: is there a *later* layer where the
edit is still efficacious (capital flips) but does NOT leak (language stays) — a genuine fact-addressable "row"?

For each model, for each edit layer L, patch the subject-token MLP store at L with the donor's and measure BOTH:
  efficacy(L)  — does the capital flip S→D? (the edit lands)
  leakage(L)   — does the *language* also flip S→D? (the edit dragged the whole entity)
A fact-addressable site = a layer with efficacy HIGH and leakage LOW (efficacy − leakage maximal). If no such layer
exists (leakage tracks efficacy at every depth), the store is entity-addressable all the way down — a hard limit on
surgical single-fact editing without weight surgery. Reuses fact_edit_xmodel's facts/templates/helpers.

Output: runs/disassembly/fact_site_sweep_summary.json (merge-safe). Findings -> docs/FINDINGS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fact_edit_xmodel import FACTS, LANG, LANG_TEMPLATE, TEMPLATE, single_tok, subj_pos  # noqa: E402


def sweep_one_model(vm, model_id, args):
    import contextlib
    tok = vm.tok; t = vm.torch; nL = vm.nL
    facts = []
    for country, cap in FACTS:
        sid = single_tok(tok, country, vm.is_gpt2); cid = single_tok(tok, cap, vm.is_gpt2)
        if sid is not None and cid is not None:
            facts.append((country, sid, cap, cid))
    if len(facts) < 4:
        return {"model": model_id.split("/")[-1], "note": f"only {len(facts)} single-token facts — skipped"}
    rng = np.random.default_rng(args.seed)

    def ids_of(template, country):
        return tok(template.format(S=country), add_special_tokens=not vm.is_gpt2)["input_ids"]

    def lp(ids, patches):
        with contextlib.ExitStack() as st:
            for (L, donor, pos) in patches:
                st.enter_context(vm.patch_mlp(L, donor, pos))
            return t.log_softmax(vm.logits(ids).float(), -1)[-1]

    def place(value, seqlen, pos):
        z = t.zeros((1, seqlen, vm.d), dtype=value.dtype, device=value.device); z[0, pos] = value; return z

    pairs = []
    for i, S in enumerate(facts):
        D = facts[(i + 1 + int(rng.integers(0, len(facts) - 1))) % len(facts)]
        Sl = single_tok(tok, LANG.get(S[0], ""), vm.is_gpt2); Dl = single_tok(tok, LANG.get(D[0], ""), vm.is_gpt2)
        if D[1] != S[1] and Sl is not None and Dl is not None and Sl != Dl:
            pairs.append((S, D, Sl, Dl))
    pairs = pairs[: args.pairs]
    if not pairs:
        return {"model": model_id.split("/")[-1], "note": "no usable pairs (need single-token capital + language)"}

    eff = np.zeros(nL); leak = np.zeros(nL); npairs = 0
    for (S, D, Sl, Dl) in pairs:
        Sc, Sid, _, Scid = S; Dc, Did, _, Dcid = D
        ids_S = ids_of(TEMPLATE, Sc); ids_D = ids_of(TEMPLATE, Dc)
        pS = subj_pos(ids_S, Sid); pD = subj_pos(ids_D, Did)
        idsL_S = ids_of(LANG_TEMPLATE, Sc); pLS = subj_pos(idsL_S, Sid)
        if pS is None or pD is None or pS != pD or pLS is None:
            continue
        D_mlp = vm.trace(ids_D)["mlp"]; npairs += 1
        for L in range(nL):
            oc = lp(ids_S, [(L, D_mlp[L], pS)])
            if float(oc[Dcid] - oc[Scid]) > 0:
                eff[L] += 1
            ol = lp(idsL_S, [(L, place(D_mlp[L][0, pD], len(idsL_S), pLS), pLS)])
            if float(ol[Dl] - ol[Sl]) > 0:
                leak[L] += 1
    if npairs == 0:
        return {"model": model_id.split("/")[-1], "note": "no aligned pairs"}
    eff /= npairs; leak /= npairs
    sep = eff - leak                                                     # fact-specificity margin per layer
    bL = int(np.argmax(sep))
    return {"model": model_id.split("/")[-1], "rope": not vm.is_gpt2, "n_layers": nL, "n_pairs": npairs,
            "efficacy_by_layer": [float(x) for x in eff], "leakage_by_layer": [float(x) for x in leak],
            "best_factspecific_layer": bL, "best_layer_frac": bL / max(nL - 1, 1),
            "best_efficacy": float(eff[bL]), "best_leakage": float(leak[bL]), "best_separation": float(sep[bL]),
            "max_efficacy": float(eff.max()), "leakage_at_max_efficacy": float(leak[int(np.argmax(eff))])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
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
            r = sweep_one_model(vm, mid, args)
            results.append(r)
            if "efficacy_by_layer" in r:
                print(f"  {r['n_pairs']} pairs | max-efficacy {r['max_efficacy']:.0%} (leakage there {r['leakage_at_max_efficacy']:.0%}) | "
                      f"best fact-specific layer L{r['best_factspecific_layer']}/{r['n_layers']} ({r['best_layer_frac']:.0%}): "
                      f"eff {r['best_efficacy']:.0%} − leak {r['best_leakage']:.0%} = sep {r['best_separation']:+.0%}")
            else:
                print(f"  {r.get('note')}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})
        finally:
            vm = None
            if dev == "cuda":
                torch.cuda.empty_cache()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "fact_site_sweep_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "fact-addressable site search: edit efficacy vs entity-leakage by layer (decompiler)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'efficacy_by_layer' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
