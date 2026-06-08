"""Weight/value surgery vs activation patch — is the entity-leakage limit the METHOD or the REPRESENTATION? (#2)

`fact_edit`/`fact_site_sweep` found: editing the early-MLP subject store flips the capital but **leaks to the
entity's other facts** (language flips 56–100%), at every depth. Open question: is that because the activation patch
is *blunt* (it swaps the WHOLE MLP output = the whole entity), or because capital and language are genuinely
entangled in the representation? A ROME-style **targeted** edit answers it: instead of grafting the donor's full MLP
output, **optimize a single edit-value v** (added to the subject's MLP output at one layer) to flip *only* the
capital — then measure whether the *language* still leaks.

  v-opt (ROME)  — v = argmax_v [ logp(D_capital) − logp(S_capital) ] − λ‖v‖²  (a few Adam steps; the model frozen),
                  the smallest targeted residual write that makes the capital flip;
  EFFICACY      — does the capital flip under v?
  LEAKAGE(rome) — does the LANGUAGE also flip under the same v?  (the entity-leakage, now for a *targeted* edit)
  LEAKAGE(patch)— the blunt full-MLP-delta patch's language leakage at the *same* layer, for a head-to-head.

If LEAKAGE(rome) ≪ LEAKAGE(patch): the limit was the METHOD — a targeted edit *is* fact-specific. If they're
comparable: the limit is the REPRESENTATION — capital and language share the subject's write direction, so no edit at
this site is surgical (you'd need a different basis, not a better method).

Output: runs/disassembly/fact_rome_xmodel_summary.json (merge-safe). Findings -> docs/FINDINGS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fact_edit_xmodel import FACTS, LANG, LANG_TEMPLATE, TEMPLATE, single_tok, subj_pos  # noqa: E402


def rome_one_model(vm, model_id, args):
    import torch
    tok = vm.tok; nL = vm.nL; dev = vm.dev
    L = max(1, round(nL * args.layer_frac))                              # the ROME edit layer (mid-early store)
    try:                                                                 # gradient checkpointing: fit backprop on a small GPU
        vm.model.config.use_cache = False; vm.model.gradient_checkpointing_enable()
    except Exception:
        pass
    facts = []
    for country, cap in FACTS:
        sid = single_tok(tok, country, vm.is_gpt2); cid = single_tok(tok, cap, vm.is_gpt2)
        sl = single_tok(tok, LANG.get(country, ""), vm.is_gpt2)
        if sid is not None and cid is not None and sl is not None:
            facts.append((country, sid, cap, cid, sl))
    if len(facts) < 4:
        return {"model": model_id.split("/")[-1], "note": f"only {len(facts)} usable facts — skipped"}
    rng = np.random.default_rng(args.seed)

    def ids_of(template, country):
        return tok(template.format(S=country), add_special_tokens=not vm.is_gpt2)["input_ids"]

    def add_v_logits(ids, pos, v):                                      # forward with v added to MLP-L output at pos (grad-enabled)
        def hook(m, i, o):
            out = (o[0] if isinstance(o, tuple) else o).clone()
            out[0, pos] = out[0, pos] + v.to(out.dtype)
            return (out,) + tuple(o[1:]) if isinstance(o, tuple) else out
        h = vm.mlps[L].register_forward_hook(hook)
        try:
            return vm.model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1]
        finally:
            h.remove()

    band = list(range(max(2, nL // 5)))                                # early-MLP store band (the #113 blunt-patch baseline)

    def band_patch_logits(ids, pos, donor_per_layer, donor_pos):       # graft donor's MLP output across the band at pos
        import contextlib
        with contextlib.ExitStack() as st:
            for Lb in band:
                dv = donor_per_layer[Lb][0, donor_pos]
                z = torch.zeros((1, len(ids), vm.d), dtype=dv.dtype, device=dev); z[0, pos] = dv
                st.enter_context(vm.patch_mlp(Lb, z, pos))
            return torch.log_softmax(vm.model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float(), -1)

    pairs = []
    for i, S in enumerate(facts):
        D = facts[(i + 1 + int(rng.integers(0, len(facts) - 1))) % len(facts)]
        if D[1] != S[1] and D[4] != S[4]:
            pairs.append((S, D))
    pairs = pairs[: args.pairs]

    def essence_dist(country, sid):                                    # original next-token dist on a subject-essence prompt
        ess = ids_of("Today {S} is", country); pe = subj_pos(ess, sid)   # leading-space subject (token matches the fact prompts)
        with torch.no_grad():
            d = torch.log_softmax(vm.model(input_ids=torch.tensor([ess], device=dev)).logits[0, -1].float(), -1).exp()
        return ess, pe, d

    pres_levels = [float(x) for x in args.pres.split(",")]
    frontier = {pr: {"eff": 0, "leak": 0, "vn": []} for pr in pres_levels}
    patch_eff = 0; patch_leak = 0; n = 0
    for (S, D) in pairs:
        Sc, Sid, _, Scid, Slid = S; Dc, Did, _, Dcid, Dlid = D
        ids_S = ids_of(TEMPLATE, Sc); ids_D = ids_of(TEMPLATE, Dc)
        pS = subj_pos(ids_S, Sid); pD = subj_pos(ids_D, Did)
        idsL_S = ids_of(LANG_TEMPLATE, Sc); pLS = subj_pos(idsL_S, Sid)
        if pS is None or pD is None or pS != pD or pLS is None:
            continue
        ess_ids, pess, P_orig = essence_dist(Sc, Sid)
        if pess is None:
            continue
        n += 1
        # ---- ROME efficacy-vs-leakage FRONTIER: trace the targeted edit at each preservation strength ----
        for pr in pres_levels:
            v = torch.zeros(vm.d, device=dev, dtype=torch.float32, requires_grad=True)
            opt = torch.optim.Adam([v], lr=args.lr)
            for _ in range(args.steps):
                opt.zero_grad()
                lp = torch.log_softmax(add_v_logits(ids_S, pS, v).float(), -1)
                lpe = torch.log_softmax(add_v_logits(ess_ids, pess, v).float(), -1)
                kl = (P_orig * (P_orig.add(1e-9).log() - lpe)).sum()    # preserve essence: KL(orig ‖ edited)
                loss = -(lp[Dcid] - lp[Scid]) + pr * kl + args.reg * v.pow(2).sum()
                loss.backward(); opt.step()
            v = v.detach(); frontier[pr]["vn"].append(float(v.norm()))
            with torch.no_grad():
                lpc = torch.log_softmax(add_v_logits(ids_S, pS, v).float(), -1)
                lpl = torch.log_softmax(add_v_logits(idsL_S, pLS, v).float(), -1)
                if float(lpc[Dcid] - lpc[Scid]) > 0:
                    frontier[pr]["eff"] += 1
                if float(lpl[Dlid] - lpl[Slid]) > 0:
                    frontier[pr]["leak"] += 1
        # ---- the blunt entity-swap BAND patch (graft donor's early-MLP store), the #113 baseline reference ----
        with torch.no_grad():
            D_mlp = vm.trace(ids_D)["mlp"]
            cap = band_patch_logits(ids_S, pS, D_mlp, pD)
            if float(cap[Dcid] - cap[Scid]) > 0:
                patch_eff += 1
            lng = band_patch_logits(idsL_S, pLS, D_mlp, pD)
            if float(lng[Dlid] - lng[Slid]) > 0:
                patch_leak += 1
    if n == 0:
        return {"model": model_id.split("/")[-1], "note": "no aligned pairs"}
    return {"model": model_id.split("/")[-1], "rope": not vm.is_gpt2, "n_layers": nL, "edit_layer": L, "n_pairs": n,
            "rome_frontier": [{"pres": pr, "efficacy": f["eff"] / n, "leakage": f["leak"] / n, "v_norm": float(np.mean(f["vn"]))}
                              for pr, f in frontier.items()],
            "patch_efficacy": patch_eff / n, "patch_leakage": patch_leak / n}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--layer-frac", type=float, default=0.2, help="ROME edit layer as a fraction of depth")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--reg", type=float, default=2e-3)
    p.add_argument("--pres", default="2,8,30", help="essence-preservation (KL) weights — traces the efficacy/leakage frontier")
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
            vm = ResidualVM(mid, device=dev, dtype="bf16")               # bf16: fit the backprop graph on a small GPU
            r = rome_one_model(vm, mid, args)
            results.append(r)
            if "rome_frontier" in r:
                fr = " ; ".join(f"pres{p['pres']:g}: eff {p['efficacy']:.0%}/leak {p['leakage']:.0%}" for p in r["rome_frontier"])
                print(f"  {r['n_pairs']} edits @ L{r['edit_layer']} | PATCH(entity) eff {r['patch_efficacy']:.0%}/leak "
                      f"{r['patch_leakage']:.0%} | ROME frontier — {fr}")
            else:
                print(f"  {r.get('note')}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})
        finally:
            vm = None
            if dev == "cuda":
                torch.cuda.empty_cache()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "fact_rome_xmodel_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "targeted (ROME v-opt) vs blunt (full-delta patch) edit — leakage head-to-head, cross-model",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'rome_efficacy' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
