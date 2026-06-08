"""Verified knowledge edits cross-model — "write a row to the database" and check the write (decompiler goal #2).

The knowledge arc has READ (causal_tracing: a fact lives in the early-MLP store at the subject) and a crude WRITE
(fact_patching: transplant donor D's store into subject S's run → S answers D's fact). This adds the two things a
*verified* database write needs, cross-model, on the ResidualVM (patch_mlp + trace + logits):

  EFFICACY      — patch S's subject-token MLP store with D's, sweep the layer; at the best layer does the capital
                  prompt flip S→D? (logit(D_capital) − logit(S_capital) crosses 0). Re-finds the store layer AND edits.
  GENERALIZATION — does the same store-edit hold under a PARAPHRASE prompt (different template, same subject)?
  LOCALIZATION  — patch a NON-subject token instead (the relation/"is" token): the flip should NOT happen, confirming
                  the fact is written at the subject store, not smeared across the prompt.

So each model gets edit efficacy + generalization + a localization control — the verified write that turns
"transplant works" into "the fact is an editable, localized row." Subjects/capitals filtered to single tokens per
tokenizer so the subject position aligns between S and D (the patch is one activation at one position).

Output: runs/disassembly/fact_edit_xmodel_summary.json (merge-safe). Findings -> docs/circuits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# (country, capital) — common, mostly single-token; filtered per tokenizer below.
FACTS = [("France", "Paris"), ("Italy", "Rome"), ("Japan", "Tokyo"), ("Spain", "Madrid"), ("Germany", "Berlin"),
         ("Russia", "Moscow"), ("Egypt", "Cairo"), ("Greece", "Athens"), ("Poland", "Warsaw"), ("Cuba", "Havana"),
         ("Norway", "Oslo"), ("Iraq", "Baghdad"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Chile", "Santiago")]
# country -> its language (a SECOND relation, for the entity-leakage / fact-specificity test); filtered to single-token.
LANG = {"France": "French", "Italy": "Italian", "Japan": "Japanese", "Spain": "Spanish", "Germany": "German",
        "Russia": "Russian", "Greece": "Greek", "Poland": "Polish", "Norway": "Norwegian"}
TEMPLATE = "The capital of {S} is"
PARAPHRASE = "The capital city of {S} is called"
LANG_TEMPLATE = "The language of {S} is"


def single_tok(tok, word, is_gpt2):
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def subj_pos(ids, subj_id):
    return next((i for i, t in enumerate(ids) if t == subj_id), None)


def fact_edit_one_model(vm, model_id, args):
    tok = vm.tok; t = vm.torch
    facts = []
    for country, cap in FACTS:
        sid = single_tok(tok, country, vm.is_gpt2); cid = single_tok(tok, cap, vm.is_gpt2)
        if sid is not None and cid is not None:
            facts.append((country, sid, cap, cid))
    if len(facts) < 4:
        return {"model": model_id.split("/")[-1], "note": f"only {len(facts)} single-token facts — skipped"}
    rng = np.random.default_rng(args.seed)
    nL = vm.nL

    def ids_of(template, country):
        return tok(template.format(S=country), add_special_tokens=not vm.is_gpt2)["input_ids"]

    import contextlib

    def lp(ids, patches=()):
        with contextlib.ExitStack() as st:
            for (L, donor, pos) in patches:
                st.enter_context(vm.patch_mlp(L, donor, pos))
            return t.log_softmax(vm.logits(ids).float(), -1)[-1]
    band = list(range(max(2, nL // 5)))                                  # early-MLP store band (matches fact_patching)

    # pairs: each subject S edited toward a random different donor D
    pairs = []
    for i, S in enumerate(facts):
        D = facts[(i + 1 + int(rng.integers(0, len(facts) - 1))) % len(facts)]
        if D[1] != S[1]:
            pairs.append((S, D))
    pairs = pairs[: args.pairs]

    def place(value, seqlen, pos):                                       # donor tensor with `value` at `pos`
        z = t.zeros((1, seqlen, vm.d), dtype=value.dtype, device=value.device); z[0, pos] = value; return z

    eff_layers = []; gen_flips = 0; loc_flips = 0; band_flips = 0; single_flips = 0; n = 0; best_layers = []
    leak_flips = 0; leak_n = 0                                            # entity-leakage (does language flip too?)
    for (S, D) in pairs:
        Sc, Sid, Scap, Scid = S; Dc, Did, Dcap, Dcid = D
        ids_S = ids_of(TEMPLATE, Sc); ids_D = ids_of(TEMPLATE, Dc)
        pS = subj_pos(ids_S, Sid); pD = subj_pos(ids_D, Did)
        if pS is None or pD is None or pS != pD:                         # need aligned subject positions
            continue
        D_mlp = vm.trace(ids_D)["mlp"]                                   # per-layer donor MLP outputs
        n += 1
        # EFFICACY (primary): patch the early-MLP BAND at the subject store (matches fact_patching)
        ob = lp(ids_S, [(L, D_mlp[L], pS) for L in band])
        if float(ob[Dcid] - ob[Scid]) > 0:
            band_flips += 1
        # CONCENTRATION: does a SINGLE best layer already flip it? (high = concentrated store, GPT-2-like)
        best = (-1, -1e9)
        for L in range(nL):
            o = lp(ids_S, [(L, D_mlp[L], pS)])
            gap = float(o[Dcid] - o[Scid])
            if gap > best[1]:
                best = (L, gap)
        bL, bgap = best; best_layers.append(bL); eff_layers.append(bL / max(nL - 1, 1))
        if bgap > 0:
            single_flips += 1
        # GENERALIZATION: band edit on a PARAPHRASE (recapture donor store for the paraphrase template)
        idsP_S = ids_of(PARAPHRASE, Sc); idsP_D = ids_of(PARAPHRASE, Dc)
        pPS = subj_pos(idsP_S, Sid); pPD = subj_pos(idsP_D, Did)
        if pPS is not None and pPD is not None and pPS == pPD:
            DP_mlp = vm.trace(idsP_D)["mlp"]
            outP = lp(idsP_S, [(L, DP_mlp[L], pPS) for L in band])
            if float(outP[Dcid] - outP[Scid]) > 0:
                gen_flips += 1
        # LOCALIZATION control: band-patch a NON-subject RELATION token (the token before the subject, e.g. "of") —
        # NOT the last/prediction token (which trivially carries the answer). Should NOT flip if the fact is at the subject.
        cpos = max(0, pS - 1)
        outL = lp(ids_S, [(L, D_mlp[L], cpos) for L in band])
        if float(outL[Dcid] - outL[Scid]) > 0:
            loc_flips += 1
        # ENTITY-LEAKAGE: with the SAME subject-store band, does S's LANGUAGE prompt also flip to D's language?
        # flip = entity-swap (the whole row moved); no flip = the capital is a fact-specific row.
        Slang = single_tok(tok, LANG.get(Sc, ""), vm.is_gpt2); Dlang = single_tok(tok, LANG.get(Dc, ""), vm.is_gpt2)
        if Slang is not None and Dlang is not None and Slang != Dlang:
            idsL_S = ids_of(LANG_TEMPLATE, Sc); pLS = subj_pos(idsL_S, Sid)
            if pLS is not None:
                outLk = lp(idsL_S, [(L, place(D_mlp[L][0, pD], len(idsL_S), pLS), pLS) for L in band])
                leak_n += 1
                if float(outLk[Dlang] - outLk[Slang]) > 0:
                    leak_flips += 1

    if n == 0:
        return {"model": model_id.split("/")[-1], "note": "no aligned-position pairs"}
    return {"model": model_id.split("/")[-1], "rope": not vm.is_gpt2, "n_layers": nL, "n_pairs": n,
            "n_facts": len(facts), "band_layers": len(band),
            "efficacy_band_flip_rate": band_flips / n, "single_layer_flip_rate": single_flips / n,
            "generalization_flip_rate": gen_flips / n, "localization_control_flip_rate": loc_flips / n,
            "mean_single_edit_layer_frac": float(np.mean(eff_layers)), "median_single_edit_layer": int(np.median(best_layers)),
            "entity_leakage_rate": (leak_flips / leak_n) if leak_n else None, "n_leakage": leak_n}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--pairs", type=int, default=10)
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
            r = fact_edit_one_model(vm, mid, args)
            results.append(r)
            if "efficacy_band_flip_rate" in r:
                lk = f"{r['entity_leakage_rate']:.0%}" if r["entity_leakage_rate"] is not None else "n/a"
                print(f"  {r['n_pairs']} edits | efficacy(band) {r['efficacy_band_flip_rate']:.0%} | single-layer "
                      f"{r['single_layer_flip_rate']:.0%} | generalization {r['generalization_flip_rate']:.0%} | "
                      f"localization-ctrl {r['localization_control_flip_rate']:.0%} | entity-leakage {lk} | "
                      f"single-edit layer ~{r['median_single_edit_layer']}/{r['n_layers']}")
            else:
                print(f"  {r.get('note')}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})
        finally:
            vm = None
            if dev == "cuda":
                torch.cuda.empty_cache()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "fact_edit_xmodel_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "verified cross-model knowledge edits (efficacy / generalization / localization, on ResidualVM)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'efficacy_band_flip_rate' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
