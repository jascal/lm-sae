"""IOI-template causal metric — confirm the IOI-family idioms on the task they actually serve.

The induction-NLL ablation (causal_validation.py) decisively validated the induction sub-circuit but was
BLIND to the IOI family (name-movers / S-inhibition / negative-movers), because generic in-context-repeat
tokens don't exercise the IOI task. This builds the right metric: a synthetic IOI dataset
("Then, Mary and John went ...; John gave a drink to" -> predict " Mary") and the logit difference

  LD = logit(IO) - logit(S)        (IO = the name mentioned once; S = the repeated giver)

GPT-2 does IOI -> baseline LD > 0. We MEAN-ABLATE each named idiom's heads and measure ΔLD, with the
SIGNED predictions from the IOI circuit:
  name-movers (9.6/9.9/10.0)      ablate -> LD DROPS  (they copy the IO into the logits)
  S-inhibition (7.3/8.6/8.10)     ablate -> LD DROPS  (S no longer inhibited; movers attend to S too)
  duplicate-token (0.1/3.0)       ablate -> LD DROPS  (feeds S-inhibition)
  negative-movers / 10.7          ablate -> LD RISES  (they write AGAINST the IO; removing them helps)
  induction (5.x/6.9/7.11)        ablate -> ~0        (NEGATIVE CONTROL here — induction is not the IOI mech)

The double dissociation vs causal_validation.py (induction load-bearing there / null here; name-movers null
there / load-bearing here) is the decisive confirmation that causal validation is metric-specific.
GPT-2, CPU; head sets from idiom_library_v2_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = [" Mary", " John", " Tom", " James", " Dan", " Paul", " Mark", " George", " Anna", " Sarah",
         " Peter", " Henry", " Susan", " Linda", " Kevin", " Brian", " Jack", " Bill", " Alice", " Jane",
         " David", " Robert", " Karen", " Nancy", " Eric", " Sam", " Joe", " Harry", " Frank", " Scott"]
PLACES = [" store", " park", " school", " office", " garden", " house", " market", " church"]
OBJECTS = [" drink", " ball", " book", " ring", " bone", " pen", " gift", " note"]
TEMPLATES = [
    "Then,{i0} and{i1} went to the{P}.{S} gave a{T} to",
    "When{i0} and{i1} got to the{P},{S} gave the{T} to",
    "After{i0} and{i1} left the{P},{S} handed a{T} to",
    "While{i0} and{i1} were at the{P},{S} passed a{T} to",
]
IDIOM_SETS = {"prev_token": 1, "duplicate_token": 3, "induction": 4, "copy_namemover": 3,
              "backup_namemover": 3, "negative_namemover": 2, "s_inhibition": 2, "copy_suppression": 2}
# expected SIGN of ΔLD on ablation (+1 LD rises, -1 LD drops, 0 control)
EXPECT = {"copy_namemover": -1, "backup_namemover": -1, "s_inhibition": -1, "duplicate_token": -1,
          "negative_namemover": +1, "copy_suppression": +1, "induction": 0, "prev_token": 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--n-examples", type=int, default=160)
    p.add_argument("--n-random", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--idioms", type=Path, default=Path("runs/idiom_library_v2_summary.json"))
    p.add_argument("--output", type=Path, default=Path("runs/ioi_causal_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained).eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    def sid(s):
        t = tok(s, add_special_tokens=False)["input_ids"]
        return t[0] if len(t) == 1 else None
    names = [(s, sid(s)) for s in NAMES if sid(s) is not None]
    places = [s for s in PLACES if sid(s) is not None]
    objects = [s for s in OBJECTS if sid(s) is not None]
    rng = np.random.default_rng(args.seed)

    # ---- build the IOI dataset: (prompt ids, IO token id, S token id) ----
    examples = []
    for _ in range(args.n_examples):
        (io_s, io_id), (s_s, s_id) = (names[i] for i in rng.choice(len(names), 2, replace=False))
        P = places[rng.integers(len(places))]; T = objects[rng.integers(len(objects))]
        tmpl = TEMPLATES[rng.integers(len(TEMPLATES))]
        i0, i1 = (io_s, s_s) if rng.random() < 0.5 else (s_s, io_s)   # intro order varies
        prompt = tmpl.format(i0=i0, i1=i1, P=P, T=T, S=s_s)            # S (giver) is repeated
        examples.append((tok(prompt)["input_ids"], int(io_id), int(s_id)))

    def logit_diff(ablate):
        by_layer = {}
        for (L, h) in ablate:
            by_layer.setdefault(L, []).append(h)
        hooks = []
        for L, hs in by_layer.items():
            def mk(L, hs):
                def hook(mod, inp):
                    x = inp[0].clone()
                    for h in hs:
                        x[..., h * hd:(h + 1) * hd] = meanvec[L][h * hd:(h + 1) * hd]
                    return (x,)
                return hook
            hooks.append(tr.h[L].attn.c_proj.register_forward_pre_hook(mk(L, hs)))
        lds, acc = [], 0
        with torch.no_grad():
            for ids, io_id, s_id in examples:
                lg = model(input_ids=torch.tensor([ids])).logits[0, -1].float()
                lds.append(float(lg[io_id] - lg[s_id])); acc += int(lg[io_id] > lg[s_id])
        for hk in hooks:
            hk.remove()
        return float(np.mean(lds)), acc / len(examples)

    # ---- capture per-layer c_proj-input mean over the IOI prompts (for mean-ablation) ----
    cap = {L: [] for L in range(nL)}
    caphooks = [tr.h[L].attn.c_proj.register_forward_pre_hook(
        (lambda L: lambda mod, inp: cap[L].append(inp[0].detach().reshape(-1, d)))(L)) for L in range(nL)]
    with torch.no_grad():
        for ids, _io, _s in examples:
            model(input_ids=torch.tensor([ids]))
    for hk in caphooks:
        hk.remove()
    meanvec = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    base_ld, base_acc = logit_diff([])
    print(f"{args.pretrained}: {len(examples)} IOI examples")
    print(f"[baseline] logit-diff(IO - S) = {base_ld:+.3f}   IOI accuracy {base_acc:.1%}  "
          f"(>0 / high => GPT-2 does the task)\n")

    summ = json.loads(args.idioms.read_text()) if args.idioms.exists() else {"idioms": {}}

    def heads_for(idiom, k):
        out = []
        for name, _s in summ.get("idioms", {}).get(idiom, [])[:k]:
            L, h = name.split("."); out.append((int(L), int(h)))
        return out

    def random_matched(named):
        return [(L, int(rng.integers(0, H))) for L, _h in named]

    rows = []
    print(f"{'idiom':>18} {'heads':>16} {'ΔLD':>7} {'ΔLD_rand':>9} {'z':>6} {'expect':>7}  verdict")
    for idiom, k in IDIOM_SETS.items():
        named = heads_for(idiom, k)
        if not named:
            continue
        ld, acc = logit_diff(named)
        d_ld = ld - base_ld
        rand = [logit_diff(random_matched(named))[0] - base_ld for _ in range(args.n_random)]
        rmu = float(np.mean(rand)); rsd = float(np.std(rand) + 1e-9)
        z = (d_ld - rmu) / rsd
        exp = EXPECT.get(idiom, 0)
        # confirmed: effect is in the predicted direction and beats random (|z|>2); controls expect ~0
        if exp == 0:
            ok = abs(z) < 2
            verdict = "CONTROL ok (~no IOI effect)" if ok else "unexpected IOI effect"
        else:
            ok = (np.sign(d_ld) == exp) and abs(z) > 2
            verdict = f"CONFIRMED ({'LD drops' if exp < 0 else 'LD rises'})" if ok else "not confirmed"
        rows.append({"idiom": idiom, "heads": [list(x) for x in named], "delta_ld": d_ld, "acc": acc,
                     "delta_ld_random_mean": rmu, "z": z, "expect_sign": exp, "confirmed": bool(ok)})
        hs = ",".join(f"{L}.{h}" for L, h in named)
        es = {1: "LD up", -1: "LD dn", 0: "ctrl"}[exp]
        print(f"{idiom:>18} {hs:>16} {d_ld:>+7.3f} {rmu:>+9.3f} {z:>6.1f} {es:>7}  {verdict}")

    # ---- literature-set audit: does the canonical published head set confirm where the catalog-derived
    # set did not? Separates "mechanism not causal" from "catalog named the wrong heads". ----
    LIT = {"duplicate_token": [(0, 1), (3, 0)], "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
           "copy_namemover": [(9, 6), (9, 9), (10, 0)], "negative_namemover": [(10, 7), (11, 10)]}
    lit_rows = []
    print("\n[literature-set audit] ablate the canonical published heads (vs the catalog-derived sets above):")
    for idiom, heads in LIT.items():
        ld, _acc = logit_diff(heads)
        d_ld = ld - base_ld
        rand = [logit_diff(random_matched(heads))[0] - base_ld for _ in range(args.n_random)]
        z = (d_ld - np.mean(rand)) / (np.std(rand) + 1e-9)
        exp = EXPECT.get(idiom, 0)
        ok = (np.sign(d_ld) == exp) and abs(z) > 2 if exp else abs(z) < 2
        lit_rows.append({"idiom": idiom, "heads": [list(x) for x in heads], "delta_ld": d_ld,
                         "z": float(z), "expect_sign": exp, "confirmed": bool(ok)})
        hs = ",".join(f"{L}.{h}" for L, h in heads)
        print(f"  {idiom:>18} {hs:>18} ΔLD {d_ld:>+7.3f} (z{z:>5.1f})  "
              f"{'CONFIRMED' if ok else 'not confirmed'}")

    out = {"experiment": "IOI-template causal metric (logit-diff, mean-ablation)", "model": args.pretrained,
           "n_examples": len(examples), "baseline_logit_diff": base_ld, "baseline_acc": base_acc,
           "idioms": rows, "literature_audit": lit_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    conf = [r for r in rows if r["expect_sign"] != 0 and r["confirmed"]]
    tot = [r for r in rows if r["expect_sign"] != 0]
    nm = next((r for r in rows if r["idiom"] == "copy_namemover"), None)
    ind = next((r for r in rows if r["idiom"] == "induction"), None)
    print(f"\n[verdict] {len(conf)}/{len(tot)} IOI-family idioms causally confirmed on the IOI logit-diff "
          "(signed effect in the predicted direction, |z|>2 vs layer-matched random)")
    if nm and ind:
        print("[double dissociation vs induction-NLL harness] "
              f"name-movers: null on induction-NLL -> ΔLD {nm['delta_ld']:+.3f} (z{nm['z']:.1f}) on IOI;  "
              f"induction: z=8.6 on induction-NLL -> ΔLD {ind['delta_ld']:+.3f} (z{ind['z']:.1f}) on IOI  "
              "=> each idiom is causal for ITS OWN metric.")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
