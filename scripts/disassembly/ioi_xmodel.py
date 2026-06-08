"""Cross-model IOI dossier — is the name-mover / negative-mover structure architecture-invariant? (closes a gap)

The circuit catalog's IOI page is GPT-2-only ("no published head-set off GPT-2"). But the **behaviour** — indirect-
object identification — exists in every model, and the unified ResidualVM can now *find* the circuit's heads
behaviourally instead of relying on a literature head-set. This builds the IOI circuit's cross-model dossier the
same way `operator_dossier_xmodel.py` did for the universal ops:

  TASK     — templated IOI prompts ("When{IO} and{S} went..., {S} gave a drink to" -> IO), logit-diff = logit(IO) −
             logit(S) at the end position; single-token names filtered per tokenizer.
  NAME-MOVERS — attribution sweep: ablate each head alone, rank by how much it RAISES (S−IO) — the heads that write
             the IO name. Their copy-attention (END -> IO token) confirms the move.
  NEGATIVE-MOVERS — the other end of the same sweep: heads whose ablation INCREASES the logit-diff (they write
             AGAINST the IO = copy-suppression), the canonical IOI negative/backup structure.
  NECESSITY — ablate the top name-movers together -> logit-diff collapse (is the circuit load-bearing?).
  DUPLICATE — is there a duplicate-token head (S2 -> S1) feeding the circuit (the chain's initiator)?

Output: runs/disassembly/circuits/ioi_xmodel_summary.json (merge-safe). Findings -> docs/circuits + FINDINGS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

NAMES = ["John", "Mary", "Tom", "James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
         "Mark", "Paul", "Anna", "Sarah", "Laura", "Peter", "Susan", "Karen", "Linda", "Nancy",
         "George", "Edward", "Henry", "Frank", "Carol", "Helen", "Alice", "Emma", "Jack", "Kate"]
TEMPLATE = "When{A} and{B} went to the store,{C} gave a drink to"


def single_token_names(tok):
    """Names that encode to exactly one token with a leading space (so logit-diff is a single-token contrast)."""
    out = []
    for n in NAMES:
        ids = tok(" " + n, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            out.append((n, ids[0]))
    return out


def ioi_one_model(vm, model_id, args):
    rng = np.random.default_rng(args.seed)
    names = single_token_names(vm.tok)
    if len(names) < 4:
        return {"model": model_id.split("/")[-1], "note": f"only {len(names)} single-token names — IOI skipped"}
    prompts = []                                                            # (ids, io_tok, s_tok, io_pos)
    for _ in range(args.prompts):
        i, j = (int(x) for x in rng.choice(len(names), 2, replace=False))
        (ion, iot), (sn, st) = names[i], names[j]
        for io_first in (True, False):                                      # ABB / BAB symmetry
            A, B = (ion, sn) if io_first else (sn, ion)
            text = TEMPLATE.format(A=" " + A, B=" " + B, C=" " + sn)
            ids = vm.tok(text, add_special_tokens=not vm.is_gpt2)["input_ids"]
            io_pos = next((k for k, t in enumerate(ids) if t == iot), None)
            prompts.append((ids, iot, st, io_pos))
    H = vm.H; nL = vm.nL
    vm.fit_means([ids for ids, *_ in prompts])                             # mean-ablation reference = the IOI distribution

    def logit_diff():                                                       # mean logit(IO) − logit(S) at end
        tot = 0.0
        for ids, iot, st, _ in prompts:
            lg = vm.logits(ids)[-1].float(); tot += float(lg[iot] - lg[st])
        return tot / len(prompts)

    def neg_ld(v):                                                          # metric for attribution: S − IO
        tot = 0.0
        for ids, iot, st, _ in prompts:
            lg = v.logits(ids)[-1].float(); tot += float(lg[st] - lg[iot])
        return tot / len(prompts)

    base_ld = logit_diff()
    # NAME-MOVERS by END->IO copy-attention (the DIRECT move). Marginal ablation under-ranks them — the backup
    # name-movers self-repair — so attention, not ablation, is the clean name-mover signal.
    copy = np.zeros(nL * H); n = 0
    for ids, iot, st, io_pos in prompts[: args.attn_probes]:
        if io_pos is None:
            continue
        att = vm.attn(ids); n += 1
        for L in range(nL):
            copy[L * H:(L + 1) * H] += att[L][:, -1, io_pos].float().cpu().numpy()
    copy /= max(n, 1)
    # the ablation sweep (Δ(S−IO)) splits the END->IO movers by SIGN: ablating a positive name-mover DROPS the
    # logit-diff (d_logit_diff<0); ablating a negative/suppression mover RAISES it (d_logit_diff>0).
    ranked = vm.attribution(neg_ld, kind="heads")
    dld = {hh: -d for hh, d in ranked}                                      # d_logit_diff per head (ablated − base)
    load_bearing = ranked[: args.top]                                       # most logit-diff-load-bearing (S-inhibition mix)
    cand = [(i // H, i % H) for i in np.argsort(-copy)[: args.top * 4]]     # the END->IO movers (both signs)
    name_movers = sorted([(hh, copy[hh[0] * H + hh[1]]) for hh in cand if dld.get(hh, 0) < 0],
                         key=lambda m: -m[1])[: args.top]                   # copy + ablation drops ld -> positive mover
    neg_movers = sorted([(hh, copy[hh[0] * H + hh[1]], dld.get(hh, 0)) for hh in cand if dld.get(hh, 0) > 0],
                        key=lambda m: -m[1])[: args.top]                    # copy + ablation raises ld -> suppressor

    nm_heads = [hh for hh, _ in name_movers]
    with vm.ablate_heads(nm_heads):
        ld_ablate_nm = logit_diff()
    # duplicate-token head: attends the subject's 2nd mention back to its 1st (the chain initiator)
    dmass = np.zeros(nL * H); cnt = 0
    for ids, iot, st, _ in prompts[: args.attn_probes]:
        occ = [k for k, t in enumerate(ids) if t == st]
        if len(occ) < 2:
            continue
        s1, s2 = occ[0], occ[-1]; cnt += 1; att = vm.attn(ids)
        for L in range(nL):
            dmass[L * H:(L + 1) * H] += att[L][:, s2, s1].float().cpu().numpy()
    dup = None
    if cnt:
        dmass /= cnt; dtop = int(dmass.argmax()); dup = {"head": f"{dtop // H}.{dtop % H}", "s2_to_s1_attn": float(dmass[dtop])}

    return {"model": model_id.split("/")[-1], "rope": not vm.is_gpt2, "n_names": len(names),
            "baseline_logit_diff": base_ld, "n_prompts": len(prompts),
            "name_movers": [{"head": f"{hh[0]}.{hh[1]}", "copy_attn_to_io": ca} for hh, ca in name_movers],
            "load_bearing_heads": [{"head": f"{hh[0]}.{hh[1]}", "d_logit_diff": -d} for hh, d in load_bearing],
            "negative_movers": [{"head": f"{hh[0]}.{hh[1]}", "copy_attn_to_io": ca, "d_logit_diff": dl} for hh, ca, dl in neg_movers],
            "necessity_ablate_namemovers": {"base": base_ld, "ablated": ld_ablate_nm, "delta": ld_ablate_nm - base_ld,
                                            "frac_collapse": (base_ld - ld_ablate_nm) / (abs(base_ld) + 1e-9)},
            "duplicate_head": dup}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--prompts", type=int, default=24, help="name pairs (×2 orders each)")
    p.add_argument("--top", type=int, default=4)
    p.add_argument("--attn-probes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/circuits"))
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
            r = ioi_one_model(vm, mid, args)
            results.append(r)
            if "name_movers" in r:
                nm = ", ".join(f"{m['head']}({m['copy_attn_to_io']:.2f})" for m in r["name_movers"][:4])
                lb = ", ".join(m["head"] for m in r["load_bearing_heads"][:3])
                neg = ", ".join(m["head"] for m in r["negative_movers"][:3]) or "—"
                nec = r["necessity_ablate_namemovers"]
                print(f"  IOI logit-diff {r['baseline_logit_diff']:+.2f} | name-movers(copy→IO) {nm} | "
                      f"load-bearing {lb} | neg-movers {neg} | ablate-NM Δ {nec['delta']:+.2f} ({nec['frac_collapse']:+.0%}) | "
                      f"dup {r['duplicate_head']['head'] if r['duplicate_head'] else '—'}")
            else:
                print(f"  {r.get('note')}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})
        finally:
            vm = None
            if dev == "cuda":
                torch.cuda.empty_cache()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "ioi_xmodel_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "cross-model IOI dossier (name-movers / negative-movers / necessity / duplicate, on ResidualVM)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'name_movers' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
