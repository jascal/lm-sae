"""Context vs memory — when an in-context (false) fact contradicts the stored fact, which wins? (cross-model)

This synthesizes the two main behaviour threads: **induction** (in-context copy) and **factual recall** (in-weights
memory). Prompt: "The capital of France is **Berlin**. The capital of France is ___" — the in-context answer
(Berlin) contradicts the model's stored fact (Paris). We measure which the model predicts:

    margin = logit(in-context answer) − logit(stored answer)        (>0 → context wins, <0 → memory wins)

Then the causal link: **mean-ablate the model's induction heads** (the in-context-copy mechanism, from the
cross-model dossier) and re-measure — if induction is what makes context win, ablating it should swing the margin
toward memory. Arch-generic; induction heads read from `xmodel_dossiers_summary.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
from circuit_content_patch import _arch  # noqa: E402

FACTS = [
    ("France", " Paris"), ("Japan", " Tokyo"), ("Italy", " Rome"), ("Russia", " Moscow"),
    ("China", " Beijing"), ("Egypt", " Cairo"), ("Spain", " Madrid"), ("Germany", " Berlin"),
    ("Canada", " Ottawa"), ("Greece", " Athens"), ("Cuba", " Havana"), ("Peru", " Lima"),
    ("Iran", " Tehran"), ("Austria", " Vienna"), ("Poland", " Warsaw"), ("Norway", " Oslo"),
]
TEMPLATE = "The capital of {subj} is{false}. The capital of {subj} is"


def build_prompt(subj, false_obj, reps):
    return " ".join([f"The capital of {subj} is{false_obj}."] * reps) + f" The capital of {subj} is"


def run_model(model_id, ind_heads, args, dev):
    import torch
    is_gpt2 = "gpt2" in model_id.lower()
    from transformers import AutoTokenizer
    if is_gpt2:
        from transformers import GPT2LMHeadModel
        m = GPT2LMHeadModel.from_pretrained(model_id, attn_implementation="eager").eval().to(dev)
    else:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    a = _arch(m); H = a["H"]; hd = a["hd"]; nL = m.config.num_hidden_layers; oproj = a["oproj"]
    tok = AutoTokenizer.from_pretrained(model_id)

    objs = {subj: tok(obj, add_special_tokens=False)["input_ids"] for subj, obj in FACTS}

    def build_items(reps):
        out = []
        for subj, obj in FACTS:
            if len(objs[subj]) != 1:
                continue
            false_subj, false_obj = next((s, o) for s, o in FACTS if s != subj and len(objs[s]) == 1)
            ids = tok(build_prompt(subj, false_obj, reps), add_special_tokens=not is_gpt2)["input_ids"]
            out.append((ids, objs[subj][0], objs[false_subj][0]))
        return out
    items = build_items(1)
    if len(items) < 6:
        raise RuntimeError(f"only {len(items)} usable items for {model_id}")

    # corpus mean for ablation
    cap = {L: [] for L in range(nL)}
    hk = [oproj[L].register_forward_pre_hook((lambda L: lambda mod, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for ids, _, _ in items:
            m(input_ids=torch.tensor([ids], device=dev))
    for h in hk:
        h.remove()
    meanv = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    def ablate(heads):
        by = {}
        for (L, h) in heads:
            by.setdefault(L, []).append(h)
        hs = []
        for L, hss in by.items():
            def mk(L, hss):
                def hook(mod, inp):
                    x = inp[0].clone()
                    for h in hss:
                        x[..., h * hd:(h + 1) * hd] = meanv[L][h * hd:(h + 1) * hd].to(x.dtype)
                    return (x,)
                return hook
            hs.append(oproj[L].register_forward_pre_hook(mk(L, hss)))
        return hs

    def margins(items_, heads=()):
        hs = ablate(heads); ctx_win = 0; tot = 0; msum = 0.0
        try:
            with torch.no_grad():
                for ids, true_t, false_t in items_:
                    lg = m(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()
                    margin = float(lg[false_t] - lg[true_t])                     # context(false) − memory(true)
                    msum += margin; tot += 1
                    if margin > 0:
                        ctx_win += 1
        finally:
            for x in hs:
                x.remove()
        return msum / max(tot, 1), ctx_win / max(tot, 1)

    base_margin, base_ctxwin = margins(items)
    ab_margin, ab_ctxwin = margins(items, ind_heads)
    rng = np.random.default_rng(args.seed)                                       # random-head control of same size
    rk = [(int(i) // H, int(i) % H) for i in rng.choice(nL * H, len(ind_heads), replace=False)]
    rnd_margin, rnd_ctxwin = margins(items, rk)
    sweep = {reps: margins(build_items(reps))[1] for reps in args.reps}          # context-win vs # in-context repetitions
    print(f"  {len(items)} items | context-win {base_ctxwin:.0%} (margin {base_margin:+.2f}) | "
          f"−induction → {ab_ctxwin:.0%} ({ab_margin:+.2f}) | −random → {rnd_ctxwin:.0%} ({rnd_margin:+.2f}) | "
          f"reps-sweep {[(r, f'{sweep[r]:.0%}') for r in args.reps]}")
    return {"model": model_id.split("/")[-1], "rope": not is_gpt2, "n_items": len(items), "n_induction_heads": len(ind_heads),
            "context_win_rate": base_ctxwin, "context_margin": base_margin,
            "context_win_minus_induction": ab_ctxwin, "margin_minus_induction": ab_margin,
            "context_win_minus_random": rnd_ctxwin, "margin_minus_random": rnd_margin,
            "reps_sweep": {str(r): sweep[r] for r in args.reps}}


def write_doc(out, docs):
    L = ["---", "title: Context vs memory", "---", "", "# Context vs memory — when an in-context fact contradicts the stored one", "",
         "Synthesizes the two behaviour threads: **induction** (in-context copy) vs **factual recall** (in-weights "
         "memory). Prompt: \"The capital of France is **Berlin**. The capital of France is ___\" — the in-context "
         "answer (Berlin) contradicts the stored fact (Paris). **margin = logit(context) − logit(memory)** (>0 → "
         "context wins). Then the causal link: mean-ablate the model's **induction heads** (the in-context-copy "
         "mechanism) — if induction is what makes context win, ablating it swings the margin back toward memory; a "
         "random same-size head-set is the control.", "",
         "| model | context-win rate (margin) | − induction heads | − random heads |",
         "|---|---|---|---|"]
    for r in out["results"]:
        if "context_win_rate" not in r:
            continue
        L.append(f"| {r['model']} | **{r['context_win_rate']:.0%}** ({r['context_margin']:+.2f}) | "
                 f"{r['context_win_minus_induction']:.0%} ({r['margin_minus_induction']:+.2f}) | "
                 f"{r['context_win_minus_random']:.0%} ({r['margin_minus_random']:+.2f}) |")
    reps = sorted({int(k) for r in out["results"] if r.get("reps_sweep") for k in r["reps_sweep"]})
    if reps:
        L += ["", "## How much context to override memory? (repetition sweep)", "",
              "Context-win rate as the false fact is repeated **N times** in-context before the query — does *more* "
              "in-context evidence eventually flip the memory-dominant models?", "",
              "| model | " + " | ".join(f"×{n}" for n in reps) + " |", "|---|" + "---|" * len(reps)]
        for r in out["results"]:
            if not r.get("reps_sweep"):
                continue
            L.append(f"| {r['model']} | " + " | ".join(f"{r['reps_sweep'].get(str(n), 0):.0%}" for n in reps) + " |")
        L += ["", "_**This reframes the memory-dominance.** At **2 repetitions every model flips to 100% context-win** "
              "— including the one-shot-memory-dominant RoPE models (Gemma 6%→100%, Llama/Qwen 0%→100%). So the "
              "RoPE \"memory-dominance\" is purely a **one-shot** effect: their induction needs **≥2 prior occurrences** "
              "to fire strongly enough to override memory, where GPT-2's fires on a single one. It is induction's "
              "**evidence threshold**, not a refusal to trust context — and once induction fires (≥2 reps) it overrides "
              "stored memory universally._"]
    L += ["", "_**Finding — two regimes.** (1) The **GPT-2 family is context-swayable** (context-win 44–81%) and "
          "**induction is the mechanism**: ablating the induction heads collapses context-win to 0–19% (memory wins), "
          "far more than ablating a random same-size head-set (which leaves it ≈baseline or higher). So induction is "
          "what lets a fresh in-context statement override stored memory. (2) The **RoPE family is memory-dominant**: "
          "Llama and Qwen **ignore the contradicting in-context fact entirely** (0% context-win) and Gemma nearly so "
          "(6%) — on a **single** in-context statement. But the repetition sweep above shows this is a one-shot "
          "effect: at ≥2 repetitions all six flip to 100% context-win. So the difference is induction's **evidence "
          "threshold** (RoPE needs ≥2 occurrences, GPT-2 fires on 1), not a standing refusal to trust context; "
          "induction is the shared override mechanism once it fires. Provisional, ~16 capital facts, single-token answers. Data: "
          "[context_vs_memory_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/context_vs_memory_summary.json). "
          "Regenerate: [context_vs_memory.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/context_vs_memory.py). "
          "See [induction](../operators/induction.md) + [where facts live](factual_recall.md)._"]
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "context_vs_memory.md").write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dossiers", type=Path, default=Path("runs/disassembly/operators/xmodel_dossiers_summary.json"))
    p.add_argument("--model-ids", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3, 5, 8], help="# in-context repetitions to sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/circuits"))
    p.add_argument("--docs", type=Path, default=Path("docs/circuits"))
    args = p.parse_args(argv)
    doss = {r["model"]: r for r in json.loads(args.dossiers.read_text())["results"] if "ops" in r}

    def ind_for(short):
        d = doss.get(short)
        return [tuple(int(x) for x in hh.split(".")) for hh in d["ops"]["induction"]["heads"]] if d else []

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.model_ids.split(",") if m.strip()]:
        short = mid.split("/")[-1]; ind = ind_for(short)
        if not ind:
            print(f"[skip] {short}: no induction heads"); continue
        print(f"\n=== {mid} (induction heads {ind}) ===")
        try:
            results.append(run_model(mid, ind, args, dev))
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": short, "error": str(e)})
        if dev == "cuda":
            torch.cuda.empty_cache()
    out = {"experiment": "context vs memory (in-context fact vs stored fact)", "results": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "context_vs_memory_summary.json").write_text(json.dumps(out, indent=2, default=float))
    write_doc(out, args.docs)
    print(f"\n[done] {len([r for r in results if 'context_win_rate' in r])} models → {args.outdir / 'context_vs_memory_summary.json'}")
    return out


if __name__ == "__main__":
    main()
