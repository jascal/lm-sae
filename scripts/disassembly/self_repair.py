"""IOI self-repair, made causal — why name-movers read ~0 under mean-ablation (the #33 caveat, quantified).

`instruction_reuse.py` (#33) found name-movers cause ~0 IOI damage under mean-ablation, and flagged the known
self-repair: ablating the primary name-movers wakes the BACKUP name-movers, which compensate so the logit-diff
barely drops. This demonstrates it causally, without direct-logit-attribution: a head class compensates if it is
load-bearing ONLY when the primaries are gone. Measure the IOI logit-diff LD = logit(IO) − logit(S) under:

  full | −primaries | −backups | −(primaries+backups) | −negative-movers

and the KEY contrast — the backups' causal importance WITH vs WITHOUT the primaries present:
  ΔLD(backups | full)              = LD(full) − LD(−backups)            (small: backups idle when primaries present)
  ΔLD(backups | primaries ablated) = LD(−primaries) − LD(−prim+backups) (large: backups carry IOI once primaries go)
self-repair ⇔ the second ≫ the first. GPT-2; literature primary/backup/negative name-mover sets; ioi_causal templates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ioi_causal import NAMES, OBJECTS, PLACES, TEMPLATES  # noqa: E402

PRIMARY = [(9, 6), (9, 9), (10, 0)]                                            # primary name-movers (Wang et al.)
BACKUP = [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2)]        # backup name-movers (self-repair)
NEGATIVE = [(10, 7), (11, 10)]                                                # negative / copy-suppression movers


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--n-ioi", type=int, default=160)
    p.add_argument("--mean-chunks", type=int, default=25)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/self_repair_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/self_repair.png"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    dev = args.device if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(args.pretrained).eval().to(dev)
    tr = model.transformer; cfg = model.config; H = cfg.n_head; hd = cfg.n_embd // H; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    rng = np.random.default_rng(args.seed)
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:120000]
    ids_all = tok(txt)["input_ids"]
    mchunks = [ids_all[i:i + args.ctx] for i in range(0, len(ids_all), args.ctx)
               if len(ids_all[i:i + args.ctx]) >= 8][: args.mean_chunks]

    cap = {L: [] for L in range(nL)}
    hk = [tr.h[L].attn.c_proj.register_forward_pre_hook(
        (lambda L: lambda m, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in mchunks:
            model(input_ids=torch.tensor([c], device=dev))
    for h in hk:
        h.remove()
    meanv = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    def ablate_hooks(ablate):
        by = {}
        for (L, h) in ablate:
            by.setdefault(L, []).append(h)
        hs = []
        for L, hss in by.items():
            def mk(L, hss):
                def hook(m, inp):
                    x = inp[0].clone()
                    for h in hss:
                        x[..., h * hd:(h + 1) * hd] = meanv[L][h * hd:(h + 1) * hd].to(x.dtype)
                    return (x,)
                return hook
            hs.append(tr.h[L].attn.c_proj.register_forward_pre_hook(mk(L, hss)))
        return hs

    def single(strs):
        out = []
        for s in strs:
            i = tok(s, add_special_tokens=False)["input_ids"]
            if len(i) == 1:
                out.append(i[0])
        return out
    names = single(NAMES); places = single(PLACES); objs = single(OBJECTS)
    prompts = []
    for _ in range(args.n_ioi):
        a, b = rng.choice(len(names), 2, replace=False)
        IO = names[a]; S = names[b]; pl = places[int(rng.integers(0, len(places)))]; ob = objs[int(rng.integers(0, len(objs)))]
        tpl = TEMPLATES[int(rng.integers(0, len(TEMPLATES)))]
        text = tpl.format(i0=tok.decode([IO]), i1=tok.decode([S]), P=tok.decode([pl]), S=tok.decode([S]), T=tok.decode([ob]))
        prompts.append((tok(text)["input_ids"], IO, S))

    def ld(ablate):
        hs = ablate_hooks(ablate); vals = []
        try:
            with torch.no_grad():
                for idsq, io, s in prompts:
                    lg = model(input_ids=torch.tensor([idsq], device=dev)).logits[0, -1].float()
                    vals.append(float(lg[io] - lg[s]))
        finally:
            for h in hs:
                h.remove()
        return float(np.mean(vals))

    full = ld(set())
    ld_prim = ld(set(PRIMARY)); ld_back = ld(set(BACKUP)); ld_both = ld(set(PRIMARY) | set(BACKUP)); ld_neg = ld(set(NEGATIVE))
    # the self-repair contrast
    dlb_full = full - ld_back                                                  # backups' effect when primaries present
    dlb_noprim = ld_prim - ld_both                                            # backups' effect when primaries ablated
    print(f"{args.pretrained}: IOI LD baseline {full:+.3f}  ({len(prompts)} prompts)")
    print(f"  −primaries {ld_prim:+.3f} (Δ {full - ld_prim:+.3f})  | −backups {ld_back:+.3f} (Δ {full - ld_back:+.3f}) "
          f"| −both {ld_both:+.3f} (Δ {full - ld_both:+.3f}) | −negatives {ld_neg:+.3f} (Δ {full - ld_neg:+.3f})")
    print(f"  backups' causal LD-effect: WITH primaries {dlb_full:+.3f}  vs  WITHOUT primaries {dlb_noprim:+.3f}  "
          f"({dlb_noprim / max(abs(dlb_full), 1e-9):.1f}x larger once primaries are gone)")

    out = {"experiment": "IOI self-repair (backup name-movers compensate)", "model": args.pretrained, "n_prompts": len(prompts),
           "ld_full": full, "ld_minus_primaries": ld_prim, "ld_minus_backups": ld_back, "ld_minus_both": ld_both,
           "ld_minus_negatives": ld_neg, "primaries": [f"{L}.{h}" for L, h in PRIMARY], "backups": [f"{L}.{h}" for L, h in BACKUP],
           "backups_effect_with_primaries": dlb_full, "backups_effect_without_primaries": dlb_noprim,
           "drop_primaries": full - ld_prim, "drop_both": full - ld_both}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    self_repair = dlb_noprim > 0.1 and dlb_noprim > 3 * max(dlb_full, 0.0) and (full - ld_both) > 1.8 * (full - ld_prim)
    if self_repair:
        verdict = (f"SELF-REPAIR CONFIRMED: ablating the primary name-movers drops IOI LD by just {full - ld_prim:+.3f} "
                   f"({full:+.2f}->{ld_prim:+.2f}) — the circuit looks robust — but ablating primaries AND backups together "
                   f"drops it {full - ld_both:+.3f} ({full:+.2f}->{ld_both:+.2f}). The backups are causally idle when "
                   f"primaries are present (Δ {dlb_full:+.3f}) and carry the IOI logit-diff once primaries are removed (Δ "
                   f"{dlb_noprim:+.3f}, {dlb_noprim / max(abs(dlb_full), 1e-9):.0f}x larger). This is exactly why name-movers read ~0 in #33's "
                   f"mean-ablation matrix — the op IS load-bearing, but a redundant backup pathway masks it under single-class "
                   f"ablation. A clean instance of the program-wide redundancy: the named circuit has hot spares.")
    else:
        verdict = f"self-repair not cleanly isolated (Δbackups: with-prim {dlb_full:+.3f}, no-prim {dlb_noprim:+.3f}) — see table"
    print(f"\n[verdict] {verdict}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axB, axR) = plt.subplots(1, 2, figsize=(12.4, 5.0))
        bars = [("full", full, "#444444"), ("−primaries", ld_prim, "#d62728"), ("−backups", ld_back, "#1f77b4"),
                ("−both", ld_both, "#9467bd"), ("−negatives", ld_neg, "#2ca02c")]
        axB.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], edgecolor="k")
        axB.axhline(0, color="k", lw=0.6, ls=":"); axB.set_xticks(range(len(bars)))
        axB.set_xticklabels([b[0] for b in bars], fontsize=8, rotation=12, ha="right")
        axB.set_ylabel("IOI logit-diff  logit(IO) − logit(S)")
        axB.set_title("−primaries barely drops LD (self-repair); −both collapses it", fontsize=10)
        axR.bar([0, 1], [dlb_full, dlb_noprim], color=["#bbbbbb", "#d62728"], edgecolor="k")
        axR.set_xticks([0, 1]); axR.set_xticklabels(["backups' effect\nWITH primaries", "backups' effect\nWITHOUT primaries"], fontsize=8)
        axR.set_ylabel("Δ IOI logit-diff from ablating backups")
        axR.set_title("backups are hot spares: idle until primaries fail", fontsize=10)
        fig.suptitle("IOI self-repair: backup name-movers mask the primaries under single-class ablation (the #33 caveat)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
