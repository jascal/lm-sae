"""Causal validation of the named idioms — are the heads we NAMED actually load-bearing?

The idiom library names heads by behavioral / weight signatures (correlational). This confirms causation:
MEAN-ABLATE each idiom's named heads (replace the head's contribution to the residual with its corpus
mean — removes its position-varying signal, keeps its average) and measure the damage to the circuit's
own behaviour. The shared metric for the INDUCTION CIRCUIT (prev-token -> duplicate -> induction ->
name-mover, all of which serve in-context copying) is the induction-predictable NLL:

  effect = ΔNLL on induction-predictable tokens  (should rise)   vs
           ΔNLL on the complement                (should barely move — specificity in WHAT breaks)   vs
           ΔNLL from ablating |S| LAYER-MATCHED RANDOM heads      (specificity in WHICH heads matter)

A named set is causally load-bearing if Δind >> Δcomplement AND Δind(named) >> Δind(random). Non-circuit
idioms (copy_suppression, succession) are NEGATIVE CONTROLS: they should NOT specifically damage induction.
Head sets are read from idiom_library_v2_summary.json (the catalog's own outputs). GPT-2, CPU; one forward
pass per ablation config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# idiom -> how many of its top named heads to ablate (from the idiom summary ranking)
IDIOM_SETS = {
    "prev_token": 1, "duplicate_token": 3, "induction": 4, "copy_namemover": 3,
    "s_inhibition": 2, "copy_suppression": 2, "succession": 3,
}
CIRCUIT = {"prev_token", "duplicate_token", "induction", "copy_namemover", "s_inhibition"}


def _induction_predictable(c):
    n = len(c); pred = np.zeros(n, bool)
    for t in range(2, n):
        ps = [p for p in range(t - 1) if c[p] == c[t - 1]]
        if ps and c[ps[-1] + 1] == c[t]:
            pred[t] = True
    return pred


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-random", type=int, default=4, help="layer-matched random control sets per idiom")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--idioms", type=Path, default=Path("runs/disassembly/idiom_library_v2_summary.json"))
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/causal_validation_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained).eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    masks = [_induction_predictable(c)[1:].astype(bool) for c in chunks]
    print(f"{args.pretrained}: {len(chunks)} chunks, induction-rate {np.concatenate(masks).mean():.3f}")

    # ---- pass 0: capture per-layer mean of the c_proj input (concatenated head outputs) ----
    cap = {L: [] for L in range(nL)}
    caphooks = []
    for L in range(nL):
        def mk(L):
            def hook(mod, inp):
                cap[L].append(inp[0].detach().reshape(-1, d))
            return hook
        caphooks.append(tr.h[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    with torch.no_grad():
        for c in chunks:
            model(input_ids=torch.tensor([c]))
    for hk in caphooks:
        hk.remove()
    meanvec = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}   # (d,) per layer

    # ---- mean-ablation runner: replace the named heads' c_proj-input slice with the captured mean ----
    def run(ablate):
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
        ind_s = ind_n = com_s = com_n = 0.0
        with torch.no_grad():
            for c, m in zip(chunks, masks):
                lg = model(input_ids=torch.tensor([c])).logits[0, :-1]
                lp = torch.log_softmax(lg.float(), -1)
                tgt = torch.tensor(c[1:])
                nll = (-lp[torch.arange(len(c) - 1), tgt]).numpy()
                ind_s += nll[m].sum(); ind_n += int(m.sum())
                com_s += nll[~m].sum(); com_n += int((~m).sum())
        for hk in hooks:
            hk.remove()
        return ind_s / max(ind_n, 1), com_s / max(com_n, 1)

    base_ind, base_com = run([])
    print(f"\n[baseline] induction-NLL {base_ind:.3f}  complement-NLL {base_com:.3f}\n")

    # ---- head sets from the idiom catalog ----
    summ = json.loads(args.idioms.read_text()) if args.idioms.exists() else {"idioms": {}}

    def heads_for(idiom, k):
        members = summ.get("idioms", {}).get(idiom, [])[:k]
        out = []
        for name, _s in members:
            L, h = name.split(".")
            out.append((int(L), int(h)))
        return out

    rng = np.random.default_rng(args.seed)

    def random_matched(named):
        layers = [L for L, _h in named]
        return [(L, int(rng.integers(0, H))) for L in layers]

    rows = []
    print(f"{'idiom':>16} {'heads':>16} {'Δind':>7} {'Δcomp':>7} {'Δind_rand':>10} {'spec':>6}  verdict")
    for idiom, k in IDIOM_SETS.items():
        named = heads_for(idiom, k)
        if not named:
            continue
        ind, com = run(named)
        d_ind = ind - base_ind; d_com = com - base_com
        rand_d = []
        for _ in range(args.n_random):
            ri, _rc = run(random_matched(named))
            rand_d.append(ri - base_ind)
        rand_mu = float(np.mean(rand_d)); rand_sd = float(np.std(rand_d) + 1e-9)
        spec = (d_ind - rand_mu) / rand_sd                       # z of named effect vs random control
        is_circuit = idiom in CIRCUIT
        # load-bearing: induction-specific (Δind > Δcomp) AND beats random (spec > 2)
        load_bearing = bool(d_ind > max(d_com, 0) and spec > 2.0)
        verdict = ("LOAD-BEARING" if load_bearing else
                   ("(neg-control: ~no ind effect)" if not is_circuit and d_ind < 2 * rand_sd + rand_mu
                    else "weak/ambiguous"))
        rows.append({"idiom": idiom, "heads": [list(x) for x in named], "is_circuit": is_circuit,
                     "delta_ind": d_ind, "delta_comp": d_com, "delta_ind_random_mean": rand_mu,
                     "delta_ind_random_sd": rand_sd, "specificity_z": spec, "load_bearing": load_bearing})
        hs = ",".join(f"{L}.{h}" for L, h in named)
        print(f"{idiom:>16} {hs:>16} {d_ind:>+7.3f} {d_com:>+7.3f} {rand_mu:>+10.3f} {spec:>6.1f}  {verdict}")

    # ---- self-repair test: name-movers are backed up (Wang et al.). If ablating the PRIMARY movers is
    # a null but ablating PRIMARIES + BACKUPS together breaks induction, the null was backup compensation. --
    nm = heads_for("copy_namemover", 3)
    bk = heads_for("backup_namemover", 4)
    combined = sorted({tuple(x) for x in nm + bk})
    if combined:
        ind, com = run(combined)
        d_ind = ind - base_ind; d_com = com - base_com
        rand_d = [run(random_matched(combined))[0] - base_ind for _ in range(args.n_random)]
        rand_mu = float(np.mean(rand_d)); spec = (d_ind - rand_mu) / (np.std(rand_d) + 1e-9)
        nm_d = next((r["delta_ind"] for r in rows if r["idiom"] == "copy_namemover"), 0.0)
        rows.append({"idiom": "namemovers+backups", "heads": [list(x) for x in combined], "is_circuit": True,
                     "delta_ind": d_ind, "delta_comp": d_com, "delta_ind_random_mean": rand_mu,
                     "specificity_z": float(spec), "load_bearing": bool(d_ind > max(d_com, 0) and spec > 2)})
        hs = ",".join(f"{L}.{h}" for L, h in combined)
        print(f"\n[self-repair test] primary name-movers alone Δind {nm_d:+.3f} (null) -> "
              f"primaries+backups Δind {d_ind:+.3f} (z{spec:.1f})")
        print(f"  {'namemovers+backups':>18}: {hs}  -> "
              f"{'BACKUP COMPENSATION confirmed (combined breaks what primaries-alone did not)' if d_ind > 2 * nm_d and spec > 2 else 'no clear combined effect'}")

    out = {"experiment": "causal validation of named idioms (mean-ablation, induction-NLL)",
           "model": args.pretrained, "baseline_ind_nll": base_ind, "baseline_comp_nll": base_com,
           "n_random": args.n_random, "idioms": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    circ = [r for r in rows if r["is_circuit"]]
    nlb = sum(1 for r in circ if r["load_bearing"])
    print(f"\n[verdict] {nlb}/{len(circ)} induction-circuit idioms are causally LOAD-BEARING "
          f"(Δind > Δcomp and specificity z>2 vs layer-matched random)")
    negc = [r for r in rows if not r["is_circuit"]]
    if negc:
        print("[neg-controls] non-circuit idioms (should show ~no induction-specific damage): "
              + ", ".join(f"{r['idiom']} Δind{r['delta_ind']:+.3f}(z{r['specificity_z']:.1f})" for r in negc))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
