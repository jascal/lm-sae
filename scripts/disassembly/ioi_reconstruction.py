"""Executable decompilation of the IOI circuit (GPT-2) — is the literature's "complete circuit" sufficient?

Induction turned out NOT to be an isolable subgraph (`circuit_reconstruction.py`: no compact head-set reconstructs
it). IOI is the field's most-celebrated *complete* circuit (Wang et al. 2022, ~26 heads). This tests its
SUFFICIENCY on the metric it serves — the IOI logit-difference LD = logit(IO) − logit(S): keep ONLY the IOI
circuit's heads (prev-token + duplicate + induction + S-inhibition + name-movers + backups + negatives),
mean-ablate every other attention head (MLPs intact), and measure how much of the IOI logit-diff survives.

    coverage = (LD_circuit-only − LD_all-attn-ablated) / (LD_full − LD_all-attn-ablated)

Appends an `ioi_gpt2` block to circuit_reconstruction_summary.json + an IOI section on the reconstruction page.
GPT-2-only (no published IOI head-set elsewhere). CPU-runnable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ioi_causal import NAMES, OBJECTS, PLACES, TEMPLATES  # noqa: E402

LIT_DUP = [(0, 1), (0, 5), (3, 0), (1, 5)]
LIT_PREV = [(4, 11)]
LIT_IND = [(5, 0), (5, 1), (5, 5), (6, 9), (7, 11)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atlas", type=Path, default=Path("runs/disassembly/operators/atlas_summary.json"))
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--mean-chunks", type=int, default=24)
    p.add_argument("--n-task", type=int, default=90)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/circuits"))
    p.add_argument("--docs", type=Path, default=Path("docs/circuits"))
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    tr = model.transformer; H = model.config.n_head; hd = model.config.n_embd // H; nL = model.config.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); rng = np.random.default_rng(args.seed)

    co = json.loads(args.atlas.read_text()).get("gpt2_circuit_ops", {})
    ioi_heads = set(LIT_DUP) | set(LIT_PREV) | set(LIT_IND)
    for op in ("name_mover", "backup_name_mover", "negative_mover", "s_inhibition"):
        for h in co.get(op, []):
            ioi_heads.add(tuple(int(x) for x in h.split(".")))

    def single(strs):
        out = []
        for s in strs:
            i = tok(s, add_special_tokens=False)["input_ids"]
            out.append(i[0] if len(i) == 1 else None)
        return [x for x in out if x is not None]
    names = single(NAMES); places = single(PLACES); objs = single(OBJECTS)
    ioi = []
    for _ in range(args.n_task):
        a, b = rng.choice(len(names), 2, replace=False)
        P = names[a]; S = names[b]; pl = places[int(rng.integers(0, len(places)))]; ob = objs[int(rng.integers(0, len(objs)))]
        tpl = TEMPLATES[int(rng.integers(0, len(TEMPLATES)))]
        text = tpl.format(i0=tok.decode([P]), i1=tok.decode([S]), P=tok.decode([pl]), S=tok.decode([S]), T=tok.decode([ob]))
        ioi.append((tok(text)["input_ids"], P, S))

    # corpus mean per layer for mean-ablation
    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8][: args.mean_chunks]
    cap = {L: [] for L in range(nL)}
    hks = [tr.h[L].attn.c_proj.register_forward_pre_hook((lambda L: lambda m, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in chunks:
            model(input_ids=torch.tensor([c]))
    for h in hks:
        h.remove()
    meanv = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    def ablate_hooks(heads):
        by = {}
        for (L, h) in heads:
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

    def ioi_ld(ablate=()):
        hs = ablate_hooks(ablate); lds = []
        try:
            with torch.no_grad():
                for idsq, io, s in ioi:
                    lg = model(input_ids=torch.tensor([idsq])).logits[0, -1].float()
                    lds.append(float(lg[io] - lg[s]))
        finally:
            for x in hs:
                x.remove()
        return float(np.mean(lds))

    all_heads = [(L, h) for L in range(nL) for h in range(H)]
    non_circ = [hh for hh in all_heads if hh not in ioi_heads]
    full = ioi_ld(); allabl = ioi_ld(all_heads); circ = ioi_ld(non_circ)

    def cov(ld):
        return (ld - allabl) / (full - allabl + 1e-9)
    rnd = []
    for _ in range(args.n_random):
        rk = {tuple(int(x) for x in divmod(int(i), H)) for i in rng.choice(nL * H, len(ioi_heads), replace=False)}
        rnd.append(cov(ioi_ld([hh for hh in all_heads if hh not in rk])))
    res = {"circuit_size": len(ioi_heads), "n_heads_total": nL * H, "ld_full": full, "ld_all_attn_ablated": allabl,
           "ld_circuit_only": circ, "coverage": cov(circ), "random_coverage_mean": float(np.mean(rnd)),
           "random_coverage_std": float(np.std(rnd))}
    print(f"IOI circuit {len(ioi_heads)}/{nL * H} heads: LD full {full:+.2f} | circuit {circ:+.2f} | all-ablated {allabl:+.2f} "
          f"| coverage {res['coverage']:+.0%} (random {res['random_coverage_mean']:+.0%})")

    sumpath = args.outdir / "circuit_reconstruction_summary.json"
    summary = json.loads(sumpath.read_text()) if sumpath.exists() else {"results": []}
    summary["ioi_gpt2"] = res
    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath.write_text(json.dumps(summary, indent=2, default=float))
    import circuit_reconstruction as cr
    cr.write_doc(summary, args.docs)
    print(f"[done] IOI reconstruction → {sumpath} + {args.docs / 'reconstruction.md'}")
    return res


if __name__ == "__main__":
    main()
