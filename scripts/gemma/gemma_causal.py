"""Causal validation of Gemma-2's behaviorally-detected induction heads (mean-ablation, induction-NLL).

Confirms the universal circuits the portable disassembly *found* on Gemma-2 are causally LOAD-BEARING,
not just correlational — the recent-model analog of causal_validation.py (GPT-2). Detect the top induction
and prev-token heads behaviorally (one pass), then MEAN-ABLATE each set (replace the head's o_proj-input
slice with its corpus mean) and measure the rise in induction-predictable NLL vs the complement vs a
layer-matched RANDOM head set. Load-bearing <=> Δind > Δcomplement AND Δind(named) >> Δind(random).
Gemma-2 hook point = self_attn.o_proj (concatenated head outputs). Runs on cuda (bf16).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _induction_predictable(c):
    n = len(c); pred = np.zeros(n, bool)
    for t in range(2, n):
        ps = [p for p in range(t - 1) if c[p] == c[t - 1]]
        if ps and c[ps[-1] + 1] == c[t]:
            pred[t] = True
    return pred


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-induct", type=int, default=6)
    p.add_argument("--n-prev", type=int, default=4)
    p.add_argument("--n-random", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--corpus", default="wikitext")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("runs/gemma/gemma_causal_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    cfg = model.config
    nL, H = cfg.num_hidden_layers, cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    CORPORA = {"shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
               "wikitext": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"}
    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(CORPORA.get(args.corpus, args.corpus),
                                 headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:400000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    masks = [_induction_predictable(c)[1:].astype(bool) for c in chunks]
    print(f"{args.model}: {nL}L x {H}H, {len(chunks)} chunks, induction-rate {np.concatenate(masks).mean():.3f}")

    # ---- detect induction + prev-token heads behaviorally + capture o_proj-input means ----
    ind = np.zeros((nL, H)); indn = 0; pt = np.zeros((nL, H)); ptn = 0
    cap = {L: [] for L in range(nL)}
    hooks = [model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(
        (lambda L: lambda m, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L))
        for L in range(nL)]
    with torch.no_grad():
        for c in chunks:
            o = model(input_ids=torch.tensor([c], device=dev), output_attentions=True)
            Lc = len(c); ca = np.array(c); qi = np.arange(Lc)
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]
            IM = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
            indn += int(IM.any(1).sum()); ptn += Lc - 1
            for L in range(nL):
                a = o.attentions[L][0].float().cpu().numpy()
                ind[L] += (a * IM[None]).sum((1, 2))
                pt[L] += np.diagonal(a, offset=-1, axis1=1, axis2=2).sum(1)
    for hk in hooks:
        hk.remove()
    meanvec = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}
    heads = [(L, h) for L in range(nL) for h in range(H)]
    indf = ind.reshape(-1); ptf = pt.reshape(-1)
    induct = [heads[i] for i in np.argsort(-indf)[:args.n_induct]]
    prev = [heads[i] for i in np.argsort(-ptf)[:args.n_prev]]
    print(f"  induction heads: {[f'{a}.{b}' for a, b in induct]}; prev-token: {[f'{a}.{b}' for a, b in prev]}")

    def run(ablate):
        by_layer = {}
        for (L, h) in ablate:
            by_layer.setdefault(L, []).append(h)
        hs = []
        for L, hss in by_layer.items():
            def mk(L, hss):
                def hook(m, inp):
                    x = inp[0].clone()
                    for h in hss:
                        x[..., h * hd:(h + 1) * hd] = meanvec[L][h * hd:(h + 1) * hd].to(x.dtype)
                    return (x,)
                return hook
            hs.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L, hss)))
        ms = mn = cs = cn = 0.0
        with torch.no_grad():
            for c, m in zip(chunks, masks):
                lg = model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float()
                lp = torch.log_softmax(lg, -1)
                nll = (-lp[torch.arange(len(c) - 1), torch.tensor(c[1:], device=dev)]).cpu().numpy()
                ms += nll[m].sum(); mn += int(m.sum()); cs += nll[~m].sum(); cn += int((~m).sum())
        for hk in hs:
            hk.remove()
        return ms / max(mn, 1), cs / max(cn, 1)

    base_ind, base_com = run([])
    print(f"\n[baseline] induction-NLL {base_ind:.3f}  complement-NLL {base_com:.3f}")
    rng = np.random.default_rng(args.seed)
    rows = []
    for name, hset in [("induction", induct), ("prev_token", prev)]:
        ik, ck = run(hset); d_ind = ik - base_ind; d_com = ck - base_com
        rand = []
        for _ in range(args.n_random):
            rh = [(L, int(rng.integers(0, H))) for L, _h in hset]
            rand.append(run(rh)[0] - base_ind)
        rmu = float(np.mean(rand)); rsd = float(np.std(rand) + 1e-9)
        z = (d_ind - rmu) / rsd
        lb = bool(d_ind > max(d_com, 0) and z > 2)
        rows.append({"set": name, "heads": [list(x) for x in hset], "delta_ind": d_ind, "delta_comp": d_com,
                     "delta_ind_random_mean": rmu, "z": z, "load_bearing": lb})
        print(f"  {name:>10}: Δind {d_ind:+.3f}  Δcomp {d_com:+.3f}  random Δind {rmu:+.3f}  z {z:+.1f}  "
              f"-> {'LOAD-BEARING' if lb else 'weak'}")

    out = {"experiment": f"causal validation (induction-NLL mean-ablation): {args.model}", "model": args.model,
           "baseline_ind_nll": base_ind, "baseline_comp_nll": base_com, "sets": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    nlb = sum(r["load_bearing"] for r in rows)
    print(f"\n[verdict] {nlb}/{len(rows)} head sets causally load-bearing for induction -> "
          f"{f'the universal circuits are CAUSAL on {args.model}, not just correlational' if nlb >= 1 else 'no clear causal effect'}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
