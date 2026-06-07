"""Cross-model discovery sweep — the debugger's discovery engine, generalized to every model.

`residual_vm_debugger.py` ranks every head + MLP by causal effect and flags the load-bearing components the named
catalog has NOT named (candidate new operators) — but it is GPT-2-only. This runs that **comprehensively across
every model**, multi-seed for rigour: for each (model, component) it mean-ablates the component (arch-generic
o_proj/c_proj head-slice → corpus mean; MLP block → corpus mean) and measures the causal ΔNLL on **induction**
(repeated-random 2nd-copy, multi-seed) and **generic** prose. Components are ranked; each is flagged **named** (a
member of the behavioural operator catalog for that model — prev-token / induction / duplicate / sink / self /
local / structural, found by attention-mask mass — or, for GPT-2, a literature circuit head) or **UNNAMED**
(a candidate new operator). The UNNAMED load-bearing components per model are the discovery output.

Output: [runs/disassembly/operators/discovered_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/discovered_summary.json) + a generated `docs/operators/discovered.md`. This
grows the catalog the rigorous way — every component in every model, not one bespoke probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GPT2_CIRCUIT = {  # GPT-2 literature circuit heads (DLA-defined), so they count as "named" though attention-mask-invisible
    (9, 6), (9, 9), (10, 0), (10, 10), (9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (11, 2), (10, 7), (11, 10),
    (7, 3), (7, 9), (8, 6), (8, 10),
}


def op_masks(toks):
    ca = np.array(toks); n = len(ca); qi = np.arange(n); pv = np.full(n, -1); pv[1:] = ca[:-1]
    qq = qi[:, None]; kk = qi[None, :]
    return {
        "prevtok": (kk == qq - 1) & (qq >= 1),
        "induction": (pv[None, :] == ca[:, None]) & (kk >= 1) & (kk < qq),
        "duplicate": (ca[None, :] == ca[:, None]) & (kk < qq),
        "sink": (kk == 0) & (qq >= 1),
        "self": (kk == qq) & (qq >= 1),
        "local": (kk < qq) & (kk >= qq - 3) & (qq >= 1),
    }


def arch(model):
    cfg = model.config
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blks = model.transformer.h
        return dict(is_gpt2=True, oproj=[b.attn.c_proj for b in blks], mlps=[b.mlp for b in blks],
                    H=cfg.n_head, hd=cfg.n_embd // cfg.n_head, nL=len(blks))
    blks = model.model.layers
    H = cfg.num_attention_heads; hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    return dict(is_gpt2=False, oproj=[ly.self_attn.o_proj for ly in blks], mlps=[ly.mlp for ly in blks],
                H=H, hd=hd, nL=len(blks))


def run_model(model_id, args, dev):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    a = arch(model); H, hd, nL = a["H"], a["hd"], a["nL"]; NH = nL * H
    oproj = a["oproj"]; mlps = a["mlps"]
    V = model.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]

    # ---- named-component set: behavioural operator heads (attention-mask mass) ----
    rng0 = np.random.default_rng(0)
    probe = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.probe_chunks]
    repprobe = [(lambda s: s + s)([int(x) for x in rng0.integers(lo, hi, args.rep_len)]) for _ in range(args.probe_chunks)]
    mass = {op: np.zeros(NH) for op in op_masks([0, 0])}; ntot = {op: 0 for op in mass}
    with torch.no_grad():
        for op_is_rep, pset in ((False, probe), (True, repprobe)):
            for s in pset:
                o = model(input_ids=torch.tensor([s], device=dev), output_attentions=True)
                M = op_masks(s)
                for op in mass:
                    if (op in ("induction", "duplicate")) != op_is_rep:
                        continue
                    Msk = M[op]; ntot[op] += int(Msk.sum())
                    if Msk.sum() == 0:
                        continue
                    for L in range(nL):
                        at = o.attentions[L][0].float().cpu().numpy()
                        mass[op][L * H:(L + 1) * H] += (at * Msk[None]).sum((1, 2))
    named = {}                                                                      # head index -> op name (behavioural)
    for op in mass:
        m = mass[op] / max(ntot[op], 1)
        for i in np.where(m > args.named_thr)[0]:
            named.setdefault(int(i), op)
    if a["is_gpt2"]:
        for (L, h) in GPT2_CIRCUIT:
            named.setdefault(L * H + h, "ioi-circuit")

    # ---- mean values: per-head c_proj/o_proj input slice, per-layer MLP output ----
    caph = {L: [] for L in range(nL)}; capm = {L: [] for L in range(nL)}
    hk = []
    for L in range(nL):
        hk.append(oproj[L].register_forward_pre_hook((lambda L: lambda m, inp: caph[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)))
        hk.append(mlps[L].register_forward_hook((lambda L: lambda m, i, o: capm[L].append((o[0] if isinstance(o, tuple) else o).detach().reshape(-1, (o[0] if isinstance(o, tuple) else o).shape[-1])))(L)))
    with torch.no_grad():
        for c in chunks:
            model(input_ids=torch.tensor([c], device=dev))
    for h in hk:
        h.remove()
    mean_h = {L: torch.cat(caph[L], 0).mean(0) for L in range(nL)}
    mean_m = {L: torch.cat(capm[L], 0).mean(0) for L in range(nL)}

    def head_hook(L, h):
        def hook(m, inp):
            x = inp[0].clone(); x[..., h * hd:(h + 1) * hd] = mean_h[L][h * hd:(h + 1) * hd].to(x.dtype); return (x,)
        return hook

    def mlp_hook(L):
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            r = mean_m[L].to(t.dtype).expand_as(t)
            return (r,) + tuple(o[1:]) if isinstance(o, tuple) else r
        return hook

    def gen_nll():
        tot = 0.0; n = 0
        with torch.no_grad():
            for c in chunks:
                lp = F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                y = torch.tensor(c[1:], device=dev); tot += float(-lp[torch.arange(len(y)), y].sum()); n += len(y)
        return tot / max(n, 1)

    def ind_nll_seed(seed):
        rng = np.random.default_rng(seed)
        seqs = [(lambda s: s + s)([int(x) for x in rng.integers(lo, hi, args.rep_len)]) for _ in range(args.n_ind)]
        tot = 0.0; n = 0
        with torch.no_grad():
            for s in seqs:
                lp = F.log_softmax(model(input_ids=torch.tensor([s], device=dev)).logits[0].float(), -1)
                L = len(s) // 2
                for p in range(L, 2 * L - 1):
                    tot += float(-lp[p, s[p + 1]]); n += 1
        return tot / max(n, 1)

    # ---- baselines (multi-seed induction) ----
    seeds = list(range(args.seeds))
    base_g = gen_nll()
    base_i = np.array([ind_nll_seed(s) for s in seeds])

    def component(label, kind, L, h):
        if kind == "head":
            hk = oproj[L].register_forward_pre_hook(head_hook(L, h))
        else:
            hk = mlps[L].register_forward_hook(mlp_hook(L))
        try:
            g = gen_nll() - base_g
            iv = np.array([ind_nll_seed(s) for s in seeds]) - base_i
        finally:
            hk.remove()
        idx = (L * H + h) if kind == "head" else None
        nm = (named.get(idx) if kind == "head" else "mlp")
        return {"comp": label, "kind": kind, "L": L, "depth": round(L / max(nL - 1, 1), 2),
                "generic_dNLL": g, "induction_dNLL_mean": float(iv.mean()), "induction_dNLL_std": float(iv.std()),
                "named": nm}

    rows = []
    for L in range(nL):
        for h in range(H):
            rows.append(component(f"{L}.{h}", "head", L, h))
    for L in range(nL):
        rows.append(component(f"mlp{L}", "mlp", L, None))
    rows.sort(key=lambda r: -r["induction_dNLL_mean"])
    unnamed = [r for r in rows if r["named"] is None and r["induction_dNLL_mean"] > args.cand_thr]
    return {"model": model_id.split("/")[-1], "arch": "GPT-2/absolute" if a["is_gpt2"] else "RoPE", "n_layers": nL,
            "n_heads": H, "base_generic": base_g, "base_induction": float(base_i.mean()), "seeds": len(seeds),
            "top_components": rows[:15], "candidate_unnamed": unnamed[:15], "n_unnamed_load_bearing": len(unnamed)}


def write_doc(out, docs):
    models = [r for r in out["results"] if "top_components" in r]
    lines = ["---", "title: Discovered components", "---", "",
             "# Discovered components — the debugger run across every model",
             "",
             "A **working catalog** (amateur, exploratory, provisional) of the load-bearing components the "
             "[discovery engine](../DECOMPILATION.md) surfaces — ranked by causal effect on **induction** "
             "(multi-seed) and **generic** prose — with each flagged **named** (already in the behavioural operator "
             "catalog for that model) or **UNNAMED** (a candidate new operator). The UNNAMED load-bearing components "
             "are the leads to dossier next.", "",
             f"_{len(models)} models · every head + every MLP mean-ablated · induction over {out.get('seeds','?')} seeds._", ""]
    for r in models:
        lines += [f"## {r['model']} ({r['arch']}, {r['n_layers']}L × {r['n_heads']}H) — {r['n_unnamed_load_bearing']} unnamed load-bearing",
                  "",
                  "Top components by induction ΔNLL (named in *italics*; **UNNAMED** = candidate op):", "",
                  "| component | kind | induction ΔNLL (μ±σ) | generic ΔNLL | status |",
                  "|---|---|---|---|---|"]
        for c in r["top_components"][:10]:
            status = f"*{c['named']}*" if c["named"] else "**UNNAMED — candidate**"
            lines.append(f"| `{c['comp']}` | {c['kind']} | {c['induction_dNLL_mean']:+.2f} ± {c['induction_dNLL_std']:.2f} | {c['generic_dNLL']:+.2f} | {status} |")
        cand = ", ".join(f"`{c['comp']}` ({c['induction_dNLL_mean']:+.2f})" for c in r["candidate_unnamed"][:8]) or "—"
        lines += ["", f"**Candidate operators (UNNAMED, load-bearing):** {cand}", ""]
    lines += ["## How to read this", "",
              "- *named* = the component is already a member of a catalogued operator class for that model "
              "(prev-token / induction / duplicate / sink / self / local, by attention-mask mass; or a GPT-2 IOI "
              "circuit head). *(Mean-ablation under-counts self-repairing classes like name-movers — see the IOI "
              "dossier — so a 'named' reading low is expected.)*",
              "- **UNNAMED** = load-bearing on induction but not yet in any catalogued class — a candidate operator "
              "to give a [dossier](README.md). MLPs are flagged `mlp` (the COMPUTE class, see "
              "[MLP / COMPUTE](mlp_compute.md)).",
              "- Provisional and descriptive — measurements to be checked, not settled results.", "",
              "_Data: [runs/disassembly/operators/discovered_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/discovered_summary.json). Regenerate: [discovery_atlas.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/discovery_atlas.py)._"]
    (docs / "discovered.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--probe-chunks", type=int, default=12)
    p.add_argument("--n-ind", type=int, default=20)
    p.add_argument("--rep-len", type=int, default=22)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--named-thr", type=float, default=0.15)
    p.add_argument("--cand-thr", type=float, default=0.10, help="induction ΔNLL above which an UNNAMED head is a candidate")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/operators"))
    p.add_argument("--docs", type=Path, default=Path("docs/operators"))
    args = p.parse_args(argv)

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = run_model(mid, args, dev)
            if dev == "cuda":
                torch.cuda.empty_cache()
            results.append(r)
            top = r["top_components"][0]
            print(f"  {r['arch']} {r['n_layers']}L×{r['n_heads']}H | top {top['comp']} ind {top['induction_dNLL_mean']:+.2f} "
                  f"({top['named'] or 'UNNAMED'}) | {r['n_unnamed_load_bearing']} unnamed load-bearing | "
                  f"candidates: {', '.join(c['comp'] for c in r['candidate_unnamed'][:6])}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid, "error": str(e)})

    out = {"experiment": "cross-model discovery sweep — every head+MLP ranked by causal effect (multi-seed induction)",
           "seeds": args.seeds, "results": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "discovered_summary.json").write_text(json.dumps(out, indent=2, default=float))
    args.docs.mkdir(parents=True, exist_ok=True)
    write_doc(out, args.docs)
    ok = [r for r in results if "top_components" in r]
    print(f"\n[done] {sum(r['n_unnamed_load_bearing'] for r in ok)} unnamed load-bearing candidates across "
          f"{len(ok)} models → {args.outdir / 'discovered_summary.json'} + {args.docs / 'discovered.md'}")
    return out


if __name__ == "__main__":
    main()
