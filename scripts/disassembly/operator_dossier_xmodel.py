"""Arch-generic CROSS-MODEL operator dossier — extend the deep dossier beyond GPT-2.

`operator_dossier.py` runs the deep A-F battery only on GPT-2: it is written against `GPT2LMHeadModel`'s fused
`c_attn` QKV (slicing `[..., d:2d]` / `[..., 2d:3d]`) + `transformer.h[L].attn.c_proj` layout, which the RoPE
family (separate q/k/v_proj + GQA + RMSNorm) does not share. But the universal *behavioural* operators
(induction, prev-token, duplicate, sink) exist in EVERY model, so this runs the arch-generic core of that battery
across the RoPE family too:

  IDENTITY  — behaviourally find the op's heads (rank by attention mass on the op's reference mask) + depth;
  CAUSAL    — mean-ablate the op's top heads -> induction-NLL + generic-NLL damage (is it load-bearing, on what?);
  CHANNEL   — for the op's top reader (content ops only), the faithful key-only path-patch (model re-applies its
              own RoPE): top upstream KEY writer (collapse) + the VALUE/move channel (ΔV-out).

so each operator page carries a cross-model deep dossier, not just the GPT-2 one. The literature/output ops
(name-movers, S-inhibition, …) stay GPT-2-only — no published head-set off GPT-2 — which is intrinsic, not a
tooling gap. Reuses `circuit_content_patch._arch` + the same faithful patch as `xmodel_candidate.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
from circuit_content_patch import _arch  # noqa: E402


def op_masks(toks):
    """(query, key) boolean mask each universal behavioural op attends along, for a token list."""
    ca = np.array(toks); n = len(ca); qi = np.arange(n); pv = np.full(n, -1); pv[1:] = ca[:-1]
    return {
        "induction": (pv[None, :] == ca[:, None]) & (qi[None, :] >= 1) & (qi[None, :] < qi[:, None]),
        "duplicate": (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]),
        "prevtok": (qi[None, :] == (qi[:, None] - 1)) & (qi[:, None] >= 1),
        "sink": (qi[None, :] == 0) & (qi[:, None] >= 1),
    }


# op -> (kind, probes-on-repeated-random?, channel-applies?). content ops read repeated-random; positional/
# addressing ops read prose; channel (key-writer collapse) is only meaningful for the content match ops.
OPS = {
    "induction": ("content", True, True),
    "duplicate": ("content", True, True),
    "prevtok": ("positional", False, False),
    "sink": ("addressing", False, False),
}


def dossier_one_model(model_id, ops, args, dev):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    is_gpt2 = "gpt2" in model_id.lower()
    if is_gpt2:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_id, attn_implementation="eager").eval().to(dev)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    a = _arch(model); H = a["H"]; hd = a["hd"]; nL = model.config.num_hidden_layers
    oproj = a["oproj"]; norm = a["norm"]; d = a.get("d", model.config.hidden_size); NH = nL * H
    tok = AutoTokenizer.from_pretrained(model_id)
    rng = np.random.default_rng(args.seed)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:160000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]
    V = model.config.vocab_size
    if is_gpt2:                                                                    # GPT-2: sample from frequent prose tokens
        cnt = {}
        for t in pids:
            cnt[t] = cnt.get(t, 0) + 1
        vocab = [t for t, _ in sorted(cnt.items(), key=lambda r: -r[1])[:400]]
        rep = lambda L: [int(vocab[i]) for i in rng.integers(0, len(vocab), L)]    # noqa: E731
    else:
        lo, hi = int(0.02 * V), int(0.4 * V)
        rep = lambda L: [int(x) for x in rng.integers(lo, hi, L)]                  # noqa: E731
    rep_seqs = [(lambda s: s + s)(rep(args.rep_len)) for _ in range(args.probes)]

    def name_of(i):
        return f"{i // H}.{i % H}"

    # ---- per-head attention mass on every op's reference mask (one pass over probes) ----
    masses = {op: np.zeros(NH) for op in ops}; ntot = {op: 0 for op in ops}
    rep_set = {op for op in ops if OPS[op][1]}
    prose_set = set(ops) - rep_set
    with torch.no_grad():
        for grp, seqs in (("rep", rep_seqs[: args.id_probes]), ("prose", chunks[: args.id_probes])):
            want_ops = rep_set if grp == "rep" else prose_set
            if not want_ops:
                continue
            for s in seqs:
                o = model(input_ids=torch.tensor([s], device=dev), output_attentions=True)
                M = op_masks(s)
                for op in want_ops:
                    wm = M[op]; ntot[op] += int(wm.sum())
                    for L in range(nL):
                        at = o.attentions[L][0].float().cpu().numpy()
                        masses[op][L * H:(L + 1) * H] += (at * wm[None]).sum((1, 2))
    for op in ops:
        masses[op] /= max(ntot[op], 1)

    # ---- mean-ablation harness (replace a head's o_proj-input slice with the corpus mean) ----
    cap = {L: [] for L in range(nL)}
    hks = [oproj[L].register_forward_pre_hook((lambda L: lambda m, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in chunks:
            model(input_ids=torch.tensor([c], device=dev))
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
            hs.append(oproj[L].register_forward_pre_hook(mk(L, hss)))
        return hs

    def gen_nll(heads=()):
        hs = ablate_hooks(heads); tot = 0.0; k = 0
        try:
            with torch.no_grad():
                for c in chunks:
                    lp = F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                    y = torch.tensor(c[1:], device=dev); tot += float(-lp[torch.arange(len(y)), y].sum()); k += len(y)
        finally:
            for h in hs:
                h.remove()
        return tot / max(k, 1)

    def ind_nll(heads=()):
        hs = ablate_hooks(heads); tot = 0.0; k = 0
        try:
            with torch.no_grad():
                for s in rep_seqs:
                    lp = F.log_softmax(model(input_ids=torch.tensor([s], device=dev)).logits[0].float(), -1)
                    L = len(s) // 2
                    for p in range(L, 2 * L - 1):
                        tot += float(-lp[p, s[p + 1]]); k += 1
        finally:
            for h in hs:
                h.remove()
        return tot / max(k, 1)
    base_gen = gen_nll(); base_ind = ind_nll()

    # ---- channel: faithful key-only path-patch for a content op's top reader ----
    def channel(reader):
        LB, hB = reader // H, reader % H
        if LB == 0:
            return {"note": "reader in layer 0 — no upstream; channel skipped"}
        upstream = [(L, h) for L in range(LB) for h in range(H)][: args.max_upstream]
        kvB = hB // (H // a["nkv"])

        def head_contrib(L, captured, h):
            x = torch.zeros_like(captured); x[..., h * hd:(h + 1) * hd] = captured[..., h * hd:(h + 1) * hd]
            return oproj[L](x) - oproj[L](torch.zeros_like(captured[..., :1, :]))

        def b_out(inp_normed, attnB):
            v = (a["cattn"][LB](inp_normed)[..., 2 * d:3 * d] if a["is_gpt2"] else a["vproj"][LB](inp_normed))
            ho = attnB.to(v.dtype) @ v[0, :, kvB * hd:(kvB + 1) * hd]
            x = torch.zeros((1, ho.shape[0], H * hd), dtype=v.dtype, device=v.device); x[0, :, hB * hd:(hB + 1) * hd] = ho
            return oproj[LB](x) - oproj[LB](x[:, :1] * 0)
        capk = {}; hooks = [a["layers"][LB].register_forward_pre_hook(lambda m, inp: capk.__setitem__("r", inp[0].detach()))]
        for L in range(LB):
            hooks.append(oproj[L].register_forward_pre_hook((lambda L: lambda m, inp: capk.__setitem__(L, inp[0].detach()))(L)))
        clean = 0.0; kp = {u: 0.0 for u in upstream}; vtot = 0.0; vp = {u: 0.0 for u in upstream}; tot = 0
        patt = "induction" if reader_op == "induction" else "duplicate"
        with torch.no_grad():
            for s in rep_seqs[: args.patch_probes]:
                capk.clear(); o = model(input_ids=torch.tensor([s], device=dev), output_attentions=True)
                Msk = op_masks(s)[patt]; attnB = o.attentions[LB][0, hB]
                clean += float((attnB.float().cpu().numpy() * Msk).sum())
                resid = capk["r"]; bclean = b_out(norm[LB](resid), attnB); vtot += float(torch.linalg.norm(bclean.float()))
                for (La, ha) in upstream:
                    kin = norm[LB](resid - head_contrib(La, capk[La], ha))
                    if a["is_gpt2"]:
                        ksl = a["cattn"][LB](kin)[..., d:2 * d]

                        def hook(m, inp, o2, _k=ksl):
                            o2 = o2.clone(); o2[..., d:2 * d] = _k; return o2
                        hh = a["cattn"][LB].register_forward_hook(hook)
                    else:
                        def pre(m, inp, _ki=kin):
                            return (_ki,) + inp[1:]
                        hh = a["kproj"][LB].register_forward_pre_hook(pre)
                    try:
                        ap = model(input_ids=torch.tensor([s], device=dev), output_attentions=True).attentions[LB][0, hB]
                        kp[(La, ha)] += float((ap.float().cpu().numpy() * Msk).sum())
                    finally:
                        hh.remove()
                    vp[(La, ha)] += float(torch.linalg.norm((bclean - b_out(kin, attnB)).float()))
                tot += 1
        for h in hooks:
            h.remove()
        clean /= max(tot, 1)
        krows = sorted([{"head": f"{La}.{ha}", "collapse": (clean - kp[(La, ha)] / max(tot, 1)) / clean if clean > 1e-6 else 0.0} for (La, ha) in upstream], key=lambda r: -r["collapse"])
        vrows = sorted([{"head": f"{La}.{ha}", "dvout": vp[(La, ha)] / max(vtot, 1e-9)} for (La, ha) in upstream], key=lambda r: -r["dvout"])
        kmed = float(np.median([r["collapse"] for r in krows]))
        return {"key_top": krows[0], "key_median": kmed, "key_concentration": krows[0]["collapse"] / (abs(kmed) + 1e-9),
                "value_top": vrows[0], "value_median": float(np.median([r["dvout"] for r in vrows])),
                "key_rows": krows[:5], "value_rows": vrows[:5]}

    out = {}
    for op in ops:
        kind, _, chan = OPS[op]
        order = np.argsort(-masses[op])
        top = [int(i) for i in order[: args.top_heads] if masses[op][int(i)] > args.min_mass]
        if not top:
            top = [int(order[0])]
        reader_op = op
        top_hh = [(i // H, i % H) for i in top]
        ind_d = ind_nll(set(top_hh)) - base_ind
        gen_d = gen_nll(set(top_hh)) - base_gen
        # E. redundancy: per-head solo induction effect + cumulative-ablation curve → bottleneck vs distributed
        solo = sorted([(f"{L}.{h}", ind_nll({(L, h)}) - base_ind) for (L, h) in top_hh], key=lambda r: -r[1])
        acc = []; curve = []
        for nm, _ in solo:
            acc.append(tuple(int(x) for x in nm.split("."))); curve.append({"n": len(acc), "effect": ind_nll(set(acc)) - base_ind})
        max_solo = max((e for _, e in solo), default=0.0)
        redundancy = {"solo": solo, "curve": curve, "full": ind_d, "max_solo": max_solo,
                      "bottleneck": bool(ind_d <= 1.4 * max_solo and max_solo > 0.1)}
        rec = {"op": op, "kind": kind, "heads": [name_of(i) for i in top],
               "top_head": name_of(top[0]), "top_mass": float(masses[op][top[0]]),
               "top_depth": (top[0] // H) / (nL - 1) if nL > 1 else 0.0,
               "causal_induction_dNLL": ind_d, "causal_generic_dNLL": gen_d, "redundancy": redundancy,
               "base_induction": base_ind, "base_generic": base_gen, "n_heads_mass": int((masses[op] > args.min_mass).sum())}
        rec["channel"] = channel(top[0]) if chan else {"note": f"{kind} op — addresses by position/key-0, not an upstream key writer; channel N/A"}
        out[op] = rec
    return {"model": model_id.split("/")[-1], "rope": not a["is_gpt2"], "n_layers": nL, "n_heads": H, "ops": out}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ops", default="induction,prevtok,duplicate,sink")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=20)
    p.add_argument("--probes", type=int, default=28)
    p.add_argument("--id-probes", type=int, default=16, help="probes for behavioural head-ID")
    p.add_argument("--patch-probes", type=int, default=10)
    p.add_argument("--rep-len", type=int, default=22)
    p.add_argument("--top-heads", type=int, default=4)
    p.add_argument("--min-mass", type=float, default=0.05)
    p.add_argument("--max-upstream", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/operators"))
    args = p.parse_args(argv)
    ops = [o.strip() for o in args.ops.split(",") if o.strip() in OPS]

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = dossier_one_model(mid, ops, args, dev)
            if dev == "cuda":
                torch.cuda.empty_cache()
            results.append(r)
            for op in ops:
                rec = r["ops"][op]; ch = rec["channel"]
                kt = f"KEY {ch['key_top']['head']} {ch['key_top']['collapse']:+.0%}" if "key_top" in ch else "channel n/a"
                print(f"  {op:>10}: heads {rec['heads']} | causal ind {rec['causal_induction_dNLL']:+.2f} gen {rec['causal_generic_dNLL']:+.2f} | {kt}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid, "error": str(e)})

    out = {"experiment": "arch-generic cross-model operator dossier (universal behavioural ops)",
           "ops": ops, "results": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "xmodel_dossiers_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'ops' in r])} models → {args.outdir / 'xmodel_dossiers_summary.json'}")
    return out


if __name__ == "__main__":
    main()
