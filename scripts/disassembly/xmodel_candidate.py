"""Arch-generic candidate profiler — dossier the discovered RoPE candidate operators (any model, any head).

`operator_dossier.py` is GPT-2-only, so the discovery sweep's RoPE candidates (Llama 0.31, Gemma 13.7, …) were
left undossiered. This profiles a given (model, head) arch-generically — the key questions for a discovered op:
  - CAUSAL    : mean-ablate the head → induction + generic ΔNLL (is it load-bearing, and on what?);
  - CHANNEL   : path-patch each upstream head out of the candidate's KEY (faithful key-only patch, model re-applies
                its own RoPE) → top collapser (what addresses it); + the VALUE/move channel (ΔV-out, what it moves);
Reuses `circuit_content_patch._arch` + the same faithful patch. Emits docs/operators/discovered_xmodel.md.

Targets are grouped by model and each model is loaded **once** (then freed) — profiling N heads of a model costs
one load, not N. This matters on a small GPU: a fresh 2B-param bf16 copy per target OOMs after a few. `--docs-only`
re-renders the page from the committed summary JSON with no GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
from circuit_content_patch import _arch  # noqa: E402


def masks(toks):
    ca = np.array(toks); n = len(ca); qi = np.arange(n); pv = np.full(n, -1); pv[1:] = ca[:-1]
    return {"induction": (pv[None, :] == ca[:, None]) & (qi[None, :] >= 1) & (qi[None, :] < qi[:, None]),
            "duplicate": (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])}


def setup(model_id, args, dev):
    """Load a model ONCE + everything shared across its candidate heads (arch, corpus, per-layer mean, baselines)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    a = _arch(model); H = a["H"]; hd = a["hd"]; nL = model.config.num_hidden_layers
    oproj = a["oproj"]; d = a.get("d", model.config.hidden_size); V = model.config.vocab_size
    rng = np.random.default_rng(args.seed)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    tok = AutoTokenizer.from_pretrained(model_id); pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]
    lo, hi = int(0.02 * V), int(0.4 * V)
    seqs = [(lambda s: s + s)([int(x) for x in rng.integers(lo, hi, args.rep_len)]) for _ in range(args.probes)]

    cap = {L: [] for L in range(nL)}
    hks = [oproj[L].register_forward_pre_hook((lambda L: lambda m, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in chunks:
            model(input_ids=torch.tensor([c], device=dev))
    for h in hks:
        h.remove()
    meanv = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    def ablate_hook_factory(LB, hB):
        def hook(m, inp):
            x = inp[0].clone(); x[..., hB * hd:(hB + 1) * hd] = meanv[LB][hB * hd:(hB + 1) * hd].to(x.dtype); return (x,)
        return hook

    def gen_nll(LB=None, hB=None):
        h = oproj[LB].register_forward_pre_hook(ablate_hook_factory(LB, hB)) if LB is not None else None
        tot = 0.0; k = 0
        try:
            with torch.no_grad():
                for c in chunks:
                    lp = F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                    y = torch.tensor(c[1:], device=dev); tot += float(-lp[torch.arange(len(y)), y].sum()); k += len(y)
        finally:
            if h:
                h.remove()
        return tot / max(k, 1)

    def ind_nll(LB=None, hB=None):
        h = oproj[LB].register_forward_pre_hook(ablate_hook_factory(LB, hB)) if LB is not None else None
        tot = 0.0; k = 0
        try:
            with torch.no_grad():
                for s in seqs:
                    lp = F.log_softmax(model(input_ids=torch.tensor([s], device=dev)).logits[0].float(), -1)
                    L = len(s) // 2
                    for p in range(L, 2 * L - 1):
                        tot += float(-lp[p, s[p + 1]]); k += 1
        finally:
            if h:
                h.remove()
        return tot / max(k, 1)
    return dict(model=model, a=a, tok=tok, chunks=chunks, seqs=seqs, meanv=meanv, nL=nL, H=H, hd=hd, d=d,
                base_gen=gen_nll(), base_ind=ind_nll(), gen_nll=gen_nll, ind_nll=ind_nll, model_id=model_id)


def profile_head(ctx, head, args, dev):
    import torch
    model = ctx["model"]; a = ctx["a"]; H = ctx["H"]; hd = ctx["hd"]; d = ctx["d"]
    oproj = a["oproj"]; norm = a["norm"]; LB, hB = head

    # which content pattern does the candidate read most (induction vs duplicate)?
    mass = {"induction": 0.0, "duplicate": 0.0}; n = {"induction": 0, "duplicate": 0}
    with torch.no_grad():
        for s in ctx["seqs"][: args.patch_probes]:
            at = model(input_ids=torch.tensor([s], device=dev), output_attentions=True).attentions[LB][0, hB].float().cpu().numpy()
            M = masks(s)
            for pat in mass:
                mass[pat] += float((at * M[pat]).sum()); n[pat] += int(M[pat].sum())
    sig = {pat: mass[pat] / max(n[pat], 1) for pat in mass}
    patt = "induction" if sig["induction"] >= sig["duplicate"] else "duplicate"

    ind_d = ctx["ind_nll"](LB, hB) - ctx["base_ind"]
    gen_d = ctx["gen_nll"](LB, hB) - ctx["base_gen"]

    channel = {"note": "reader in layer 0 — no upstream"}
    if LB > 0:
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
        with torch.no_grad():
            for s in ctx["seqs"][: args.patch_probes]:
                capk.clear(); o = model(input_ids=torch.tensor([s], device=dev), output_attentions=True)
                Msk = masks(s)[patt]; attnB = o.attentions[LB][0, hB]
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
        channel = {"pattern": patt, "key_top": krows[0], "key_median": kmed, "value_top": vrows[0],
                   "value_median": float(np.median([r["dvout"] for r in vrows])), "key_rows": krows[:5], "value_rows": vrows[:5]}
    return {"model": ctx["model_id"].split("/")[-1], "rope": not a["is_gpt2"], "head": f"{LB}.{hB}", "n_layers": ctx["nL"],
            "reads_pattern": patt, "induction_mass": sig["induction"], "duplicate_mass": sig["duplicate"],
            "causal_induction_dNLL": ind_d, "causal_generic_dNLL": gen_d, "channel": channel}


def write_doc(results, docs, root):
    """Render docs/operators/discovered_xmodel.md from the profiles (CPU-only; cross-refs the discovery sweep)."""
    disc = json.loads((root / "discovered_summary.json").read_text()) if (root / "discovered_summary.json").exists() else {}
    seen = {}  # (model, head) -> discovery induction dNLL, for cross-reference
    for r in disc.get("results", []):
        for c in r.get("candidate_unnamed", []) + r.get("top_components", []):
            seen[(r["model"], c["comp"])] = c.get("induction_dNLL_mean")
    ok = [r for r in results if "channel" in r]
    by_model = OrderedDict()
    for r in ok:
        by_model.setdefault(r["model"], []).append(r)
    lines = ["---", "title: Discovered candidates (cross-model)", "---", "",
             "# Discovered candidate operators — cross-model profiles", "",
             "Arch-generic dossiers of the **UNNAMED** load-bearing candidates the [discovery sweep](discovered.md) "
             "surfaced in the RoPE models (which `operator_dossier.py`, GPT-2-only, could not reach). Each: which "
             "content pattern it reads, its causal ΔNLL (mean-ablation), and the channel decomposition (what addresses "
             "its key vs what it moves). Two independent harnesses agree where the **discovery ind ΔNLL** "
             "(multi-seed sweep) and the **profiled causal ind ΔNLL** (this run, fresh probes) line up. Provisional.", "",
             f"**{len(ok)} candidates profiled** across {len(by_model)} models. Sorted by profiled induction ΔNLL.", ""]
    for model, rows in by_model.items():
        rows.sort(key=lambda r: -r["causal_induction_dNLL"])
        lines += [f"## {model}", "",
                  "| head | reads | discovery ind ΔNLL | profiled causal ΔNLL (ind / gen) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            ch = r["channel"]
            kc = f"{ch['key_top']['head']} ({ch['key_top']['collapse']:+.0%})" if "key_top" in ch else "— (layer 0)"
            vc = f"{ch['value_top']['head']} ({ch['value_top']['dvout']:.2f})" if "value_top" in ch else "—"
            dv = seen.get((model, r["head"]))
            ds = f"{dv:+.2f}" if dv is not None else "—"
            lines.append(f"| `{r['head']}` | {r['reads_pattern']} | {ds} | {r['causal_induction_dNLL']:+.2f} / {r['causal_generic_dNLL']:+.2f} | {kc} | {vc} |")
        lines.append("")
    lines += ["_Data: [xmodel_candidates_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/xmodel_candidates_summary.json). "
              "Regenerate (GPU): [xmodel_candidate.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/xmodel_candidate.py); re-render the page (CPU): `xmodel_candidate.py --docs-only`. "
              "The full per-op battery for these is the RoPE-dossier port (future); this is the channel + causal core._"]
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "discovered_xmodel.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", default="google/gemma-2-2b:13.7,unsloth/Llama-3.2-1B:0.31,unsloth/Llama-3.2-1B:1.31,Qwen/Qwen2.5-1.5B:1.6",
                   help="comma-sep model:L.h candidates")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--probes", type=int, default=24)
    p.add_argument("--patch-probes", type=int, default=10)
    p.add_argument("--rep-len", type=int, default=22)
    p.add_argument("--max-upstream", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/operators"))
    p.add_argument("--docs", type=Path, default=Path("docs/operators"))
    p.add_argument("--docs-only", action="store_true", help="re-render the page from the committed summary JSON; no GPU")
    args = p.parse_args(argv)

    if args.docs_only:
        out = json.loads((args.outdir / "xmodel_candidates_summary.json").read_text())
        write_doc(out["results"], args.docs, args.outdir)
        print(f"[docs-only] re-rendered {args.docs / 'discovered_xmodel.md'} from {len(out['results'])} profiles")
        return out

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    # group targets by model so each model loads once (then is freed) — N heads, one load.
    grouped = OrderedDict()
    for tgt in [t.strip() for t in args.targets.split(",") if t.strip()]:
        mid, hd_s = tgt.rsplit(":", 1); grouped.setdefault(mid, []).append(tuple(int(x) for x in hd_s.split(".")))
    results = []
    for mid, heads in grouped.items():
        print(f"\n=== {mid}: {len(heads)} candidate heads ===")
        try:
            ctx = setup(mid, args, dev)
        except Exception as e:  # pragma: no cover
            print(f"  [skip model] {e}")
            for h in heads:
                results.append({"target": f"{mid}:{h[0]}.{h[1]}", "error": str(e)})
            continue
        for (L, h) in heads:
            try:
                r = profile_head(ctx, (L, h), args, dev)
                results.append(r); ch = r["channel"]
                kt = (f"KEY top {ch['key_top']['head']} collapse {ch['key_top']['collapse']:+.0%}; VALUE top {ch['value_top']['head']} ΔV-out {ch['value_top']['dvout']:.2f}"
                      if "key_top" in ch else ch.get("note", ""))
                print(f"  {L}.{h}: reads {r['reads_pattern']} | causal ind {r['causal_induction_dNLL']:+.2f} gen {r['causal_generic_dNLL']:+.2f} | {kt}")
            except Exception as e:  # pragma: no cover
                print(f"  {L}.{h}: [skip] {e}"); results.append({"target": f"{mid}:{L}.{h}", "error": str(e)})
        del ctx
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    out = {"experiment": "arch-generic dossier of discovered RoPE candidate operators", "results": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "xmodel_candidates_summary.json").write_text(json.dumps(out, indent=2, default=float))
    write_doc(results, args.docs, args.outdir)
    ok = [r for r in results if "channel" in r]
    print(f"\n[done] {len(ok)} candidate profiles → {args.outdir / 'xmodel_candidates_summary.json'} + {args.docs / 'discovered_xmodel.md'}")
    return out


if __name__ == "__main__":
    main()
