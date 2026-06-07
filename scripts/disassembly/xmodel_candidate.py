"""Arch-generic candidate profiler — dossier the discovered RoPE candidate operators (any model, any head).

`operator_dossier.py` is GPT-2-only, so the discovery sweep's RoPE candidates (Llama 0.31, Gemma 13.7, …) were
left undossiered. This profiles a given (model, head) arch-generically — the key questions for a discovered op:
  - CAUSAL    : mean-ablate the head → induction + generic ΔNLL (is it load-bearing, and on what?);
  - CHANNEL   : path-patch each upstream head out of the candidate's KEY (faithful key-only patch, model re-applies
                its own RoPE) → top collapser (what addresses it); + the VALUE/move channel (ΔV-out, what it moves);
  - COMPOSITION: weight-space in-edges (which earlier heads' OV feed the candidate's key / value).
Reuses `circuit_content_patch._arch` + the same faithful patch. Emits docs/operators/dossiers/<model>_<head>/ pages.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
from circuit_content_patch import _arch  # noqa: E402


def masks(toks):
    ca = np.array(toks); n = len(ca); qi = np.arange(n); pv = np.full(n, -1); pv[1:] = ca[:-1]
    return {"induction": (pv[None, :] == ca[:, None]) & (qi[None, :] >= 1) & (qi[None, :] < qi[:, None]),
            "duplicate": (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])}


def profile(model_id, head, args, dev):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    a = _arch(model); H = a["H"]; hd = a["hd"]; nL = model.config.num_hidden_layers
    oproj = a["oproj"]; norm = a["norm"]; d = a.get("d", model.config.hidden_size)
    LB, hB = head; V = model.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
    rng = np.random.default_rng(args.seed)
    import urllib.request
    from transformers import AutoTokenizer
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    tok = AutoTokenizer.from_pretrained(model_id); pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]
    seqs = [(lambda s: s + s)([int(x) for x in rng.integers(lo, hi, args.rep_len)]) for _ in range(args.probes)]

    # which pattern does the candidate read most (induction vs duplicate)?
    mass = {"induction": 0.0, "duplicate": 0.0}; n = {"induction": 0, "duplicate": 0}
    with torch.no_grad():
        for s in seqs[: args.patch_probes]:
            o = model(input_ids=torch.tensor([s], device=dev), output_attentions=True)
            at = o.attentions[LB][0, hB].float().cpu().numpy(); M = masks(s)
            for p in mass:
                mass[p] += float((at * M[p]).sum()); n[p] += int(M[p].sum())
    sig = {p: mass[p] / max(n[p], 1) for p in mass}
    patt = "induction" if sig["induction"] >= sig["duplicate"] else "duplicate"

    # ---- causal: mean-ablate the head's o_proj/c_proj slice -> corpus mean ----
    cap = []
    hk = oproj[LB].register_forward_pre_hook(lambda m, inp: cap.append(inp[0].detach().reshape(-1, inp[0].shape[-1])))
    with torch.no_grad():
        for c in chunks:
            model(input_ids=torch.tensor([c], device=dev))
    hk.remove(); mean_slice = torch.cat(cap, 0).mean(0)[hB * hd:(hB + 1) * hd]

    def ablate_hook(m, inp):
        x = inp[0].clone(); x[..., hB * hd:(hB + 1) * hd] = mean_slice.to(x.dtype); return (x,)

    def gen_nll(ablate=False):
        h = oproj[LB].register_forward_pre_hook(ablate_hook) if ablate else None
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

    def ind_nll(ablate=False):
        h = oproj[LB].register_forward_pre_hook(ablate_hook) if ablate else None
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
    gen_d = gen_nll(True) - gen_nll(); ind_d = ind_nll(True) - ind_nll()

    # ---- channel: key-collapse (top upstream writer) + value ΔV-out ----
    channel = {"note": "reader in layer 0 — no upstream"}
    if LB > 0:
        upstream = [(L, h) for L in range(LB) for h in range(H)][: args.max_upstream]

        def head_contrib(L, captured, h):
            x = torch.zeros_like(captured); x[..., h * hd:(h + 1) * hd] = captured[..., h * hd:(h + 1) * hd]
            return oproj[L](x) - oproj[L](torch.zeros_like(captured[..., :1, :]))
        kvB = hB // (H // a["nkv"])

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
            for s in seqs[: args.patch_probes]:
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
    return {"model": model_id.split("/")[-1], "rope": not a["is_gpt2"], "head": f"{LB}.{hB}", "n_layers": nL,
            "reads_pattern": patt, "induction_mass": sig["induction"], "duplicate_mass": sig["duplicate"],
            "causal_induction_dNLL": ind_d, "causal_generic_dNLL": gen_d, "channel": channel}


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
    args = p.parse_args(argv)

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for tgt in [t.strip() for t in args.targets.split(",") if t.strip()]:
        mid, hd_s = tgt.rsplit(":", 1); L, h = (int(x) for x in hd_s.split("."))
        print(f"\n=== {mid} head {L}.{h} ===")
        try:
            r = profile(mid, (L, h), args, dev)
            if dev == "cuda":
                torch.cuda.empty_cache()
            results.append(r)
            ch = r["channel"]
            kt = (f"KEY top {ch['key_top']['head']} collapse {ch['key_top']['collapse']:+.0%}; VALUE top {ch['value_top']['head']} ΔV-out {ch['value_top']['dvout']:.2f}"
                  if "key_top" in ch else ch.get("note", ""))
            print(f"  reads {r['reads_pattern']} (ind-mass {r['induction_mass']:.2f}) | causal induction {r['causal_induction_dNLL']:+.2f} generic {r['causal_generic_dNLL']:+.2f} | {kt}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"target": tgt, "error": str(e)})

    out = {"experiment": "arch-generic dossier of discovered RoPE candidate operators", "results": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "xmodel_candidates_summary.json").write_text(json.dumps(out, indent=2, default=float))
    ok = [r for r in results if "channel" in r]
    lines = ["---", "title: Discovered candidates (cross-model)", "---", "",
             "# Discovered candidate operators — cross-model profiles", "",
             "Arch-generic dossiers of the strongest **UNNAMED** load-bearing candidates the "
             "[discovery sweep](discovered.md) surfaced in the RoPE models (which `operator_dossier.py`, GPT-2-only, "
             "could not reach). Each: which content pattern it reads, its causal ΔNLL (mean-ablation), and the "
             "channel decomposition (what addresses its key vs what it moves). Provisional.", "",
             "| model | head | reads | causal ΔNLL (ind / gen) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |",
             "|---|---|---|---|---|---|"]
    for r in ok:
        ch = r["channel"]
        kc = f"{ch['key_top']['head']} ({ch['key_top']['collapse']:+.0%})" if "key_top" in ch else "—"
        vc = f"{ch['value_top']['head']} ({ch['value_top']['dvout']:.2f})" if "value_top" in ch else "—"
        lines.append(f"| {r['model']} | {r['head']} | {r['reads_pattern']} | {r['causal_induction_dNLL']:+.2f} / {r['causal_generic_dNLL']:+.2f} | {kc} | {vc} |")
    lines += ["", "_Data: [runs/disassembly/operators/xmodel_candidates_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/xmodel_candidates_summary.json). Regenerate: [xmodel_candidate.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/xmodel_candidate.py). "
              "The full per-op battery for these is the RoPE-dossier port (future); this is the channel + causal core._"]
    args.docs.mkdir(parents=True, exist_ok=True)
    (args.docs / "discovered_xmodel.md").write_text("\n".join(lines))
    print(f"\n[done] {len(ok)} candidate profiles → {args.outdir / 'xmodel_candidates_summary.json'} + {args.docs / 'discovered_xmodel.md'}")
    return out


if __name__ == "__main__":
    main()
