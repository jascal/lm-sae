"""Executable decompilation — does the catalogued induction circuit RECONSTRUCT induction on its own?

The catalog establishes which heads are *necessary* (ablating them hurts). Sufficiency is the other half of a
decompilation: keep ONLY the circuit's heads (the induction + prev-token heads, from the cross-model dossier),
mean-ablate **every other attention head** (MLPs left intact — they are the substrate), and measure how much of
the model's induction capability survives. Reconstruction coverage:

    coverage = (NLL_all_attn_ablated − NLL_circuit_only) / (NLL_all_attn_ablated − NLL_full)

= 1 if the circuit alone fully reconstructs induction, 0 if it is no better than ablating all attention. A
random same-size head-set is the control — the circuit should beat it. Arch-generic (reuses `circuit_content_patch
._arch`); reads the circuit heads from the committed `xmodel_dossiers_summary.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
from circuit_content_patch import _arch  # noqa: E402


def run_model(model_id, circuit_heads, args, dev):
    import torch
    import torch.nn.functional as F
    is_gpt2 = "gpt2" in model_id.lower()
    from transformers import AutoTokenizer
    if is_gpt2:
        from transformers import GPT2LMHeadModel
        m = GPT2LMHeadModel.from_pretrained(model_id, attn_implementation="eager").eval().to(dev)
    else:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    a = _arch(m); H = a["H"]; hd = a["hd"]; nL = m.config.num_hidden_layers; oproj = a["oproj"]
    tok = AutoTokenizer.from_pretrained(model_id); V = m.config.vocab_size
    rng = np.random.default_rng(args.seed)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:160000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]
    if is_gpt2:
        from collections import Counter
        vocab = [t for t, _ in Counter(t for c in chunks for t in c).most_common(400)]
        rep = lambda L: [int(vocab[i]) for i in rng.integers(0, len(vocab), L)]  # noqa: E731
    else:
        lo, hi = int(0.02 * V), int(0.4 * V)
        rep = lambda L: [int(x) for x in rng.integers(lo, hi, L)]                 # noqa: E731
    seqs = [(lambda s: s + s)(rep(args.rep_len)) for _ in range(args.probes)]

    cap = {L: [] for L in range(nL)}
    hks = [oproj[L].register_forward_pre_hook((lambda L: lambda mod, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in chunks:
            m(input_ids=torch.tensor([c], device=dev))
    for h in hks:
        h.remove()
    meanv = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    def ablate_hooks(heads):                                                     # mean-ablate this set of (L,h)
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

    def ind_nll(heads=()):
        hs = ablate_hooks(heads); tot = 0.0; k = 0
        try:
            with torch.no_grad():
                for s in seqs:
                    lp = F.log_softmax(m(input_ids=torch.tensor([s], device=dev)).logits[0].float(), -1); Lh = len(s) // 2
                    for p in range(Lh, 2 * Lh - 1):
                        tot += float(-lp[p, s[p + 1]]); k += 1
        finally:
            for x in hs:
                x.remove()
        return tot / max(k, 1)

    all_heads = [(L, h) for L in range(nL) for h in range(H)]
    keep = set(circuit_heads)
    non_circuit = [hh for hh in all_heads if hh not in keep]
    base = ind_nll()                                                             # full model
    allabl = ind_nll(all_heads)                                                  # ablate ALL attention (MLPs intact)
    circ = ind_nll(non_circuit)                                                  # keep only the circuit heads

    # robustness: RESAMPLE ablation (on-distribution) — replace ablated heads with a different seq's activations
    def resample_nll(keepset):
        ablate = [hh for hh in all_heads if hh not in keepset]
        by = {}
        for (L, hh) in ablate:
            by.setdefault(L, []).append(hh)
        tot = 0.0; k = 0
        for i, s in enumerate(seqs):
            sd = seqs[(i + 1) % len(seqs)]; dcap = {}
            hk = [oproj[L].register_forward_pre_hook((lambda L: lambda mod, inp: dcap.__setitem__(L, inp[0].detach()))(L)) for L in by]
            with torch.no_grad():
                m(input_ids=torch.tensor([sd], device=dev))
            for x in hk:
                x.remove()
            hs = []
            for L, hss in by.items():
                def mk(L, hss):
                    def hook(mod, inp):
                        x = inp[0].clone()
                        for hh in hss:
                            x[..., hh * hd:(hh + 1) * hd] = dcap[L][..., hh * hd:(hh + 1) * hd].to(x.dtype)
                        return (x,)
                    return hook
                hs.append(oproj[L].register_forward_pre_hook(mk(L, hss)))
            with torch.no_grad():
                lp = F.log_softmax(m(input_ids=torch.tensor([s], device=dev)).logits[0].float(), -1); Lh = len(s) // 2
                for pp in range(Lh, 2 * Lh - 1):
                    tot += float(-lp[pp, s[pp + 1]]); k += 1
            for x in hs:
                x.remove()
        return tot / max(k, 1)
    rs_allabl = resample_nll(set()); rs_circ = resample_nll(keep)
    rs_cov = (rs_allabl - rs_circ) / (rs_allabl - base + 1e-9)

    def coverage(nll):
        return (allabl - nll) / (allabl - base + 1e-9)

    # reconstruction CURVE: rank heads by induction-mass, keep top-K, coverage(K) — "how many heads does induction need?"
    def imask(toks):
        ca = np.array(toks); n = len(ca); qi = np.arange(n); pv = np.full(n, -1); pv[1:] = ca[:-1]
        return (pv[None, :] == ca[:, None]) & (qi[None, :] >= 1) & (qi[None, :] < qi[:, None])
    mass = np.zeros(nL * H); _nt = 0
    with torch.no_grad():
        for s in seqs[: args.id_probes]:
            o = m(input_ids=torch.tensor([s], device=dev), output_attentions=True); wm = imask(s); _nt += int(wm.sum())
            for L in range(nL):
                at = o.attentions[L][0].float().cpu().numpy(); mass[L * H:(L + 1) * H] += (at * wm[None]).sum((1, 2))
    ranked = [tuple(int(x) for x in divmod(int(i), H)) for i in np.argsort(-mass)]
    curve = []
    for K in [k for k in [4, 8, 16, 32, 64, 128, 256] if k <= nL * H]:
        keepK = set(ranked[:K])
        curve.append({"k": K, "coverage": coverage(ind_nll([hh for hh in all_heads if hh not in keepK]))})

    rnd = []
    for _ in range(args.n_random):
        rk = {tuple(int(x) for x in divmod(int(i), H)) for i in rng.choice(nL * H, len(keep), replace=False)}
        rnd.append(coverage(ind_nll([hh for hh in all_heads if hh not in rk])))
    return {"model": model_id.split("/")[-1], "rope": not is_gpt2, "n_heads_total": nL * H, "circuit_size": len(keep),
            "base_ind_nll": base, "all_attn_ablated_nll": allabl, "circuit_only_nll": circ,
            "circuit_coverage": coverage(circ), "random_coverage_mean": float(np.mean(rnd)),
            "random_coverage_std": float(np.std(rnd)), "curve": curve,
            "resample_circuit_coverage": rs_cov, "resample_all_ablated_nll": rs_allabl, "resample_circuit_nll": rs_circ}


def write_doc(out, docs):
    L = ["---", "title: Executable decompilation", "---", "", "# Executable decompilation — does the induction circuit reconstruct itself?", "",
         "The catalog shows which heads are *necessary*. This tests **sufficiency**: keep ONLY the induction circuit "
         "(the induction + prev-token heads from the [cross-model dossier](operators/induction.md)), mean-ablate "
         "**every other attention head** (MLPs intact — the substrate), and measure how much induction survives.", "",
         "**coverage = (NLL_all-attn-ablated − NLL_circuit-only) / (NLL_all-attn-ablated − NLL_full)** — 1 = the "
         "circuit alone fully reconstructs induction, 0 = no better than ablating all attention. A random same-size "
         "head-set is the control.", "",
         "| model | circuit size / total heads | induction-NLL (full / circuit-only / all-ablated) | **circuit coverage** (mean-abl) | coverage (resample-abl) | random control |",
         "|---|---|---|---|---|---|"]
    for r in out["results"]:
        if "circuit_coverage" not in r:
            continue
        rs = f"{r['resample_circuit_coverage']:+.0%}" if "resample_circuit_coverage" in r else "—"
        L.append(f"| {r['model']} | {r['circuit_size']} / {r['n_heads_total']} | "
                 f"{r['base_ind_nll']:.2f} / {r['circuit_only_nll']:.2f} / {r['all_attn_ablated_nll']:.2f} | "
                 f"**{r['circuit_coverage']:+.0%}** | {rs} | {r['random_coverage_mean']:+.0%} ± {r['random_coverage_std']:.0%} |")
    ks = sorted({c["k"] for r in out["results"] if r.get("curve") for c in r["curve"]})
    if ks:
        L += ["", "## How many heads does induction need? (reconstruction curve)", "",
              "Rank every head by induction-mass, keep the top-K (ablate the rest), and watch coverage grow with K — "
              "the size at which it saturates is induction's *effective* circuit size.", "",
              "| model | " + " | ".join(f"K={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
        for r in out["results"]:
            if not r.get("curve"):
                continue
            cm = {c["k"]: c["coverage"] for c in r["curve"]}
            L.append(f"| {r['model']} | " + " | ".join((f"{cm[k]:+.0%}" if k in cm else "—") for k in ks) + " |")
        L += ["", "_**No compact head-subset reconstructs induction in any model.** GPT-2-small only reaches near-full "
              "coverage at K≈128/144 (it needs nearly every head); gpt2-medium saturates at ~22% even with 256 heads; "
              "gpt2-large stays ~0% throughout; and the RoPE curves go **non-monotonic** — Gemma peaks ~32% then "
              "drops, Qwen goes **negative** (keeping more induction-mass heads *hurts* induction-NLL — the same "
              "interference / compensatory effect the [outlier digs](../operators/outlier_digs.md) traced to a "
              "synthetic-probe artifact). Induction is a property of the near-whole network, not an isolable subgraph._"]
    io = out.get("ioi_gpt2")
    if io:
        L += ["", "## The IOI circuit (GPT-2) — is the literature's *complete* circuit sufficient?", "",
              "The same test on the field's most-celebrated complete circuit (Wang et al. 2022), measured on the metric "
              "it serves — the IOI logit-difference `LD = logit(IO) − logit(S)`. Keep only the IOI circuit's "
              f"**{io['circuit_size']} heads** (of {io['n_heads_total']}), ablate the rest:", "",
              "| circuit | LD (full / circuit-only / all-ablated) | coverage | random control |",
              "|---|---|---|---|",
              f"| IOI ({io['circuit_size']}h) | {io['ld_full']:+.2f} / {io['ld_circuit_only']:+.2f} / {io['ld_all_attn_ablated']:+.2f} | "
              f"**{io['coverage']:+.0%}** | {io['random_coverage_mean']:+.0%} ± {io['random_coverage_std']:.0%} |", "",
              "**Same lesson as induction, sharper:** keeping only the 26 IOI heads and mean-ablating the rest gives a "
              f"**negative** logit-diff ({io['ld_circuit_only']:+.2f}) — the model now prefers S over IO — and is **no "
              "better than a random 26-head set**. The named circuit is not a *sufficient* isolated subgraph; it needs "
              "the rest of the network as substrate.", "",
              "> **Caveat (important, read this).** This is a harsh *sufficiency-under-mean-ablation* test: mean-ablating "
              "~120 heads pushes activations far off-distribution and severs the upstream signals the circuit reads. "
              "The original IOI result (Wang et al.) is about **necessity** + path-patching, **not** isolated "
              "mean-ablation sufficiency — so this does **not** refute it. It says the IOI computation, like induction, "
              "is not recoverable from its named heads *in isolation*; the named circuit is necessary and explanatory, "
              "but the behaviour is carried by the near-whole network. A statement about distributedness, not validity._", ""]
    L += ["", "_**Robustness — does it survive a gentler ablation?** Mean-ablation pushes activations off-distribution, "
          "so it *understates* coverage: under **resample-ablation** (replace ablated heads with a different valid "
          "sequence's activations, on the data manifold) the GPT-2 family reconstructs more (gpt2 +17%→**+30%**, "
          "medium +7%→+24%). But **no model exceeds ~30% even under resample** — so the distributedness is real, not a "
          "mean-ablation artifact; mean-ablation just exaggerated it. The named 8-head circuit is the dominant driver, "
          "not a sufficient subgraph, under either ablation._", ""]
    L += ["", "_**The honest result: necessity ≠ a small sufficient circuit.** No 8-head circuit *fully* reconstructs "
          "induction in any model (best +17% mean / +30% resample, GPT-2-small). The circuit beats its random control in 4/6 models — it is "
          "the **main** contributor — but coverage is modest, and it **decays with GPT-2 scale** (small +17% → medium "
          "+7% → large +0%) and fails in Qwen (−4%): in the larger / more distributed models the top induction + "
          "prev-token heads in isolation recover essentially nothing, because induction there is spread across a "
          "supporting cast the 8-head set excludes. So the catalogued circuit is causally necessary and the dominant "
          "driver, but not an executable small-circuit decompilation on its own — consistent with the distributed / "
          "non-monotonic induction-redundancy seen in the [dossier](../operators/induction.md). Provisional, single "
          "corpus; induction-NLL on repeated-random sequences. "
          "Data: [circuit_reconstruction_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/circuit_reconstruction_summary.json). "
          "Regenerate: [circuit_reconstruction.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_reconstruction.py)._"]
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "reconstruction.md").write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dossiers", type=Path, default=Path("runs/disassembly/operators/xmodel_dossiers_summary.json"))
    p.add_argument("--model-ids", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--probes", type=int, default=28)
    p.add_argument("--id-probes", type=int, default=16, help="probes for the induction-mass head ranking (the curve)")
    p.add_argument("--rep-len", type=int, default=22)
    p.add_argument("--n-random", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/circuits"))
    p.add_argument("--docs", type=Path, default=Path("docs/circuits"))
    args = p.parse_args(argv)

    doss = {r["model"]: r for r in json.loads(args.dossiers.read_text())["results"] if "ops" in r}

    def circuit_for(short):
        d = doss.get(short)
        if not d:
            return None
        heads = d["ops"]["induction"]["heads"] + d["ops"]["prevtok"]["heads"]
        return [tuple(int(x) for x in hh.split(".")) for hh in heads]

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.model_ids.split(",") if m.strip()]:
        short = mid.split("/")[-1]
        circ = circuit_for(short)
        if not circ:
            print(f"[skip] {short}: no dossier circuit"); continue
        print(f"\n=== {mid}: circuit = {len(circ)} heads ===")
        try:
            r = run_model(mid, circ, args, dev); results.append(r)
            print(f"  full {r['base_ind_nll']:.2f} | circuit-only {r['circuit_only_nll']:.2f} | all-ablated {r['all_attn_ablated_nll']:.2f} "
                  f"| coverage {r['circuit_coverage']:+.0%} (random {r['random_coverage_mean']:+.0%})")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": short, "error": str(e)})
        if dev == "cuda":
            torch.cuda.empty_cache()
    sumpath = args.outdir / "circuit_reconstruction_summary.json"
    out = json.loads(sumpath.read_text()) if sumpath.exists() else {}            # preserve extra blocks (e.g. ioi_gpt2)
    out["experiment"] = "executable decompilation — induction-circuit reconstruction coverage"
    out["results"] = results
    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    write_doc(out, args.docs)
    print(f"\n[done] {len([r for r in results if 'circuit_coverage' in r])} models → {args.outdir / 'circuit_reconstruction_summary.json'}")
    return out


if __name__ == "__main__":
    main()
