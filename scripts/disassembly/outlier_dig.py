"""Mechanism digs on the cross-model outliers the catalog surfaced (both reuse `circuit_content_patch._arch`).

DIG 1 — `mlp0`: why is Llama-3.2-1B's MLP0 *context*-determined (token-determinism η²≈0.01) when GPT-2 / Gemma /
Qwen MLP0 is *token*-determined (0.6–0.9)? Decompose what enters MLP0: the layer-0 **attention** contribution vs
the raw token embedding (residual entering layer 0). We report, per model, (a) the relative norm of the L0
attention write vs the embedding at MLP0's input, and (b) the token-determinism (η²) of the embedding, of the L0
attention output, and of MLP0's output. Hypothesis: Llama's early heads inject a large, context-determined
component into MLP0's input (so MLP0 processes a context-mixed residual), where GPT-2's L0 attention is small/sink.

DIG 2 — `compensatory`: Gemma & gpt2-large induction redundancy is COMPENSATORY — ablating the top induction heads
*together* recovers induction (the cumulative curve is non-monotonic). Leave-one-out marginals of the induction
top-k identify WHICH head, when ablated, triggers the recovery: a **negative LOO marginal** (effect(full) <
effect(full∖{h})) means ablating h *reduces* induction damage — a net suppressor / self-repair trigger — vs a
distributed control (gpt2-small, where every marginal is positive).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_content_patch import _arch  # noqa: E402
from mlp_atlas import mlp_blocks  # noqa: E402


def eta2(stats):
    """token-determinism η² from accumulated per-token {n, sum_vec, sumsq_scalar} + globals, or None."""
    n = stats["n"]
    if n == 0:
        return None
    total = stats["ss"] / n - float((stats["sum"] / n) @ (stats["sum"] / n))
    wv = 0.0; wn = 0
    for _t, (nt, s, ss) in stats["tok"].items():
        if nt < stats["min_count"]:
            continue
        wv += (ss / nt - float((s / nt) @ (s / nt))) * nt; wn += nt
    return float(1 - (wv / max(wn, 1)) / total) if total > 1e-9 else None


def new_stats(min_count):
    return {"tok": defaultdict(lambda: [0, None, 0.0]), "n": 0, "sum": None, "ss": 0.0, "min_count": min_count}


def accum(stats, vecs, toks, freq):
    nsq = (vecs * vecs).sum(-1)
    for pos, tid in enumerate(toks):
        if tid not in freq:
            continue
        v = vecs[pos]; rec = stats["tok"][tid]
        rec[0] += 1; rec[1] = v.copy() if rec[1] is None else rec[1] + v; rec[2] += float(nsq[pos])
        stats["n"] += 1; stats["sum"] = v.copy() if stats["sum"] is None else stats["sum"] + v; stats["ss"] += float(nsq[pos])


def corpus(model_id, tok, args):
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:300000]
    pids = tok(prose)["input_ids"]
    return [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]


def load(model_id, dev):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    is_gpt2 = "gpt2" in model_id.lower()
    if is_gpt2:
        from transformers import GPT2LMHeadModel
        m = GPT2LMHeadModel.from_pretrained(model_id, attn_implementation="eager").eval().to(dev)
    else:
        m = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    return m, AutoTokenizer.from_pretrained(model_id), _arch(m)


def dig_mlp0(model_id, args, dev):
    """Decompose MLP0's input: L0-attention contribution vs embedding, + token-determinism of each."""
    import numpy as np
    import torch
    m, tok, a = load(model_id, dev)
    oproj = a["oproj"]; layers = a["layers"]; mlp0 = mlp_blocks(m)[0]
    chunks = corpus(model_id, tok, args)
    freq = set(t for t, _ in Counter(t for c in chunks for t in c).most_common(args.n_tok))
    st_emb = new_stats(args.min_count); st_attn = new_stats(args.min_count); st_mlp = new_stats(args.min_count)
    cap = {}
    h_pre = layers[0].register_forward_pre_hook(lambda mod, inp: cap.__setitem__("resid_pre", inp[0].detach()))   # residual entering layer 0
    h_attn = oproj[0].register_forward_hook(lambda mod, i, o: cap.__setitem__("attn", (o[0] if isinstance(o, tuple) else o).detach()))
    h_mlp = mlp0.register_forward_hook(lambda mod, i, o: cap.__setitem__("mlp", (o[0] if isinstance(o, tuple) else o).detach()))
    ratios = []
    with torch.no_grad():
        for c in chunks:
            cap.clear(); m(input_ids=torch.tensor([c], device=dev))
            emb = cap["resid_pre"][0].float().cpu().numpy(); at = cap["attn"][0].float().cpu().numpy(); mp = cap["mlp"][0].float().cpu().numpy()
            ratios.append(float(np.linalg.norm(at, axis=-1).mean() / (np.linalg.norm(emb, axis=-1).mean() + 1e-9)))
            accum(st_emb, emb, c, freq); accum(st_attn, at, c, freq); accum(st_mlp, mp, c, freq)
    for h in (h_pre, h_attn, h_mlp):
        h.remove()
    return {"model": model_id.split("/")[-1], "attn_over_emb_norm": float(np.mean(ratios)),
            "det_embedding": eta2(st_emb), "det_L0_attn_out": eta2(st_attn), "det_mlp0_out": eta2(st_mlp)}


def dig_compensatory(model_id, args, dev):
    """Leave-one-out marginals of the induction top-k → which head's ablation triggers the recovery."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    m, tok, a = load(model_id, dev)
    H = a["H"]; hd = a["hd"]; nL = m.config.num_hidden_layers; oproj = a["oproj"]
    chunks = corpus(model_id, tok, args); V = m.config.vocab_size
    rng = np.random.default_rng(args.seed)
    if "gpt2" in model_id.lower():
        cnt = Counter(t for c in chunks for t in c); vocab = [t for t, _ in cnt.most_common(400)]
        rep = lambda L: [int(vocab[i]) for i in rng.integers(0, len(vocab), L)]  # noqa: E731
    else:
        lo, hi = int(0.02 * V), int(0.4 * V)
        rep = lambda L: [int(x) for x in rng.integers(lo, hi, L)]                 # noqa: E731
    seqs = [(lambda s: s + s)(rep(args.rep_len)) for _ in range(args.probes)]

    # induction-mass ranking → top-k heads
    def imask(toks):
        ca = np.array(toks); n = len(ca); qi = np.arange(n); pv = np.full(n, -1); pv[1:] = ca[:-1]
        return (pv[None, :] == ca[:, None]) & (qi[None, :] >= 1) & (qi[None, :] < qi[:, None])
    mass = np.zeros(nL * H); ntot = 0
    with torch.no_grad():
        for s in seqs[: args.id_probes]:
            o = m(input_ids=torch.tensor([s], device=dev), output_attentions=True); wm = imask(s); ntot += int(wm.sum())
            for L in range(nL):
                at = o.attentions[L][0].float().cpu().numpy(); mass[L * H:(L + 1) * H] += (at * wm[None]).sum((1, 2))
    mass /= max(ntot, 1)
    top = [int(i) for i in np.argsort(-mass)[: args.topk]]
    top_hh = [(i // H, i % H) for i in top]

    cap = {L: [] for L in range(nL)}
    hks = [oproj[L].register_forward_pre_hook((lambda L: lambda mod, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in chunks:
            m(input_ids=torch.tensor([c], device=dev))
    for h in hks:
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

    def ind_nll(heads=()):
        hs = ablate(heads); tot = 0.0; k = 0
        try:
            with torch.no_grad():
                for s in seqs:
                    lp = F.log_softmax(m(input_ids=torch.tensor([s], device=dev)).logits[0].float(), -1); L = len(s) // 2
                    for p in range(L, 2 * L - 1):
                        tot += float(-lp[p, s[p + 1]]); k += 1
        finally:
            for h in hs:
                h.remove()
        return tot / max(k, 1)
    base = ind_nll(); full = ind_nll(set(top_hh)) - base
    solo = {f"{L}.{h}": ind_nll({(L, h)}) - base for (L, h) in top_hh}
    loo = {f"{L}.{h}": full - (ind_nll(set(top_hh) - {(L, h)}) - base) for (L, h) in top_hh}   # marginal of h within the full set
    comp = sorted(loo.items(), key=lambda r: r[1])  # most-negative marginal first = the suppressor / recovery trigger
    return {"model": model_id.split("/")[-1], "induction_top": [f"{L}.{h}" for (L, h) in top_hh],
            "base_induction": base, "full_effect": full, "solo": solo, "loo_marginal": loo,
            "compensator": comp[0][0] if comp and comp[0][1] < 0 else None, "min_marginal": comp[0][1] if comp else None}


def ov_copy_score(m, a, head, dev):
    """First-order OV→unembed copy-score sign for a head: +ve = copies the attended token (raises its logit),
    −ve = suppresses it (copy-suppression / negative head). Arch-generic; folds no LN gain (sign estimate)."""
    import numpy as np
    LB, hB = head; H = a["H"]; hd = a["hd"]; nkv = a["nkv"]; d = a.get("d", m.config.hidden_size)
    kvB = hB // (H // nkv)
    if a["is_gpt2"]:
        Wv = a["cattn"][LB].weight.detach().float().cpu().numpy()[:, 2 * d:3 * d]    # (d, d) value block
        Wv_h = Wv[:, hB * hd:(hB + 1) * hd]                                          # (d, hd)
        Wo_h = a["oproj"][LB].weight.detach().float().cpu().numpy()[hB * hd:(hB + 1) * hd, :]   # (hd, d)
    else:
        Wv_h = a["vproj"][LB].weight.detach().float().cpu().numpy()[kvB * hd:(kvB + 1) * hd, :].T   # (d, hd)
        Wo_h = a["oproj"][LB].weight.detach().float().cpu().numpy()[:, hB * hd:(hB + 1) * hd].T     # (hd, d)
    OV = Wv_h @ Wo_h                                                                 # (d, d)
    E = m.get_input_embeddings().weight.detach().float().cpu().numpy()
    U = m.get_output_embeddings().weight.detach().float().cpu().numpy()
    idx = np.random.default_rng(0).choice(E.shape[0], min(400, E.shape[0]), replace=False)
    sc = []
    for t in idx:
        ov = E[t] @ OV
        sc.append(float(ov @ U[t]) / (np.linalg.norm(ov) * np.linalg.norm(U[t]) + 1e-9))
    return float(np.mean(sc))


def _ind_nll_on(m, seqs, oproj, hd, meanv, dev, ablate=None):
    import torch
    import torch.nn.functional as F
    hs = []
    if ablate is not None:
        L, h = ablate

        def hook(mod, inp):
            x = inp[0].clone(); x[..., h * hd:(h + 1) * hd] = meanv[L][h * hd:(h + 1) * hd].to(x.dtype); return (x,)
        hs = [oproj[L].register_forward_pre_hook(hook)]
    tot = 0.0; k = 0
    try:
        with torch.no_grad():
            for s in seqs:
                lp = F.log_softmax(m(input_ids=torch.tensor([s], device=dev)).logits[0].float(), -1); half = len(s) // 2
                for p in range(half, 2 * half - 1):
                    tot += float(-lp[p, s[p + 1]]); k += 1
    finally:
        for x in hs:
            x.remove()
    return tot / max(k, 1)


def _meanv(m, chunks, oproj, nL, dev):
    import torch
    cap = {L: [] for L in range(nL)}
    hks = [oproj[L].register_forward_pre_hook((lambda L: lambda mod, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in chunks:
            m(input_ids=torch.tensor([c], device=dev))
    for h in hks:
        h.remove()
    return {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}


def dig_suppressor(model_id, heads, args, dev):
    """For each suppressor (+ workhorse for contrast): OV copy-score sign + ablation effect on induction over
    NATURAL-repeated text (real passage + itself) vs SYNTHETIC-repeated (random tokens + itself)."""
    import numpy as np
    m, tok, a = load(model_id, dev)
    hd = a["hd"]; nL = m.config.num_hidden_layers; oproj = a["oproj"]; V = m.config.vocab_size
    chunks = corpus(model_id, tok, args); rng = np.random.default_rng(args.seed)
    nat = [c + c for c in chunks[: args.probes]]                                      # real text repeated
    if "gpt2" in model_id.lower():
        vocab = [t for t, _ in Counter(t for c in chunks for t in c).most_common(400)]
        syn = [(lambda s: s + s)([int(vocab[i]) for i in rng.integers(0, len(vocab), args.rep_len)]) for _ in range(args.probes)]
    else:
        lo, hi = int(0.02 * V), int(0.4 * V)
        syn = [(lambda s: s + s)([int(x) for x in rng.integers(lo, hi, args.rep_len)]) for _ in range(args.probes)]
    meanv = _meanv(m, chunks, oproj, nL, dev)
    base_nat = _ind_nll_on(m, nat, oproj, hd, meanv, dev); base_syn = _ind_nll_on(m, syn, oproj, hd, meanv, dev)
    recs = []
    for (L, h) in heads:
        rec = {"head": f"{L}.{h}", "ov_copy_score": ov_copy_score(m, a, (L, h), dev),
               "nat_induction_dNLL": _ind_nll_on(m, nat, oproj, hd, meanv, dev, (L, h)) - base_nat,
               "syn_induction_dNLL": _ind_nll_on(m, syn, oproj, hd, meanv, dev, (L, h)) - base_syn}
        recs.append(rec)
        print(f"  {L}.{h}: OV copy {rec['ov_copy_score']:+.3f} | ablate ΔNLL natural {rec['nat_induction_dNLL']:+.2f} synthetic {rec['syn_induction_dNLL']:+.2f}")
    return {"model": model_id.split("/")[-1], "base_nat": base_nat, "base_syn": base_syn, "heads": recs}


def dig_llama_l0(model_id, heads, args, dev):
    """Are the layer-0 induction-mass heads single-layer inductors? induction-mass vs duplicate-mass + OV copy sign."""
    import numpy as np
    import torch
    m, tok, a = load(model_id, dev)
    V = m.config.vocab_size; rng = np.random.default_rng(args.seed)
    lo, hi = int(0.02 * V), int(0.4 * V)
    seqs = [(lambda s: s + s)([int(x) for x in rng.integers(lo, hi, args.rep_len)]) for _ in range(args.id_probes)]

    def masks(toks):
        ca = np.array(toks); n = len(ca); qi = np.arange(n); pv = np.full(n, -1); pv[1:] = ca[:-1]
        return {"induction": (pv[None, :] == ca[:, None]) & (qi[None, :] >= 1) & (qi[None, :] < qi[:, None]),
                "duplicate": (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])}
    im = {f"{L}.{h}": 0.0 for (L, h) in heads}; dm = {f"{L}.{h}": 0.0 for (L, h) in heads}; ni = nd = 0
    with torch.no_grad():
        for s in seqs:
            o = m(input_ids=torch.tensor([s], device=dev), output_attentions=True); M = masks(s)
            ni += int(M["induction"].sum()); nd += int(M["duplicate"].sum())
            for (L, h) in heads:
                at = o.attentions[L][0, h].float().cpu().numpy()
                im[f"{L}.{h}"] += float((at * M["induction"]).sum()); dm[f"{L}.{h}"] += float((at * M["duplicate"]).sum())
    recs = [{"head": f"{L}.{h}", "induction_mass": im[f"{L}.{h}"] / max(ni, 1), "duplicate_mass": dm[f"{L}.{h}"] / max(nd, 1),
             "ov_copy_score": ov_copy_score(m, a, (L, h), dev)} for (L, h) in heads]
    for r in recs:
        print(f"  {r['head']}: induction-mass {r['induction_mass']:.3f} duplicate-mass {r['duplicate_mass']:.3f} OV copy {r['ov_copy_score']:+.3f}")
    return {"model": model_id.split("/")[-1], "heads": recs}


def write_doc(out, docs):
    L = ["---", "title: Outlier mechanism digs", "---", "", "# Outlier mechanism digs", "",
         "Targeted follow-ups on the two recurring outliers the cross-model dossier surfaced. Provisional.", ""]
    if out.get("mlp0"):
        L += ["## Why is Llama-3.2-1B's MLP0 context-determined?", "",
              "MLP0's input is the token embedding **plus** the layer-0 attention write. If the early attention "
              "injects a large, context-determined component, MLP0 processes a context-mixed residual and its output "
              "is no longer token-determined. Per model: the L0-attention write size relative to the embedding, and "
              "the token-determinism (η²) of the embedding, the L0 attention output, and MLP0's output.", "",
              "| model | ‖L0 attn‖ / ‖embedding‖ | η² embedding | η² L0 attn-out | η² MLP0-out |",
              "|---|---|---|---|---|"]
        for r in out["mlp0"]:
            def f(x):
                return f"{x:.2f}" if x is not None else "n/a"
            L.append(f"| {r['model']} | {r['attn_over_emb_norm']:.2f} | {f(r['det_embedding'])} | {f(r['det_L0_attn_out'])} | {f(r['det_mlp0_out'])} |")
        L += ["", "_Reading: it is the **determinism of the L0 attention output**, not its size, that distinguishes "
              "Llama. Qwen's L0 attention is far *larger* (≈12× the embedding) yet token-determined (η² 0.84), so its "
              "MLP0 stays token-ish; Gemma's L0 attention is tiny (0.18×) so MLP0 ≈ the pure embedding (0.91). "
              "**Llama's L0 attention is both comparable in size to the embedding and the most context-determined "
              "(η² 0.49)** — its layer-0 induction-mass head cluster does genuine context-mixing — so MLP0 ingests a "
              "large context-laden component and its output carries ~no token-determinism (η²≈0; small negatives are "
              "estimation noise). The context-dependence is inherited from the early heads, not intrinsic to MLP0. "
              "(GPT-2's embedding η²<1 is its absolute positional embedding adding position variance to the residual; "
              "the RoPE models read 1.00 — no positional component in the residual.)_", ""]
    if out.get("compensatory"):
        L += ["## Compensatory induction: which head triggers the recovery?", "",
              "For each model's top-k induction heads (by induction-mass), the **leave-one-out marginal** of head h "
              "= effect(ablate-all) − effect(ablate-all-except-h). A **negative** marginal means ablating h *reduces* "
              "induction damage — a net suppressor / self-repair trigger; a distributed op has all-positive marginals.", "",
              "| model | induction top-k | full ΔNLL | most-negative LOO marginal (head) | distributed? |",
              "|---|---|---|---|---|"]
        for r in out["compensatory"]:
            comp = f"{r['min_marginal']:+.2f} (`{r['compensator']}`)" if r["compensator"] else f"{r['min_marginal']:+.2f} (none<0)"
            dist = "no — has a suppressor" if r["compensator"] else "yes (all marginals ≥0)"
            L.append(f"| {r['model']} | {', '.join('`' + h + '`' for h in r['induction_top'])} | {r['full_effect']:+.2f} | {comp} | {dist} |")
        L += ["", "_Per-head solo + LOO marginals are in the JSON. The suppressor head is the one whose removal lets "
              "a backup carry induction (the non-monotonic cumulative curve in the [operator catalog](induction.md)). "
              "**Caveat (see the next section):** the apparent suppression is largely a *synthetic repeated-random "
              "probe artifact* — these heads have positive OV and are ~neutral on natural-text induction._", ""]
    if out.get("suppressor"):
        L += ["## Is the suppressor a genuine negative head, or a synthetic-probe artifact?", "",
              "For each identified suppressor (+ the workhorse for contrast): the **OV copy-score** sign (+ve copies "
              "the attended token → a real copy/induction head; −ve suppresses it → a copy-suppression / negative "
              "head), and the head's ablation effect on induction measured over **natural**-repeated text (a real "
              "passage + itself) vs **synthetic**-repeated (random tokens + itself). A probe artifact would help "
              "natural induction (ablation ΔNLL > 0) but hurt synthetic (< 0); a genuine suppressor is −ve OV and "
              "helps both (ablation ΔNLL < 0).", "",
              "| model | head | role | OV copy-score | ablate ΔNLL natural | ablate ΔNLL synthetic |",
              "|---|---|---|---|---|---|"]
        for r in out["suppressor"]:
            for i, h in enumerate(r["heads"]):
                role = "suppressor" if i == 0 else "workhorse (contrast)"
                L.append(f"| {r['model']} | `{h['head']}` | {role} | {h['ov_copy_score']:+.3f} | {h['nat_induction_dNLL']:+.2f} | {h['syn_induction_dNLL']:+.2f} |")
        L += ["", "_+ve ablation ΔNLL = the head HELPS induction (removing it hurts); −ve = the head SUPPRESSES it "
              "(removing it helps). **Finding:** both suppressors have **positive** OV copy-scores — they are "
              "copy/induction heads, **not** copy-suppression / negative heads. The suppression shows up **only on the "
              "synthetic repeated-random probe** (Gemma 4.4: ΔNLL synthetic −0.60 but natural ≈0; gpt2-large 16.0 "
              "likewise marginal/positive on natural). So the *compensatory* redundancy is substantially a "
              "**repeated-random probe artifact** — these heads interfere with the degenerate synthetic-induction task "
              "but are ~neutral on real-text induction — not a genuine negative-head self-repair mechanism._", ""]
    if out.get("llama_l0"):
        L += ["## Do Llama's layer-0 heads do single-layer (RoPE-enabled) induction?", "",
              "GPT-2 needs a two-layer chain (a prev-token head feeds an induction head's key). Llama has "
              "induction-load-bearing heads at **layer 0** — where there is no prior layer to supply a prev-token "
              "signal. RoPE puts relative position in the key, so a single head can match *token-after-previous-"
              "occurrence* directly. If these heads carry **induction-mass ≫ duplicate-mass** and a **+ve OV "
              "copy-score**, they are single-layer inductors (no upstream writer needed).", "",
              "| model | head | induction-mass | duplicate-mass | OV copy-score | single-layer inductor? |",
              "|---|---|---|---|---|---|"]
        for r in out["llama_l0"]:
            for h in r["heads"]:
                yes = "yes" if (h["induction_mass"] > 0.1 and h["induction_mass"] > 1.3 * h["duplicate_mass"] and h["ov_copy_score"] > 0) else "**no** — enabler, not inductor"
                L.append(f"| {r['model']} | `{h['head']}` | {h['induction_mass']:.3f} | {h['duplicate_mass']:.3f} | {h['ov_copy_score']:+.3f} | {yes} |")
        L += ["", "_**Finding (hypothesis not supported):** these layer-0 heads do **not** behave as single-layer "
              "inductors — their induction-mass is weak (~0.03) and ≈ their duplicate-mass, even though 0.31 is "
              "strongly induction-*causal* (+7.99 when ablated, per the [discovered candidates](discovered_xmodel.md)). "
              "So they are induction **enablers**, not inductors: they don't attend induction-style themselves, but "
              "their early context-mixing (Dig 1 — Llama's L0 attention is the most context-determined) sets up the "
              "residual that later heads read. Llama's actual induction *reader* is a later head (10.23 in the "
              "[dossier](induction.md)). A clean reminder that high causal effect ≠ doing the named operation._", ""]
    L += ["_Data: [outlier_digs_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/outlier_digs_summary.json). "
          "Regenerate: [outlier_dig.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/outlier_dig.py)._"]
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "outlier_digs.md").write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dig", default="both", choices=["mlp0", "compensatory", "suppressor", "llama_l0", "threads", "both"])
    p.add_argument("--mlp0-models", default="gpt2,gpt2-large,google/gemma-2-2b,Qwen/Qwen2.5-1.5B,unsloth/Llama-3.2-1B")
    p.add_argument("--comp-models", default="gpt2,gpt2-large,google/gemma-2-2b")
    p.add_argument("--suppressor", default="google/gemma-2-2b:4.4,22.4|gpt2-large:16.0,16.9",
                   help="model:L.h,L.h|... — first head per model is the suppressor, rest are workhorse contrasts")
    p.add_argument("--llama-l0", default="unsloth/Llama-3.2-1B:0.31,0.29,0.13,0.14,1.31,1.29",
                   help="model:L.h,... — the layer-0/1 induction-mass cluster to test for single-layer induction")
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--chunks", type=int, default=30)
    p.add_argument("--n-tok", type=int, default=300)
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--probes", type=int, default=28)
    p.add_argument("--id-probes", type=int, default=16)
    p.add_argument("--rep-len", type=int, default=22)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/operators"))
    p.add_argument("--docs", type=Path, default=Path("docs/operators"))
    p.add_argument("--docs-only", action="store_true")
    args = p.parse_args(argv)
    if args.docs_only:
        out = json.loads((args.outdir / "outlier_digs_summary.json").read_text()); write_doc(out, args.docs)
        print(f"[docs-only] re-rendered {args.docs / 'outlier_digs.md'}"); return out
    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    existing = args.outdir / "outlier_digs_summary.json"
    out = json.loads(existing.read_text()) if existing.exists() else {}   # merge: keep prior digs not re-run

    def parse_targets(spec):
        groups = []
        for g in spec.split("|"):
            g = g.strip()
            if not g:
                continue
            mid, hs = g.rsplit(":", 1)
            groups.append((mid, [tuple(int(x) for x in h.split(".")) for h in hs.split(",") if h.strip()]))
        return groups

    if args.dig in ("mlp0", "both"):
        out["mlp0"] = []
        for mid in [s.strip() for s in args.mlp0_models.split(",") if s.strip()]:
            print(f"\n[mlp0] {mid}")
            try:
                r = dig_mlp0(mid, args, dev); out["mlp0"].append(r)
                print(f"  attn/emb {r['attn_over_emb_norm']:.2f} | η² emb {r['det_embedding']} attn {r['det_L0_attn_out']} mlp0 {r['det_mlp0_out']}")
            except Exception as e:  # pragma: no cover
                print(f"  [skip] {e}"); out["mlp0"].append({"model": mid, "error": str(e)})
            if dev == "cuda":
                torch.cuda.empty_cache()
    if args.dig in ("compensatory", "both"):
        out["compensatory"] = []
        for mid in [s.strip() for s in args.comp_models.split(",") if s.strip()]:
            print(f"\n[compensatory] {mid}")
            try:
                r = dig_compensatory(mid, args, dev); out["compensatory"].append(r)
                print(f"  top {r['induction_top']} | full {r['full_effect']:+.2f} | min-marginal {r['min_marginal']:+.2f} comp {r['compensator']}")
            except Exception as e:  # pragma: no cover
                print(f"  [skip] {e}"); out["compensatory"].append({"model": mid, "error": str(e)})
            if dev == "cuda":
                torch.cuda.empty_cache()
    if args.dig in ("suppressor", "threads"):
        out["suppressor"] = []
        for mid, heads in parse_targets(args.suppressor):
            print(f"\n[suppressor] {mid} {heads}")
            try:
                out["suppressor"].append(dig_suppressor(mid, heads, args, dev))
            except Exception as e:  # pragma: no cover
                print(f"  [skip] {e}"); out["suppressor"].append({"model": mid, "error": str(e)})
            if dev == "cuda":
                torch.cuda.empty_cache()
    if args.dig in ("llama_l0", "threads"):
        out["llama_l0"] = []
        for mid, heads in parse_targets(args.llama_l0):
            print(f"\n[llama_l0] {mid} {heads}")
            try:
                out["llama_l0"].append(dig_llama_l0(mid, heads, args, dev))
            except Exception as e:  # pragma: no cover
                print(f"  [skip] {e}"); out["llama_l0"].append({"model": mid, "error": str(e)})
            if dev == "cuda":
                torch.cuda.empty_cache()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "outlier_digs_summary.json").write_text(json.dumps(out, indent=2, default=float))
    write_doc(out, args.docs)
    print(f"\n[done] → {args.outdir / 'outlier_digs_summary.json'} + {args.docs / 'outlier_digs.md'}")
    return out


if __name__ == "__main__":
    main()
