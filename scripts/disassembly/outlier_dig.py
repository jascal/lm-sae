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
              "a backup carry induction (the non-monotonic cumulative curve in the [operator catalog](induction.md))._", ""]
    L += ["_Data: [outlier_digs_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/outlier_digs_summary.json). "
          "Regenerate: [outlier_dig.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/outlier_dig.py)._"]
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "outlier_digs.md").write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dig", default="both", choices=["mlp0", "compensatory", "both"])
    p.add_argument("--mlp0-models", default="gpt2,gpt2-large,google/gemma-2-2b,Qwen/Qwen2.5-1.5B,unsloth/Llama-3.2-1B")
    p.add_argument("--comp-models", default="gpt2,gpt2-large,google/gemma-2-2b")
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
    out = {}
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
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "outlier_digs_summary.json").write_text(json.dumps(out, indent=2, default=float))
    write_doc(out, args.docs)
    print(f"\n[done] → {args.outdir / 'outlier_digs_summary.json'} + {args.docs / 'outlier_digs.md'}")
    return out


if __name__ == "__main__":
    main()
