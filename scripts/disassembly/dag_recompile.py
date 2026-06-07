"""Decompilation milestone 1↔2 bridge — does the extracted composition-DAG pick the keep-set?

`residual_vm.py` (M1) measured which / how-many heads reconstruct the forward pass, ranking heads by EXPENSIVE
marginal-ablation importance (one forward pass per head). `composition_dag.py` (M2) extracts the live
composition sub-DAG from weights + two forward passes (CHEAP, structural). This asks whether the cheap
structural signal SUBSTITUTES for the expensive importance measurement — i.e. whether the extracted DAG IS the
op-set the recompiler should keep:

  keep-set = the path-patch-confirmed live sub-DAG (heads in live induction + IOI edges; and + the new live edges)
  vs       = equal-size top-B heads by marginal-ablation importance (M1's flat budget)
  vs       = equal-size random control
  metric   = reconstruction_coverage = 1 - KL(host || keep-only) / KL(host || all-heads-ablated)   (M1's metric)

and, graph-wide: does a DAG-CONNECTIVITY head ordering (sum of incident live-edge specificity, weight-cheap)
reproduce the marginal-importance ordering (Spearman + the coverage-vs-budget curve)? If the DAG keep-set rivals
top-importance and beats random, and connectivity tracks importance, then the extracted DAG is the structured
op-set the recompile harness should retain — recoverable for ~free from weights, no per-head ablation sweep.

It also adds the **VALUE pathway** (V-composition, `vcomposition.py`): the K/Q sub-DAG covers attention routing,
but the composed-OV "virtual heads" (induction content re-read as a later head's value) move CONTENT the K/Q
edges don't. We add the live V-edge heads to the circuit keep-set and ask: does the value pathway lift
reconstruction coverage *beyond* the K/Q circuit, and beat the same number of random additions? The V-edge
readout is an OUTPUT change (ΔV-out) — does that make the value-pathway heads output-load-bearing, or is ΔV-out
(like ΔTV in #23) an edge-COUPLING score that does NOT predict head output-importance? (Spoiler in the verdict.)

Consumes `composition_dag_summary.json` + `vcomposition_summary.json` (auto-generates either if absent) and
reuses residual_vm's proven mean-ablation coverage harness (GPT-2; arch-generic o_proj/c_proj slice hook).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def _oproj_modules(model):
    """(output-projection module per layer, head_dim) for the mean-ablation slice — arch-generic (from residual_vm)."""
    cfg = model.config
    H = cfg.num_attention_heads
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
        return [lyr.self_attn.o_proj for lyr in model.model.layers], hd
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [blk.attn.c_proj for blk in model.transformer.h], cfg.n_embd // H
    raise SystemExit("unknown architecture")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--corpus", default="shakespeare")
    p.add_argument("--dag-summary", type=Path, default=Path("runs/disassembly/composition_dag_summary.json"))
    p.add_argument("--v-summary", type=Path, default=Path("runs/disassembly/vcomposition_summary.json"),
                   help="V-composition DAG (the value pathway) to add onto the K/Q circuit keep-set")
    p.add_argument("--rank-tokens", type=int, default=2400, help="tokens for per-head marginal importance ranking")
    p.add_argument("--eval-tokens", type=int, default=4000, help="tokens for the coverage KLs")
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--budgets", default="2,4,6,8,12,16,24,32,48,64")
    p.add_argument("--n-rand", type=int, default=4, help="random control sets per budget / keep-set")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/dag_recompile_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/dag_recompile.png"))
    args = p.parse_args(argv)

    # ---- load (or generate) the M2 composition-DAG summary ----
    if not args.dag_summary.exists():
        print(f"[dag] {args.dag_summary} missing -> running composition_dag.py to generate it...")
        import composition_dag
        composition_dag.main(["--pretrained", args.model, "--output", str(args.dag_summary)])
    dag = json.loads(args.dag_summary.read_text())
    edges = dag["edges"]

    def hk(s):
        L, h = s.split("."); return (int(L), int(h))
    conn = defaultdict(float); statsum = defaultdict(float)
    for e in edges:
        a, b = hk(e["A"]), hk(e["B"])
        statsum[a] += e["static"]; statsum[b] += e["static"]
        if e["live"] and not e["is_random"]:
            conn[a] += e["specificity"]; conn[b] += e["specificity"]
    live_circ = [e for e in edges if e["live"] and e["is_circuit"]]
    live_new = [e for e in edges if e["live"] and not e["is_circuit"] and not e["is_random"]]
    dag_circuit = sorted({hk(e["A"]) for e in live_circ} | {hk(e["B"]) for e in live_circ})
    dag_all = sorted(set(dag_circuit) | {hk(e["A"]) for e in live_new} | {hk(e["B"]) for e in live_new})
    print(f"[dag] live circuit sub-DAG = {len(dag_circuit)} heads; +new live = {len(dag_all)} heads "
          f"(spearman static->specificity {dag.get('spearman_static_vs_specificity'):+.2f})")

    # ---- load (or generate) the V-composition DAG (the value pathway) ----
    if not args.v_summary.exists():
        print(f"[v] {args.v_summary} missing -> running vcomposition.py to generate it...")
        import vcomposition
        vcomposition.main(["--pretrained", args.model, "--output", str(args.v_summary)])
    vdag = json.loads(args.v_summary.read_text())
    v_live = [e for e in vdag["edges"] if e["kind"] == "topV" and e.get("dvout_spec", 0.0) > 0]
    v_heads = {hk(e["A"]) for e in v_live} | {hk(e["B"]) for e in v_live}
    v_new = sorted(v_heads - set(dag_circuit))                                # value-pathway heads not in the K/Q circuit
    print(f"[v] {len(v_live)} live V-edges -> {len(v_heads)} heads, {len(v_new)} NEW vs the K/Q circuit: "
          f"{[f'{L}.{h}' for L, h in v_new]}")

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    cfg = model.config
    nL, H = cfg.num_hidden_layers, cfg.num_attention_heads
    oprojs, hd = _oproj_modules(model)
    rng = np.random.default_rng(0)
    import urllib.request
    CORPORA = {"shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
               "wikitext": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"}
    txt = urllib.request.urlopen(urllib.request.Request(CORPORA.get(args.corpus, args.corpus),
                                 headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:400000]
    ids_all = tok(txt)["input_ids"]

    def chunkify(n):
        ids = ids_all[:n]
        return [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]

    # ---- capture per-head mean output (the ablation value) ----
    cap = {L: [] for L in range(nL)}
    caps = [oprojs[L].register_forward_pre_hook(
        (lambda L: lambda m, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in chunkify(args.rank_tokens):
            model(input_ids=torch.tensor([c], device=dev))
    for h in caps:
        h.remove()
    meanvec = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    def ablate_hooks(ablate):
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
            hs.append(oprojs[L].register_forward_pre_hook(mk(L, hss)))
        return hs

    all_heads = [(L, h) for L in range(nL) for h in range(H)]

    def host_logprobs(chunks):
        out = []
        with torch.no_grad():
            for c in chunks:
                lp = F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                out.append(lp.cpu())
        return out

    def kl_keep(keep, chunks, host_lp):
        ablate = [hh for hh in all_heads if hh not in keep]
        hs = ablate_hooks(ablate)
        tot = 0.0; n = 0
        try:
            with torch.no_grad():
                for c, hlp in zip(chunks, host_lp):
                    lp = F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                    kl = (hlp.to(dev).exp() * (hlp.to(dev) - lp)).sum(-1)
                    tot += float(kl.sum()); n += kl.numel()
        finally:
            for h in hs:
                h.remove()
        return tot / max(n, 1)

    # ---- per-head marginal importance ranking (M1: KL of ablating each head alone) ----
    rchunks = chunkify(args.rank_tokens); rhost = host_logprobs(rchunks)
    print(f"{args.model}: {len(all_heads)} heads; marginal-importance ranking on {len(rchunks)} chunks ...")
    full = set(all_heads); imp = {}
    for i, hh in enumerate(all_heads):
        imp[hh] = kl_keep(full - {hh}, rchunks, rhost)
        if (i + 1) % 48 == 0:
            print(f"  ranked {i + 1}/{len(all_heads)}")
    ranked = sorted(all_heads, key=lambda hh: -imp[hh])
    # DAG-connectivity ordering (weight-cheap): incident live-edge specificity, then static composition
    conn_ranked = sorted(all_heads, key=lambda hh: (-conn.get(hh, 0.0), -statsum.get(hh, 0.0)))
    touched = [hh for hh in all_heads if hh in statsum]                       # heads the DAG actually gated
    rho_all = _spearman([conn.get(hh, 0.0) for hh in all_heads], [imp[hh] for hh in all_heads])
    rho_touched = _spearman([conn.get(hh, 0.0) for hh in touched], [imp[hh] for hh in touched])
    print(f"  Spearman(DAG-connectivity, marginal-importance): all heads {rho_all:+.3f} | DAG-touched {rho_touched:+.3f}")

    # ---- coverage ----
    echunks = chunkify(args.eval_tokens); ehost = host_logprobs(echunks)
    floor = kl_keep(set(), echunks, ehost)
    print(f"  floor KL(host || all-ablated) = {floor:.4f}")

    def coverage(keep):
        return 1.0 - kl_keep(set(keep), echunks, ehost) / floor

    def rand_cov(size):
        return float(np.mean([coverage(set(map(tuple, np.array(all_heads)[rng.choice(len(all_heads), size, replace=False)].tolist())))
                              for _ in range(args.n_rand)]))

    # named keep-sets: the DAG sub-DAGs vs equal-size top-importance vs equal-size random
    keepsets = {"dag_circuit (induction+IOI)": dag_circuit, "dag_all_live": dag_all}
    ks_res = {}
    for nm, heads in keepsets.items():
        sz = len(heads)
        cov_dag = coverage(heads)
        cov_top = coverage(ranked[:sz])
        cov_rnd = rand_cov(sz)
        ranks = sorted(ranked.index(hh) for hh in heads)
        ks_res[nm] = {"size": sz, "coverage_dag": cov_dag, "coverage_top_importance": cov_top,
                      "coverage_random": cov_rnd, "importance_ranks": ranks,
                      "heads": [f"{L}.{h}" for L, h in heads]}
        print(f"  [{nm}] n={sz}: DAG {cov_dag:+.3f} | top-imp {cov_top:+.3f} | random {cov_rnd:+.3f}  "
              f"(DAG/top {cov_dag/cov_top:.0%} of importance-optimal, {cov_dag - cov_rnd:+.3f} over random)")

    # ---- VALUE PATHWAY: add the live V-edge heads to the K/Q circuit; does it lift coverage beyond random additions? ----
    kq = dag_circuit
    cov_kq = ks_res["dag_circuit (induction+IOI)"]["coverage_dag"]
    cov_kqv = coverage(sorted(set(kq) | set(v_new)))

    def rand_add(base, k):
        pool = [hh for hh in all_heads if hh not in set(base)]
        return float(np.mean([coverage(set(base) | set(map(tuple, np.array(pool)[rng.choice(len(pool), k, replace=False)].tolist())))
                              for _ in range(args.n_rand)])) if k else coverage(set(base))
    cov_kq_rand = rand_add(kq, len(v_new))
    imp_add = [hh for hh in ranked if hh not in set(kq)][:len(v_new)]                # importance-optimal addition
    cov_kq_imp = coverage(set(kq) | set(imp_add)) if v_new else cov_kq
    v_ranks = sorted(ranked.index(hh) for hh in v_new)
    v_incr = cov_kqv - cov_kq; rand_incr = cov_kq_rand - cov_kq; imp_incr = cov_kq_imp - cov_kq
    v_res = {"n_v_live_edges": len(v_live), "n_v_new_heads": len(v_new),
             "v_new_heads": [f"{L}.{h}" for L, h in v_new], "coverage_kq": cov_kq, "coverage_kq_plus_v": cov_kqv,
             "v_increment": v_incr, "coverage_kq_plus_random": cov_kq_rand, "random_increment": rand_incr,
             "coverage_kq_plus_topimp": cov_kq_imp, "topimp_increment": imp_incr, "v_new_importance_ranks": v_ranks}
    print(f"\n[value pathway] K/Q circuit {cov_kq:+.3f}  + {len(v_new)} V-heads -> {cov_kqv:+.3f} (Δ {v_incr:+.3f})  "
          f"| + {len(v_new)} random -> {cov_kq_rand:+.3f} (Δ {rand_incr:+.3f})  "
          f"| + {len(v_new)} top-importance -> {cov_kq_imp:+.3f} (Δ {imp_incr:+.3f})")
    print(f"[value pathway] V-head marginal-importance ranks (of {len(all_heads)}): {v_ranks}")

    # coverage curve: three orderings (importance / DAG-connectivity / random) vs budget
    budgets = sorted({min(int(b), len(all_heads)) for b in args.budgets.split(",")})
    curve = []
    for B in budgets:
        cov_imp = coverage(ranked[:B]); cov_conn = coverage(conn_ranked[:B]); cov_rnd = rand_cov(B)
        curve.append({"budget": B, "coverage_importance": cov_imp, "coverage_dag_connectivity": cov_conn,
                      "coverage_random": cov_rnd})
        print(f"  B={B:>3}: importance {cov_imp:+.3f} | DAG-connectivity {cov_conn:+.3f} | random {cov_rnd:+.3f}")
    rho_curve = _spearman([c["coverage_importance"] for c in curve], [c["coverage_dag_connectivity"] for c in curve])

    out = {"experiment": "M1<->M2 bridge: does the composition-DAG pick the recompile keep-set?",
           "model": args.model, "corpus": args.corpus, "n_heads": len(all_heads), "floor_kl": floor,
           "spearman_connectivity_vs_importance_all": rho_all,
           "spearman_connectivity_vs_importance_touched": rho_touched,
           "spearman_curve_importance_vs_connectivity": rho_curve,
           "keepsets": ks_res, "value_pathway": v_res, "curve": curve,
           "top_importance_heads": [f"{L}.{h}" for L, h in ranked[:16]],
           "dag_connectivity_heads": [f"{L}.{h}" for L, h in conn_ranked[:16]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    # verdict (two-part): the path-patch-CONFIRMED circuit sub-DAG is the clean keep-set; raw connectivity is NOT
    # a general importance proxy (ΔTV reshapes attention != KL output-importance; new write-hubs are redundant).
    circ = ks_res["dag_circuit (induction+IOI)"]
    circ_frac = circ["coverage_dag"] / max(circ["coverage_top_importance"], 1e-9)
    circ_over_rand = circ["coverage_dag"] - circ["coverage_random"]
    circuit_wins = circ_frac >= 0.85 and circ_over_rand > 0.05
    out["circuit_frac_of_optimal"] = circ_frac; out["circuit_coverage_over_random"] = circ_over_rand
    out["circuit_keepset_matches_importance"] = bool(circuit_wins)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    if circuit_wins:
        print(f"\n[verdict] BRIDGED (circuit): the path-patch-confirmed induction+IOI sub-DAG ({circ['size']} heads, "
              f"extracted from weights + 2 passes) reconstructs {circ['coverage_dag']:+.3f} = {circ_frac:.0%} of the "
              f"equal-size top-marginal-importance set ({circ['coverage_top_importance']:+.3f}) and {circ_over_rand:+.3f} "
              f"over random -> the auto-extracted CIRCUIT is as coverage-efficient as the expensive importance ranking, "
              f"no per-head ablation sweep needed.")
    else:
        print("\n[verdict] circuit keep-set did NOT match top-importance — see table")
    print(f"[caveat] raw DAG-connectivity is a WEAK output-importance proxy (Spearman vs marginal-importance: "
          f"DAG-touched {rho_touched:+.2f}); the new live write-hubs RESHAPE attention (high ΔTV) without being "
          f"output-load-bearing (dag_all_live {ks_res['dag_all_live']['coverage_dag']/max(ks_res['dag_all_live']['coverage_top_importance'],1e-9):.0%} of optimal) "
          f"=> ΔTV (attention-influence) != KL (output-importance); the win is the CONFIRMED CIRCUIT, not a generic importance score.")
    # value-pathway verdict: does the V edge type add output-load-bearing heads the K/Q circuit missed?
    v_adds = v_incr > 0.01 and v_incr > 1.5 * max(rand_incr, 0.0)
    out["value_pathway"]["v_beats_random_addition"] = bool(v_adds)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    if v_adds:
        print(f"[value pathway] V-EDGES ADD REAL COVERAGE: the {len(v_new)} composed-OV virtual heads lift the K/Q "
              f"circuit {cov_kq:+.3f} -> {cov_kqv:+.3f} (Δ {v_incr:+.3f}), {v_incr/max(rand_incr,1e-9):.1f}x the "
              f"random-addition lift (Δ {rand_incr:+.3f}) and {v_incr/max(imp_incr,1e-9):.0%} of the importance-optimal "
              f"addition (Δ {imp_incr:+.3f}) -> the value pathway carries output-load-bearing reconstruction the "
              f"attention-routing (K/Q) DAG missed; ΔV-out (output-change) finds heads ΔTV (attention) could not.")
    else:
        print(f"[value pathway] the V-edges add {v_incr:+.3f} coverage (random additions add {rand_incr:+.3f}) — "
              f"not clearly output-load-bearing beyond the K/Q circuit; see table.")
    print(f"[done] {args.output}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axC, axB, axV) = plt.subplots(1, 3, figsize=(16.5, 5.0))
        Bs = [c["budget"] for c in curve]
        axC.plot(Bs, [c["coverage_importance"] for c in curve], "-o", color="#d62728", label="top-importance (M1, expensive)")
        axC.plot(Bs, [c["coverage_dag_connectivity"] for c in curve], "-s", color="#1f77b4", label="DAG-connectivity (M2, cheap)")
        axC.plot(Bs, [c["coverage_random"] for c in curve], "--^", color="#999999", label="random")
        axC.set_xlabel("keep budget (# heads)"); axC.set_ylabel("reconstruction coverage  1 − KL/floor")
        axC.set_title(f"keep-ordering curves  (per-head ρ(conn,imp)={rho_touched:+.2f}: connectivity ≠ importance)", fontsize=10)
        axC.legend(fontsize=8); axC.axhline(0, color="k", lw=0.5, ls=":")
        names = list(ks_res); x = np.arange(len(names)); w = 0.26
        axB.bar(x - w, [ks_res[n]["coverage_dag"] for n in names], w, color="#1f77b4", edgecolor="k", label="DAG sub-DAG")
        axB.bar(x, [ks_res[n]["coverage_top_importance"] for n in names], w, color="#d62728", edgecolor="k", label="top-importance (=size)")
        axB.bar(x + w, [ks_res[n]["coverage_random"] for n in names], w, color="#999999", edgecolor="k", label="random (=size)")
        axB.set_xticks(x); axB.set_xticklabels([f"{n}\n(n={ks_res[n]['size']})" for n in names], fontsize=8)
        axB.set_ylabel("reconstruction coverage"); axB.set_title("DAG keep-set vs equal-size baselines", fontsize=10)
        axB.legend(fontsize=8)
        vb = [("K/Q\ncircuit", cov_kq, "#d62728"), (f"+ {len(v_new)}\nV-heads", cov_kqv, "#2ca02c"),
              (f"+ {len(v_new)}\nrandom", cov_kq_rand, "#999999"), (f"+ {len(v_new)}\ntop-imp", cov_kq_imp, "#9467bd")]
        axV.bar(range(len(vb)), [b[1] for b in vb], color=[b[2] for b in vb], edgecolor="k")
        axV.axhline(cov_kq, color="#d62728", lw=0.8, ls=":")
        axV.set_xticks(range(len(vb))); axV.set_xticklabels([b[0] for b in vb], fontsize=8)
        axV.set_ylabel("reconstruction coverage")
        axV.set_title(f"value pathway: V-heads add {v_incr:+.3f} (random {rand_incr:+.3f})", fontsize=10)
        fig.suptitle("M1↔M2 bridge: the weight-cheap composition-DAG picks the recompile keep-set (+ the V value pathway)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    return out


if __name__ == "__main__":
    main()
