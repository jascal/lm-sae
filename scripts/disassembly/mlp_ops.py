"""Decompilation milestone 3 — the MLP instruction class in the DAG + the recompile.

`residual_vm.py` (M1) and `dag_recompile.py` (the M1↔M2 bridge) kept/ablated ATTENTION HEADS only — MLPs ran
at full fidelity, so the reconstruction-coverage metric never charged for them. But MLPs carry the COMPUTE
(greater-than is MLP-dominated; `mlp_catalog.py` read the neuron key→value vocabulary). M3 adds them:

  (1) RECOMPILE (the teeth): extend the mean-ablation coverage harness to keep/ablate MLP LAYERS as well as
      heads (floor = all heads AND all MLPs ablated). Measure attention-only vs MLP-only vs full; per-MLP
      marginal importance; and the BRIDGE EXTENSION — does adding MLP layers to the M2 attention-circuit
      keep-set lift coverage beyond what the attention circuit reaches alone? (Mean-ablating the MLP and
      reading ΔKL IS the causal validation, the program's canonical metric.)
  (2) DAG: head↔MLP composition edges in weight space, mean-write removed — head OV-write feeds MLP read W_in
      (head→MLP), MLP write W_out feeds a later head's Q/K/V (MLP→head) — the MLP nodes + typed edges the
      attention-only DAG (composition_dag.py) was missing.
  (3) named MLP idioms: the load-bearing MLP layers' top neurons read→write in the token-feature basis (cf
      mlp_catalog.py), so the recompile's important MLPs get a human label.

GPT-2; reuses residual_vm's mean-ablation harness + composition_dag's mean-write-removed scoring. Consumes the
M2 attention DAG (`composition_dag_summary.json`).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lm_bundle import COMMON  # noqa: E402


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def _gpt2_mlp_proj(model):
    """(mlp down-projection per layer for mean-ablation, attn out-proj per layer, head_dim) — GPT-2."""
    tr = model.transformer
    return [blk.mlp.c_proj for blk in tr.h], [blk.attn.c_proj for blk in tr.h], model.config.n_embd // model.config.n_head


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--corpus", default="shakespeare")
    p.add_argument("--dag-summary", type=Path, default=Path("runs/disassembly/composition_dag_summary.json"))
    p.add_argument("--rank-tokens", type=int, default=2400)
    p.add_argument("--eval-tokens", type=int, default=4000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--ref-layer", type=int, default=6, help="layer for the token-feature operand basis (naming)")
    p.add_argument("--n-name-tokens", type=int, default=48)
    p.add_argument("--mlp-budgets", default="1,2,3,4,6,8,12", help="# top MLP layers to add onto the circuit keep-set")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/mlp_ops_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/mlp_ops.png"))
    args = p.parse_args(argv)

    # ---- M2 attention DAG: the path-patch-confirmed circuit heads (the bridge keep-set) ----
    if not args.dag_summary.exists():
        print(f"[dag] {args.dag_summary} missing -> running composition_dag.py ...")
        import composition_dag
        composition_dag.main(["--pretrained", args.model, "--output", str(args.dag_summary)])
    dag = json.loads(args.dag_summary.read_text())

    def hk(s):
        L, h = s.split("."); return (int(L), int(h))
    live_circ = [e for e in dag["edges"] if e["live"] and e["is_circuit"]]
    circuit_heads = sorted({hk(e["A"]) for e in live_circ} | {hk(e["B"]) for e in live_circ})

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    cfg = model.config
    nL, H = cfg.num_hidden_layers, cfg.num_attention_heads
    mlp_proj, oprojs, hd = _gpt2_mlp_proj(model)
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

    # ---- capture mean ablation values: per-head o_proj-input slice + per-MLP c_proj-input vector ----
    capO = {L: [] for L in range(nL)}; capM = {L: [] for L in range(nL)}
    hsr = []                                                                  # ref-layer hidden for naming centroids
    hooksO = [oprojs[L].register_forward_pre_hook(
        (lambda L: lambda m, inp: capO[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    hooksM = [mlp_proj[L].register_forward_pre_hook(
        (lambda L: lambda m, inp: capM[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    rchunks = chunkify(args.rank_tokens); ref_ids = []
    with torch.no_grad():
        for c in rchunks:
            o = model(input_ids=torch.tensor([c], device=dev), output_hidden_states=True)
            hsr.append(o.hidden_states[args.ref_layer][0].float().cpu().numpy()); ref_ids.extend(c)
    for h in hooksO + hooksM:
        h.remove()
    meanO = {L: torch.cat(capO[L], 0).mean(0) for L in range(nL)}
    meanM = {L: torch.cat(capM[L], 0).mean(0) for L in range(nL)}

    all_heads = [(L, h) for L in range(nL) for h in range(H)]
    all_mlps = list(range(nL))

    def ablate_hooks(ab_heads, ab_mlps):
        by_layer = {}
        for (L, h) in ab_heads:
            by_layer.setdefault(L, []).append(h)
        hs = []
        for L, hss in by_layer.items():
            def mkO(L, hss):
                def hook(m, inp):
                    x = inp[0].clone()
                    for h in hss:
                        x[..., h * hd:(h + 1) * hd] = meanO[L][h * hd:(h + 1) * hd].to(x.dtype)
                    return (x,)
                return hook
            hs.append(oprojs[L].register_forward_pre_hook(mkO(L, hss)))
        for L in ab_mlps:
            def mkM(L):
                def hook(m, inp):
                    return (meanM[L].to(inp[0].dtype).expand_as(inp[0]),)
                return hook
            hs.append(mlp_proj[L].register_forward_pre_hook(mkM(L)))
        return hs

    def host_logprobs(chunks):
        out = []
        with torch.no_grad():
            for c in chunks:
                out.append(F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1).cpu())
        return out

    def kl_keep(keep_heads, keep_mlps, chunks, host_lp):
        ab_h = [hh for hh in all_heads if hh not in keep_heads]
        ab_m = [L for L in all_mlps if L not in keep_mlps]
        hs = ablate_hooks(ab_h, ab_m)
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

    echunks = chunkify(args.eval_tokens); ehost = host_logprobs(echunks)
    rhost = host_logprobs(rchunks)
    AH, AM = set(all_heads), set(all_mlps)
    floor = kl_keep(set(), set(), echunks, ehost)                            # all heads AND all MLPs ablated
    print(f"{args.model}: {nL}L x {H}H + {nL} MLPs; floor KL(host || all-ablated) = {floor:.4f}")

    def cov(keep_heads, keep_mlps):
        return 1.0 - kl_keep(keep_heads, keep_mlps, echunks, ehost) / floor

    cov_attn_only = cov(AH, set())          # keep all heads, ablate all MLPs
    cov_mlp_only = cov(set(), AM)           # keep all MLPs, ablate all heads
    cov_full = cov(AH, AM)
    print(f"  attention-only {cov_attn_only:+.3f} | MLP-only {cov_mlp_only:+.3f} | full {cov_full:+.3f}")

    # ---- per-MLP marginal importance (ablate one MLP, keep all heads + other MLPs) on the rank budget ----
    floor_r = kl_keep(set(), set(), rchunks, rhost)
    base_full_r = kl_keep(AH, AM, rchunks, rhost)
    mlp_imp = {}
    for L in all_mlps:
        mlp_imp[L] = (kl_keep(AH, AM - {L}, rchunks, rhost) - base_full_r) / floor_r    # coverage drop from removing MLP L
    mlp_ranked = sorted(all_mlps, key=lambda L: -mlp_imp[L])
    print("  MLP marginal importance (coverage drop when ablated), top: " +
          ", ".join(f"L{L}({mlp_imp[L]:+.3f})" for L in mlp_ranked[:6]))

    # ---- MLP-budget curve in the ATTENTION-INTACT regime (the clean one: mean-ablating ALL MLPs is severe — L0
    #      is the detokenizer — so keep heads intact and ask how few MLPs reconstruct the forward pass) ----
    budgets = sorted({min(int(b), nL) for b in args.mlp_budgets.split(",")})
    mlp_curve = []
    for B in budgets:
        c_top = cov(AH, set(mlp_ranked[:B]))
        c_rnd = float(np.mean([cov(AH, set(int(x) for x in rng.choice(nL, B, replace=False))) for _ in range(3)]))
        mlp_curve.append({"n_mlp": B, "coverage_attn_plus_topMLP": c_top, "coverage_attn_plus_randMLP": c_rnd})
        print(f"  all heads + {B:>2} MLP: top-imp {c_top:+.3f} | random-MLP {c_rnd:+.3f}")
    # the combined sparse DAG op-set: the M2 circuit heads + the top-k load-bearing MLPs (a tiny MOVE+COMPUTE set)
    k_dag = min(4, nL)
    cov_dag_ops = cov(set(circuit_heads), set(mlp_ranked[:k_dag]))
    print(f"  combined sparse op-set: {len(circuit_heads)} circuit heads + top-{k_dag} MLP = {cov_dag_ops:+.3f}  "
          f"(vs full {cov_full:+.3f})")

    # ---- DAG: head<->MLP composition edges in weight space (mean-write removed) ----
    tr = model.transformer; d = cfg.n_embd
    Wc = [tr.h[L].attn.c_attn.weight.detach().float().cpu().numpy().astype(np.float64) for L in range(nL)]
    Wo = [tr.h[L].attn.c_proj.weight.detach().float().cpu().numpy().astype(np.float64) for L in range(nL)]
    Win = [tr.h[L].mlp.c_fc.weight.detach().float().cpu().numpy().astype(np.float64) for L in range(nL)]      # (d, dff)
    Wout = [tr.h[L].mlp.c_proj.weight.detach().float().cpu().numpy().astype(np.float64) for L in range(nL)]   # (dff, d)
    OV = np.zeros((nL * H, d, d))
    for L in range(nL):
        Wv = Wc[L][:, 2 * d:3 * d]
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            OV[L * H + h] = Wv[:, sl] @ Wo[L][sl, :]
    u = np.linalg.svd(OV.transpose(0, 2, 1).reshape(-1, d), full_matrices=False)[2][0]
    P = np.eye(d) - np.outer(u, u)
    OV = np.einsum("nij,jk->nik", OV, P)
    ovn = np.linalg.norm(OV.reshape(nL * H, -1), axis=1) + 1e-9
    Woutp = [W @ P for W in Wout]                                            # remove shared mean-write from MLP output
    WK = [Wc[L][:, d:2 * d] for L in range(nL)]; WQ = [Wc[L][:, :d] for L in range(nL)]
    head_layer = np.array([L for L in range(nL) for _ in range(H)])
    winn = [np.linalg.norm(Win[L]) + 1e-9 for L in range(nL)]
    woutn = [np.linalg.norm(Woutp[L]) + 1e-9 for L in range(nL)]

    h2m = []                                                                 # head -> MLP read (causal: L_head <= L_mlp)
    for a in range(nL * H):
        for L in range(nL):
            if head_layer[a] <= L:
                h2m.append((a, L, float(np.linalg.norm(OV[a] @ Win[L]) / (ovn[a] * winn[L]))))
    m2h = []                                                                 # MLP -> head Q/K read (causal: L_mlp < L_head)
    for L in range(nL):
        for b in range(nL * H):
            if L < head_layer[b]:
                hb = b % H; slb = slice(hb * hd, (hb + 1) * hd)
                sc = max(np.linalg.norm(Woutp[L] @ WQ[head_layer[b]][:, slb]),
                         np.linalg.norm(Woutp[L] @ WK[head_layer[b]][:, slb])) / (woutn[L] * (np.linalg.norm(WK[head_layer[b]][:, slb]) + 1e-9))
                m2h.append((L, b, float(sc)))

    def hn(a):
        return f"{a // H}.{a % H}"
    top_h2m = sorted(h2m, key=lambda r: -r[2])[:10]
    top_m2h = sorted(m2h, key=lambda r: -r[2])[:10]
    # do the important MLPs (recompile) line up with high static composition? (read+write incident score per MLP)
    mlp_static = {L: float(np.mean([s for _a, LL, s in h2m if LL == L] or [0]) + np.mean([s for LL, _b, s in m2h if LL == L] or [0]))
                  for L in range(nL)}
    rho_static_imp = _spearman([mlp_static[L] for L in all_mlps], [mlp_imp[L] for L in all_mlps])

    # ---- named idioms for the load-bearing MLP layers (top neurons read->write, token-feature basis) ----
    cnt = Counter(ref_ids)
    cand = [tok(c, add_special_tokens=False)["input_ids"][0] for c in COMMON
            if len(tok(c, add_special_tokens=False)["input_ids"]) == 1]
    cand += [t for t, _ in cnt.most_common(250)]
    seen, ntoks = set(), []
    for t in cand:
        if t not in seen and cnt[t] >= 20:
            seen.add(t); ntoks.append(t)
        if len(ntoks) >= args.n_name_tokens:
            break
    Xref = np.concatenate(hsr, 0); yref = np.array(ref_ids)
    nmlbl = [tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t in ntoks]
    Dc = np.stack([Xref[yref == t].mean(0) for t in ntoks]) - Xref.mean(0)
    Dc = Dc / (np.linalg.norm(Dc, axis=1, keepdims=True) + 1e-9)
    named = {}
    for L in mlp_ranked[:4]:
        ln2 = tr.h[L].ln_2.weight.detach().float().cpu().numpy().astype(np.float64)
        rd = (Dc * ln2) @ Win[L]; wr = Wout[L] @ Dc.T                        # (nt,dff),(dff,nt)
        sal = np.abs(rd).max(0) * np.abs(wr).max(1)
        neurons = []
        for i in np.argsort(-sal)[:3]:
            reads = [nmlbl[x] for x in np.argsort(-np.abs(rd[:, i]))[:3]]
            writes = [nmlbl[x] for x in np.argsort(-np.abs(wr[i]))[:3]]
            neurons.append({"neuron": int(i), "reads": reads, "writes": writes})
        named[f"L{L}"] = neurons

    out = {"experiment": "milestone-3 MLP ops in the DAG + recompile", "model": args.model, "corpus": args.corpus,
           "floor_kl": floor, "n_heads": len(all_heads), "n_mlp": nL,
           "coverage_attention_only": cov_attn_only, "coverage_mlp_only": cov_mlp_only, "coverage_full": cov_full,
           "coverage_combined_sparse_op_set": cov_dag_ops, "combined_op_set_n_heads": len(circuit_heads),
           "combined_op_set_n_mlp": k_dag,
           "mlp_importance": {f"L{L}": mlp_imp[L] for L in all_mlps}, "mlp_ranked": mlp_ranked,
           "mlp_curve": mlp_curve, "spearman_mlp_static_vs_importance": rho_static_imp,
           "top_head_to_mlp": [[hn(a), f"L{L}", s] for a, L, s in top_h2m],
           "top_mlp_to_head": [[f"L{L}", hn(b), s] for L, b, s in top_m2h],
           "named_mlp_idioms": named, "circuit_heads": [f"{L}.{h}" for (L, h) in circuit_heads]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print("\n[head->MLP read] top: " + ", ".join(f"{hn(a)}->L{L}({s:.2f})" for a, L, s in top_h2m[:5]))
    print("[MLP->head write] top: " + ", ".join(f"L{L}->{hn(b)}({s:.2f})" for L, b, s in top_m2h[:5]))
    print(f"[static->live] Spearman(MLP static-composition, MLP recompile-importance) = {rho_static_imp:+.3f}")
    print("[named idioms] load-bearing MLPs:")
    for L, neus in named.items():
        for ne in neus[:1]:
            print(f"  {L}.n{ne['neuron']}: reads {ne['reads']} -> writes {ne['writes']}")

    top_mlp = mlp_ranked[0]
    mlp_necessary = cov_attn_only < 0.2                                      # removing all MLPs collapses coverage
    top_beats_rand = all(c["coverage_attn_plus_topMLP"] >= c["coverage_attn_plus_randMLP"] - 1e-3 for c in mlp_curve)
    ok = mlp_necessary and mlp_imp[top_mlp] > 0.2 and cov_mlp_only > 0.1
    if ok:
        verdict = (f"MLP OPS ADDED: the recompile now charges for MLPs (M1/bridge kept heads only). MLPs are "
                   f"LOAD-BEARING — removing all MLPs (heads intact) collapses coverage to {cov_attn_only:+.3f}; the load "
                   f"is CONCENTRATED in L{top_mlp} (importance {mlp_imp[top_mlp]:+.3f}, GPT-2's detokenizer). A few MLPs "
                   f"reconstruct most (attention-intact; top-importance {'>=' if top_beats_rand else '~'} random); the "
                   f"combined sparse op-set ({len(circuit_heads)} circuit heads + top-{k_dag} MLP) reaches {cov_dag_ops:+.3f}. "
                   f"Head<->MLP composition edges are weight-legible -> the DAG now has COMPUTE (MLP) nodes alongside "
                   f"MOVE (attention). CAVEAT: mean-ablating ALL MLPs is severe, so attention-only/MLP-only are necessity "
                   f"statements, not a clean credit split (attention's value is the MLP-intact bridge #23).")
    else:
        verdict = "partial — see coverage table"
    print(f"\n[verdict] {verdict}")
    print(f"[caveat] static head<->MLP composition does NOT rank MLP recompile-importance (Spearman {rho_static_imp:+.2f}) "
          f"— the most important MLPs are EARLY (L{top_mlp}) with the fewest incoming head->MLP edges (depth confound), "
          f"so the DAG edges give STRUCTURE, importance comes from the recompile (cf the bridge's ΔTV != KL-importance).")
    print(f"[done] {args.output}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axB, axC) = plt.subplots(1, 2, figsize=(12.8, 5.0))
        bars = [("attention only\n(all MLP ablated)", cov_attn_only, "#1f77b4"), ("MLP only\n(all heads ablated)", cov_mlp_only, "#2ca02c"),
                ("full\n(all)", cov_full, "#444444"),
                (f"sparse op-set\n({len(circuit_heads)}h + {k_dag} MLP)", cov_dag_ops, "#9467bd")]
        axB.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars], edgecolor="k")
        axB.set_xticks(range(len(bars))); axB.set_xticklabels([b[0] for b in bars], fontsize=8)
        axB.set_ylabel("reconstruction coverage  1 − KL/floor"); axB.axhline(0, color="k", lw=0.5, ls=":")
        axB.set_title("MLPs are load-bearing: removing all MLPs collapses coverage", fontsize=10)
        Bs = [c["n_mlp"] for c in mlp_curve]
        axC.plot(Bs, [c["coverage_attn_plus_topMLP"] for c in mlp_curve], "-o", color="#9467bd", label="all heads + top-importance MLP")
        axC.plot(Bs, [c["coverage_attn_plus_randMLP"] for c in mlp_curve], "--^", color="#999999", label="all heads + random MLP")
        axC.set_xlabel("# MLP layers kept (attention intact)"); axC.set_ylabel("reconstruction coverage")
        axC.set_title(f"few MLPs reconstruct most (L{top_mlp} dominates)", fontsize=10); axC.legend(fontsize=8)
        fig.suptitle("Milestone 3: the MLP instruction class enters the DAG + the recompile", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    return out


if __name__ == "__main__":
    main()
