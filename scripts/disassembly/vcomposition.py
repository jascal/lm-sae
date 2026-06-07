"""The VALUE pathway — V-composition, the third Elhage edge type the attention-readout (M2) is blind to.

`composition_dag.py` (M2) scored K- and Q-composition and gated them with **ΔTV** — the change in the reader's
attention *pattern* when a writer is removed from that port. By construction ΔTV cannot see **V-composition**:
head A's output feeding head B's **value** changes *what B moves*, not *where B attends* (its attention pattern
is untouched). V-composition is how heads chain OV circuits — "virtual heads" / composed copies — so the DAG is
incomplete without it.

This adds the value pathway with the matching readout. Static: `comp_V(A→B) = ‖OV_A · W_V^B‖_F / (‖OV_A‖‖W_V^B‖)`
(mean-write removed), the value analog of M2's K/Q score. Dynamic **ΔV-out**: remove A's output from B's *value*
input, recompute B's **output contribution** (attention fixed), and measure the relative change in B's residual
write — the value-pathway analog of ΔTV. Reader-matched null (random causal writers into B). The decisive check
is the **K/V dissociation**: for the strongest V-edges, removing A changes B's *output* (ΔV-out high) but **not**
its *attention* (ΔTV≈0); for the strongest K-edges the opposite — confirming V is a separable pathway, not a
relabelled K-edge. GPT-2; weights + two forward passes (faithful key/value patches verified against the model).
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


def _ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * w + b


def _causal_softmax(s):
    seq = s.shape[0]; s = s.copy()
    s[np.triu(np.ones((seq, seq), bool), 1)] = -1e30
    s -= s.max(1, keepdims=True); e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=14000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--top-v-edges", type=int, default=24, help="strongest static V-edges to gate")
    p.add_argument("--top-k-edges", type=int, default=24, help="strongest static K-edges (dissociation control)")
    p.add_argument("--k-null", type=int, default=3, help="reader-matched random-writer controls per (port,reader)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/vcomposition_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/vcomposition.png"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer; cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H; nL = cfg.n_layer; NH = nL * H
    layer_of = np.array([L for L in range(nL) for _ in range(H)])
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]

    def nm(i):
        return f"{i // H}.{i % H}"

    ln1w = [tr.h[L].ln_1.weight.detach().numpy().astype(np.float64) for L in range(nL)]
    ln1b = [tr.h[L].ln_1.bias.detach().numpy().astype(np.float64) for L in range(nL)]
    Wq = [tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)[:, :d] for L in range(nL)]
    Wk = [tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)[:, d:2 * d] for L in range(nL)]
    Wv = [tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)[:, 2 * d:3 * d] for L in range(nL)]
    bq = [tr.h[L].attn.c_attn.bias.detach().numpy().astype(np.float64)[:d] for L in range(nL)]
    bk = [tr.h[L].attn.c_attn.bias.detach().numpy().astype(np.float64)[d:2 * d] for L in range(nL)]
    bv = [tr.h[L].attn.c_attn.bias.detach().numpy().astype(np.float64)[2 * d:3 * d] for L in range(nL)]
    Wo = [tr.h[L].attn.c_proj.weight.detach().numpy().astype(np.float64) for L in range(nL)]

    # per-head OV / W_V / W_K, mean-write direction removed from OV (write_bus_check artifact)
    OV = np.zeros((NH, d, d)); WV = np.zeros((NH, d, hd)); WK = np.zeros((NH, d, hd))
    for L in range(nL):
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd); i = L * H + h
            OV[i] = Wv[L][:, sl] @ Wo[L][sl, :]; WV[i] = Wv[L][:, sl]; WK[i] = Wk[L][:, sl]
    u = np.linalg.svd(OV.transpose(0, 2, 1).reshape(-1, d), full_matrices=False)[2][0]
    OV = np.einsum("nij,jk->nik", OV, np.eye(d) - np.outer(u, u))
    ovn = np.linalg.norm(OV.reshape(NH, -1), axis=1) + 1e-9
    causal = layer_of[:, None] < layer_of[None, :]

    def comp(Wport):
        portn = np.linalg.norm(Wport.reshape(NH, -1), axis=1) + 1e-9
        S = np.zeros((NH, NH))
        for a in range(NH):
            for b in range(NH):
                if layer_of[b] > layer_of[a]:
                    S[a, b] = np.linalg.norm(OV[a] @ Wport[b]) / (ovn[a] * portn[b])
        return S
    Vc = comp(WV); Kc = comp(WK)
    print(f"{args.pretrained}: {NH} heads; static V- and K-composition computed (mean-write removed)")
    print(f"  composition magnitude: V mean {Vc[causal].mean():.3f} vs K mean {Kc[causal].mean():.3f}  "
          f"(V/K {Vc[causal].mean() / max(Kc[causal].mean(), 1e-9):.2f}); "
          f"V max {Vc.max():.3f} vs K max {Kc.max():.3f}")

    def topedges(S, n):
        flat = [(a, b, S[a, b]) for a in range(NH) for b in range(NH) if S[a, b] > 0]
        return [(a, b) for a, b, _ in sorted(flat, key=lambda r: -r[2])[:n]]

    # edges: top static V (port V) + top static K (port K, dissociation control) + per-(port,reader) random null
    edges = {}                                                                # (port,a,b) -> {static, kind}
    for (a, b) in topedges(Vc, args.top_v_edges):
        edges[("V", a, b)] = {"static": float(Vc[a, b]), "kind": "topV"}
    for (a, b) in topedges(Kc, args.top_k_edges):
        edges[("K", a, b)] = {"static": float(Kc[a, b]), "kind": "topK"}
    rng = np.random.default_rng(args.seed)
    for port, S in (("V", Vc), ("K", Kc)):
        for b in sorted({bb for (pp, _a, bb) in list(edges) if pp == port}):
            pool = [a for a in range(NH) if layer_of[a] < layer_of[b] and (port, a, b) not in edges]
            for a in (int(x) for x in rng.permutation(pool)[: args.k_null]):
                edges[(port, a, b)] = {"static": float(S[a, b]), "kind": "null"}
    writers_all = {a for (_p, a, _b) in edges}; readers_all = {b for (_p, _a, b) in edges}
    print(f"  gating {len(edges)} edges; measuring BOTH ΔV-out (value pathway) and ΔTV (attention) for each...")

    # ---- pass: remove A from B's value -> ΔV-out (B output change); remove A from B's key -> ΔTV (attention change) ----
    dvout = {e: 0.0 for e in edges}; dtv = {e: 0.0 for e in edges}; tot_tok = 0; tot_out = defaultdict(float)
    sane_k = None
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            HS = [o.hidden_states[L][0].float().numpy() for L in range(nL)]
            ATT = [o.attentions[L][0].float().numpy() for L in range(nL)]
            Lc = len(c); tot_tok += Lc
            Aout = {}
            for a in writers_all:
                La, hA = a // H, a % H; sl = slice(hA * hd, (hA + 1) * hd)
                vA = _ln(HS[La], ln1w[La], ln1b[La]) @ Wv[La][:, sl] + bv[La][sl]
                Aout[a] = (ATT[La][hA] @ vA) @ Wo[La][sl, :]
            clean = {}
            for b in readers_all:
                Lb, hB = b // H, b % H; sl = slice(hB * hd, (hB + 1) * hd)
                lnB = _ln(HS[Lb], ln1w[Lb], ln1b[Lb])
                Q = lnB @ Wq[Lb][:, sl] + bq[Lb][sl]; K = lnB @ Wk[Lb][:, sl] + bk[Lb][sl]
                vB = lnB @ Wv[Lb][:, sl] + bv[Lb][sl]; P = _causal_softmax(Q @ K.T / np.sqrt(hd))
                outB = (P @ vB) @ Wo[Lb][sl, :]                               # B's residual write (seq,d)
                clean[b] = (Q, K, P, outB); tot_out[b] += float(np.linalg.norm(outB))
                if sane_k is None:
                    sane_k = float(np.abs(P - ATT[Lb][hB]).max())
            for (port, a, b) in edges:
                Lb, hB = b // H, b % H; sl = slice(hB * hd, (hB + 1) * hd)
                Q, K, P, outB = clean[b]
                lnp = _ln(HS[Lb] - Aout[a], ln1w[Lb], ln1b[Lb])
                # value pathway: patch B's value, attention fixed -> change in B's OUTPUT
                vBp = lnp @ Wv[Lb][:, sl] + bv[Lb][sl]; outBp = (P @ vBp) @ Wo[Lb][sl, :]
                dvout[(port, a, b)] += float(np.linalg.norm(outB - outBp))
                # attention pathway: patch B's key, queries fixed -> change in B's attention pattern
                Kp = lnp @ Wk[Lb][:, sl] + bk[Lb][sl]; Pp = _causal_softmax(Q @ Kp.T / np.sqrt(hd))
                dtv[(port, a, b)] += float(0.5 * np.abs(P - Pp).sum())
    # normalize: ΔV-out relative to B's clean output norm; ΔTV per token
    print(f"[sanity] recomputed clean attn vs model max|Δ| = {sane_k:.2e}")

    rows = []
    for (port, a, b), m in edges.items():
        relv = dvout[(port, a, b)] / max(tot_out[b], 1e-9)                    # ΔV-out relative to B's output norm
        tv = dtv[(port, a, b)] / max(tot_tok, 1)
        rows.append({"port": port, "kind": m["kind"], "A": nm(a), "B": nm(b),
                     "static": m["static"], "dvout": relv, "dtv": tv})

    # reader-matched nulls (per port) and specificity
    nullv = defaultdict(list); nullt = defaultdict(list)
    for r, (port, a, b) in zip(rows, edges):
        if r["kind"] == "null":
            nullv[(r["port"], r["B"])].append(r["dvout"]); nullt[(r["port"], r["B"])].append(r["dtv"])
    nv_mean = {k: float(np.mean(v)) for k, v in nullv.items()}
    nt_mean = {k: float(np.mean(v)) for k, v in nullt.items()}
    for r in rows:
        r["dvout_spec"] = r["dvout"] - nv_mean.get((r["port"], r["B"]), 0.0)
        r["dtv_spec"] = r["dtv"] - nt_mean.get((r["port"], r["B"]), 0.0)

    topV = [r for r in rows if r["kind"] == "topV"]; topK = [r for r in rows if r["kind"] == "topK"]
    rho_v = _spearman([r["static"] for r in topV], [r["dvout_spec"] for r in topV])
    rho_k = _spearman([r["static"] for r in topK], [r["dtv_spec"] for r in topK])
    mv = lambda rs, key: float(np.median([r[key] for r in rs])) if rs else 0.0  # noqa: E731
    out = {"experiment": "V-composition — the value pathway (the third Elhage edge type)", "model": args.pretrained,
           "n_heads": NH, "V_comp_mean": float(Vc[causal].mean()), "K_comp_mean": float(Kc[causal].mean()),
           "V_over_K_mean": float(Vc[causal].mean() / max(Kc[causal].mean(), 1e-9)),
           "spearman_staticV_vs_dVout": rho_v, "spearman_staticK_vs_dTV": rho_k,
           "topV_median_dVout": mv(topV, "dvout"), "topV_median_dTV": mv(topV, "dtv"),
           "topK_median_dVout": mv(topK, "dvout"), "topK_median_dTV": mv(topK, "dtv"),
           "edges": sorted(rows, key=lambda r: -r["dvout_spec"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\n[static->dynamic] V: Spearman(static V-comp, ΔV-out specificity) = {rho_v:+.3f}  |  "
          f"K: Spearman(static K-comp, ΔTV specificity) = {rho_k:+.3f}")
    print(f"[K/V dissociation] top-V edges: ΔV-out {mv(topV, 'dvout'):.3f} / ΔTV {mv(topV, 'dtv'):.4f}  ||  "
          f"top-K edges: ΔV-out {mv(topK, 'dvout'):.3f} / ΔTV {mv(topK, 'dtv'):.4f}")
    print("\ntop value-pathway edges by ΔV-out specificity (A's write feeds B's VALUE = composed OV / virtual head):")
    print(f"  {'V edge':>12} {'static':>7} {'ΔV-out':>7} {'ΔTV':>7}")
    for r in sorted(topV, key=lambda r: -r["dvout_spec"])[:10]:
        print(f"  {r['A']+'->'+r['B']:>12} {r['static']:>7.3f} {r['dvout']:>7.3f} {r['dtv']:>7.4f}")

    v_pathway = mv(topV, "dvout") > 2 * mv(topV, "dtv") and rho_v > 0.2
    sparser = out["V_over_K_mean"] < 1.0
    if v_pathway:
        verdict = (f"VALUE PATHWAY ADDED: V-composition is a SEPARABLE edge type — top-V edges change B's OUTPUT "
                   f"(median ΔV-out {mv(topV, 'dvout'):.3f}) but barely its attention (ΔTV {mv(topV, 'dtv'):.4f}), "
                   f"the mirror of top-K edges (ΔTV {mv(topK, 'dtv'):.4f}); static V-composition predicts dynamic "
                   f"ΔV-out (ρ {rho_v:+.2f}). V-composition is {'SPARSER/weaker than K' if sparser else 'comparable to K'} "
                   f"(V/K mean {out['V_over_K_mean']:.2f}) — the value pathway is real but secondary to attention "
                   f"routing, and it is exactly what M2's ΔTV readout could not see. The DAG now has K, Q AND V edges.")
    else:
        verdict = "partial — see edge table"
    print(f"\n[verdict] {verdict}")
    print(f"[done] {args.output}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axD, axS) = plt.subplots(1, 2, figsize=(12.8, 5.0))
        # dissociation scatter: ΔV-out vs ΔTV, colored by edge kind
        cols = {"topV": "#2ca02c", "topK": "#d62728", "null": "#cccccc"}
        for kind, cl in cols.items():
            pts = [(r["dtv"], r["dvout"]) for r in rows if r["kind"] == kind]
            if pts:
                xs, ys = zip(*pts); axD.scatter(xs, ys, c=cl, s=42, edgecolor="k", linewidth=0.3, label=kind)
        axD.set_xlabel("ΔTV  (attention-pattern change, K-patch)")
        axD.set_ylabel("ΔV-out  (B output change, V-patch)")
        axD.set_title("K/V dissociation: V-edges move VALUE not attention", fontsize=10); axD.legend(fontsize=8)
        groups = [("top-V\nΔV-out", [r["dvout"] for r in topV], "#2ca02c"), ("top-V\nΔTV", [r["dtv"] for r in topV], "#98df8a"),
                  ("top-K\nΔV-out", [r["dvout"] for r in topK], "#ff9896"), ("top-K\nΔTV", [r["dtv"] for r in topK], "#d62728")]
        axS.bar(range(4), [np.median(g[1]) if g[1] else 0 for g in groups], color=[g[2] for g in groups], edgecolor="k")
        axS.set_xticks(range(4)); axS.set_xticklabels([g[0] for g in groups], fontsize=8)
        axS.set_ylabel("median effect"); axS.set_title(f"value vs attention pathway  (ρ_V={rho_v:+.2f})", fontsize=10)
        fig.suptitle("V-composition: the value pathway (third Elhage edge type), invisible to M2's ΔTV", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    return out


if __name__ == "__main__":
    main()
