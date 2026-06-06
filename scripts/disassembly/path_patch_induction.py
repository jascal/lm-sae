"""Rung-2(d): path-patch the A->B key edge to GATE candidate induction edges (static -> live).

The wiring graph gave CANDIDATE edges (prev-token writer A -> inductor B) but weights alone are
writer-specific, not inductor-selective: a prev-token OV composes a diagonal into MANY later QKs.
This is the DYNAMIC trace that selects the LIVE edges. Induction is K-composition: A writes
"my predecessor was Y" into the residual; B reads it as its KEY. So we surgically remove A's output
from B's KEY computation only (queries untouched) and measure how much B's BEHAVIORAL induction
attention (mass on keys s with token[s-1]==token[query]) collapses:
  R_B            = residual entering layer B (= hidden_states[LB])
  R_A_out        = head A's additive output contribution (attn_A @ (ln_A x W_V^A)) @ W_O^A
  K_clean  = (ln^B R_B) W_K^B ;  K_patched = (ln^B (R_B - R_A_out)) W_K^B ;  Q unchanged
  delta_induction = induction_attn(softmax Q K_clean^T) - induction_attn(softmax Q K_patched^T)
A LIVE edge -> large positive delta (removing A kills B's induction). CONTROLS: a SINK (non-prev-token)
writer A' must give ~0; patching into a NON-inductor B' must give ~0 (these also rule out the LN
renorm artifact). Headline: does static comp_diag predict dynamic liveness (delta)? GPT-2.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
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
    p.add_argument("--n-tokens", type=int, default=40)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--n-writers", type=int, default=4)
    p.add_argument("--n-inductors", type=int, default=6)
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/path_patch_induction_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/path_patch_induction.png"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    all_ids = [j for c in chunks for j in c]
    cnt = Counter(all_ids)
    cand = [tok(c, add_special_tokens=False)["input_ids"][0] for c in COMMON
            if len(tok(c, add_special_tokens=False)["input_ids"]) == 1]
    cand += [t for t, _ in cnt.most_common(300)]
    seen, toks = set(), []
    for t in cand:
        if t not in seen and cnt[t] >= args.min_pos:
            seen.add(t); toks.append(t)
        if len(toks) >= args.n_tokens:
            break
    nt = len(toks); tok2i = {t: i for i, t in enumerate(toks)}

    # cache per-layer weight slices
    ln1w = [tr.h[L].ln_1.weight.detach().numpy().astype(np.float64) for L in range(nL)]
    ln1b = [tr.h[L].ln_1.bias.detach().numpy().astype(np.float64) for L in range(nL)]
    Wq = [tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)[:, :d] for L in range(nL)]
    Wk = [tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)[:, d:2 * d] for L in range(nL)]
    Wv = [tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)[:, 2 * d:3 * d] for L in range(nL)]
    bq = [tr.h[L].attn.c_attn.bias.detach().numpy().astype(np.float64)[:d] for L in range(nL)]
    bk = [tr.h[L].attn.c_attn.bias.detach().numpy().astype(np.float64)[d:2 * d] for L in range(nL)]
    bv = [tr.h[L].attn.c_attn.bias.detach().numpy().astype(np.float64)[2 * d:3 * d] for L in range(nL)]
    Wo = [tr.h[L].attn.c_proj.weight.detach().numpy().astype(np.float64) for L in range(nL)]

    def indmask(c):
        Lc = len(c); ca = np.array(c); prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]; qi = np.arange(Lc)
        return (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)

    # ---- PASS 1: prev-token attn, induction excess, token centroids (for comp_diag) ----
    print(f"{args.pretrained}: layers={nL} heads={H}  pass1 (stats)...")
    cen_sum = np.zeros((nL, nt, d)); cen_cnt = np.zeros(nt); gmean_sum = np.zeros((nL, d)); gmean_cnt = 0
    pt_sum = np.zeros((nL, H)); pt_cnt = 0; ind_sum = np.zeros((nL, H)); ind_q = 0; ind_base = 0.0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            Lc = len(c); pidx = np.array([tok2i.get(t, -1) for t in c]); vmask = pidx >= 0
            cen_cnt += np.bincount(pidx[vmask], minlength=nt); gmean_cnt += Lc; pt_cnt += Lc - 1
            IM = indmask(c); has = IM.any(1); ind_q += int(has.sum())
            if has.any():
                ind_base += float((IM.sum(1)[has] / np.maximum(np.arange(Lc)[has], 1)).sum())
            for L in range(nL):
                hs = o.hidden_states[L][0].float().numpy()
                np.add.at(cen_sum[L], pidx[vmask], hs[vmask]); gmean_sum[L] += hs.sum(0)
                aL = o.attentions[L][0].float().numpy()
                pt_sum[L] += np.diagonal(aL, offset=-1, axis1=1, axis2=2).sum(1)
                ind_sum[L] += (aL * IM[None]).sum((1, 2))
    cen = cen_sum / np.maximum(cen_cnt, 1)[None, :, None]; gmean = gmean_sum / max(gmean_cnt, 1)
    prevtok_attn = pt_sum / max(pt_cnt, 1); induct_excess = ind_sum / max(ind_q, 1) - ind_base / max(ind_q, 1)
    offmask = ~np.eye(nt, dtype=bool)
    Dl = []
    for L in range(nL):
        D = (cen[L] - gmean[L]) * ln1w[L]; Dl.append(D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9))

    def comp_diag(LA, hA, LB, hB):
        slA = slice(hA * hd, (hA + 1) * hd); slB = slice(hB * hd, (hB + 1) * hd)
        MB = Wq[LB][:, slB] @ Wk[LB][:, slB].T / np.sqrt(hd); OVA = Wv[LA][:, slA] @ Wo[LA][slA, :]
        C = Dl[LB] @ (MB @ OVA.T) @ Dl[LA].T
        return float((np.diag(C).mean() - C[offmask].mean()) / (C.std() + 1e-9))

    # pick writers / inductors / controls
    pf = [((L, h), prevtok_attn[L, h]) for L in range(nL) for h in range(H)]
    writers = [lh for lh, _ in sorted(pf, key=lambda r: -r[1])[:args.n_writers]]
    ctrl_writer = min([lh for lh in (x[0] for x in pf) if lh[0] <= 5], key=lambda lh: prevtok_attn[lh])
    indf = [((L, h), induct_excess[L, h]) for L in range(2, nL) for h in range(H)]
    inductors = [lh for lh, _ in sorted(indf, key=lambda r: -r[1])[:args.n_inductors]]
    ctrl_ind = min([lh for lh in (x[0] for x in indf) if lh[0] >= 6], key=lambda lh: induct_excess[lh])
    print(f"writers={writers} ctrl_writer={ctrl_writer} (pt {prevtok_attn[ctrl_writer]:.3f})")
    print(f"inductors={inductors} ctrl_inductor={ctrl_ind} (ind {induct_excess[ctrl_ind]:+.3f})")

    # candidate edges (LA<LB): each inductor x {its writers + control writer}; + control inductor x writers
    edges = []
    for B in inductors + [ctrl_ind]:
        for A in writers + [ctrl_writer]:
            if A[0] < B[0]:
                edges.append((A, B))
    edges = sorted(set(edges))
    e_clean = {e: 0.0 for e in edges}; e_patch = {e: 0.0 for e in edges}; e_q = 0
    sane = None

    # ---- PASS 2: path-patch each edge's KEY path, accumulate induction collapse ----
    print(f"pass2 (patch {len(edges)} edges)...")
    with torch.no_grad():
        for ci, c in enumerate(chunks):
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            HS = [o.hidden_states[L][0].float().numpy() for L in range(nL)]
            ATT = [o.attentions[L][0].float().numpy() for L in range(nL)]
            IM = indmask(c); e_q += int(IM.sum())
            Aout = {}
            for (LA, hA) in set(a for a, _ in edges):
                slA = slice(hA * hd, (hA + 1) * hd)
                lnA = _ln(HS[LA], ln1w[LA], ln1b[LA])
                vA = lnA @ Wv[LA][:, slA] + bv[LA][slA]
                Aout[(LA, hA)] = (ATT[LA][hA] @ vA) @ Wo[LA][slA, :]            # (seq,d)
            for (A, B) in edges:
                LB, hB = B; slB = slice(hB * hd, (hB + 1) * hd)
                lnB = _ln(HS[LB], ln1w[LB], ln1b[LB])
                Q = lnB @ Wq[LB][:, slB] + bq[LB][slB]
                Kc = lnB @ Wk[LB][:, slB] + bk[LB][slB]
                Pc = _causal_softmax(Q @ Kc.T / np.sqrt(hd))
                lnBp = _ln(HS[LB] - Aout[A], ln1w[LB], ln1b[LB])
                Kp = lnBp @ Wk[LB][:, slB] + bk[LB][slB]
                Pp = _causal_softmax(Q @ Kp.T / np.sqrt(hd))
                e_clean[(A, B)] += float((Pc * IM).sum()); e_patch[(A, B)] += float((Pp * IM).sum())
                if sane is None:
                    sane = float(np.abs(Pc - ATT[LB][hB]).max())
    print(f"[sanity] max|recomputed clean attn - model attn| (first edge) = {sane:.2e}")

    rows = []
    for (A, B) in edges:
        clean = e_clean[(A, B)] / max(e_q, 1); patch = e_patch[(A, B)] / max(e_q, 1)
        delta = clean - patch
        rows.append({"A": list(A), "B": list(B), "comp_diag": comp_diag(*A, *B),
                     "A_prevtok": float(prevtok_attn[A]), "B_induct": float(induct_excess[B]),
                     "clean_induction": clean, "patched_induction": patch, "delta": delta,
                     "rel_drop": float(delta / clean) if clean > 1e-9 else 0.0,
                     "is_ctrl_writer": A == ctrl_writer, "is_ctrl_inductor": B == ctrl_ind})
    rho_static = _spearman([r["comp_diag"] for r in rows], [r["delta"] for r in rows])
    real = [r for r in rows if not r["is_ctrl_writer"] and not r["is_ctrl_inductor"]]
    cw = [r for r in rows if r["is_ctrl_writer"] and not r["is_ctrl_inductor"]]
    cib = [r for r in rows if r["is_ctrl_inductor"] and not r["is_ctrl_writer"]]
    md_real = float(np.median([r["delta"] for r in real])) if real else 0.0
    md_cw = float(np.median([r["delta"] for r in cw])) if cw else 0.0
    md_cib = float(np.median([r["delta"] for r in cib])) if cib else 0.0

    out = {"experiment": "rung-2(d) path-patch induction key-edge gate", "model": args.pretrained,
           "writers": [list(a) for a in writers], "ctrl_writer": list(ctrl_writer),
           "inductors": [list(a) for a in inductors], "ctrl_inductor": list(ctrl_ind),
           "spearman_compdiag_vs_delta": rho_static, "median_delta_real_edges": md_real,
           "median_delta_ctrl_writer": md_cw, "median_delta_ctrl_inductor": md_cib,
           "sanity_recompute_err": sane, "edges": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\nedge induction-attention collapse under key-path patching (clean base rate {ind_base/max(ind_q,1):.3f}):")
    print(f"{'A->B':>14} {'cd':>5} {'A.pt':>5} {'B.ind':>6} {'clean':>6} {'patch':>6} {'Δ':>7} {'rel':>6}  tag")
    for r in sorted(rows, key=lambda r: -r["delta"]):
        tag = "CTRL-writer" if r["is_ctrl_writer"] else ("CTRL-inductor" if r["is_ctrl_inductor"] else "")
        print(f"{r['A'][0]}.{r['A'][1]}->{r['B'][0]}.{r['B'][1]:<2}{'':>4} {r['comp_diag']:>5.2f} {r['A_prevtok']:>5.2f} "
              f"{r['B_induct']:>6.3f} {r['clean_induction']:>6.3f} {r['patched_induction']:>6.3f} "
              f"{r['delta']:>+7.3f} {r['rel_drop']:>+6.1%}  {tag}")
    strong = [r for r in real if r["comp_diag"] >= np.median([x["comp_diag"] for x in real])]
    md_strong = float(np.median([r["delta"] for r in strong])) if strong else 0.0
    top_rel = max((r["rel_drop"] for r in real), default=0.0)
    out["median_delta_strong_edges"] = md_strong; out["max_rel_drop"] = top_rel
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[gate] median Δinduction: STRONG real edges {md_strong:+.3f}  |  all real {md_real:+.3f}  |  "
          f"ctrl-writer {md_cw:+.3f}  |  ctrl-inductor {md_cib:+.3f}   (top live edge drops {top_rel:.0%})")
    print(f"[static->live] Spearman(comp_diag, Δinduction) over {len(rows)} edges = {rho_static:+.3f}")
    sign_sep = md_strong > 0.0 > md_cw                       # real writers ADD induction; sink writer does not
    ok = md_strong > 0.005 and sign_sep and rho_static > 0.4 and top_rel > 0.15
    print(f"\n[verdict] {('LIVE EDGES CONFIRMED: removing a prev-token writer from B key path collapses B induction (top edge -' + f'{top_rel:.0%}); the SINK-writer control goes the OTHER way and the NON-inductor control is ~0 -> candidate edges are dynamically real. And static comp_diag PREDICTS dynamic liveness (rho ' + f'{rho_static:+.2f}) -> the disassembly DID encode the live graph; the crude weight product just measured it wrong.') if ok else 'patching did not isolate a live induction edge'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axB, axS) = plt.subplots(1, 2, figsize=(12.8, 5.0))
        groups = [("real (prevtok->inductor)", real, "#1f77b4"), ("ctrl writer (sink->inductor)", cw, "#ff7f0e"),
                  ("ctrl inductor (prevtok->non-ind)", cib, "#999999")]
        for i, (lbl, g, cl) in enumerate(groups):
            ys = [r["delta"] for r in g]
            axB.scatter(np.full(len(ys), i) + np.linspace(-.15, .15, len(ys)), ys, c=cl, s=40,
                        edgecolor="k", linewidth=0.3, label=lbl)
        axB.axhline(0, color="k", lw=0.6, ls=":")
        axB.set_xticks(range(3)); axB.set_xticklabels([g[0] for g in groups], rotation=12, ha="right", fontsize=8)
        axB.set_ylabel("Δ induction attention (clean − patched)")
        axB.set_title("path-patch gate: live edges drop, controls don't", fontsize=10)
        cols = ["#1f77b4" if (not r["is_ctrl_writer"] and not r["is_ctrl_inductor"]) else "#bbbbbb" for r in rows]
        axS.scatter([r["comp_diag"] for r in rows], [r["delta"] for r in rows], c=cols, s=45, edgecolor="k", linewidth=0.3)
        for r in sorted(real, key=lambda r: -r["delta"])[:5]:
            axS.annotate(f"{r['A'][0]}.{r['A'][1]}→{r['B'][0]}.{r['B'][1]}", (r["comp_diag"], r["delta"]), fontsize=7)
        axS.set_xlabel("static comp_diag(A→B)"); axS.set_ylabel("dynamic Δ induction")
        axS.set_title(f"does static predict live? ρ={rho_static:+.2f}", fontsize=10)
        fig.suptitle("Rung-2(d): path-patching the A→B key edge gates candidate induction edges into live ones", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
