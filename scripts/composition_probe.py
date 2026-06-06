"""Rung-2 composition probe: read the INDUCTION MACRO as a 2-head circuit from the weights.

Single-head reads are disassembly. Induction is the first META-instruction: a previous-token head A
(e.g. 4.11) writes "my predecessor was Y" into the residual via OV_A = W_V^A W_O^A; a later head B reads
that as its KEY and, querying on the current token X, attends to positions whose predecessor was X, then
copies what followed (K-composition). The composed pre-softmax binding in token-feature coords is
  C_AB[X,Y] = (d_X W_Q^B) · ((d_Y OV_A) W_K^B)^T = d_X · M_B · OV_A^T · d_Y      (M_B = W_Q^B W_K^B^T/√hd)
DIAGONAL-DOMINANT C_AB  ==  "attend to a key whose previous token equals my current token" = induction.
We (1) read C_AB for prev-token A x later B from the WEIGHTS, (2) VALIDATE against each B's realized
induction attention on text (mass on keys s with token[s-1]==token[query]), and (3) CONTROL: composing
through a SINK head A' (not a prev-token head) must NOT produce the induction diagonal. If the weight
composition through 4.11 predicts the real induction heads, the macro is read from the weights. GPT-2.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=14000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=40)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--n-prevtok", type=int, default=5, help="# prev-token heads A to use as writers")
    p.add_argument("--output", type=Path, default=Path("runs/composition_probe_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/composition_probe.png"))
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
    print(f"{args.pretrained}: layers={nL} heads={H} head_dim={hd}  operands={nt}")

    # ---- one pass: per-layer token centroids + per-head prev-token & INDUCTION attention ----
    cen_sum = np.zeros((nL, nt, d)); cen_cnt = np.zeros(nt); gmean_sum = np.zeros((nL, d)); gmean_cnt = 0
    pt_sum = np.zeros((nL, H)); pt_cnt = 0
    ind_sum = np.zeros((nL, H)); ind_q = 0; ind_base = 0.0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            Lc = len(c); ca = np.array(c)
            pidx = np.array([tok2i.get(t, -1) for t in c]); vmask = pidx >= 0
            cen_cnt += np.bincount(pidx[vmask], minlength=nt); gmean_cnt += Lc; pt_cnt += Lc - 1
            # induction key-mask: IndMask[q,s] = 1 if 1<=s<q and token[s-1]==token[q]
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]
            qi = np.arange(Lc)
            IndMask = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
            hasind = IndMask.any(1)
            ind_q += int(hasind.sum())
            if hasind.any():
                ind_base += float((IndMask.sum(1)[hasind] / np.maximum(qi[hasind], 1)).sum())
            for L in range(nL):
                hs = o.hidden_states[L][0].float().numpy()
                np.add.at(cen_sum[L], pidx[vmask], hs[vmask]); gmean_sum[L] += hs.sum(0)
                aL = o.attentions[L][0].float().numpy()                       # (H, Lc, Lc)
                pt_sum[L] += np.diagonal(aL, offset=-1, axis1=1, axis2=2).sum(1)
                ind_sum[L] += (aL * IndMask[None]).sum((1, 2))
    cen = cen_sum / np.maximum(cen_cnt, 1)[None, :, None]; gmean = gmean_sum / max(gmean_cnt, 1)
    prevtok_attn = pt_sum / max(pt_cnt, 1)
    induct_attn = ind_sum / max(ind_q, 1)                                     # mean mass on induction keys / ind-query
    induct_base = ind_base / max(ind_q, 1)
    induct_excess = induct_attn - induct_base                                # over chance
    offmask = ~np.eye(nt, dtype=bool)

    # ln-folded normalized token directions per layer
    Dl = []
    for L in range(nL):
        ln_w = tr.h[L].ln_1.weight.detach().numpy().astype(np.float64)
        D = (cen[L] - gmean[L]) * ln_w
        Dl.append(D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9))

    def ov(L, h):
        Wc = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wv = Wc[:, 2 * d:3 * d]; Wo = tr.h[L].attn.c_proj.weight.detach().numpy().astype(np.float64)
        sl = slice(h * hd, (h + 1) * hd); return Wv[:, sl] @ Wo[sl, :]

    def qk(L, h):
        Wc = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        sl = slice(h * hd, (h + 1) * hd)
        return Wc[:, sl] @ Wc[:, d:2 * d][:, sl].T / np.sqrt(hd)

    def comp_diag(LA, hA, LB, hB):
        """normalized diagonal dominance of C_AB = D_B M_B OV_A^T D_A^T."""
        C = Dl[LB] @ (qk(LB, hB) @ ov(LA, hA).T) @ Dl[LA].T
        return float((np.diag(C).mean() - C[offmask].mean()) / (C.std() + 1e-9))

    # B's OWN copy diagonal (single-head; induction NOT readable here)
    own_diag = {}
    for LB in range(nL):
        for hB in range(H):
            C = Dl[LB] @ qk(LB, hB) @ Dl[LB].T
            own_diag[(LB, hB)] = float((np.diag(C).mean() - C[offmask].mean()) / (C.std() + 1e-9))

    # writers: top prev-token heads A ; control: top sink head A' (low prevtok, we proxy by lowest prevtok among high-attn)
    flat = [((L, h), prevtok_attn[L, h]) for L in range(nL) for h in range(H)]
    A_heads = [lh for lh, _ in sorted(flat, key=lambda r: -r[1])[:args.n_prevtok]]
    A_star = A_heads[0]
    # a non-prev-token control writer in an early layer (lowest prev-token attn among layers <= A_star layer+1)
    early = [((L, h), prevtok_attn[L, h]) for L in range(A_star[0] + 1) for h in range(H)]
    A_ctrl = min(early, key=lambda r: r[1])[0]
    print(f"prev-token writer heads A: {A_heads}  (A*={A_star}, prevtok_attn {prevtok_attn[A_star]:.2f})")
    print(f"control writer A' (non-prev-token): {A_ctrl} (prevtok_attn {prevtok_attn[A_ctrl]:.3f})")

    # composed diagonal for A* x all later B, and control A' x all later B
    rows = []
    for LB in range(A_star[0] + 1, nL):
        for hB in range(H):
            rows.append({"B": [LB, hB], "comp_diag_via_Astar": comp_diag(*A_star, LB, hB),
                         "comp_diag_via_ctrl": comp_diag(*A_ctrl, LB, hB),
                         "own_copy_diag": own_diag[(LB, hB)],
                         "induct_excess": float(induct_excess[LB, hB]),
                         "induct_attn": float(induct_attn[LB, hB])})
    rho_star = _spearman([r["comp_diag_via_Astar"] for r in rows], [r["induct_excess"] for r in rows])
    rho_ctrl = _spearman([r["comp_diag_via_ctrl"] for r in rows], [r["induct_excess"] for r in rows])
    rho_own = _spearman([r["own_copy_diag"] for r in rows], [r["induct_excess"] for r in rows])

    # best (A,B) over ALL prev-token writers, ranked by composed diagonal
    allpairs = []
    for (LA, hA) in A_heads:
        for LB in range(LA + 1, nL):
            for hB in range(H):
                allpairs.append({"A": [LA, hA], "B": [LB, hB], "comp_diag": comp_diag(LA, hA, LB, hB),
                                 "induct_excess": float(induct_excess[LB, hB])})
    allpairs.sort(key=lambda r: -r["comp_diag"])

    out = {"experiment": "rung-2 composition probe (induction macro)", "model": args.pretrained,
           "A_prevtoken_heads": [list(a) for a in A_heads], "A_star": list(A_star),
           "A_control": list(A_ctrl), "induction_base_rate": induct_base,
           "spearman_compdiag_vs_induction_via_Astar": rho_star,
           "spearman_compdiag_vs_induction_via_control": rho_ctrl,
           "spearman_own_copydiag_vs_induction": rho_own,
           "top_behavioral_induction_heads": sorted(
               [{"B": [L, h], "induct_excess": float(induct_excess[L, h])} for L in range(nL) for h in range(H)],
               key=lambda r: -r["induct_excess"])[:8],
           "top_composed_pairs": allpairs[:12], "per_B_via_Astar": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\ntop BEHAVIORAL induction heads (attn mass on induction keys, base rate {induct_base:.3f}):")
    for r in out["top_behavioral_induction_heads"][:6]:
        print(f"  {r['B'][0]}.{r['B'][1]:<2} induct_excess {r['induct_excess']:+.3f}")
    print(f"\ntop COMPOSED (A->B) pairs by weight-read induction diagonal:")
    print(f"{'A(prev-tok)':>12} {'B':>6} {'comp_diag':>10} {'B induct_excess':>15}")
    for r in allpairs[:10]:
        print(f"{str(tuple(r['A'])):>12} {str(tuple(r['B'])):>6} {r['comp_diag']:>10.2f} {r['induct_excess']:>15.3f}")
    print(f"\n[validation] Spearman(weight composed-diagonal via A*={tuple(A_star)}, behavioral induction) over "
          f"{len(rows)} later heads:")
    print(f"   via prev-token A*  : {rho_star:+.3f}")
    print(f"   via control A'={tuple(A_ctrl)} (sink/non-prev): {rho_ctrl:+.3f}")
    print(f"   via B's OWN single-head copy diagonal: {rho_own:+.3f}  (induction NOT a single-head read)")
    ok = rho_star > 0.3 and rho_star > rho_ctrl + 0.15
    print(f"\n[verdict] {'INDUCTION MACRO READ FROM WEIGHTS: composing prev-token OV_A through B QK predicts the real induction heads, and the SINK control does not -> the 2-head meta-instruction is legible (rung-2 of the tower)' if ok else 'composition does not cleanly predict induction'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0))
        ce = np.array([r["comp_diag_via_Astar"] for r in rows]); ie = np.array([r["induct_excess"] for r in rows])
        cc = np.array([r["comp_diag_via_ctrl"] for r in rows])
        axA.scatter(ce, ie, s=45, c="#1f77b4", edgecolor="k", linewidth=0.4, label=f"via prev-tok A*={tuple(A_star)}")
        axA.scatter(cc, ie, s=30, c="#cccccc", edgecolor="k", linewidth=0.3, label=f"via sink control A'={tuple(A_ctrl)}")
        for r in sorted(rows, key=lambda r: -r["induct_excess"])[:4]:
            axA.annotate(f"{r['B'][0]}.{r['B'][1]}", (r["comp_diag_via_Astar"], r["induct_excess"]), fontsize=8)
        axA.set_xlabel("weight composed-diagonal C_AB  (induction binding via A)")
        axA.set_ylabel("behavioral induction attention (excess)")
        axA.set_title(f"induction macro from the weights\nρ via prev-tok {rho_star:+.2f} vs sink control {rho_ctrl:+.2f}", fontsize=10)
        axA.legend(fontsize=8)
        # C_AB heatmap for the top composed pair
        tp = allpairs[0]; LA, hA = tp["A"]; LB, hB = tp["B"]
        C = Dl[LB] @ (qk(LB, hB) @ ov(LA, hA).T) @ Dl[LA].T
        v = np.abs(C).max()
        names = [tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t in toks]
        oo = np.argsort(names)
        im = axB.imshow(C[np.ix_(oo, oo)], cmap="RdBu_r", vmin=-v, vmax=v)
        axB.set_title(f"C_AB top pair: A={LA}.{hA} (prev-tok) -> B={LB}.{hB}\ndiagonal = induction (query-tok == key's prev-tok)", fontsize=9)
        axB.set_xlabel("key prev-token (written by A)"); axB.set_ylabel("query current-token (B)")
        axB.set_xticks([]); axB.set_yticks([]); fig.colorbar(im, ax=axB, fraction=0.046)
        fig.suptitle("Rung-2: the induction META-instruction read as OV_A composed into QK_B", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        args.fig.parent.mkdir(parents=True, exist_ok=True); fig.savefig(args.fig, dpi=130)
        print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
