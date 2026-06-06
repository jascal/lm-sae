"""Induction wiring graph: map the full prev-token-writer -> inductor bipartite circuit.

Rung-2 read ONE edge (prev-token head 4.11 -> later induction heads, rho +0.78). This maps the whole
graph: for EVERY writer head A and EVERY later head B compute the composed induction binding
  comp_diag(A,B) = normalized diagonal of  C_AB = D_B (M_B OV_A^T) D_A^T
and overlay it with the behavioral types (prevtok_attn[A] = is-A-a-prev-token-head, induct_excess[B] =
is-B-an-induction-head). Questions: (1) does the composed binding concentrate exactly on
prev-token-writer x inductor cells (global specificity)? (2) for each real induction head, who is its
best writer -- and do different inductors branch off DIFFERENT writers (e.g. is 7.11 fed by a later
prev-token head than 4.11)? GPT-2.
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=14000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=40)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--pt-thresh", type=float, default=0.20, help="prev-token-attn to count A as a writer")
    p.add_argument("--ind-thresh", type=float, default=0.05, help="induction-excess to count B as an inductor")
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/induction_graph_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/induction_graph.png"))
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
    print(f"{args.pretrained}: layers={nL} heads={H}  operands={nt}")

    cen_sum = np.zeros((nL, nt, d)); cen_cnt = np.zeros(nt); gmean_sum = np.zeros((nL, d)); gmean_cnt = 0
    pt_sum = np.zeros((nL, H)); pt_cnt = 0
    ind_sum = np.zeros((nL, H)); ind_q = 0; ind_base = 0.0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            Lc = len(c); ca = np.array(c)
            pidx = np.array([tok2i.get(t, -1) for t in c]); vmask = pidx >= 0
            cen_cnt += np.bincount(pidx[vmask], minlength=nt); gmean_cnt += Lc; pt_cnt += Lc - 1
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]; qi = np.arange(Lc)
            IndMask = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
            hasind = IndMask.any(1); ind_q += int(hasind.sum())
            if hasind.any():
                ind_base += float((IndMask.sum(1)[hasind] / np.maximum(qi[hasind], 1)).sum())
            for L in range(nL):
                hs = o.hidden_states[L][0].float().numpy()
                np.add.at(cen_sum[L], pidx[vmask], hs[vmask]); gmean_sum[L] += hs.sum(0)
                aL = o.attentions[L][0].float().numpy()
                pt_sum[L] += np.diagonal(aL, offset=-1, axis1=1, axis2=2).sum(1)
                ind_sum[L] += (aL * IndMask[None]).sum((1, 2))
    cen = cen_sum / np.maximum(cen_cnt, 1)[None, :, None]; gmean = gmean_sum / max(gmean_cnt, 1)
    prevtok_attn = pt_sum / max(pt_cnt, 1)
    induct_excess = ind_sum / max(ind_q, 1) - ind_base / max(ind_q, 1)
    offmask = ~np.eye(nt, dtype=bool)

    Dl = []
    for L in range(nL):
        ln_w = tr.h[L].ln_1.weight.detach().numpy().astype(np.float64)
        D = (cen[L] - gmean[L]) * ln_w; Dl.append(D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9))

    def ov(L, h):
        Wc = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wo = tr.h[L].attn.c_proj.weight.detach().numpy().astype(np.float64)
        sl = slice(h * hd, (h + 1) * hd); return Wc[:, 2 * d:3 * d][:, sl] @ Wo[sl, :]

    def qk(L, h):
        Wc = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        sl = slice(h * hd, (h + 1) * hd); return Wc[:, sl] @ Wc[:, d:2 * d][:, sl].T / np.sqrt(hd)

    OVt = {(L, h): ov(L, h).T for L in range(nL) for h in range(H)}
    QK = {(L, h): qk(L, h) for L in range(nL) for h in range(H)}

    def comp_diag(LA, hA, LB, hB):
        C = Dl[LB] @ (QK[(LB, hB)] @ OVt[(LA, hA)]) @ Dl[LA].T
        return float((np.diag(C).mean() - C[offmask].mean()) / (C.std() + 1e-9))

    # ---- full pairwise composed-induction-diagonal for L_A < L_B ----
    pairs = []
    for LA in range(nL):
        for hA in range(H):
            for LB in range(LA + 1, nL):
                for hB in range(H):
                    pairs.append((LA, hA, LB, hB, comp_diag(LA, hA, LB, hB)))
    cd = np.array([p[4] for p in pairs])
    pa = np.array([prevtok_attn[p[0], p[1]] for p in pairs])
    ie = np.array([induct_excess[p[2], p[3]] for p in pairs])
    rho_pa = _spearman(cd, pa)
    rho_ie = _spearman(cd, ie)
    rho_prod = _spearman(cd, pa * np.clip(ie, 0, None))

    # ---- per-inductor best writer (over ALL earlier heads) ----
    inductors = sorted([(L, h) for L in range(1, nL) for h in range(H) if induct_excess[L, h] >= args.ind_thresh],
                       key=lambda lh: -induct_excess[lh])
    per_ind = []
    for (LB, hB) in inductors:
        writers = sorted([((LA, hA), comp_diag(LA, hA, LB, hB))
                          for LA in range(LB) for hA in range(H)], key=lambda r: -r[1])[:3]
        per_ind.append({"B": [LB, hB], "induct_excess": float(induct_excess[LB, hB]),
                        "top_writers": [{"A": list(a), "comp_diag": float(v),
                                         "A_prevtok_attn": float(prevtok_attn[a[0], a[1]])} for a, v in writers]})
    best_writer_is_prevtok = float(np.mean([w["top_writers"][0]["A_prevtok_attn"] >= args.pt_thresh
                                            for w in per_ind])) if per_ind else 0.0
    distinct_best = sorted({tuple(w["top_writers"][0]["A"]) for w in per_ind})

    writers_all = sorted([(L, h) for L in range(nL) for h in range(H) if prevtok_attn[L, h] >= args.pt_thresh],
                         key=lambda lh: -prevtok_attn[lh])
    out = {"experiment": "induction wiring graph", "model": args.pretrained,
           "induction_base_rate": float(ind_base / max(ind_q, 1)),
           "spearman_compdiag_vs_writerType": rho_pa, "spearman_compdiag_vs_inductorType": rho_ie,
           "spearman_compdiag_vs_product": rho_prod,
           "prevtoken_writers": [{"A": list(a), "prevtok_attn": float(prevtok_attn[a])} for a in writers_all],
           "best_writer_is_prevtoken_frac": best_writer_is_prevtok,
           "distinct_best_writers": [list(x) for x in distinct_best], "per_inductor": per_ind}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\nprev-token writer heads (attn>{args.pt_thresh}): "
          f"{[(f'{a[0]}.{a[1]}', round(float(prevtok_attn[a]),2)) for a in writers_all]}")
    print(f"\nper-inductor BEST writer (does the graph branch?):")
    print(f"{'inductor B':>11} {'ind_exc':>8}  best writer A (comp_diag, A.prevtok)")
    for w in per_ind:
        bw = w["top_writers"][0]
        alt = w["top_writers"][1]
        print(f"{w['B'][0]:>2}.{w['B'][1]:<2}{'':>5} {w['induct_excess']:>8.3f}  "
              f"{bw['A'][0]}.{bw['A'][1]} (cd {bw['comp_diag']:.2f}, pt {bw['A_prevtok_attn']:.2f})"
              f"   2nd: {alt['A'][0]}.{alt['A'][1]} (cd {alt['comp_diag']:.2f}, pt {alt['A_prevtok_attn']:.2f})")
    print(f"\n[global specificity] Spearman(comp_diag, ...) over {len(pairs)} ordered pairs:")
    print(f"   vs writer-is-prevtoken  {rho_pa:+.3f}   vs inductor-type {rho_ie:+.3f}   vs product {rho_prod:+.3f}")
    print(f"[graph] best writer is a prev-token head for {best_writer_is_prevtok:.0%} of {len(per_ind)} inductors; "
          f"distinct best writers = {[f'{a[0]}.{a[1]}' for a in distinct_best]}")
    branches = len(distinct_best) > 1
    ok = rho_prod > 0.3 and best_writer_is_prevtok > 0.5
    print(f"\n[verdict] {'induction is a STRUCTURED bipartite circuit: composed binding concentrates on prev-token-writer x inductor cells' if ok else 'no clean bipartite structure'}; "
          f"{'the graph BRANCHES (different inductors fed by different prev-token writers)' if branches else 'all inductors share one writer'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Wr = writers_all[:8] if len(writers_all) >= 2 else writers_all
        Bs = [tuple(w["B"]) for w in per_ind][:10]
        M = np.array([[comp_diag(a[0], a[1], b[0], b[1]) if a[0] < b[0] else np.nan for b in Bs] for a in Wr])
        fig, (axH, axS) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                       gridspec_kw={"width_ratios": [1.25, 1]})
        im = axH.imshow(M, cmap="viridis", aspect="auto")
        axH.set_xticks(range(len(Bs))); axH.set_xticklabels([f"{b[0]}.{b[1]}" for b in Bs], rotation=45, fontsize=8)
        axH.set_yticks(range(len(Wr))); axH.set_yticklabels([f"{a[0]}.{a[1]} (pt{prevtok_attn[a]:.2f})" for a in Wr], fontsize=8)
        axH.set_xlabel("inductor head B (sorted by induction strength)"); axH.set_ylabel("prev-token writer A")
        axH.set_title("composed induction binding comp_diag(A->B)", fontsize=10)
        for j, b in enumerate(Bs):                      # mark each inductor's best writer
            col = M[:, j]
            if np.isfinite(col).any():
                i = int(np.nanargmax(col)); axH.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="red", lw=1.6))
        fig.colorbar(im, ax=axH, fraction=0.046)
        axS.scatter(pa, cd, s=6, c=np.clip(ie, 0, None), cmap="Reds", alpha=0.5)
        axS.set_xlabel("writer A prev-token attention"); axS.set_ylabel("comp_diag(A->B)")
        axS.set_title(f"specificity: comp_diag vs writer-type\nρ(comp, prevtok×induct)={rho_prod:+.2f}", fontsize=10)
        fig.suptitle("Induction wiring graph: which prev-token writer feeds which inductor (red box = best writer)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
