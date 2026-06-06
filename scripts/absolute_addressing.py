"""Absolute/structural-position channel: explain the 112 'neither' heads.

The content (B_h) + relative-offset (P_h[Δ]) split left 112/144 heads sharp on NEITHER -- predicted to be
a THIRD addressing mode: ABSOLUTE position (attend to a fixed key index, esp. the position-0 attention
SINK) and STRUCTURAL position (attend to landmark tokens -- newlines / line starts, the Ċ-attractors the
opcode table flagged). GPT-2 has LEARNED ABSOLUTE positions, so these are real modes the relative-Δ
channel cannot see. This adds two readers:
  absolute  Spos_col[s] = mean_t (p_t·M_h·p_s)   -- columnar (key-position) component of the positional
            score; a SINK head peaks at s=0. Behavioral: colbeh[s] = mean attention position s receives
            from later queries; abs_sink = colbeh[0].
  structural newline-excess = mean attention to newline keys minus the newline base rate.
Then bucket all 144 heads by DOMINANT addressing mode {content, relative-pos, absolute-sink,
structural-newline, none} and ask: do absolute+structural account for the 'neither' mass? GPT-2.
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


def _softmax_offdiag(B):
    M = B.copy(); np.fill_diagonal(M, -np.inf)
    M = M - np.nanmax(np.where(np.isfinite(M), M, -np.inf), axis=1, keepdims=True)
    E = np.exp(M); E[~np.isfinite(E)] = 0.0
    s = E.sum(1, keepdims=True)
    return E / np.where(s > 0, s, 1.0)


def _z(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=14000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=40)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--max-delta", type=int, default=24)
    p.add_argument("--output", type=Path, default=Path("runs/absolute_addressing_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/absolute_addressing.png"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    nl_ids = set()
    for s in ("\n", "\n\n", " \n"):
        ii = tok(s, add_special_tokens=False)["input_ids"]
        if len(ii) == 1:
            nl_ids.add(ii[0])
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    all_ids = [j for c in chunks for j in c]
    cnt = Counter(all_ids)
    C = args.ctx; maxД = args.max_delta
    print(f"{args.pretrained}: layers={nL} heads={H}  newline-ids={sorted(nl_ids)}  chunks={len(chunks)}")

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

    # ---- one pass: content + position centroids, relative-Δ attn, absolute-key attn, newline attn ----
    cen_sum = np.zeros((nL, nt, d)); cen_cnt = np.zeros(nt)
    pos_sum = np.zeros((nL, C, d)); pos_cnt = np.zeros(C)
    gmean_sum = np.zeros((nL, d)); gmean_cnt = 0
    apos_sum = np.zeros((nL, H, maxД + 1)); apos_cnt = np.zeros(maxД + 1)
    abskey_sum = np.zeros((nL, H, C)); abskey_cnt = np.zeros(C)
    nl_sum = np.zeros((nL, H)); nq_sum = 0; nl_base_sum = 0.0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            Lc = len(c)
            pidx = np.array([tok2i.get(t, -1) for t in c]); vmask = pidx >= 0
            cen_cnt += np.bincount(pidx[vmask], minlength=nt)
            pos_cnt[:Lc] += 1; gmean_cnt += Lc; nq_sum += Lc
            abskey_cnt[:Lc] += np.maximum(Lc - 1 - np.arange(Lc), 0)
            for dl in range(1, min(maxД, Lc - 1) + 1):
                apos_cnt[dl] += Lc - dl
            km = np.array([1.0 if t in nl_ids else 0.0 for t in c])           # newline key-mask
            cumnl = np.cumsum(km)                                             # available newlines up to (incl) t
            nl_base_sum += float((cumnl / (np.arange(Lc) + 1.0)).sum())
            for L in range(nL):
                hs = o.hidden_states[L][0].float().numpy()
                np.add.at(cen_sum[L], pidx[vmask], hs[vmask])
                pos_sum[L, :Lc] += hs; gmean_sum[L] += hs.sum(0)
                aL = o.attentions[L][0].float().numpy()                       # (H, Lc, Lc)
                diag = np.diagonal(aL, axis1=1, axis2=2)                      # (H, Lc) self-attn
                abskey_sum[L, :, :Lc] += aL.sum(1) - diag                     # attn each key gets from LATER queries
                nl_sum[L] += (aL @ km).sum(1)                                 # attn mass to newline keys, summed over queries
                for dl in range(1, min(maxД, Lc - 1) + 1):
                    apos_sum[L, :, dl] += np.diagonal(aL, offset=-dl, axis1=1, axis2=2).sum(1)

    cen = cen_sum / np.maximum(cen_cnt, 1)[None, :, None]
    posc = pos_sum / np.maximum(pos_cnt, 1)[None, :, None]
    gmean = gmean_sum / max(gmean_cnt, 1)
    offmask = ~np.eye(nt, dtype=bool)
    Abeh = apos_sum / np.maximum(apos_cnt, 1)[None, None, :]
    Abeh_mean = Abeh.reshape(-1, maxД + 1).mean(0)
    colbeh = abskey_sum / np.maximum(abskey_cnt, 1)[None, None, :]            # (nL,H,C) attn position s gets
    nl_beh = nl_sum / max(nq_sum, 1)                                          # mean attn to newline keys
    nl_base = nl_base_sum / max(nq_sum, 1)

    heads = []
    for L in range(nL):
        blk = tr.h[L]
        ln_w = blk.ln_1.weight.detach().numpy().astype(np.float64)
        Dc = (cen[L] - gmean[L]) * ln_w; Dc = Dc / (np.linalg.norm(Dc, axis=1, keepdims=True) + 1e-9)
        Pp = (posc[L] - gmean[L]) * ln_w
        W = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wq, Wk = W[:, :d], W[:, d:2 * d]
        for h in range(H):
            Mh = Wq[:, h * hd:(h + 1) * hd] @ Wk[:, h * hd:(h + 1) * hd].T / np.sqrt(hd)
            B = Dc @ Mh @ Dc.T
            diag_score = float(np.diag(B).mean() - B[offmask].mean())
            soft = _softmax_offdiag(B); col_mass = soft.sum(0)
            row_peak = float(soft.max(1).mean()); broadcast = float(col_mass.max() / col_mass.sum())
            # relative
            Spos = Pp @ Mh @ Pp.T
            Ph = np.array([np.diagonal(Spos, offset=-dl).mean() for dl in range(1, maxД + 1)])
            ab = Abeh[L, h, 1:]; pos_excess = float((ab - Abeh_mean[1:]).max())
            pos_leg = _spearman(Ph, ab)
            # absolute: columnar score (key-position), sink = position 0
            iu = np.triu_indices(C, k=1)                                      # t<s upper? want mean over t>s per key s
            Spos_col = np.array([Spos[s + 1:, s].mean() if s + 1 < C else 0.0 for s in range(C)])
            cb = colbeh[L, h]
            abs_sink_beh = float(cb[0])
            abs_sink_weight = float((Spos_col[0] - Spos_col.mean()) / (Spos_col.std() + 1e-9))
            abs_leg = _spearman(Spos_col[:60], cb[:60])
            struct_excess = float(nl_beh[L, h] - nl_base)
            heads.append({"layer": L, "head": h, "diag_score": diag_score, "row_peak": row_peak,
                          "broadcast": broadcast, "pos_excess": pos_excess, "pos_leg": pos_leg,
                          "abs_sink_beh": abs_sink_beh, "abs_sink_weight": abs_sink_weight,
                          "abs_leg": abs_leg, "struct_newline_excess": struct_excess})

    # ---- content-structure + dominant-mode bucketing ----
    zds = _z([x["diag_score"] for x in heads]); zrp = _z([x["row_peak"] for x in heads])
    zbc = _z([x["broadcast"] for x in heads])
    for i, x in enumerate(heads):
        x["content_structure"] = float(max(zds[i], zrp[i], zbc[i]))
    # z-score each modality and take the argmax as the dominant addressing mode
    zc = _z([x["content_structure"] for x in heads]); zr = _z([x["pos_excess"] for x in heads])
    za = _z([x["abs_sink_beh"] for x in heads]); zs = _z([x["struct_newline_excess"] for x in heads])
    modes = ["content", "relative-pos", "absolute-sink", "structural-newline"]
    Zstack = np.stack([zc, zr, za, zs], 1)
    for i, x in enumerate(heads):
        j = int(np.argmax(Zstack[i]))
        x["dominant_mode"] = modes[j] if Zstack[i, j] > 0.5 else "none"
        x["mode_scores"] = {m: float(Zstack[i, k]) for k, m in enumerate(modes)}
    dom = Counter(x["dominant_mode"] for x in heads)

    # of the original 'neither' heads (content-flat AND relative-flat), where do they go now?
    neither = [x for x in heads if x["content_structure"] <= 1.0 and x["pos_excess"] <= 0.10]
    neither_modes = Counter(x["dominant_mode"] for x in neither)
    abs_legible = [x for x in heads if x["dominant_mode"] == "absolute-sink"]
    abs_predmatch = float(np.mean([x["abs_sink_weight"] > 0.5 for x in abs_legible])) if abs_legible else 0.0

    out = {"experiment": "absolute/structural-position channel", "model": args.pretrained,
           "n_heads": len(heads), "newline_base_rate": nl_base, "dominant_modes": dict(dom),
           "neither_set_size": len(neither), "neither_resolved_modes": dict(neither_modes),
           "abs_sink_weight_predicts_beh_frac": abs_predmatch, "heads": heads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\ntop ABSOLUTE-SINK heads (attention to key position 0):")
    print(f"{'L.h':>5} {'sink_beh':>8} {'sink_wt_z':>9} {'abs_leg':>7} {'content':>7} {'pos_exc':>7}")
    for x in sorted(heads, key=lambda r: -r["abs_sink_beh"])[:10]:
        print(f"{x['layer']:>2}.{x['head']:<2} {x['abs_sink_beh']:>8.3f} {x['abs_sink_weight']:>9.2f} "
              f"{x['abs_leg']:>7.2f} {x['content_structure']:>7.2f} {x['pos_excess']:>7.3f}")
    print(f"\ntop STRUCTURAL-NEWLINE heads (attn to newline keys over base rate {nl_base:.3f}):")
    print(f"{'L.h':>5} {'nl_excess':>9} {'content':>7} {'pos_exc':>7} {'sink_beh':>8}")
    for x in sorted(heads, key=lambda r: -r["struct_newline_excess"])[:8]:
        print(f"{x['layer']:>2}.{x['head']:<2} {x['struct_newline_excess']:>9.3f} {x['content_structure']:>7.2f} "
              f"{x['pos_excess']:>7.3f} {x['abs_sink_beh']:>8.3f}")

    print(f"\n[dominant addressing mode over all {len(heads)} heads] {dict(dom)}")
    print(f"[the {len(neither)} content+relative 'neither' heads now resolve to] {dict(neither_modes)}")
    print(f"  absolute-sink heads: weight column-score predicts the sink for {abs_predmatch:.0%}")
    resolved = neither_modes.get("absolute-sink", 0) + neither_modes.get("structural-newline", 0)
    frac = resolved / max(len(neither), 1)
    print(f"\n[verdict] absolute-sink + structural-newline reclaim {resolved}/{len(neither)} ({frac:.0%}) of the "
          f"previously-unexplained heads -> "
          f"{'the THIRD register is real: most neither heads address by ABSOLUTE/STRUCTURAL position' if frac > 0.5 else 'absolute/structural only partly explains the neither set'}")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axB, axC) = plt.subplots(1, 2, figsize=(12.5, 5.0))
        order = ["content", "relative-pos", "absolute-sink", "structural-newline", "none"]
        cols = {"content": "#1f77b4", "relative-pos": "#ff7f0e", "absolute-sink": "#9467bd",
                "structural-newline": "#8c564b", "none": "#cccccc"}
        axB.bar(range(len(order)), [dom.get(m, 0) for m in order], color=[cols[m] for m in order],
                edgecolor="k")
        axB.set_xticks(range(len(order))); axB.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
        axB.set_ylabel("# heads"); axB.set_title("dominant addressing mode (144 heads)", fontsize=10)
        for i, m in enumerate(order):
            axB.text(i, dom.get(m, 0) + 0.5, str(dom.get(m, 0)), ha="center", fontsize=9)
        # colbeh[s] profiles: top sink heads spike at s=0; a relative head is flat
        sinks = sorted(heads, key=lambda r: -r["abs_sink_beh"])[:3]
        rel = sorted(heads, key=lambda r: -r["pos_excess"])[0]
        ss = np.arange(C)
        for x in sinks:
            axC.plot(ss, colbeh[x["layer"], x["head"]], lw=1.3,
                     label=f"sink {x['layer']}.{x['head']}")
        axC.plot(ss, colbeh[rel["layer"], rel["head"]], "k--", lw=1.2,
                 label=f"relative {rel['layer']}.{rel['head']}")
        axC.set_xlabel("absolute key position s"); axC.set_ylabel("mean attention received from later queries")
        axC.set_title("absolute channel: sink heads peak at position 0", fontsize=10)
        axC.legend(fontsize=8)
        fig.suptitle("Third addressing register: absolute position (sink @ s=0) + structural (newline anchors)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130)
        print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
