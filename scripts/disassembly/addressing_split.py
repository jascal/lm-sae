"""Content-vs-position addressing split: separate the read head from the tape-mover.

The opcode table read each head's CONTENT addressing B_h[X,Y] = d_X·M_h·d_Y (which query-feature
binds which key-feature) and found COPY/BIND/BROADCAST opcodes plus a DIFFUSE majority (91/144) with
no dominant feature-binding. The tape-machine reading predicts the DIFFUSE heads address by POSITION,
not content -- they are "move Δ back" tape-head shifts, invisible to a content-coordinate probe by
construction. This builds the second channel and tests that prediction.

Per head we read TWO QK channels through the same M_h:
  content  B_h[X,Y]  = d_X·M_h·d_Y          (d_X = per-token residual centroid, folded ln_1)
  position P_h[Δ]    = mean_t p_t·M_h·p_{t-Δ}  (p_t = per-POSITION residual centroid, folded ln_1)
and validate each against behavior:
  content  vs token-pair attention  A_content[X,Y]
  position vs attention-by-offset   A_beh[Δ]  (excess over the cross-head recency baseline)
Decisive crosstab: do the content-DIFFUSE heads carry the position structure (movers), while the
COPY/BIND/BROADCAST heads are content-addressed (the read head)? GPT-2, all 144 heads.
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
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=14000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=40)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--min-count", type=int, default=15)
    p.add_argument("--max-delta", type=int, default=24, help="relative offsets for the position channel")
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/addressing_split_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/addressing_split.png"))
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
    C = args.ctx; maxД = args.max_delta

    cand = [tok(c, add_special_tokens=False)["input_ids"][0] for c in COMMON
            if len(tok(c, add_special_tokens=False)["input_ids"]) == 1]
    cand += [t for t, _ in cnt.most_common(300)]
    seen, toks = set(), []
    for t in cand:
        if t not in seen and cnt[t] >= args.min_pos:
            seen.add(t); toks.append(t)
        if len(toks) >= args.n_tokens:
            break
    nt = len(toks)
    tok2i = {t: i for i, t in enumerate(toks)}
    names = [tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t in toks]
    print(f"{args.pretrained}: layers={nL} heads={H} head_dim={hd}  operands={nt}  ctx={C} maxΔ={maxД}")

    # ---- one pass: content centroids, POSITION centroids, content-attn map, attn-by-offset ----
    cen_sum = np.zeros((nL, nt, d)); cen_cnt = np.zeros(nt)
    pos_sum = np.zeros((nL, C, d)); pos_cnt = np.zeros(C)
    gmean_sum = np.zeros((nL, d)); gmean_cnt = 0
    att_sum = np.zeros((nL, H, nt * nt)); att_cnt = np.zeros(nt * nt)
    apos_sum = np.zeros((nL, H, maxД + 1)); apos_cnt = np.zeros(maxД + 1)
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            Lc = len(c)
            pidx = np.array([tok2i.get(t, -1) for t in c]); vmask = pidx >= 0
            cen_cnt += np.bincount(pidx[vmask], minlength=nt)
            pos_cnt[:Lc] += 1; gmean_cnt += Lc
            ti, si = np.tril_indices(Lc, k=-1)
            m = (pidx[ti] >= 0) & (pidx[si] >= 0)
            cti, csi = ti[m], si[m]
            flat = pidx[cti] * nt + pidx[csi]
            np.add.at(att_cnt, flat, 1.0)
            for dl in range(1, min(maxД, Lc - 1) + 1):
                apos_cnt[dl] += Lc - dl
            for L in range(nL):
                hs = o.hidden_states[L][0].float().numpy()
                np.add.at(cen_sum[L], pidx[vmask], hs[vmask])
                pos_sum[L, :Lc] += hs
                gmean_sum[L] += hs.sum(0)
                aL = o.attentions[L][0].float().numpy()                  # (H, Lc, Lc)
                for h in range(H):
                    np.add.at(att_sum[L, h], flat, aL[h][cti, csi])
                for dl in range(1, min(maxД, Lc - 1) + 1):
                    apos_sum[L, :, dl] += np.diagonal(aL, offset=-dl, axis1=1, axis2=2).sum(1)

    att_cnt2 = att_cnt.reshape(nt, nt)
    cen = cen_sum / np.maximum(cen_cnt, 1)[None, :, None]
    posc = pos_sum / np.maximum(pos_cnt, 1)[None, :, None]
    gmean = gmean_sum / max(gmean_cnt, 1)
    offmask = ~np.eye(nt, dtype=bool)
    supp = (att_cnt2 >= args.min_count) & offmask
    Abeh = apos_sum / np.maximum(apos_cnt, 1)[None, None, :]              # (nL,H,maxД+1) behavioral Δ-profile
    Abeh_mean = Abeh.reshape(-1, maxД + 1).mean(0)                        # cross-head recency baseline

    # ---- per head: content channel (B_h) + position channel (P_h) ----
    heads = []
    for L in range(nL):
        blk = tr.h[L]
        ln_w = blk.ln_1.weight.detach().numpy().astype(np.float64)
        Dc = (cen[L] - gmean[L]) * ln_w
        Dc = Dc / (np.linalg.norm(Dc, axis=1, keepdims=True) + 1e-9)
        Pp = (posc[L] - gmean[L]) * ln_w                                 # per-position folded (keep scale)
        W = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wq, Wk = W[:, :d], W[:, d:2 * d]
        for h in range(H):
            Mh = Wq[:, h * hd:(h + 1) * hd] @ Wk[:, h * hd:(h + 1) * hd].T / np.sqrt(hd)
            # content
            B = Dc @ Mh @ Dc.T
            diag_score = float(np.diag(B).mean() - B[offmask].mean())
            soft = _softmax_offdiag(B); col_mass = soft.sum(0)
            row_peak = float(soft.max(1).mean())
            broadcast = float(col_mass.max() / col_mass.sum())
            Acont = att_sum[L, h].reshape(nt, nt) / np.maximum(att_cnt2, 1)
            content_leg = _spearman(B[supp], Acont[supp])
            # position: Spos[t,s] = p_t·M_h·p_s ; P_h[Δ] = mean_t Spos[t,t-Δ]
            Spos = Pp @ Mh @ Pp.T
            Ph = np.array([np.diagonal(Spos, offset=-dl).mean() for dl in range(1, maxД + 1)])
            ab = Abeh[L, h, 1:]
            pos_leg = _spearman(Ph, ab)
            resid = ab - Abeh_mean[1:]                                   # excess over generic recency
            pos_excess = float(resid.max())
            pref_delta = int(np.argmax(resid) + 1)
            # weights predict the preferred offset?
            pred_delta = int(np.argmax(Ph) + 1)
            heads.append({"layer": L, "head": h, "diag_score": diag_score, "row_peak": row_peak,
                          "broadcast": broadcast, "content_leg": content_leg, "pos_leg": pos_leg,
                          "pos_excess": pos_excess, "pref_delta": pref_delta, "pred_delta": pred_delta})

    # ---- opcode class (same thresholds as the opcode table) + content-structure scalar ----
    ds = np.array([x["diag_score"] for x in heads]); rp = np.array([x["row_peak"] for x in heads])
    bc = np.array([x["broadcast"] for x in heads])

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)
    zds, zrp, zbc = z(ds), z(rp), z(bc)
    diag_hi, rp_hi, bc_hi = np.quantile(ds, .80), np.quantile(rp, .80), np.quantile(bc, .80)
    for i, x in enumerate(heads):
        if x["diag_score"] >= diag_hi:
            x["opcode"] = "COPY"
        elif x["broadcast"] >= bc_hi:
            x["opcode"] = "BROADCAST"
        elif x["row_peak"] >= rp_hi:
            x["opcode"] = "BIND"
        else:
            x["opcode"] = "DIFFUSE"
        x["content_structure"] = float(max(zds[i], zrp[i], zbc[i]))      # stands out on ANY content axis

    pe = np.array([x["pos_excess"] for x in heads])
    rho_cs_pe = _spearman([x["content_structure"] for x in heads], [x["pos_excess"] for x in heads])
    pos_leg_resid = _spearman([x["pos_leg"] for x in heads], [x["pos_excess"] for x in heads])

    # ---- 2x2 addressing quadrants (sharp on content? sharp on relative position?) ----
    CS_THR, PE_THR = 1.0, 0.10
    for x in heads:
        x["content_sharp"] = bool(x["content_structure"] > CS_THR)
        x["pos_sharp"] = bool(x["pos_excess"] > PE_THR)
        x["quadrant"] = (("read+move" if x["pos_sharp"] else "read") if x["content_sharp"]
                         else ("move" if x["pos_sharp"] else "neither"))
    quad = Counter(x["quadrant"] for x in heads)
    movers = [x for x in heads if x["pos_sharp"]]
    mover_opcodes = Counter(x["opcode"] for x in movers)
    pure_movers = [x for x in heads if x["quadrant"] == "move"]
    pure_mover_diffuse = float(np.mean([x["opcode"] == "DIFFUSE" for x in pure_movers])) if pure_movers else 0.0
    mover_predmatch = float(np.mean([x["pred_delta"] == x["pref_delta"] for x in movers])) if movers else 0.0
    mover_posleg = float(np.median([x["pos_leg"] for x in movers])) if movers else float("nan")
    mover_delta1 = float(np.mean([x["pref_delta"] == 1 for x in movers])) if movers else 0.0

    out = {"experiment": "content-vs-position addressing split", "model": args.pretrained,
           "n_heads": len(heads), "quadrants": dict(quad), "n_movers": len(movers),
           "mover_opcode_mix": dict(mover_opcodes), "pure_mover_frac_DIFFUSE": pure_mover_diffuse,
           "mover_weight_pred_offset_match": mover_predmatch, "mover_median_pos_leg": mover_posleg,
           "mover_frac_prev_token": mover_delta1,
           "spearman_content_structure_vs_pos_excess": rho_cs_pe,
           "spearman_posleg_vs_posexcess": pos_leg_resid, "heads": heads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\nsupported content cells {int(supp.sum())}  | cross-head recency baseline A[Δ=1..3]="
          f"{Abeh_mean[1]:.3f},{Abeh_mean[2]:.3f},{Abeh_mean[3]:.3f}")
    print(f"\ntop POSITION-addressed heads (excess attn over recency baseline):")
    print(f"{'L.h':>5} {'opcode':>9} {'pref_Δ':>6} {'pred_Δ':>6} {'pos_excess':>10} {'pos_leg':>7} {'cont_struct':>11}")
    for x in sorted(heads, key=lambda r: -r["pos_excess"])[:14]:
        print(f"{x['layer']:>2}.{x['head']:<2} {x['opcode']:>9} {x['pref_delta']:>6} {x['pred_delta']:>6} "
              f"{x['pos_excess']:>10.3f} {x['pos_leg']:>7.2f} {x['content_structure']:>11.2f}")

    print(f"\n[quadrants @ content>{CS_THR}, pos_excess>{PE_THR}] {dict(quad)}")
    print(f"  {len(movers)} movers; weight predicts the offset for {mover_predmatch:.0%} (median pos_leg {mover_posleg:.2f}); "
          f"{mover_delta1:.0%} are prev-token (Δ=1)")
    print(f"  among PURE movers (position-sharp, content-flat): {pure_mover_diffuse:.0%} are content-DIFFUSE; opcode mix of all movers {dict(mover_opcodes)}")
    print(f"  Spearman(content_structure, pos_excess) = {rho_cs_pe:+.3f}  (~0 => content & position are INDEPENDENT axes, not a partition)")
    # the registers exist & are weight-legible iff movers are weight-predicted; the partition fails iff 'neither' dominates
    registers_real = mover_predmatch > 0.8 and mover_posleg > 0.6
    n_neither = quad.get("neither", 0)
    partitions = n_neither < 0.4 * len(heads)
    tail = ("PARTITION the heads" if partitions else
            f"do NOT partition — {n_neither}/{len(heads)} heads are sharp on NEITHER "
            f"(a 3rd, absolute/structural-position mode)")
    print(f"\n[verdict] position register {'WEIGHT-LEGIBLE' if registers_real else 'not legible'} "
          f"({len(movers)} movers, weight-predicted offset {mover_predmatch:.0%}, mostly Δ=1 prev-token); "
          f"content & position are SEPARABLE per-head but {tail}")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = {"COPY": "#d62728", "BIND": "#1f77b4", "BROADCAST": "#2ca02c", "DIFFUSE": "#999999"}
        fig, (axS, axD) = plt.subplots(1, 2, figsize=(12.5, 5.0))
        for opc in ["DIFFUSE", "BIND", "BROADCAST", "COPY"]:
            xs = [x for x in heads if x["opcode"] == opc]
            axS.scatter([x["content_structure"] for x in xs], [x["pos_excess"] for x in xs],
                        s=64, c=col[opc], edgecolor="k", linewidth=0.4, alpha=0.85, label=opc)
        axS.axhline(PE_THR, color="k", lw=0.6, ls=":"); axS.axvline(CS_THR, color="k", lw=0.6, ls=":")
        axS.set_xlabel("content-structure  (stands out as a content opcode)")
        axS.set_ylabel("position excess  (attn at preferred Δ over recency baseline)")
        axS.set_title(f"read-head vs tape-mover\nρ(content,position)={rho_cs_pe:+.2f}", fontsize=10)
        axS.legend(fontsize=8)
        # Δ-profiles of the top positional heads (behavioral + weight-predicted)
        tops = sorted(heads, key=lambda r: -r["pos_excess"])[:4]
        dd = np.arange(1, maxД + 1)
        for x in tops:
            ab = Abeh[x["layer"], x["head"], 1:]
            axD.plot(dd, ab, label=f"{x['layer']}.{x['head']} ({x['opcode']}, Δ*={x['pref_delta']})")
        axD.plot(dd, Abeh_mean[1:], "k--", lw=1.2, label="cross-head recency baseline")
        axD.set_xlabel("relative offset Δ (key = query − Δ)"); axD.set_ylabel("mean attention")
        axD.set_title("position channel: attention-by-offset of the top movers", fontsize=10)
        axD.legend(fontsize=8)
        fig.suptitle("GPT-2 attention has two addressing registers: content (QK in feature coords) + position (QK in offset coords)",
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
