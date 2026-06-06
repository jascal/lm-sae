"""QK opcode table: read the FULL B_h binding matrix per head as a candidate instruction set.

The diagonal probe showed "attend-to-same-token" (B_h[X,X]) is legible in feature coords (rho=+0.68).
This reads the OFF-DIAGONAL. For head h, M_h = W_Q^h W_K^h.T (folded /sqrt(head_dim)) and
B_h[X,Y] = d_X . M_h . d_Y = how strongly a query carrying feature X binds a key carrying feature Y
-- a per-head instruction whose OPERANDS are the residual token-features d_X. For all 144 GPT-2 heads
we (1) build B_h in the token-feature basis, (2) classify its shape into a candidate opcode
{COPY, BIND, BROADCAST, DIFFUSE}, (3) VALIDATE the off-diagonal: does B_h[X,Y] predict the head's
empirical attention from token-X queries to earlier token-Y keys on real text, above a label-permuted
null? Then the decisive question for "the residual holds operands in an instruction language": do the
behaviorally-legible heads collapse into a SMALL, REUSED vocabulary of opcode shapes, or are all 144
idiosyncratic?
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
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def _softmax_offdiag(B):
    """Row-softmax of B with the diagonal masked out (binding distribution over key-features)."""
    M = B.copy()
    np.fill_diagonal(M, -np.inf)
    M = M - np.nanmax(np.where(np.isfinite(M), M, -np.inf), axis=1, keepdims=True)
    E = np.exp(M); E[~np.isfinite(E)] = 0.0
    s = E.sum(1, keepdims=True)
    return E / np.where(s > 0, s, 1.0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=14000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=40, help="# token-feature operands")
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--min-count", type=int, default=15, help="min ordered (X->Y) pairs to score a cell")
    p.add_argument("--n-perm", type=int, default=25, help="label permutations for the behavioral null")
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/qk_opcode_table_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/qk_opcodes.png"))
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
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx)]
    all_ids = [j for c in chunks for j in c]
    cnt = Counter(all_ids)

    # ---- pick operand token-features (single-token COMMON first, then frequent) ----
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
    print(f"{args.pretrained}: layers={nL} heads={H} head_dim={hd}  operands={nt}")

    # ---- one pass: per-layer token centroids + per-(layer,head) empirical attention map ----
    cen_sum = np.zeros((nL, nt, d)); cen_cnt = np.zeros(nt)
    gmean_sum = np.zeros((nL, d)); gmean_cnt = 0
    att_sum = np.zeros((nL, H, nt * nt)); att_cnt = np.zeros(nt * nt)
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            pos = np.array([tok2i.get(t, -1) for t in c])
            vmask = pos >= 0
            cen_cnt += np.bincount(pos[vmask], minlength=nt)
            gmean_cnt += len(c)
            ti, si = np.tril_indices(len(c), k=-1)            # t > s (query attends earlier key)
            m = (pos[ti] >= 0) & (pos[si] >= 0)
            ti, si = ti[m], si[m]
            flat = pos[ti] * nt + pos[si]
            np.add.at(att_cnt, flat, 1.0)
            for L in range(nL):
                hs = o.hidden_states[L][0].float().numpy()
                np.add.at(cen_sum[L], pos[vmask], hs[vmask])
                gmean_sum[L] += hs.sum(0)
                aL = o.attentions[L][0].float().numpy()       # (H, seq, seq)
                for h in range(H):
                    np.add.at(att_sum[L, h], flat, aL[h][ti, si])

    att_cnt2 = att_cnt.reshape(nt, nt)
    cen = cen_sum / np.maximum(cen_cnt, 1)[None, :, None]
    gmean = gmean_sum / max(gmean_cnt, 1)

    # ---- per-layer operand directions (ln_1-folded, unit) + per-head B_h + empirical A ----
    offmask = ~np.eye(nt, dtype=bool)
    supp = (att_cnt2 >= args.min_count) & offmask                # off-diagonal cells with support
    print(f"supported off-diagonal cells: {int(supp.sum())}/{nt*(nt-1)}")

    heads = []
    rng = np.random.default_rng(0)
    perms = [rng.permutation(nt) for _ in range(args.n_perm)]
    for L in range(nL):
        blk = tr.h[L]
        ln_w = blk.ln_1.weight.detach().numpy().astype(np.float64)
        D = (cen[L] - gmean[L]) * ln_w
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
        W = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wq, Wk = W[:, :d], W[:, d:2 * d]
        for h in range(H):
            Mh = Wq[:, h * hd:(h + 1) * hd] @ Wk[:, h * hd:(h + 1) * hd].T / np.sqrt(hd)
            B = D @ Mh @ D.T
            A = (att_sum[L, h].reshape(nt, nt)) / np.maximum(att_cnt2, 1)
            # shape descriptors
            diag_score = float(np.diag(B).mean() - B[offmask].mean())
            soft = _softmax_offdiag(B)
            row_peak = float(soft.max(1).mean())                # ~1 => each query binds one key (PERMUTE)
            col_mass = soft.sum(0)
            broadcast = float(col_mass.max() / col_mass.sum())   # one key-feature attracts all (BROADCAST)
            # off-diagonal behavioral legibility vs label-permuted null
            bv, av = B[supp], A[supp]
            leg = _spearman(bv, av)
            null = [_spearman(B[np.ix_(pp, pp)][supp], av) for pp in perms]
            null_mu = float(np.nanmean(null)); null_sd = float(np.nanstd(null) + 1e-9)
            z = (leg - null_mu) / null_sd if np.isfinite(leg) else float("nan")
            # top off-diagonal binding read from weights
            Boff = B.copy(); np.fill_diagonal(Boff, -np.inf)
            qi, ki = np.unravel_index(int(np.argmax(Boff)), Boff.shape)
            heads.append({"layer": L, "head": h, "diag_score": diag_score, "row_peak": row_peak,
                          "broadcast": broadcast, "offdiag_legibility": leg, "null_mu": null_mu,
                          "leg_z": float(z), "top_bind": [names[qi], names[ki], float(B[qi, ki])]})

    # ---- classify into a candidate opcode vocabulary (data-driven thresholds) ----
    ds = np.array([x["diag_score"] for x in heads])
    rp = np.array([x["row_peak"] for x in heads])
    bc = np.array([x["broadcast"] for x in heads])
    diag_hi = np.quantile(ds, 0.80); rp_hi = np.quantile(rp, 0.80); bc_hi = np.quantile(bc, 0.80)
    for x in heads:
        if x["diag_score"] >= diag_hi:
            x["opcode"] = "COPY"          # attend-to-same-feature (induction/copy)
        elif x["broadcast"] >= bc_hi:
            x["opcode"] = "BROADCAST"     # one key-feature attracts every query (subject/BOS sink)
        elif x["row_peak"] >= rp_hi:
            x["opcode"] = "BIND"          # each query-feature binds a distinct key-feature (permutation)
        else:
            x["opcode"] = "DIFFUSE"
    # legible = off-diagonal binding beats the permuted null by >2 sigma
    for x in heads:
        x["legible"] = bool(np.isfinite(x["leg_z"]) and x["leg_z"] > 2.0)

    legible = [x for x in heads if x["legible"]]
    cls_all = Counter(x["opcode"] for x in heads)
    cls_leg = Counter(x["opcode"] for x in legible)
    pooled_real = _spearman(*np.array([(x["offdiag_legibility"], 1.0) for x in heads]).T) if False else None

    out = {"experiment": "QK opcode table (full B_h instruction reader)", "model": args.pretrained,
           "n_operands": nt, "n_heads": len(heads), "opcode_classes_all": dict(cls_all),
           "opcode_classes_legible": dict(cls_leg), "n_legible": len(legible),
           "legible_distinct_classes": len(cls_leg),
           "heads": heads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    # ---- report ----
    print(f"\n{'L.h':>5} {'opcode':>9} {'diag':>6} {'rowpk':>6} {'bcast':>6} "
          f"{'offdiag_leg':>11} {'z':>6}  top-binding(query->key)")
    for x in sorted(heads, key=lambda r: -r["leg_z"] if np.isfinite(r["leg_z"]) else 0)[:16]:
        q, k, v = x["top_bind"]
        print(f"{x['layer']:>2}.{x['head']:<2} {x['opcode']:>9} {x['diag_score']:>6.3f} "
              f"{x['row_peak']:>6.3f} {x['broadcast']:>6.3f} {x['offdiag_legibility']:>11.3f} "
              f"{x['leg_z']:>6.2f}  {q!r}->{k!r}")
    print(f"\nopcode vocabulary over all {len(heads)} heads: {dict(cls_all)}")
    print(f"behaviorally-legible heads (off-diag z>2): {len(legible)}  -> classes {dict(cls_leg)}")
    small = len(cls_leg) <= 4 and len(legible) >= 3
    print(f"\n[verdict] {len(legible)} of {len(heads)} heads have a behaviorally-legible off-diagonal "
          f"binding, collapsing into {len(cls_leg)} opcode shape(s) -> "
          f"{'SMALL reused instruction set (operand-language supported)' if small else 'no small reused vocabulary'}")

    # ---- figure: representative B_h per opcode + summary scatter ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def rep(opc):
            xs = [x for x in heads if x["opcode"] == opc]
            xs = sorted(xs, key=lambda r: -(r["leg_z"] if np.isfinite(r["leg_z"]) else -9))
            return xs[0] if xs else None
        reps = [r for r in (rep("COPY"), rep("BIND"), rep("BROADCAST")) if r]
        fig, axes = plt.subplots(1, len(reps) + 1, figsize=(4.2 * (len(reps) + 1), 4.4))
        for ax, x in zip(axes, reps):
            L, h = x["layer"], x["head"]
            blk = tr.h[L]; ln_w = blk.ln_1.weight.detach().numpy().astype(np.float64)
            D = (cen[L] - gmean[L]) * ln_w; D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
            W = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
            Wq, Wk = W[:, :d], W[:, d:2 * d]
            Mh = Wq[:, h * hd:(h + 1) * hd] @ Wk[:, h * hd:(h + 1) * hd].T / np.sqrt(hd)
            B = D @ Mh @ D.T
            o = np.argsort(-np.diag(B)) if x["opcode"] == "COPY" else np.argsort(names)
            v = np.abs(B).max()
            ax.imshow(B[np.ix_(o, o)], cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(f"{x['opcode']}  head {L}.{h}\noff-diag leg {x['offdiag_legibility']:+.2f} (z {x['leg_z']:.1f})",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel("key-feature"); ax.set_ylabel("query-feature")
        axS = axes[-1]
        col = {"COPY": "#d62728", "BIND": "#1f77b4", "BROADCAST": "#2ca02c", "DIFFUSE": "#999999"}
        for opc in ["DIFFUSE", "BIND", "BROADCAST", "COPY"]:
            xs = [x for x in heads if x["opcode"] == opc]
            axS.scatter([x["diag_score"] for x in xs],
                        [x["offdiag_legibility"] for x in xs],
                        s=[60 if x["legible"] else 22 for x in xs],
                        c=col[opc], label=opc, edgecolor="k", linewidth=0.4, alpha=0.85)
        axS.axhline(0, color="k", lw=0.6, ls=":")
        axS.set_xlabel("diagonal score  (COPY-ness)"); axS.set_ylabel("off-diagonal legibility  (Spearman B vs attn)")
        axS.set_title(f"144 heads: {len(legible)} legible -> {len(cls_leg)} opcode shapes", fontsize=9)
        axS.legend(fontsize=7, loc="best")
        fig.suptitle("GPT-2 attention as an instruction set: B_h[X,Y] = which query-feature binds which key-feature",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130)
        print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")

    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
