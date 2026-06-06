"""OV write channel: complete each head into READ (QK) -> WRITE (OV).

The QK channels read ADDRESSING (where a head attends). The instruction isn't complete until we read
what it WRITES. Head h's output-value circuit OV_h = W_V^h W_O^h maps a source residual to the vector
added at the destination. In token-feature coords V_h[Y,Z] = d_Y·OV_h·d_Z = "attending to a source
carrying feature Y writes feature Z to the destination." Two regimes:
  COPY write   V_h diagonal-dominant (Y->Y): the head moves the source feature unchanged. Paired with a
               COPY/induction QK opcode this is the full duplication instruction (match X, copy X).
  COMPUTE write V_h off-diagonal (Y->Z, Z!=Y): the head TRANSFORMS -- reads feature Y, writes a different
               feature Z (a learned unary function / type conversion). These are the interesting ops.
We read V_h for all 144 GPT-2 heads, pair it with the QK opcode, and name each head's top read->write
map. (Behavioral confirmation of the write needs path-patching; here we read it from the weights and use
the canonical copying-head signature: positive OV diagonal = the head preserves token identity.)
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
    p.add_argument("--output", type=Path, default=Path("runs/ov_write_channel_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/ov_write_channel.png"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained).eval()
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
    names = [tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t in toks]
    print(f"{args.pretrained}: layers={nL} heads={H} head_dim={hd}  operands={nt}")

    # ---- one pass: per-layer token centroids (no attention needed; OV/QK are weight-only) ----
    cen_sum = np.zeros((nL, nt, d)); cen_cnt = np.zeros(nt); gmean_sum = np.zeros((nL, d)); gmean_cnt = 0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True)
            pidx = np.array([tok2i.get(t, -1) for t in c]); vmask = pidx >= 0
            cen_cnt += np.bincount(pidx[vmask], minlength=nt); gmean_cnt += len(c)
            for L in range(nL):
                hs = o.hidden_states[L][0].float().numpy()
                np.add.at(cen_sum[L], pidx[vmask], hs[vmask]); gmean_sum[L] += hs.sum(0)
    cen = cen_sum / np.maximum(cen_cnt, 1)[None, :, None]; gmean = gmean_sum / max(gmean_cnt, 1)
    offmask = ~np.eye(nt, dtype=bool)

    heads = []
    for L in range(nL):
        blk = tr.h[L]
        ln_w = blk.ln_1.weight.detach().numpy().astype(np.float64)
        D = (cen[L] - gmean[L]) * ln_w; D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
        Wc = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)     # (d,3d)
        Wq, Wk, Wv = Wc[:, :d], Wc[:, d:2 * d], Wc[:, 2 * d:3 * d]
        Wo = blk.attn.c_proj.weight.detach().numpy().astype(np.float64)     # (d,d)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            Mh = Wq[:, sl] @ Wk[:, sl].T / np.sqrt(hd)
            B = D @ Mh @ D.T
            qk_diag = float(np.diag(B).mean() - B[offmask].mean())
            soft = _softmax_offdiag(B); col_mass = soft.sum(0)
            row_peak = float(soft.max(1).mean()); broadcast = float(col_mass.max() / col_mass.sum())
            # OV write
            OVh = Wv[:, sl] @ Wo[sl, :]                                     # (d,d) source-residual -> dest-residual
            V = D @ OVh @ D.T                                               # V[Y,Z] = write feature Z from source feature Y
            dia = np.diag(V)
            ov_diag = float(dia.mean() - V[offmask].mean())
            ov_self_frac = float(np.mean(np.argmax(V, axis=1) == np.arange(nt)))   # source writes ITSELF
            ov_self_pos = float(np.mean(dia > 0))                          # copying-head signature
            Voff = V.copy(); np.fill_diagonal(Voff, -np.inf)
            yy, zz = np.unravel_index(int(np.argmax(Voff)), Voff.shape)
            # transform strength: how much the top write is NON-self
            trans = float(V[yy, zz] - dia[yy])
            heads.append({"layer": L, "head": h, "qk_diag": qk_diag, "row_peak": row_peak,
                          "broadcast": broadcast, "ov_diag": ov_diag, "ov_self_frac": ov_self_frac,
                          "ov_self_pos": ov_self_pos, "ov_top_map": [names[yy], names[zz], float(V[yy, zz])],
                          "transform_strength": trans})

    # ---- QK opcode (same thresholds) + READ->WRITE pairing ----
    qd = np.array([x["qk_diag"] for x in heads]); rp = np.array([x["row_peak"] for x in heads])
    bc = np.array([x["broadcast"] for x in heads])
    diag_hi, rp_hi, bc_hi = np.quantile(qd, .80), np.quantile(rp, .80), np.quantile(bc, .80)
    for x in heads:
        if x["qk_diag"] >= diag_hi:
            x["qk_opcode"] = "COPY"
        elif x["broadcast"] >= bc_hi:
            x["qk_opcode"] = "BROADCAST"
        elif x["row_peak"] >= rp_hi:
            x["qk_opcode"] = "BIND"
        else:
            x["qk_opcode"] = "DIFFUSE"
    ov_hi = np.quantile([x["ov_diag"] for x in heads], .70)
    for x in heads:
        x["write"] = "COPY" if x["ov_diag"] >= ov_hi else "TRANSFORM"
        x["instruction"] = f"{x['qk_opcode']}->{x['write']}"

    # copy-QK heads that also copy-write = full duplication/induction instruction
    copyqk = [x for x in heads if x["qk_opcode"] == "COPY"]
    copyqk_copywrite = float(np.mean([x["write"] == "COPY" for x in copyqk])) if copyqk else 0.0
    ov_self_pos_all = float(np.mean([x["ov_self_pos"] for x in heads]))
    instr = Counter(x["instruction"] for x in heads)
    rho_qk_ov = float(np.corrcoef([x["qk_diag"] for x in heads], [x["ov_diag"] for x in heads])[0, 1])

    out = {"experiment": "OV write channel (read->write instructions)", "model": args.pretrained,
           "n_heads": len(heads), "instruction_mix": dict(instr),
           "copyQK_fraction_copyWrite": copyqk_copywrite, "mean_ov_self_positive": ov_self_pos_all,
           "corr_qkdiag_ovdiag": rho_qk_ov, "heads": heads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\nfull READ->WRITE instruction mix over {len(heads)} heads: {dict(instr)}")
    print(f"copy-QK heads that ALSO copy-write (full duplication/induction): {copyqk_copywrite:.0%} of {len(copyqk)}")
    print(f"mean fraction of source-features written with POSITIVE diagonal (copying signature): {ov_self_pos_all:.2f}")
    print(f"corr(QK diag, OV diag) = {rho_qk_ov:+.2f}\n")
    print(f"top COMPUTE heads (transformative write, read feature Y -> write a DIFFERENT feature Z):")
    print(f"{'L.h':>5} {'instruction':>16} {'ov_diag':>7} {'transform':>9}  top read->write")
    for x in sorted(heads, key=lambda r: -r["transform_strength"])[:10]:
        y, z, v = x["ov_top_map"]
        print(f"{x['layer']:>2}.{x['head']:<2} {x['instruction']:>16} {x['ov_diag']:>7.2f} "
              f"{x['transform_strength']:>9.2f}  {y!r}->{z!r}")
    print(f"\ntop COPY->COPY heads (duplication: match a feature, write it forward):")
    for x in sorted([h for h in heads if h["instruction"] == "COPY->COPY"], key=lambda r: -r["ov_diag"])[:6]:
        print(f"{x['layer']:>2}.{x['head']:<2}  qk_diag {x['qk_diag']:>5.2f}  ov_diag {x['ov_diag']:>5.2f}  ov_self_frac {x['ov_self_frac']:.2f}")

    print(f"\n[verdict] read->write completed: {instr.get('COPY->COPY',0)} full-duplication heads, "
          f"{sum(v for k,v in instr.items() if k.endswith('TRANSFORM'))} compute/transform heads; "
          f"the OV diagonal is {'mostly positive (copying preserves identity)' if ov_self_pos_all>0.6 else 'mixed'} -> "
          f"each head is a (where-to-read, what-to-write) pair")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axS, axV) = plt.subplots(1, 2, figsize=(12.5, 5.2))
        col = {"COPY": "#d62728", "BIND": "#1f77b4", "BROADCAST": "#2ca02c", "DIFFUSE": "#999999"}
        for opc in ["DIFFUSE", "BIND", "BROADCAST", "COPY"]:
            xs = [x for x in heads if x["qk_opcode"] == opc]
            axS.scatter([x["qk_diag"] for x in xs], [x["ov_diag"] for x in xs], s=55, c=col[opc],
                        edgecolor="k", linewidth=0.4, alpha=0.85, label=f"QK:{opc}")
        axS.axhline(ov_hi, color="k", lw=0.6, ls=":")
        axS.set_xlabel("QK diagonal  (read: match-same-feature)")
        axS.set_ylabel("OV diagonal  (write: copy-same-feature)")
        axS.set_title(f"READ vs WRITE  (corr {rho_qk_ov:+.2f})\nupper-right = full duplication/induction", fontsize=10)
        axS.legend(fontsize=7)
        # a transformative head's V_h heatmap
        tx = sorted(heads, key=lambda r: -r["transform_strength"])[0]
        L, h = tx["layer"], tx["head"]
        blk = tr.h[L]; ln_w = blk.ln_1.weight.detach().numpy().astype(np.float64)
        D = (cen[L] - gmean[L]) * ln_w; D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
        Wc = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wv = Wc[:, 2 * d:3 * d]; Wo = blk.attn.c_proj.weight.detach().numpy().astype(np.float64)
        sl = slice(h * hd, (h + 1) * hd); V = D @ (Wv[:, sl] @ Wo[sl, :]) @ D.T
        oo = np.argsort(names); v = np.abs(V).max()
        im = axV.imshow(V[np.ix_(oo, oo)], cmap="PuOr_r", vmin=-v, vmax=v)
        axV.set_title(f"V_h write map: head {L}.{h} ({tx['instruction']})\ntop {tx['ov_top_map'][0]!r}->{tx['ov_top_map'][1]!r}", fontsize=9)
        axV.set_xlabel("write feature Z"); axV.set_ylabel("read feature Y")
        axV.set_xticks([]); axV.set_yticks([])
        fig.colorbar(im, ax=axV, fraction=0.046)
        fig.suptitle("Completing the instruction: each head = READ (QK) -> WRITE (OV) in feature coords", fontsize=11)
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
