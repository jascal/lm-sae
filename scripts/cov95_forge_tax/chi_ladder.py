"""(c) The χ-ladder: do the weight-legible macros sit in the high-χ (entangled) band?

Closes the loop back to the forge tax. The whole arc said: knowledge factors into low-χ residual
features (cov95, SAE-readable assertions); composition does NOT factor in the residual (high-χ, the
entangled core) but IS legible in WEIGHT/attention-macro coordinates (QK opcodes, the induction macro).
This measures both axes on the SAME ladder of functions, ordered by composition depth:
  rung-0  token identity        (unary assertion)
  rung-0b lexical (cap / punct) (unary)
  rung-1  in-context repeat     (relational, partly compiled)
  rung-2  induction-predictable (composed: next tok = what followed the current token's last occurrence)
For each function: (A) RESIDUAL-χ = 1 - best single SAE-latent AUC (the cov95 forge-tax meter, the
residual basis); (B) MACRO legibility = best AUC from the attention-circuit features (induction-head
attention-to-induction-keys, duplicate-head attention-to-same-token = the weight-legible macros' outputs).
PREDICTION (knowledge=low-χ residual, computation=high-χ macro): residual-AUC FALLS up the ladder while
macro-AUC RISES -> the two CROSS. The forge tax is the residual projection (cov95 dies on composition);
the macro-reading is the orthogonal weight projection that recovers it. GPT-2.
"""
from __future__ import annotations

import argparse
import json
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from build_lm_bundle import COMMON  # noqa: E402
from forge_cov_mechanism import _column_ranks, _encode, _train_topk_sae  # noqa: E402
from preserve_hybrid_tiny import _auc_matrix  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=14000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--sae-steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--n-circuit", type=int, default=4, help="# induction/duplicate heads to pool as macros")
    p.add_argument("--output", type=Path, default=Path("runs/cov95_forge_tax/chi_ladder_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/chi_ladder.png"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer
    H = model.config.n_head
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    all_ids = [j for c in chunks for j in c]
    N = len(all_ids); cnt = Counter(all_ids)
    toks_str = tok.convert_ids_to_tokens(all_ids)
    punct = set(string.punctuation)

    def is_cap(s):
        s = s.replace("Ġ", "")
        return 1 if (s[:1].isupper()) else 0

    def is_punct(s):
        s = s.replace("Ġ", "")
        return 1 if (s and all(ch in punct for ch in s)) else 0

    # ---- forward: residual X at layer L + per-head induction/duplicate macro signals ----
    nL = model.config.n_layer
    X = np.zeros((N, model.config.n_embd), np.float32)
    finduct = np.zeros((N, nL * H), np.float32); fdup = np.zeros((N, nL * H), np.float32)
    is_rep = np.zeros(N, np.uint8); ind_pred = np.zeros(N, np.uint8)
    off = 0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            Lc = len(c); ca = np.array(c)
            X[off:off + Lc] = o.hidden_states[args.layer][0].float().numpy()
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]; qi = np.arange(Lc)
            IndMask = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
            DupMask = (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])
            seen = set()
            for i, tid in enumerate(c):
                if tid in seen:
                    is_rep[off + i] = 1
                seen.add(tid)
            for t in range(2, Lc):
                ps = [pp for pp in range(t - 1) if c[pp] == c[t - 1]]
                if ps and c[ps[-1] + 1] == c[t]:
                    ind_pred[off + t] = 1
            for L in range(nL):
                aL = o.attentions[L][0].float().numpy()
                base = L * H
                finduct[off:off + Lc, base:base + H] = (aL * IndMask[None]).sum(2).T
                fdup[off:off + Lc, base:base + H] = (aL * DupMask[None]).sum(2).T
            off += Lc
    print(f"{args.pretrained} L{args.layer}: X={X.shape}  repeat-rate {is_rep.mean():.2f}  induction-pred-rate {ind_pred.mean():.3f}")

    # pick circuit heads (macros): induction heads by induction-key attention, duplicate heads by same-tok
    ind_excess = finduct.mean(0)                              # mean induction-key attention per head
    dup_excess = fdup.mean(0)
    ind_heads = np.argsort(-ind_excess)[:args.n_circuit]
    dup_heads = np.argsort(-dup_excess)[:args.n_circuit]
    macros = np.concatenate([finduct[:, ind_heads], fdup[:, dup_heads]], 1)   # (N, 2*n_circuit)
    print(f"macro heads: induction {[ (int(h)//H, int(h)%H) for h in ind_heads]}  duplicate {[ (int(h)//H, int(h)%H) for h in dup_heads]}")

    # ---- labels grouped by rung ----
    common1 = [t for t, _ in cnt.most_common(60)
               if len(tok(tok.convert_ids_to_tokens(t).replace('Ġ', ' '), add_special_tokens=False)['input_ids']) >= 1][:10]
    label_groups = {}
    cols = []
    for t in common1:
        v = np.array([1 if j == t else 0 for j in all_ids], np.uint8)
        if args.min_pos <= v.sum() <= N - args.min_pos:
            cols.append(("rung0_token", v))
    cols.append(("rung0b_lexical", np.array([is_cap(s) for s in toks_str], np.uint8)))
    cols.append(("rung0b_lexical", np.array([is_punct(s) for s in toks_str], np.uint8)))
    cols.append(("rung1_repeat", is_rep.copy()))
    cols.append(("rung2_induction", ind_pred.copy()))

    # ---- residual SAE (the χ-meter) ----
    params = _train_topk_sae(X, args.width, args.k, args.sae_steps, 1e-3, 0)
    z = _encode(X, params, args.k)
    zr = _column_ranks(z); mr = _column_ranks(macros)

    rungs = {}
    for name, v in cols:
        if not (args.min_pos <= v.sum() <= N - args.min_pos):
            continue
        Ares, okr = _auc_matrix(zr, v.reshape(-1, 1))
        Amac, okm = _auc_matrix(mr, v.reshape(-1, 1))
        res_auc = float(Ares[0].max()) if okr[0] else float("nan")
        mac_auc = float(Amac[0].max()) if okm[0] else float("nan")
        rungs.setdefault(name, {"res": [], "mac": [], "npos": []})
        rungs[name]["res"].append(res_auc); rungs[name]["mac"].append(mac_auc); rungs[name]["npos"].append(int(v.sum()))

    order = ["rung0_token", "rung0b_lexical", "rung1_repeat", "rung2_induction"]
    summary = {}
    for name in order:
        if name not in rungs:
            continue
        r = rungs[name]
        res = float(np.nanmean(r["res"])); mac = float(np.nanmean(r["mac"]))
        summary[name] = {"residual_bestAUC": res, "macro_bestAUC": mac, "residual_chi": 1 - res,
                         "residual_cov95": float(np.mean(np.array(r["res"]) >= 0.95)), "n_labels": len(r["res"])}

    out = {"experiment": "chi-ladder (residual-χ vs macro-legibility up the composition ladder)",
           "model": args.pretrained, "layer": args.layer,
           "macro_heads": {"induction": [[int(h) // H, int(h) % H] for h in ind_heads],
                           "duplicate": [[int(h) // H, int(h) % H] for h in dup_heads]},
           "ladder": summary}
    res_seq = [summary[n]["residual_bestAUC"] for n in order if n in summary]
    mac_seq = [summary[n]["macro_bestAUC"] for n in order if n in summary]
    out["residual_AUC_falls"] = res_seq[0] - res_seq[-1]
    out["macro_AUC_rises"] = mac_seq[-1] - mac_seq[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\n{'rung':>18} {'res bestAUC':>11} {'res cov95':>9} {'res χ=1-AUC':>11} {'MACRO bestAUC':>13}")
    for name in order:
        if name not in summary:
            continue
        s = summary[name]
        print(f"{name:>18} {s['residual_bestAUC']:>11.3f} {s['residual_cov95']:>9.2f} {s['residual_chi']:>11.3f} {s['macro_bestAUC']:>13.3f}")
    cross = res_seq[0] > mac_seq[0] and res_seq[-1] < mac_seq[-1]
    print(f"\n[ladder] residual-AUC falls {res_seq[0]:.2f}→{res_seq[-1]:.2f} (Δ{out['residual_AUC_falls']:+.2f}); "
          f"macro-AUC rises {mac_seq[0]:.2f}→{mac_seq[-1]:.2f} (Δ{out['macro_AUC_rises']:+.2f})")
    print(f"[verdict] {'THE LINES CROSS: knowledge (low rung) is read by LOW-χ residual features, computation (induction) is read by HIGH-χ attention MACROS. residual-χ and macro-legibility are ORTHOGONAL axes -> the forge tax is the residual projection (cov95 dies on composition); macro-reading is the weight projection that recovers it. Loop closed.' if cross else 'no clean crossover — residual and macro track together here'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = np.arange(len(res_seq))
        labs = [n.replace("rung", "r").replace("_", "\n") for n in order if n in summary]
        fig, (axL, axB) = plt.subplots(1, 2, figsize=(12.6, 5.0))
        axL.plot(xs, res_seq, "o-", color="#1f77b4", lw=2, label="residual SAE best AUC  (low-χ, assertions)")
        axL.plot(xs, mac_seq, "s-", color="#d62728", lw=2, label="attention-macro best AUC  (high-χ, rules)")
        axL.axhline(0.5, color="k", lw=0.5, ls=":")
        axL.set_xticks(xs); axL.set_xticklabels(labs, fontsize=8)
        axL.set_ylabel("best single-feature AUC for the function"); axL.set_ylim(0.45, 1.02)
        axL.set_title("the χ-ladder: residual-features fall, attention-macros rise", fontsize=10)
        axL.legend(fontsize=8, loc="center left")
        chi = [1 - r for r in res_seq]
        axB.bar(xs, chi, color="#9467bd", edgecolor="k", width=0.6)
        for i, (cc, mc) in enumerate(zip(chi, mac_seq)):
            axB.text(i, cc + 0.01, f"χ {cc:.2f}\nmac {mc:.2f}", ha="center", fontsize=7)
        axB.set_xticks(xs); axB.set_xticklabels(labs, fontsize=8)
        axB.set_ylabel("residual-χ  (1 − best residual AUC)")
        axB.set_title("residual-χ climbs the composition ladder\n(macros stay legible — orthogonal axis)", fontsize=10)
        fig.suptitle("Closing the loop: knowledge=low-χ residual features, computation=high-χ attention macros", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
