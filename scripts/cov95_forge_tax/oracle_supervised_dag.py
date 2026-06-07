"""Does oracle-feature supervision buy CIRCUIT legibility, or only FEATURE legibility? (the oracle-supervised DAG)

#19/#20 (`host_width_sweep`, `monosemantic_aux`) showed an oracle-feature-recovery aux loss lifts native cov95 —
FEATURE legibility / monosemanticity — for ~free. But the program's thesis is that **knowledge** (residual
features, cov95-factorable) and **computation** (composition, the M2 DAG) are DIFFERENT axes: the forge tax is
exactly cov95 collapsing while the composition stays weight-legible-not-feature-legible. So the sharp question:
does the same supervision also make the COMPOSITION DAG more legible, or is circuit legibility an independent
axis the feature-recovery loss never touches?

Train tiny GPTs from scratch, unsupervised vs oracle-supervised (the #20 `linear` lever), multi-seed. On EACH
trained model measure, side by side:
  - FEATURE legibility:  cov95 (fit a TopK SAE to the residual; best-latent AUC per oracle feature) — the #20 metric.
  - CIRCUIT legibility:  (a) static→dynamic composition agreement — Spearman(weight K-composition, reader-matched
                         ΔTV) over head pairs (the M2 headline); (b) induction recovery — prev-token→induction
                         K-composition vs the causal baseline.
  - POSITIONAL machinery: the prev-token head's key variance, position- vs token-explained (the #26 probe) — does
                         supervision move the absolute-position content the prev-token head reads?

Compare none vs linear (paired by seed): if feature- and circuit-legibility move TOGETHER, supervision buys both
(cleaner features → cleaner composition); if cov95 lifts but circuit/positional metrics don't, they are
INDEPENDENT axes — you cannot supervise circuits into existence with a feature-recovery loss (the thesis).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parents[1] / "common"))
sys.path.insert(0, str(_here.parent))
from forge_cov_mechanism import _best_auc_per_label, _encode, _per_tier, _train_topk_sae  # noqa: E402
from host_width_sweep import CORPUS_URL, build_oracle_table  # noqa: E402


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
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--modes", default="none,linear")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--aux-lambda", type=float, default=1.0)
    p.add_argument("--sae-over", type=int, default=4)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--sae-steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=400000)
    p.add_argument("--eval-tokens", type=int, default=6000)
    p.add_argument("--top-k-edges", type=int, default=20, help="top static K-edges to gate for the static→dynamic ρ")
    p.add_argument("--k-null", type=int, default=3, help="reader-matched random-writer controls per reader")
    p.add_argument("--n-name-tokens", type=int, default=60)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/cov95_forge_tax/oracle_supervised_dag_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/oracle_supervised_dag.png"))
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as F
    import urllib.request
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    text = urllib.request.urlopen(CORPUS_URL, timeout=8).read().decode("utf-8", "ignore")[: args.max_chars]
    ids = np.array(tok(text)["input_ids"], dtype=np.int64)
    n_train = int(len(ids) * 0.9)
    train_ids, eval_ids = ids[:n_train], ids[n_train:]
    Ttab, tiers = build_oracle_table(tok, ids, args.min_pos)
    Ttab_t = torch.from_numpy(Ttab).to(dev); Ffeat = Ttab.shape[1]
    seeds = [int(s) for s in args.seeds.split(",")]; modes = [m.strip() for m in args.modes.split(",")]
    d = args.width; H = args.n_head; hd = d // H; nL = args.n_layer; NH = nL * H
    layer_of = np.array([L for L in range(nL) for _ in range(H)])
    print(f"corpus {len(ids)} tok; width {d}; oracle {Ffeat} features; modes {modes}; seeds {seeds}")

    def batch(g):
        st = torch.randint(0, len(train_ids) - args.ctx - 1, (args.batch,), generator=g)
        x = torch.stack([torch.from_numpy(train_ids[s:s + args.ctx]) for s in st])
        y = torch.stack([torch.from_numpy(train_ids[s + 1:s + 1 + args.ctx]) for s in st])
        return x.to(dev), y.to(dev)

    def train(mode, seed):
        torch.manual_seed(seed)
        cfg = GPT2Config(vocab_size=tok.vocab_size, n_positions=args.ctx, n_ctx=args.ctx,
                         n_embd=d, n_layer=nL, n_head=H)
        cfg._attn_implementation = "eager"
        model = GPT2LMHeadModel(cfg).to(dev).train()
        head = torch.nn.Linear(d, Ffeat).to(dev)
        params = list(model.parameters()) + (list(head.parameters()) if mode == "linear" else [])
        opt = torch.optim.AdamW(params, lr=args.lr); g = torch.Generator().manual_seed(seed)
        for _ in range(args.steps):
            x, y = batch(g)
            out = model(input_ids=x, labels=y, output_hidden_states=True)
            loss = out.loss + (args.aux_lambda * F.binary_cross_entropy_with_logits(head(out.hidden_states[-1]), Ttab_t[x])
                               if mode == "linear" else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        return model.eval()

    def eval_chunks():
        ev = eval_ids[: args.eval_tokens]
        return [ev[i:i + args.ctx] for i in range(0, len(ev) - 1, args.ctx) if len(ev[i:i + args.ctx]) >= 8]

    def cov95(model):
        chunks = eval_chunks(); acts = []; ev_ids = []
        with torch.no_grad():
            for ch in chunks:
                xx = torch.from_numpy(np.ascontiguousarray(ch))[None].to(dev)
                acts.append(model(input_ids=xx, output_hidden_states=True).hidden_states[-1][0].float().cpu().numpy())
                ev_ids.extend(ch)
        Xr = np.concatenate(acts, 0).astype(np.float32); Y = Ttab[np.array(ev_ids)]
        X = ((Xr - Xr.mean(0)) / (Xr.std(0) + 1e-6)).astype(np.float32)
        sae = _train_topk_sae(X, args.sae_over * d, args.k, args.sae_steps, 1e-3, 0)
        return _per_tier(_best_auc_per_label(_encode(X, sae, args.k), Y), tiers)["all"]["cov95"]

    def indmask(c):
        Lc = len(c); ca = np.array(c); pv = np.full(Lc, -1); pv[1:] = ca[:-1]; qi = np.arange(Lc)
        return (pv[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)

    def circuit(model, seed):
        tr = model.transformer
        Wq = [tr.h[L].attn.c_attn.weight.detach().float().cpu().numpy().astype(np.float64)[:, :d] for L in range(nL)]
        Wk = [tr.h[L].attn.c_attn.weight.detach().float().cpu().numpy().astype(np.float64)[:, d:2 * d] for L in range(nL)]
        Wv = [tr.h[L].attn.c_attn.weight.detach().float().cpu().numpy().astype(np.float64)[:, 2 * d:3 * d] for L in range(nL)]
        bq = [tr.h[L].attn.c_attn.bias.detach().float().cpu().numpy().astype(np.float64)[:d] for L in range(nL)]
        bk = [tr.h[L].attn.c_attn.bias.detach().float().cpu().numpy().astype(np.float64)[d:2 * d] for L in range(nL)]
        bv = [tr.h[L].attn.c_attn.bias.detach().float().cpu().numpy().astype(np.float64)[2 * d:3 * d] for L in range(nL)]
        Wo = [tr.h[L].attn.c_proj.weight.detach().float().cpu().numpy().astype(np.float64) for L in range(nL)]
        ln1w = [tr.h[L].ln_1.weight.detach().float().cpu().numpy().astype(np.float64) for L in range(nL)]
        ln1b = [tr.h[L].ln_1.bias.detach().float().cpu().numpy().astype(np.float64) for L in range(nL)]
        OV = np.zeros((NH, d, d)); WK = np.zeros((NH, d, hd))
        for L in range(nL):
            for h in range(H):
                sl = slice(h * hd, (h + 1) * hd); i = L * H + h
                OV[i] = Wv[L][:, sl] @ Wo[L][sl, :]; WK[i] = Wk[L][:, sl]
        u = np.linalg.svd(OV.transpose(0, 2, 1).reshape(-1, d), full_matrices=False)[2][0]
        OV = np.einsum("nij,jk->nik", OV, np.eye(d) - np.outer(u, u))
        ovn = np.linalg.norm(OV.reshape(NH, -1), axis=1) + 1e-9
        wkn = np.linalg.norm(WK.reshape(NH, -1), axis=1) + 1e-9
        causal = layer_of[:, None] < layer_of[None, :]
        Kc = np.zeros((NH, NH))
        for a in range(NH):
            for b in range(NH):
                if layer_of[b] > layer_of[a]:
                    Kc[a, b] = np.linalg.norm(OV[a] @ WK[b]) / (ovn[a] * wkn[b])

        # behavioural prev-token + induction per head
        chunks = eval_chunks()
        pt = np.zeros(NH); ind = np.zeros(NH); ptn = 0; indn = 0
        with torch.no_grad():
            for c in chunks:
                o = tr(input_ids=torch.tensor([c], device=dev), output_attentions=True)
                Lc = len(c); IM = indmask(c); ptn += Lc - 1; indn += int(IM.sum())
                for L in range(nL):
                    a = o.attentions[L][0].float().cpu().numpy()
                    pt[L * H:(L + 1) * H] += np.diagonal(a, offset=-1, axis1=1, axis2=2).sum(1)
                    ind[L * H:(L + 1) * H] += (a * IM[None]).sum((1, 2))
        prevtok = pt / max(ptn, 1); induct = ind / max(indn, 1)
        writers = [int(i) for i in np.argsort(-prevtok)[:3]]
        inductors = [int(i) for i in np.argsort(-induct)[:4]]
        base_k = float(Kc[causal].mean())
        pt2ind = float(np.mean([Kc[a, b] for a in writers for b in inductors if layer_of[a] < layer_of[b]] or [0.0]))
        induction_recovery = pt2ind / max(base_k, 1e-9)

        # gated edges: top static K + prev→ind + per-reader random-writer null
        rng = np.random.default_rng(seed)
        flat = sorted([(a, b, Kc[a, b]) for a in range(NH) for b in range(NH) if Kc[a, b] > 0], key=lambda r: -r[2])
        edges = {(a, b): {"static": Kc[a, b], "rand": False} for a, b, _ in flat[: args.top_k_edges]}
        for a in writers:
            for b in inductors:
                if layer_of[a] < layer_of[b]:
                    edges[(a, b)] = {"static": Kc[a, b], "rand": False}
        for b in sorted({bb for (_a, bb) in list(edges)}):
            pool = [a for a in range(NH) if layer_of[a] < layer_of[b] and (a, b) not in edges]
            for a in (int(x) for x in rng.permutation(pool)[: args.k_null]):
                edges[(a, b)] = {"static": Kc[a, b], "rand": True}
        wall = {a for (a, _b) in edges}

        # ΔTV path-patch (remove writer A from reader B key) + prev-token-head key pos/tok variance
        B = writers[0]; LB, hB = B // H, B % H; slB = slice(hB * hd, (hB + 1) * hd)
        dtv = {e: 0.0 for e in edges}; tot = 0
        pos_sum = np.zeros((args.ctx, hd)); pos_cnt = np.zeros(args.ctx)
        freq = [int(t) for t, _ in __import__("collections").Counter(int(j) for c in chunks for j in c).most_common(args.n_name_tokens)]
        tix = {t: i for i, t in enumerate(freq)}; tok_sum = np.zeros((len(freq), hd)); tok_cnt = np.zeros(len(freq))
        gsum = np.zeros(hd); gsq = 0.0; Nk = 0
        with torch.no_grad():
            for c in chunks:
                o = tr(input_ids=torch.tensor([c], device=dev), output_hidden_states=True, output_attentions=True)
                HS = [o.hidden_states[L][0].float().cpu().numpy() for L in range(nL)]
                ATT = [o.attentions[L][0].float().cpu().numpy() for L in range(nL)]
                Lc = len(c); tot += Lc
                Aout = {}
                for a in wall:
                    La, hA = a // H, a % H; sl = slice(hA * hd, (hA + 1) * hd)
                    vA = _ln(HS[La], ln1w[La], ln1b[La]) @ Wv[La][:, sl] + bv[La][sl]
                    Aout[a] = (ATT[La][hA] @ vA) @ Wo[La][sl, :]
                lnB = _ln(HS[LB], ln1w[LB], ln1b[LB])
                Q = lnB @ Wq[LB][:, slB] + bq[LB][slB]; Kcl = lnB @ Wk[LB][:, slB] + bk[LB][slB]
                Pc = _causal_softmax(Q @ Kcl.T / np.sqrt(hd))
                pos_sum[:Lc] += Kcl; pos_cnt[:Lc] += 1                            # prev-token head key stats
                for s in range(Lc):
                    if c[s] in tix:
                        j = tix[c[s]]; tok_sum[j] += Kcl[s]; tok_cnt[j] += 1
                gsum += Kcl.sum(0); gsq += float((Kcl ** 2).sum()); Nk += Lc
                for (a, b) in edges:
                    Lb, hb = b // H, b % H; sl = slice(hb * hd, (hb + 1) * hd)
                    lnp = _ln(HS[Lb] - Aout[a], ln1w[Lb], ln1b[Lb])
                    Kp = lnp @ Wk[Lb][:, sl] + bk[Lb][sl]
                    Qb = (lnB @ Wq[LB][:, slB] + bq[LB][slB]) if b == B else (_ln(HS[Lb], ln1w[Lb], ln1b[Lb]) @ Wq[Lb][:, sl] + bq[Lb][sl])
                    Pcl = Pc if b == B else _causal_softmax(Qb @ (_ln(HS[Lb], ln1w[Lb], ln1b[Lb]) @ Wk[Lb][:, sl] + bk[Lb][sl]).T / np.sqrt(hd))
                    Pp = _causal_softmax(Qb @ Kp.T / np.sqrt(hd))
                    dtv[(a, b)] += float(0.5 * np.abs(Pcl - Pp).sum())
        dv = {e: dtv[e] / max(tot, 1) for e in edges}
        rn = {}
        for (a, b), m in edges.items():
            if m["rand"]:
                rn.setdefault(b, []).append(dv[(a, b)])
        rnmean = {b: float(np.mean(v)) for b, v in rn.items()}
        nonrand = [(a, b) for (a, b), m in edges.items() if not m["rand"]]
        rho = _spearman([edges[(a, b)]["static"] for (a, b) in nonrand],
                        [dv[(a, b)] - rnmean.get(b, 0.0) for (a, b) in nonrand])

        gmean = gsum / Nk; total_var = gsq / Nk - float(gmean @ gmean)
        pm = pos_cnt > 0; tm = tok_cnt > 0
        bpos = float((pos_cnt[pm][:, None] * (pos_sum[pm] / pos_cnt[pm][:, None] - gmean) ** 2).sum() / Nk)
        btok = float((tok_cnt[tm][:, None] * (tok_sum[tm] / tok_cnt[tm][:, None] - gmean) ** 2).sum() / Nk)
        return {"static_dynamic_rho": rho, "induction_recovery": induction_recovery,
                "prevtok_head": f"{LB}.{hB}", "induction_strength": float(induct[inductors[0]]),
                "key_pos_frac": bpos / max(total_var, 1e-9), "key_tok_frac": btok / max(total_var, 1e-9)}

    rows = []
    for seed in seeds:
        for mode in modes:
            m = train(mode, seed)
            cv = cov95(m); cir = circuit(m, seed)
            rows.append({"mode": mode, "seed": seed, "cov95": cv, **cir})
            print(f"  seed {seed} {mode:>6}: cov95 {cv:.3f} | static→dyn ρ {cir['static_dynamic_rho']:+.3f} | "
                  f"induction-recovery {cir['induction_recovery']:.2f}x (str {cir['induction_strength']:.3f}) | "
                  f"prev-tok key pos {cir['key_pos_frac']:.0%}/tok {cir['key_tok_frac']:.0%}")

    def col(mode, key):
        return [r[key] for r in rows if r["mode"] == mode]

    def agg(key):
        return {m: {"mean": float(np.mean(col(m, key))), "std": float(np.std(col(m, key)))} for m in modes}

    def paired(key):
        if not set(modes) >= {"none", "linear"}:
            return []
        return [next(r[key] for r in rows if r["mode"] == "linear" and r["seed"] == s)
                - next(r[key] for r in rows if r["mode"] == "none" and r["seed"] == s) for s in seeds]
    metrics = ["cov95", "static_dynamic_rho", "induction_recovery", "key_pos_frac"]
    aggs = {k: agg(k) for k in metrics}; deltas = {k: paired(k) for k in metrics}
    out = {"experiment": "oracle-supervised DAG: does feature legibility buy circuit legibility?",
           "width": d, "seeds": seeds, "modes": modes, "rows": rows, "agg": aggs, "linear_minus_none": deltas}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print("\n[verdict] none → linear (mean over seeds), paired Δ (+n/total seeds positive):")
    for k in metrics:
        dd = deltas[k]; a = aggs[k]
        npos = sum(x > 0 for x in dd) if dd else 0
        print(f"  {k:>18}: {a['none']['mean']:.3f} → {a['linear']['mean']:.3f}  (Δ {np.mean(dd):+.3f} ± {np.std(dd):.3f}; +{npos}/{len(dd)})"
              if dd else f"  {k:>18}: {a}")
    def signs(key):
        return sum(x > 0 for x in deltas[key]), sum(x < 0 for x in deltas[key]), len(deltas[key])

    def robust(key):  # consistent direction across seeds AND |mean| exceeds the spread
        d = deltas[key]
        if not d:
            return 0
        pos, neg, n = signs(key)
        consistent = max(pos, neg) == n and abs(np.mean(d)) > np.std(d)
        return int(np.sign(np.mean(d))) if consistent else 0
    d_cov = float(np.mean(deltas["cov95"])) if deltas["cov95"] else 0.0
    d_rho = float(np.mean(deltas["static_dynamic_rho"])) if deltas["static_dynamic_rho"] else 0.0
    s_rho = float(np.std(deltas["static_dynamic_rho"])) if deltas["static_dynamic_rho"] else 0.0
    d_ind = float(np.mean(deltas["induction_recovery"])) if deltas["induction_recovery"] else 0.0
    d_pos = float(np.mean(deltas["key_pos_frac"])) if deltas["key_pos_frac"] else 0.0
    feat_up = robust("cov95") > 0 or (np.mean(deltas["cov95"]) > 0.03 if deltas["cov95"] else False)
    rho_robust = robust("static_dynamic_rho") != 0          # the broad circuit metric: is it even consistent?
    ind_robust = robust("induction_recovery") > 0
    pos_shift = robust("key_pos_frac")                       # +1 toward position, -1 toward token, 0 if noisy
    if deltas["cov95"]:
        substrate = (f"reshapes the FEATURE substrate — cov95 {d_cov:+.3f}" +
                     (f"; the prev-token key shifts {'POSITION-ward' if pos_shift > 0 else 'TOKEN-ward'} "
                      f"(Δkey-pos-frac {d_pos:+.3f}, {max(signs('key_pos_frac')[:2])}/{signs('key_pos_frac')[2]} seeds)"
                      if pos_shift else f"; prev-token key fraction noisy (Δ {d_pos:+.3f})"))
        circuit_clause = (f"induction-recoverability lifts (Δ {d_ind:+.2f}, {signs('induction_recovery')[0]}/"
                          f"{signs('induction_recovery')[2]} seeds)" if ind_robust else
                          f"induction-recoverability flat (Δ {d_ind:+.2f})")
        rho_clause = (f"the broad static→dynamic agreement {'lifts' if d_rho > 0 else 'falls'} but is NOISE-DOMINATED "
                      f"(Δρ {d_rho:+.3f} ± {s_rho:.2f})" if not rho_robust else f"static→dynamic ρ shifts {d_rho:+.3f}")
        if feat_up and ind_robust and not rho_robust:
            concl = (f"PARTIAL / SUBSTRATE-DOMINATED: oracle-feature supervision {substrate}; a small but consistent "
                     f"{circuit_clause}; {rho_clause}. The feature-recovery loss acts mainly on WHAT the residual "
                     f"represents (features), with only a marginal spillover to circuit recoverability — knowledge and "
                     f"computation read as largely SEPARATE axes (the thesis), on a tiny host underpowered for circuit metrics.")
        elif feat_up and (ind_robust or rho_robust):
            concl = f"COUPLED (with caveats): supervision {substrate}, and circuit legibility lifts too — {circuit_clause}; {rho_clause}."
        elif feat_up:
            concl = (f"INDEPENDENT AXES: supervision {substrate} but circuit legibility does NOT lift — {circuit_clause}; "
                     f"{rho_clause} — a feature-recovery loss does not supervise circuits into existence.")
        else:
            concl = "cov95 did not robustly lift this run — inconclusive (the supervision lever did not fire cleanly)"
        out["conclusion"] = concl
        args.output.write_text(json.dumps(out, indent=2, default=float))
        print(f"\n[conclusion] {concl}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.2))
        titles = {"cov95": "FEATURE legibility\n(cov95)", "static_dynamic_rho": "CIRCUIT legibility\n(static→dyn ρ)",
                  "induction_recovery": "induction recovery\n(prev→ind / baseline)", "key_pos_frac": "prev-tok key\nposition fraction"}
        for ax, k in zip(axes, metrics):
            mu = [aggs[k][m]["mean"] for m in modes]; sd = [aggs[k][m]["std"] for m in modes]
            ax.bar(range(len(modes)), mu, yerr=sd, capsize=4,
                   color=["#999999" if m == "none" else "#2ca02c" for m in modes], edgecolor="k")
            ax.set_xticks(range(len(modes))); ax.set_xticklabels(modes, fontsize=9)
            ax.set_title(titles.get(k, k), fontsize=10)
        axes[0].set_ylabel("metric (mean ± std over seeds)")
        fig.suptitle("Oracle-supervised DAG: does feature legibility (cov95) buy circuit legibility?", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()