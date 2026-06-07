"""Validate the new write-hub edges M2 surfaced — targeted path-patch with a BEHAVIORAL readout + naming.

`composition_dag.py` (M2) flagged 22 new live edges (high reader-matched ΔTV, not in induction/IOI) dominated by
early-layer WRITE-HUBS (0.11 / 0.9 / 1.8 → many readers, incl. the canonical prev-token head 4.11's key). ΔTV
says they reshape attention; this asks WHAT they do. For each edge A→B, surgically remove A's output from B's
port (key or query), recompute B's attention from scratch, and measure the collapse of B's NAMED behavioral
components — {prev-token (Δ=1), duplicate-token, induction, sink (pos-0)} — against a **reader-matched
random-writer null**. A pattern collapse names the edge's function (e.g. an early positional head supplying the
prev-token head's key); a live ΔTV with no named collapse = real-but-unlabeled attention-shaping. GPT-2; the
faithful key/query-path patch is the induction-style strong readout, generalized to the four idiom masks.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


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
    p.add_argument("--dag-summary", type=Path, default=Path("runs/disassembly/composition_dag_summary.json"))
    p.add_argument("--max-tokens", type=int, default=14000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--k-null", type=int, default=4, help="reader-matched random-writer controls per (reader,port)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/validate_new_edges_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/validate_new_edges.png"))
    args = p.parse_args(argv)

    if not args.dag_summary.exists():
        print(f"[dag] {args.dag_summary} missing -> running composition_dag.py ...")
        import composition_dag
        composition_dag.main(["--pretrained", args.pretrained, "--output", str(args.dag_summary)])
    dag = json.loads(args.dag_summary.read_text())
    new_edges = [(e["port"], e["A"], e["B"], e["specificity"], e["dtv"])
                 for e in dag["edges"] if e["live"] and not e["is_circuit"] and not e["is_random"]]

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer; cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]

    heads = [(L, h) for L in range(nL) for h in range(H)]
    layer_of = np.array([L for L, _ in heads]); NH = nL * H

    def hi(s):
        L, h = s.split("."); return int(L) * H + int(h)

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

    PATTERNS = ["prevtok", "duplicate", "induction", "sink"]

    def masks(c):
        Lc = len(c); ca = np.array(c); qi = np.arange(Lc)
        pv = np.full(Lc, -1); pv[1:] = ca[:-1]
        PT = (qi[None, :] == (qi[:, None] - 1)) & (qi[None, :] >= 0) & (qi[:, None] >= 1)
        DUP = (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])
        IND = (pv[None, :] == ca[:, None]) & (qi[None, :] >= 1) & (qi[None, :] < qi[:, None])
        SINK = (qi[None, :] == 0) & (qi[:, None] >= 1)
        return {"prevtok": PT, "duplicate": DUP, "induction": IND, "sink": SINK}

    # ---- PASS 1: per-head clean masses for self-labeling ----
    print(f"{args.pretrained}: {len(new_edges)} new edges to validate; pass1 (head self-labels)...")
    hmass = {p: np.zeros(NH) for p in PATTERNS}; tot_tok = 0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_attentions=True)
            M = masks(c); tot_tok += len(c)
            for L in range(nL):
                aL = o.attentions[L][0].float().numpy()
                for p in PATTERNS:
                    hmass[p][L * H:(L + 1) * H] += (aL * M[p][None]).sum((1, 2))
    for p in PATTERNS:
        hmass[p] /= max(tot_tok, 1)

    def self_label(i):
        vals = {p: hmass[p][i] for p in PATTERNS}
        best = max(vals, key=vals.get)
        return best if vals[best] > 0.05 else "diffuse"

    # ---- build edge set: the new edges + reader-matched random-writer null per (reader, port) ----
    rng = np.random.default_rng(args.seed)
    edges = {}                                                               # (port,a,b) -> {is_new, spec, dtv}
    for port, A, B, spec, dtv in new_edges:
        edges[(port, hi(A), hi(B))] = {"is_new": True, "spec": spec, "dtv_m2": dtv}
    for port in ("K", "Q"):
        readers = sorted({b for (pp, _a, b) in list(edges) if pp == port})
        for b in readers:
            pool = [a for a in range(NH) if layer_of[a] < layer_of[b] and (port, a, b) not in edges]
            for a in (int(x) for x in rng.permutation(pool)[: args.k_null]):
                edges[(port, a, b)] = {"is_new": False, "spec": 0.0, "dtv_m2": 0.0}
    writers_all = {a for (_p, a, _b) in edges}; readers_all = {b for (_p, _a, b) in edges}
    print(f"  gating {len(edges)} edges ({sum(v['is_new'] for v in edges.values())} new + reader-matched null); pass2...")

    # ---- PASS 2: targeted path-patch; clean/patched named-pattern mass + ΔTV per edge ----
    clean = {e: {p: 0.0 for p in PATTERNS} for e in edges}
    patch = {e: {p: 0.0 for p in PATTERNS} for e in edges}
    dtv = {e: 0.0 for e in edges}; tot = 0; sane = None
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            HS = [o.hidden_states[L][0].float().numpy() for L in range(nL)]
            ATT = [o.attentions[L][0].float().numpy() for L in range(nL)]
            M = masks(c); Lc = len(c); tot += Lc
            Aout = {}
            for a in writers_all:
                La, hA = heads[a]; sl = slice(hA * hd, (hA + 1) * hd)
                vA = _ln(HS[La], ln1w[La], ln1b[La]) @ Wv[La][:, sl] + bv[La][sl]
                Aout[a] = (ATT[La][hA] @ vA) @ Wo[La][sl, :]
            cleanP = {}
            for b in readers_all:
                Lb, hB = heads[b]; sl = slice(hB * hd, (hB + 1) * hd)
                lnB = _ln(HS[Lb], ln1w[Lb], ln1b[Lb])
                Q = lnB @ Wq[Lb][:, sl] + bq[Lb][sl]; K = lnB @ Wk[Lb][:, sl] + bk[Lb][sl]
                Pc = _causal_softmax(Q @ K.T / np.sqrt(hd)); cleanP[b] = (Q, K, Pc)
                if sane is None:
                    sane = float(np.abs(Pc - ATT[Lb][hB]).max())
            for (port, a, b) in edges:
                Lb, hB = heads[b]; sl = slice(hB * hd, (hB + 1) * hd)
                Q, K, Pc = cleanP[b]
                lnp = _ln(HS[Lb] - Aout[a], ln1w[Lb], ln1b[Lb])
                if port == "K":
                    Kp = lnp @ Wk[Lb][:, sl] + bk[Lb][sl]; Pp = _causal_softmax(Q @ Kp.T / np.sqrt(hd))
                else:
                    Qp = lnp @ Wq[Lb][:, sl] + bq[Lb][sl]; Pp = _causal_softmax(Qp @ K.T / np.sqrt(hd))
                dtv[(port, a, b)] += float(0.5 * np.abs(Pc - Pp).sum())
                for p in PATTERNS:
                    clean[(port, a, b)][p] += float((Pc * M[p]).sum()); patch[(port, a, b)][p] += float((Pp * M[p]).sum())
    print(f"[sanity] max|recomputed clean attn - model attn| = {sane:.2e}")

    # ---- per-(reader,port,pattern) null: random-writer delta band ----
    null = defaultdict(lambda: defaultdict(list))
    for (port, a, b), m in edges.items():
        if not m["is_new"]:
            for p in PATTERNS:
                null[(port, b)][p].append((clean[(port, a, b)][p] - patch[(port, a, b)][p]) / max(tot, 1))
    null_thr = {kp: {p: (float(np.mean(v)) + 2 * float(np.std(v))) for p, v in d.items()} for kp, d in null.items()}

    rows = []
    for (port, a, b), m in edges.items():
        if not m["is_new"]:
            continue
        pats = {}
        for p in PATTERNS:
            cl = clean[(port, a, b)][p] / max(tot, 1); pa = patch[(port, a, b)][p] / max(tot, 1)
            thr = null_thr.get((port, b), {}).get(p, 0.0)
            pats[p] = {"clean": cl, "patched": pa, "delta": cl - pa, "rel": (cl - pa) / cl if cl > 1e-6 else 0.0,
                       "null_thr": thr, "confirmed": (cl - pa) > max(thr, 1e-4) and cl > 0.02}
        confirmed = {p: v for p, v in pats.items() if v["confirmed"]}
        best = max(confirmed, key=lambda p: confirmed[p]["rel"]) if confirmed else None
        rows.append({"port": port, "A": nm(a), "B": nm(b), "A_role": self_label(a), "B_role": self_label(b),
                     "spec_m2": m["spec"], "dtv": dtv[(port, a, b)] / max(tot, 1), "patterns": pats,
                     "best_pattern": best, "named": best is not None})

    n_named = sum(r["named"] for r in rows)
    by_pat = defaultdict(int)
    for r in rows:
        if r["best_pattern"]:
            by_pat[r["best_pattern"]] += 1
    out = {"experiment": "validate new write-hub edges (targeted path-patch + behavioral naming)",
           "model": args.pretrained, "n_new_edges": len(rows), "n_named": n_named,
           "named_by_pattern": dict(by_pat), "sanity_recompute_err": sane,
           "edges": sorted(rows, key=lambda r: -r["spec_m2"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\n{'edge':>13} {'A.role':>10} {'B.role':>10} {'ΔTV':>6}  pattern collapses (rel-drop, * = beats null)")
    for r in sorted(rows, key=lambda r: -r["spec_m2"]):
        tags = []
        for p in PATTERNS:
            pv = r["patterns"][p]
            if pv["clean"] > 0.02 and pv["rel"] > 0.05:
                star = "*" if pv["confirmed"] else ""
                tags.append(f"{p}{star} -{pv['rel']:.0%}")
        label = f"  => {r['A_role']}->{r['B_role']} supplies B's {r['best_pattern'].upper()}" if r["named"] else "  => unlabeled shaping"
        print(f"  {r['port']} {r['A']+'->'+r['B']:>11} {r['A_role']:>10} {r['B_role']:>10} {r['dtv']:>6.3f}  "
              f"{', '.join(tags) if tags else '(no named pattern)'}{label}")

    print(f"\n[summary] {n_named}/{len(rows)} new edges carry a NAMED behavioral function (beat reader-matched null): "
          + ", ".join(f"{p} {n}" for p, n in sorted(by_pat.items(), key=lambda x: -x[1])))
    hubs = defaultdict(list)
    for r in rows:
        hubs[r["A"]].append(r)
    print("[write-hubs] per-writer summary:")
    for A in sorted(hubs, key=lambda a: -len(hubs[a])):
        rs = hubs[A]
        pats = defaultdict(int)
        for r in rs:
            if r["best_pattern"]:
                pats[r["best_pattern"]] += 1
        print(f"  {A} ({self_label(hi(A))}): {len(rs)} edges -> " +
              (", ".join(f"{p}x{n}" for p, n in pats.items()) or "unlabeled") +
              f"  (readers {sorted({r['B'] for r in rs})})")
    verdict = (f"VALIDATED: {n_named}/{len(rows)} write-hub edges resolve to a NAMED behavioral circuit (path-patch "
               f"collapse beats the reader-matched null); dominant function = "
               f"{max(by_pat, key=by_pat.get).upper() if by_pat else 'none'}. The early write-hubs broadcast the "
               f"positional/identity signal downstream keys read — the disassembly's position/structure register, "
               f"now edge-resolved. Remaining edges = real ΔTV but no single named pattern (distributed shaping).") \
        if n_named else "no new edge resolved to a named pattern beyond the null — they are distributed attention-shaping"
    print(f"\n[verdict] {verdict}")
    print(f"[done] {args.output}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axS, axH) = plt.subplots(1, 2, figsize=(12.8, 5.0))
        cols = {"prevtok": "#1f77b4", "duplicate": "#2ca02c", "induction": "#d62728", "sink": "#9467bd", None: "#bbbbbb"}
        srt = sorted(rows, key=lambda r: -r["spec_m2"])
        y = np.arange(len(srt))
        axS.barh(y, [r["patterns"][r["best_pattern"]]["rel"] if r["best_pattern"] else 0 for r in srt],
                 color=[cols[r["best_pattern"]] for r in srt], edgecolor="k")
        axS.set_yticks(y); axS.set_yticklabels([f"{r['port']} {r['A']}->{r['B']}" for r in srt], fontsize=7)
        axS.invert_yaxis(); axS.set_xlabel("best named-pattern rel-collapse under path-patch")
        axS.set_title(f"new edges -> named function ({n_named}/{len(rows)} named)", fontsize=10)
        handles = [plt.Rectangle((0, 0), 1, 1, color=cols[p]) for p in PATTERNS]
        axS.legend(handles, PATTERNS, fontsize=7, loc="lower right")
        labels = list(by_pat) + (["unlabeled"] if n_named < len(rows) else [])
        counts = [by_pat[p] for p in by_pat] + ([len(rows) - n_named] if n_named < len(rows) else [])
        axH.bar(range(len(labels)), counts, color=[cols.get(p, "#bbbbbb") for p in by_pat] + (["#bbbbbb"] if n_named < len(rows) else []), edgecolor="k")
        axH.set_xticks(range(len(labels))); axH.set_xticklabels(labels, fontsize=8)
        axH.set_ylabel("# new edges"); axH.set_title("what the write-hubs supply", fontsize=10)
        fig.suptitle("Validating M2's new write-hub edges: targeted path-patch + behavioral naming", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    return out


if __name__ == "__main__":
    main()
