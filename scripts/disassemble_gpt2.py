"""Disassemble GPT-2 — a per-component instruction listing from the catalogued reads.

Folds every read we built into one program listing. For each attention head: ADDRESSING field
(content-opcode / relative-Δ / absolute-sink / structural, from behaviour) x WRITE field (copy/transform,
from OV V_h), its top content binding (B_h), and circuit ROLE (prev-token mover / induction / sink / line-
anchor). For each MLP layer: the top salient COMPUTE neurons (read-features -> write-features). One forward
pass for the operand basis + behavioural signals; everything else is weights.

Honest scope: FIRST-ORDER disassembly (rung-1 single-component instructions + the induction idiom). Not a
decompilation — superposition + the imperfect centroid operand basis cap the fidelity. The shared mean-write
direction (common-token nudge every component adds; see write_bus_check) is the implicit 'default', not listed.
GPT-2.
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


def _z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-9)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=9000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=40)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--mlp-per-layer", type=int, default=3)
    p.add_argument("--listing", type=Path, default=Path("runs/gpt2_disassembly.txt"))
    p.add_argument("--output", type=Path, default=Path("runs/gpt2_disassembly.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    nl_ids = {tok(s, add_special_tokens=False)["input_ids"][0] for s in ("\n", "\n\n")
              if len(tok(s, add_special_tokens=False)["input_ids"]) == 1}
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
    cand += [t for t, _ in cnt.most_common(200)]
    seen, toks = set(), []
    for t in cand:
        if t not in seen and cnt[t] >= args.min_pos:
            seen.add(t); toks.append(t)
        if len(toks) >= args.n_tokens:
            break
    nt = len(toks); tok2i = {t: i for i, t in enumerate(toks)}
    nm = [tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t in toks]

    # one pass: per-layer centroids + per-head behavioural signals
    cen = np.zeros((nL + 1, nt, d)); ccnt = np.zeros(nt); gm = np.zeros((nL + 1, d)); gn = 0
    pt = np.zeros((nL, H)); ptn = 0
    ind = np.zeros((nL, H)); indn = 0
    sink = np.zeros((nL, H)); sinkn = 0
    nlh = np.zeros((nL, H)); nlbase = 0.0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            Lc = len(c); ca = np.array(c)
            pid = np.array([tok2i.get(t, -1) for t in c]); m = pid >= 0
            ccnt += np.bincount(pid[m], minlength=nt); gn += Lc; ptn += Lc - 1; sinkn += Lc
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]; qi = np.arange(Lc)
            IM = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
            indn += int(IM.sum())
            km = np.array([1.0 if t in nl_ids else 0.0 for t in c])
            nlbase += float(km.sum() / Lc) * Lc
            for L in range(nL + 1):
                hs = o.hidden_states[L][0].float().numpy()
                np.add.at(cen[L], pid[m], hs[m]); gm[L] += hs.sum(0)
            for L in range(nL):
                aL = o.attentions[L][0].float().numpy()
                pt[L] += np.diagonal(aL, offset=-1, axis1=1, axis2=2).sum(1)
                ind[L] += (aL * IM[None]).sum((1, 2))
                sink[L] += aL[:, :, 0].sum(1)
                nlh[L] += (aL @ km).sum(1)
    cen = cen / np.maximum(ccnt, 1)[None, :, None]; gm = gm / max(gn, 1)
    prevtok = pt / max(ptn, 1); induct = ind / max(indn, 1) - (indn / max(indn, 1))  # excess handled below
    induct = ind / max(indn, 1)
    sinkv = sink / max(sinkn, 1); nlv = nlh / max(gn, 1) - nlbase / max(gn, 1)

    # per-head weight reads
    rows = []
    for L in range(nL):
        blk = tr.h[L]
        lnw = blk.ln_1.weight.detach().numpy().astype(np.float64)
        Dl = (cen[L] - gm[L]) * lnw; Dl = Dl / (np.linalg.norm(Dl, axis=1, keepdims=True) + 1e-9)
        Wc = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wo = blk.attn.c_proj.weight.detach().numpy().astype(np.float64)
        Wq, Wk, Wv = Wc[:, :d], Wc[:, d:2 * d], Wc[:, 2 * d:3 * d]
        offm = ~np.eye(nt, dtype=bool)
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            B = Dl @ (Wq[:, sl] @ Wk[:, sl].T / np.sqrt(hd)) @ Dl.T
            V = Dl @ (Wv[:, sl] @ Wo[sl, :]) @ Dl.T
            soft = np.exp(B - B.max(1, keepdims=True)); np.fill_diagonal(soft, 0)
            soft = soft / (soft.sum(1, keepdims=True) + 1e-9)
            cstruct = max(float(np.diag(B).mean() - B[offm].mean()),
                          float(soft.sum(0).max() / soft.sum()), float(soft.max(1).mean()))
            ov_diag = float(np.diag(V).mean() - V[offm].mean())
            Bo = B.copy(); np.fill_diagonal(Bo, -np.inf)
            qi2, ki2 = np.unravel_index(int(np.argmax(Bo)), Bo.shape)
            rows.append({"L": L, "h": h, "cstruct": cstruct, "ov_diag": ov_diag,
                         "bind": (nm[qi2], nm[ki2]), "prevtok": float(prevtok[L, h]),
                         "induct": float(induct[L, h]), "sink": float(sinkv[L, h]), "nl": float(nlv[L, h])})

    # classify addressing (z across heads) + role
    cs = _z([r["cstruct"] for r in rows]); pz = _z([r["prevtok"] for r in rows])
    sz = _z([r["sink"] for r in rows]); nz = _z([r["nl"] for r in rows]); iz = _z([r["induct"] for r in rows])
    addr_names = ["content", "relative-Δ", "absolute-sink", "structural"]
    ov_hi = np.quantile([r["ov_diag"] for r in rows], 0.75)
    for i, r in enumerate(rows):
        a = np.array([cs[i], pz[i], sz[i], nz[i]])
        r["addr"] = addr_names[int(a.argmax())] if a.max() > 0.3 else "diffuse"
        r["write"] = "copy" if r["ov_diag"] >= ov_hi else "transform"
        role = []
        if pz[i] > 1.5 and r["prevtok"] > 0.25:
            role.append("prev-tok→induction-feed")
        if iz[i] > 1.5:
            role.append("induction")
        if sz[i] > 2.0:
            role.append("sink/NOP")
        if nz[i] > 1.5:
            role.append("line-anchor")
        r["role"] = ",".join(role)

    # MLP top neurons per layer
    mlp = {}
    for L in range(nL):
        blk = tr.h[L]
        ln2 = blk.ln_2.weight.detach().numpy().astype(np.float64)
        Dl = (cen[L] - gm[L]) * ln2; Dl = Dl / (np.linalg.norm(Dl, axis=1, keepdims=True) + 1e-9)
        Win = blk.mlp.c_fc.weight.detach().numpy().astype(np.float64)
        Wout = blk.mlp.c_proj.weight.detach().numpy().astype(np.float64)
        inm = (Dl @ Win).T; outm = Wout @ Dl.T
        sal = np.abs(inm).max(1) * np.abs(outm).max(1)
        mlp[L] = []
        for ni in np.argsort(-sal)[: args.mlp_per_layer]:
            rd = [nm[x] for x in np.argsort(-np.abs(inm[ni]))[:3]]
            wr = [nm[x] for x in np.argsort(-np.abs(outm[ni]))[:3]]
            mlp[L].append({"n": int(ni), "reads": rd, "writes": wr})

    # ---- emit listing ----
    lines = [f"; GPT-2 disassembly ({nL} layers x {H} heads + MLP). first-order; operand basis nt={nt}.",
             "; ADDR=where-to-read  WRITE=copy/transform (OV)  bind=top content binding (B_h)  role=circuit",
             "; (shared mean-write 'default' direction omitted — see write_bus_check)\n"]
    for L in range(nL):
        lines.append(f"--- layer {L} ---")
        for r in [x for x in rows if x["L"] == L]:
            q, k = r["bind"]
            role = f"  [{r['role']}]" if r["role"] else ""
            lines.append(f"  L{L}.H{r['h']:<2} ADDR={r['addr']:<12} WRITE={r['write']:<9} "
                         f"bind {q!r}->{k!r}{role}")
        for nrec in mlp[L]:
            lines.append(f"  L{L}.MLP.n{nrec['n']:<4} reads {{{','.join(nrec['reads'])}}} -> "
                         f"writes {{{','.join(nrec['writes'])}}}")
        lines.append("")
    args.listing.parent.mkdir(parents=True, exist_ok=True)
    args.listing.write_text("\n".join(lines))

    addr_hist = Counter(r["addr"] for r in rows)
    roles = Counter(x for r in rows for x in (r["role"].split(",") if r["role"] else []))
    out = {"experiment": "gpt2 disassembly", "model": args.pretrained, "n_heads": len(rows),
           "addr_hist": dict(addr_hist), "role_hist": dict(roles),
           "write_hist": dict(Counter(r["write"] for r in rows)),
           "heads": rows, "mlp": {str(k): v for k, v in mlp.items()}}
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print("\n".join(lines[:3]))
    print(f"[addressing] {dict(addr_hist)}")
    print(f"[write] {dict(Counter(r['write'] for r in rows))}")
    print(f"[roles] {dict(roles)}")
    print(f"\n[done] listing -> {args.listing}   json -> {args.output}")
    return out


if __name__ == "__main__":
    main()
