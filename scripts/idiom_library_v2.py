"""Idiom library v2 — repair the OV-copy idioms and extend the library toward the IOI circuit.

v1 (idiom_library.py) recovered prev-token / induction / duplicate behaviorally, but its OV-copy idioms
were DEAD: the positive-eigenvalue-MASS copy score saturated at +-0.50 for every head, so it recovered
NEITHER the name-movers (9.6/9.9/10.0) NOR copy-suppression (10.7). v2 fixes that and adds three idioms.

FIX — copy score = column-standardised diagonal of the OV->unembed direct path. For head h the token->token
logit-effect of attending to Y is C[Z,Y] = (wte[Z]*ln_f) . (OV_h @ wte[Y]) (tied GPT-2; ln_f gain folded).
copy_score_h = mean_Y (C[Y,Y] - mean_Z C[Z,Y]) / std_Z C[Z,Y]  -- how much attending to a token boosts
THAT token vs others, in std units. + = copy / name-mover, - = copy-suppression (10.7). No saturation.

ADD:
  succession      attending to an ordinal Y boosts its SUCCESSOR succ(Y) (one->two, Mon->Tue, Jan->Feb)
                  = the off-by-one diagonal of C over curated ordinal token lists (Gould et al.)
  s_inhibition    writes into the NAME-MOVERS' QUERY (Q-composition) -- the IOI head class that tells the
                  name-mover NOT to attend to the repeated subject; ranked by mean Q-comp into the copy-
                  score-discovered name-movers (Elhage comp score, mean-write removed); val vs {7.3,7.9,8.6,8.10}
  ioi_chain       the composed 3-stage idiom duplicate-token -> s_inhibition -> name-mover, scored as the
                  product of the stage composition scores; reports the strongest chain

GPT-2; one forward for the behavioural signatures, weights for copy / succession / composition.
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

# proper-name operands — name-movers are defined by COPYING NAMES (IOI). Filtered to single-token.
NAMES = [
    " John", " Mary", " James", " Robert", " Michael", " William", " David", " Richard", " Thomas",
    " Charles", " Paul", " Mark", " George", " Edward", " Brian", " Anthony", " Kevin", " Jason",
    " Gary", " Frank", " Scott", " Eric", " Andrew", " Peter", " Henry", " Carl", " Arthur", " Ryan",
    " Patricia", " Jennifer", " Linda", " Elizabeth", " Barbara", " Susan", " Sarah", " Karen", " Nancy",
    " Lisa", " Betty", " Anna", " Tom", " Sam", " Joe", " Jack", " Bill", " Harry", " Alice", " Jane",
]

# curated single-token-friendly ordinal sequences (leading space = GPT-2 word-initial BPE)
ORDINALS = [
    [" one", " two", " three", " four", " five", " six", " seven", " eight", " nine", " ten"],
    [" January", " February", " March", " April", " May", " June", " July", " August",
     " September", " October", " November", " December"],
    [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"],
    ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
]


def _z(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)


def _col_std_at(C, rows, cols):
    """mean over (row,col) pairs of (C[row,col] - colmean) / colstd -- standardised cell vs its column."""
    cm = C.mean(0)
    cs = C.std(0) + 1e-9
    return float(np.mean([(C[r, c] - cm[c]) / cs[c] for r, c in zip(rows, cols)]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=9000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=64, help="# common-token operands for the copy circuit")
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--n-namemovers", type=int, default=5, help="# top copy heads that define the name-mover set")
    p.add_argument("--late-layer", type=int, default=8, help="name-movers are sought in layers >= this")
    p.add_argument("--output", type=Path, default=Path("runs/idiom_library_v2_summary.json"))
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

    # ---- behavioural signatures: prev-token, duplicate-token, induction (v1, unchanged — these work) ----
    pt = np.zeros((nL, H)); ptn = 0
    dup = np.zeros((nL, H)); dupn = 0; dup_base = 0.0
    ind = np.zeros((nL, H)); indn = 0; ind_base = 0.0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_attentions=True)
            Lc = len(c); ca = np.array(c); qi = np.arange(Lc); ptn += Lc - 1
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]
            DM = (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])
            IM = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
            dupn += int(DM.any(1).sum()); indn += int(IM.any(1).sum())
            dq = DM.any(1); iq = IM.any(1)
            if dq.any():
                dup_base += float((DM.sum(1)[dq] / np.maximum(qi[dq], 1)).sum())
            if iq.any():
                ind_base += float((IM.sum(1)[iq] / np.maximum(qi[iq], 1)).sum())
            for L in range(nL):
                aL = o.attentions[L][0].float().numpy()
                pt[L] += np.diagonal(aL, offset=-1, axis1=1, axis2=2).sum(1)
                dup[L] += (aL * DM[None]).sum((1, 2))
                ind[L] += (aL * IM[None]).sum((1, 2))
    prevtok = pt / max(ptn, 1)
    dupv = dup / max(dupn, 1) - dup_base / max(dupn, 1)
    indv = ind / max(indn, 1) - ind_base / max(indn, 1)

    # ---- operand token sets for the OV->unembed copy circuit ----
    wte = tr.wte.weight.detach().numpy().astype(np.float64)        # (vocab, d) tied embed = unembed
    lnf = tr.ln_f.weight.detach().numpy().astype(np.float64)
    Uout_full = wte * lnf                                          # (vocab, d) ln_f-folded unembed rows

    # extended embedding (IOI copy-score convention): the OV-copy circuits read the layer-0 MLP-enriched
    # token embedding, not the raw embedding. e_ext = e + MLP0(ln_2(e)) (GPT-2 gelu_new).
    b0 = tr.h[0]
    g2 = b0.ln_2.weight.detach().numpy().astype(np.float64); b2 = b0.ln_2.bias.detach().numpy().astype(np.float64)
    Wfc = b0.mlp.c_fc.weight.detach().numpy().astype(np.float64); bfc = b0.mlp.c_fc.bias.detach().numpy().astype(np.float64)
    Wpr = b0.mlp.c_proj.weight.detach().numpy().astype(np.float64); bpr = b0.mlp.c_proj.bias.detach().numpy().astype(np.float64)
    eps = float(getattr(cfg, "layer_norm_epsilon", 1e-5))

    def ext_embed(E):
        xn = (E - E.mean(1, keepdims=True)) / np.sqrt(E.var(1, keepdims=True) + eps) * g2 + b2
        hpre = xn @ Wfc + bfc
        hact = 0.5 * hpre * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hpre + 0.044715 * hpre ** 3)))
        return E + hact @ Wpr + bpr

    def single(tokens):
        out = []
        for s in tokens:
            tid = tok(s, add_special_tokens=False)["input_ids"]
            out.append(tid[0] if len(tid) == 1 else None)
        return out

    cand = [tok(c, add_special_tokens=False)["input_ids"][0] for c in COMMON
            if len(tok(c, add_special_tokens=False)["input_ids"]) == 1]
    cand += [t for t, _ in cnt.most_common(300)]
    seen, copytoks = set(), []
    for t in cand:
        if t not in seen and cnt[t] >= args.min_pos:
            seen.add(t); copytoks.append(t)
        if len(copytoks) >= args.n_tokens:
            break
    copytoks = np.array(copytoks)

    # name operands (single-token proper names) for the name-mover copy score
    nameids = np.array([t for t in single(NAMES) if t is not None])

    # succession operand index: consecutive (Y -> succ(Y)) pairs whose BOTH tokens are single-token
    succ_pairs = []
    for seq in ORDINALS:
        sids = single(seq)
        for a, b in zip(sids[:-1], sids[1:]):
            if a is not None and b is not None:
                succ_pairs.append((int(a), int(b)))
    succ_vocab = np.array(sorted({t for pair in succ_pairs for t in pair})) if succ_pairs else np.empty(0, int)

    # ---- copy + succession scores per head (weights only) ----
    # The residual WRITTEN when attending to source token Y is contrib_Y = e_ext[Y] @ OV_h (= OV_h^T e_ext[Y]);
    # its logit effect on token Z is C[Z,Y] = Uout[Z] . contrib_Y. (Source uses the MLP0-extended embedding.)
    Ecopy = ext_embed(wte[copytoks])                               # (nc, d) MLP0-extended attended-token embeds
    Ucopy = Uout_full[copytoks]                                    # (nc, d) folded unembed
    nc = len(copytoks)
    Ename = ext_embed(wte[nameids]); Uname = Uout_full[nameids]    # name operands
    nn = len(nameids)
    copy = np.zeros((nL, H)); succ = np.zeros((nL, H)); copyname = np.zeros((nL, H))
    if succ_vocab.size:
        Esucc = ext_embed(wte[succ_vocab]); Usucc = Uout_full[succ_vocab]
        sidx = {int(t): i for i, t in enumerate(succ_vocab)}
        srows = [sidx[b] for _a, b in succ_pairs]                 # successor row
        scols = [sidx[a] for a, _b in succ_pairs]                 # attended (ordinal) col
    for L in range(nL):
        Wc = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wo = tr.h[L].attn.c_proj.weight.detach().numpy().astype(np.float64)
        Wv = Wc[:, 2 * d:3 * d]
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            OVh = Wv[:, sl] @ Wo[sl, :]                            # (d, d)
            C = Ucopy @ (Ecopy @ OVh).T                            # (nc, nc): C[Z,Y] logit-effect Z|attend Y
            copy[L, h] = _col_std_at(C, range(nc), range(nc))      # standardised DIAGONAL = copy-ness (broad)
            copyname[L, h] = _col_std_at(Uname @ (Ename @ OVh).T, range(nn), range(nn))  # NAME copy
            if succ_vocab.size:
                Cs = Usucc @ (Esucc @ OVh).T
                succ[L, h] = _col_std_at(Cs, srows, scols)         # standardised off-by-one = succession

    heads = [(L, h) for L in range(nL) for h in range(H)]
    layer_of = np.array([L for L, _ in heads])

    def name(i):
        return f"{heads[i][0]}.{heads[i][1]}"

    # ---- name-mover set (top copy) -> S-inhibition via Q-composition into those name-movers ----
    OV = np.zeros((nL * H, d, d)); WQ = np.zeros((nL * H, d, hd))
    for L in range(nL):
        Wc = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wo = tr.h[L].attn.c_proj.weight.detach().numpy().astype(np.float64)
        Wv, Wq = Wc[:, 2 * d:3 * d], Wc[:, :d]
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd); i = L * H + h
            OV[i] = Wv[:, sl] @ Wo[sl, :]; WQ[i] = Wq[:, sl]
    # remove the shared mean-write direction (write_bus_check artifact) before composition
    u = np.linalg.svd(OV.transpose(0, 2, 1).reshape(-1, d), full_matrices=False)[2][0]
    OV = np.einsum("nij,jk->nik", OV, np.eye(d) - np.outer(u, u))
    ovn = np.linalg.norm(OV.reshape(nL * H, -1), axis=1) + 1e-9
    wqn = np.linalg.norm(WQ.reshape(nL * H, -1), axis=1) + 1e-9

    copyf = copy.reshape(-1)
    # name-movers: LATE heads (a known structural fact) ranked by NAME-copy; early heads masked out
    late = layer_of >= args.late_layer
    copyname_idiom = np.where(late, copyname.reshape(-1), np.nan)
    namemovers = [int(i) for i in np.argsort(-np.nan_to_num(copyname_idiom, nan=-1e9))[:args.n_namemovers]]
    # S-inhibition score = mean Q-composition of head A into the name-movers' queries (causal: L_A < L_NM)
    sinh = np.full(nL * H, np.nan)
    for a in range(nL * H):
        cs = [np.linalg.norm(OV[a] @ WQ[b]) / (ovn[a] * wqn[b])
              for b in namemovers if layer_of[a] < layer_of[b]]
        if cs:
            sinh[a] = float(np.mean(cs))
    sinh_filled = np.nan_to_num(sinh, nan=0.0)

    # ---- IOI chain: duplicate-token -> S-inhibition -> name-mover (product of stage comp scores) ----
    dupf = dupv.reshape(-1)
    Dset = [i for i in np.argsort(-dupf)[:4]]
    Sset = [i for i in np.argsort(-sinh_filled)[:4]]

    def qcomp(a, b):
        return float(np.linalg.norm(OV[a] @ WQ[b]) / (ovn[a] * wqn[b]))
    chain = []
    for Dd in Dset:
        for Ss in Sset:
            if layer_of[Dd] >= layer_of[Ss]:
                continue
            for Nn in namemovers:
                if layer_of[Ss] >= layer_of[Nn]:
                    continue
                score = qcomp(Dd, Ss) * qcomp(Ss, Nn)
                chain.append((Dd, Ss, Nn, score))
    chain.sort(key=lambda r: -r[3])

    # ---- assemble idiom rankings ----
    flat = {"prev_token": prevtok.reshape(-1), "duplicate_token": dupf, "induction": indv.reshape(-1),
            "copy_namemover": copyname_idiom, "copy_suppression": -copyf, "succession": succ.reshape(-1),
            "s_inhibition": sinh_filled}

    def topk(v, k=6):
        return [(name(i), float(v[i])) for i in np.argsort(-np.nan_to_num(v, nan=-1e9))[:k]]

    out = {"experiment": "idiom library v2 (copy-score repair + succession + S-inhibition + IOI chain)",
           "model": args.pretrained, "idioms": {}, "name_movers_used": [name(i) for i in namemovers]}
    print(f"{args.pretrained}: idiom library v2 over {nL*H} heads  (name-movers: {[name(i) for i in namemovers]})\n")
    for idiom, v in flat.items():
        members = topk(v)
        out["idioms"][idiom] = members
        print(f"[{idiom:16}] " + ", ".join(f"{n}({s:+.2f})" for n, s in members))

    out["ioi_chain_top"] = [[name(a), name(b), name(c), float(s)] for a, b, c, s in chain[:6]]
    print("\n[ioi_chain]  duplicate -> S-inhibition -> name-mover  (product of Q-composition scores):")
    for a, b, c, s in chain[:6]:
        print(f"  {name(a):>5} -> {name(b):>5} -> {name(c):>5}   {s:.4f}")

    # ---- validation vs literature (GPT-2 small / IOI) ----
    known = {"prev_token": {"4.11"}, "induction": {"5.0", "5.1", "5.5", "6.9", "7.11"},
             "duplicate_token": {"0.1", "0.5", "3.0", "1.5"},
             "copy_namemover": {"9.9", "9.6", "10.0", "10.10", "9.0", "11.3"},
             "copy_suppression": {"10.7", "11.10"},
             "s_inhibition": {"7.3", "7.9", "8.6", "8.10"}}
    print("\n[validation vs literature]")
    val = {}
    for idiom, kset in known.items():
        found = {n for n, _ in out["idioms"][idiom]}
        hit = sorted(kset & found)
        val[idiom] = {"known": sorted(kset), "recovered_in_top6": hit}
        flag = "OK" if hit else "MISS"
        print(f"  [{flag:4}] {idiom:18} known {sorted(kset)} -> recovered {hit if hit else 'none in top-6'}")
    out["validation"] = val

    # ---- per-head idiom assignment (z>1.5) for the disassembler annotation ----
    zf = {k: _z(v) for k, v in flat.items()}
    assign = {}
    for i, (L, h) in enumerate(heads):
        tags = [k for k in flat if np.isfinite(zf[k][i]) and zf[k][i] > 1.5]
        if tags:
            assign[f"{L}.{h}"] = tags
    out["per_head_idioms"] = assign
    n_recovered = sum(1 for v in val.values() if v["recovered_in_top6"])
    out["n_idioms_validated"] = n_recovered
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[assigned] {len(assign)} heads carry >=1 idiom tag; {n_recovered}/{len(known)} idioms validated vs literature")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
