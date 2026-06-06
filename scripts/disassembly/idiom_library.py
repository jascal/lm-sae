"""Idiom library — recognise the canonical multi-component circuit idioms in GPT-2.

The disassembly lists single-component instructions; a decompiler recognises IDIOMS (composed patterns).
This scans GPT-2 for a library of known attention-head idioms by their signatures and reports which heads
implement each, then validates against the literature's known heads.

  previous-token   attends to t-1 (the induction feeder)                         [behaviour: A[t,t-1]]
  duplicate-token  attends to an earlier occurrence of the SAME token            [token[s]==token[t], s<t]
  induction        attends to the token AFTER the prev occurrence of cur token   [token[s-1]==token[t]]
  copy / name-mover OV copies the attended token to its own logit (positive)     [copy score > 0, late]
  copy-suppression OV writes the attended token NEGATIVELY (anti-copy)           [copy score < 0]

Copy score (Elhage/McDougall): C_h[Y,Z] = e_Y . (W_V^h W_O^h) . (lnf * e_Z) over common tokens; diagonal
dominance > 0 = copies the attended token to output, << 0 = suppresses it (GPT-2's famous 10.7). GPT-2;
one forward for the behavioural signatures, weights for the copy score.
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
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/idiom_library_summary.json"))
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
    cand += [t for t, _ in cnt.most_common(200)]
    seen, toks = set(), []
    for t in cand:
        if t not in seen and cnt[t] >= args.min_pos:
            seen.add(t); toks.append(t)
        if len(toks) >= args.n_tokens:
            break
    nt = len(toks)

    # ---- behavioural signatures: prev-token, duplicate-token, induction ----
    pt = np.zeros((nL, H)); ptn = 0
    dup = np.zeros((nL, H)); dupn = 0; dup_base = 0.0
    ind = np.zeros((nL, H)); indn = 0; ind_base = 0.0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_attentions=True)
            Lc = len(c); ca = np.array(c); qi = np.arange(Lc); ptn += Lc - 1
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]
            DM = (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])          # same-token earlier
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

    # ---- copy score (weights, FULL VOCAB): eigenvalue-positivity of the OV->unembed circuit ----
    # The token-space copying matrix W_U OV_h W_E (vocab x vocab) has nonzero eigenvalues equal to those of
    # OV_h @ (W_E^T W_U) (d x d) by the cyclic identity; tied GPT-2 -> W_E=W_U=wte. score = sum(real eig)/
    # sum|real eig| in [-1,+1]: +1 = pure copy/name-mover, -1 = pure copy-suppression (e.g. GPT-2 10.7).
    wte = tr.wte.weight.detach().numpy().astype(np.float64)                      # (vocab, d), tied
    lnf = tr.ln_f.weight.detach().numpy().astype(np.float64)
    G = wte.T @ (wte * lnf)                                                      # (d, d) embed-unembed gram (lnf-folded)
    copy = np.zeros((nL, H))
    for L in range(nL):
        Wc = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wo = tr.h[L].attn.c_proj.weight.detach().numpy().astype(np.float64)
        Wv = Wc[:, 2 * d:3 * d]
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            ev = np.linalg.eigvals((Wv[:, sl] @ Wo[sl, :]) @ G).real             # OV_h @ G eigenvalues
            copy[L, h] = ev[ev > 0].sum() / (np.abs(ev).sum() + 1e-9)            # positive-eigenvalue mass in [0,1]
    copy = copy - 0.5                                                            # center: >0 copy, <0 suppress

    heads = [(L, h) for L in range(nL) for h in range(H)]
    flat = {"prev_token": prevtok.reshape(-1), "duplicate_token": dupv.reshape(-1),
            "induction": indv.reshape(-1), "copy_namemover": copy.reshape(-1),
            "copy_suppression": -copy.reshape(-1)}

    def topk(v, k=6):
        return [(f"{heads[i][0]}.{heads[i][1]}", float(v[i])) for i in np.argsort(-v)[:k]]

    out = {"experiment": "idiom library", "model": args.pretrained, "idioms": {}}
    print(f"{args.pretrained}: idiom library over {nL*H} heads\n")
    for idiom, v in flat.items():
        members = topk(v)
        out["idioms"][idiom] = members
        print(f"[{idiom}]  " + ", ".join(f"{n}({s:+.2f})" for n, s in members))

    # validation vs literature (GPT-2): prev-tok 4.11; induction 5.x/6.9; copy-suppression 10.7;
    # name-movers 9.6/9.9/10.0
    known = {"prev_token": {"4.11"}, "induction": {"5.0", "5.5", "6.9", "7.11", "5.1"},
             "copy_suppression": {"10.7"}, "copy_namemover": {"9.9", "9.6", "10.0", "10.10", "9.0"}}
    print("\n[validation vs literature]")
    val = {}
    for idiom, kset in known.items():
        found = {n for n, _ in out["idioms"][idiom]}
        hit = sorted(kset & found)
        val[idiom] = {"known": sorted(kset), "recovered_in_top6": hit}
        print(f"  {idiom:18} known {sorted(kset)}  -> recovered {hit if hit else 'none in top-6'}")
    out["validation"] = val

    # per-head idiom assignment (z>1.5) for the disassembler annotation
    zf = {k: _z(v) for k, v in flat.items()}
    assign = {}
    for i, (L, h) in enumerate(heads):
        tags = [k for k in flat if zf[k][i] > 1.5]
        if tags:
            assign[f"{L}.{h}"] = tags
    out["per_head_idioms"] = assign
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[assigned] {len(assign)} heads carry >=1 idiom tag")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
