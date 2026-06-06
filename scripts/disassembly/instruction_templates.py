"""Name the dominant instruction templates — what ARE the ~5 shared QK opcodes?

The instruction-tensor test showed the 144-head QK tensor B_h[X,Y] is low-rank (~5 effective
templates vs ~132 random). This reads those templates out: head-mode SVD of the (144, nt, nt) tensor
gives, per component, a TEMPLATE matrix (the shared nt×nt binding pattern) and a head LOADING vector
(which heads use it). For each top template we characterise its shape (diagonal = copy/induction;
columnar = broadcast/sink; off-diagonal-peaked = bind/permutation), name its strongest query→key
feature bindings with token strings, and list the heads that load on it. This turns the quantitative
"low-rank" into the model's actual instruction VOCABULARY. GPT-2.
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--ref-layer", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=40)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--top", type=int, default=6, help="# templates to name")
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/instruction_templates_summary.json"))
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
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx)]
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
    names = [tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t in toks]

    csum = np.zeros((nt, d)); ccnt = np.zeros(nt); gsum = np.zeros(d); gn = 0
    with torch.no_grad():
        for c in chunks:
            hs = tr(input_ids=torch.tensor([c]), output_hidden_states=True).hidden_states[args.ref_layer][0]
            hs = hs.float().numpy()
            pid = np.array([tok2i.get(t, -1) for t in c]); m = pid >= 0
            np.add.at(csum, pid[m], hs[m]); np.add.at(ccnt, pid[m], 1)
            gsum += hs.sum(0); gn += len(c)
    D = csum / np.maximum(ccnt, 1)[:, None] - gsum / gn
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)

    heads, Bs = [], []
    for L in range(nL):
        Wc = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
        Wq, Wk = Wc[:, :d], Wc[:, d:2 * d]
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            B = D @ (Wq[:, sl] @ Wk[:, sl].T / np.sqrt(hd)) @ D.T
            Bs.append(B / (np.linalg.norm(B) + 1e-12)); heads.append((L, h))
    T = np.stack(Bs, 0)
    Hu = T.reshape(T.shape[0], -1)
    U, s, Vt = np.linalg.svd(Hu, full_matrices=False)
    print(f"{args.pretrained}: 144 heads, nt={nt} @ layer {args.ref_layer}; top singular values "
          f"{np.round(s[:6] / s.sum(), 3)}")

    off = ~np.eye(nt, dtype=bool)
    out = {"experiment": "instruction templates", "model": args.pretrained, "n_tokens": nt,
           "singular_fraction": (s[:args.top] / s.sum()).tolist(), "templates": []}
    for i in range(args.top):
        Tm = Vt[i].reshape(nt, nt)
        load = U[:, i]
        # orient sign so the dominant structure reads positive (prefer positive diagonal)
        if np.diag(Tm).mean() < 0 and abs(np.diag(Tm).mean()) > 1e-6:
            Tm = -Tm; load = -load
        elif Tm[np.unravel_index(np.argmax(np.abs(Tm)), Tm.shape)] < 0:
            Tm = -Tm; load = -load
        diag = float(np.diag(Tm).mean() - Tm[off].mean())
        colmass = np.abs(Tm).sum(0); broadcast = float(colmass.max() / colmass.sum())
        rowpk = float((np.abs(Tm) / (np.abs(Tm).sum(1, keepdims=True) + 1e-9)).max(1).mean())
        shape = ("COPY" if diag > 0.5 * np.abs(Tm).max() else
                 "BROADCAST" if broadcast > 3.0 / nt else
                 "BIND" if rowpk > 0.5 else "DIFFUSE")
        Toff = Tm.copy(); np.fill_diagonal(Toff, 0.0)
        binds = []
        for qi, ki in zip(*np.unravel_index(np.argsort(-np.abs(Toff), axis=None)[:5], Toff.shape)):
            binds.append([names[qi], names[ki], round(float(Toff[qi, ki]), 3)])
        diagtok = [names[x] for x in np.argsort(-np.diag(Tm))[:5]]
        toph = [f"{heads[x][0]}.{heads[x][1]}" for x in np.argsort(-np.abs(load))[:6]]
        out["templates"].append({"rank": i, "energy": float(s[i] / s.sum()), "shape": shape,
                                 "diag_score": diag, "broadcast": broadcast, "row_peak": rowpk,
                                 "top_offdiag_binds": binds, "top_diag_tokens": diagtok, "top_heads": toph})
        print(f"\n[template {i}] {shape}  energy {s[i]/s.sum():.3f}  diag {diag:+.3f} bcast {broadcast:.3f} rowpk {rowpk:.2f}")
        print(f"   top heads: {', '.join(toph)}")
        print(f"   diag (self-bind) tokens: {', '.join(diagtok)}")
        print("   top off-diag binds (query->key): " + "; ".join(f"{q!r}->{k!r} {v:+.2f}" for q, k, v in binds))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    shapes = Counter(t["shape"] for t in out["templates"])
    print(f"\n[vocabulary] top-{args.top} template shapes: {dict(shapes)}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
