"""QK feature-coordinate probe: read attention's BINDING (inference rules) in feature coords.

Residual-feature probes (single/within-pair/cross-pair) couldn't read induction because the
binding lives in the ATTENTION, not the residual. This reads it directly. A head's QK matrix
M_h = W_Q^h W_K^h.T gives the pre-softmax score h_t.M_h.h_s between a query at t and a key at
s. Expand h in token-feature directions d_X (empirical residual centroid of token X): the
score decomposes into B_h[X,X'] = d_X.M_h.d_X' -- WHICH query-feature binds WHICH key-feature.
The same-token DIAGONAL d_X.M_h.d_X is "attend to the same token" -- the copy/induction rule.

We (1) read same-token binding per head from the WEIGHTS (no behavior), then (2) VALIDATE it
against each head's actual same-token attention mass on real text. If the weight-read binding
predicts the behavior across heads, the inference rule IS legible in feature coordinates.
GPT-2; pick a layer with induction/duplicate heads.
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
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--layer", type=int, default=5, help="block whose attention to read")
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=40, help="# token-feature directions")
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/qk_feature_probe_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx)]
    all_ids = [j for c in chunks for j in c]

    # residual entering block L (= hidden_states[L]) + attention patterns of block L
    Hs, atts = [], []
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            Hs.append(o.hidden_states[args.layer][0].float().numpy())
            atts.append(o.attentions[args.layer][0].float().numpy())   # (H, seq, seq)
    Xr = np.concatenate(Hs, 0).astype(np.float64)
    print(f"{args.pretrained} block {args.layer}: X={Xr.shape}  heads={H} head_dim={hd}")

    # ---- token-feature directions: empirical residual centroid per common token ----
    blk = tr.h[args.layer]
    ln_w = blk.ln_1.weight.detach().numpy().astype(np.float64)        # fold ln_1 scale
    gmean = Xr.mean(0)
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
    D = []
    for t in toks:
        m = np.array([1 if j == t else 0 for j in all_ids], bool)
        v = (Xr[m].mean(0) - gmean) * ln_w
        D.append(v / (np.linalg.norm(v) + 1e-9))
    D = np.stack(D, 0)                                                # (n_tok, d)
    print(f"token-feature directions: {len(toks)}")

    # ---- QK matrices per head; B_h[X,X'] = d_X . M_h . d_X' ; same-token diagonal ----
    W = blk.attn.c_attn.weight.detach().numpy().astype(np.float64)    # (d, 3d) Conv1D
    Wq, Wk = W[:, :d], W[:, d:2 * d]
    off = np.eye(len(toks), dtype=bool)
    qk_binding = []
    for h in range(H):
        Mh = Wq[:, h * hd:(h + 1) * hd] @ Wk[:, h * hd:(h + 1) * hd].T / np.sqrt(hd)
        B = D @ Mh @ D.T
        same = float(np.diag(B).mean()); other = float(B[~off].mean())
        qk_binding.append(same - other)                              # same-token bind vs off-diagonal

    # ---- behavioral same-token attention per head (mass on earlier same-token keys) ----
    beh = np.zeros(H)
    nq = 0
    for ci, c in enumerate(chunks):
        A = atts[ci]                                                 # (H, L, L)
        L = len(c)
        same_mask = np.zeros((L, L), bool)
        for t in range(1, L):
            for s in range(t):
                if c[s] == c[t]:
                    same_mask[t, s] = True
        m = same_mask.any(1)                                         # query positions with a same-token earlier
        if m.sum() == 0:
            continue
        beh += (A * same_mask[None]).sum(2)[:, m].sum(1)
        nq += int(m.sum())
    beh = beh / max(nq, 1)

    order = np.argsort(-np.array(qk_binding))
    rho = _spearman(qk_binding, beh)
    out = {"experiment": "QK feature-coordinate probe", "model": args.pretrained, "layer": args.layer,
           "per_head": [{"head": int(h), "qk_same_token_binding": qk_binding[h],
                         "behavioral_same_token_attn": float(beh[h])} for h in range(H)],
           "spearman_binding_vs_behavior": rho,
           "top_head_by_qk": int(order[0])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n{'head':>4} {'QK same-tok binding':>20} {'behavioral same-tok attn':>26}")
    for h in order:
        print(f"{h:>4} {qk_binding[h]:>20.3f} {beh[h]:>26.3f}")
    print(f"\n[verdict] Spearman(QK-feature binding, same-token attention) = {rho:+.3f}  "
          f"-> {'the inference rule (token-match) IS legible in feature coords' if rho > 0.5 else 'weak/no correspondence'}")
    print(f"   top head by QK-feature same-token binding: head {order[0]} "
          f"(behavioral same-token attn {beh[order[0]]:.3f}, vs mean {beh.mean():.3f})")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
