"""Corpus-robustness check — which disassembly claims are corpus-INVARIANT vs corpus-CONDITIONED?

All the corpus-dependent disassembly ran on tinyshakespeare (VERSE: newline-heavy, archaic). Concern:
are the operators / numbers selected by Shakespeare? This re-runs the corpus-dependent measurements on a
CONTRASTING corpus (WikiText-2, modern encyclopedic PROSE) and compares, head by head:

  (1) behavioral idiom scores  prev / duplicate / induction        -> HEAD-IDENTITY stability (Spearman + top-k)
  (2) attention-bucket split   self/sink/prev/structural/local/lr  -> COVERAGE-NUMBER shift (verse vs prose)
  (3) QK opcode legibility z   B_h vs empirical attn (SHARED ops)  -> OPCODE stability

Predictions: head identities are weight-grounded -> stable across corpora (the literature heads were found
on OpenWebText, recovering them on both is cross-corpus replication). Bucket % are corpus-shaped -> verse's
newline structure should inflate structural/sink and deflate long-range vs prose. A SHARED operand set
(COMMON single-tokens, same for both) isolates opcode legibility from operand-selection drift. One forward
pass per corpus. GPT-2, CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lm_bundle import COMMON  # noqa: E402

CORPORA = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "wikitext": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
}
STRUCT_STR = {".", ",", ";", ":", "!", "?", "Ċ", "ĊĊ", "\n"}
BUCKETS = ["self", "sink", "prev", "structural", "local", "long_range"]


def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def _fetch(url, n_chars):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")[:n_chars]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--corpora", default="shakespeare,wikitext")
    p.add_argument("--max-tokens", type=int, default=20000, help="per corpus; needs to be large enough that "
                   "enough token types clear --op-min-freq in BOTH corpora (8k -> only ~9 shared operands)")
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-operands", type=int, default=80)
    p.add_argument("--n-chars", type=int, default=600000)
    p.add_argument("--op-min-freq", type=int, default=20, help="min per-corpus token freq to be a shared operand")
    p.add_argument("--min-count", type=int, default=12, help="min ordered (i->j) pairs to score an opcode cell")
    p.add_argument("--n-perm", type=int, default=15)
    p.add_argument("--top-k", type=int, default=5, help="top-k heads for the identity-overlap check")
    p.add_argument("--local-max", type=int, default=8)
    p.add_argument("--output", type=Path, default=Path("runs/corpus_robustness_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; H = cfg.n_head; hd = d // H; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    corpora = args.corpora.split(",")

    # ---- tokenize both corpora up front (also drives the shared operand set) ----
    from collections import Counter
    data = {}
    for name in corpora:
        ids = tok(_fetch(CORPORA[name], args.n_chars))["input_ids"][: args.max_tokens]
        chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
        data[name] = {"chunks": chunks, "cnt": Counter(t for c in chunks for t in c)}

    # ---- SHARED operand set: tokens FREQUENT IN BOTH corpora (apples-to-apples; corpus-agnostic).
    # COMMON single-tokens seed it, then fill by min cross-corpus frequency. ----
    cnts = [data[n]["cnt"] for n in corpora]
    common_ids = {tok(c, add_special_tokens=False)["input_ids"][0] for c in COMMON
                  if len(tok(c, add_special_tokens=False)["input_ids"]) == 1}
    shared = [t for t in set().union(*[set(c) for c in cnts])
              if all(cnts_i.get(t, 0) >= args.op_min_freq for cnts_i in cnts)]
    shared.sort(key=lambda t: (t not in common_ids, -min(c.get(t, 0) for c in cnts)))  # COMMON first, then freq
    ops = shared[: args.n_operands]
    nt = len(ops)
    op2i = {t: i for i, t in enumerate(ops)}
    print(f"{args.pretrained}: {nt} shared operands (freq>={args.op_min_freq} in all corpora), corpora={corpora}")

    rng = np.random.default_rng(0)
    perms = [rng.permutation(nt) for _ in range(args.n_perm)]
    offmask = ~np.eye(nt, dtype=bool)

    def measure(name):
        chunks = data[name]["chunks"]
        # behavioral
        pt = np.zeros((nL, H)); ptn = 0
        dup = np.zeros((nL, H)); dupn = 0; dupb = 0.0
        ind = np.zeros((nL, H)); indn = 0; indb = 0.0
        # buckets
        bacc = np.zeros((nL, H, len(BUCKETS))); btot = np.zeros((nL, H))
        # opcode: centroids + empirical attention over operand pairs
        cen = np.zeros((nL, nt, d)); cencnt = np.zeros(nt)
        gmean = np.zeros((nL, d)); gcnt = 0
        asum = np.zeros((nL, H, nt * nt)); acnt = np.zeros(nt * nt)
        with torch.no_grad():
            for c in chunks:
                o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
                Lc = len(c); ca = np.array(c); qi = np.arange(Lc)
                toks = tok.convert_ids_to_tokens(c)
                struct = np.array([any(s in t for s in STRUCT_STR) for t in toks])
                pos = np.array([op2i.get(t, -1) for t in c]); vmask = pos >= 0
                prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]
                DM = (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])
                IM = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
                ptn += Lc - 1; dupn += int(DM.any(1).sum()); indn += int(IM.any(1).sum())
                dq = DM.any(1); iq = IM.any(1)
                if dq.any():
                    dupb += float((DM.sum(1)[dq] / np.maximum(qi[dq], 1)).sum())
                if iq.any():
                    indb += float((IM.sum(1)[iq] / np.maximum(qi[iq], 1)).sum())
                # bucket priority masks (key index S over columns, query T over rows)
                S, T = np.meshgrid(qi, qi); delta = T - S
                bid = np.full((Lc, Lc), 5, dtype=int)
                bid[(delta >= 2) & (delta <= args.local_max)] = 4
                bid[struct[None, :].repeat(Lc, 0)] = 3
                bid[delta == 1] = 2
                bid[S == 0] = 1
                bid[S == T] = 0
                bid[S > T] = -1
                # opcode empirical pairs
                ti, si = np.tril_indices(Lc, k=-1)
                mm = (pos[ti] >= 0) & (pos[si] >= 0)
                flat = pos[ti[mm]] * nt + pos[si[mm]]
                np.add.at(acnt, flat, 1.0)
                gmean += np.stack([o.hidden_states[L][0].float().numpy().sum(0) for L in range(nL)]); gcnt += Lc
                cencnt += np.bincount(pos[vmask], minlength=nt)
                for L in range(nL):
                    hs = o.hidden_states[L][0].float().numpy()
                    np.add.at(cen[L], pos[vmask], hs[vmask])
                    aL = o.attentions[L][0].float().numpy()
                    pt[L] += np.diagonal(aL, offset=-1, axis1=1, axis2=2).sum(1)
                    dup[L] += (aL * DM[None]).sum((1, 2)); ind[L] += (aL * IM[None]).sum((1, 2))
                    btot[L] += aL.sum((1, 2))
                    for b in range(len(BUCKETS)):
                        bacc[L, :, b] += (aL * (bid == b)[None]).sum((1, 2))
                    for h in range(H):
                        np.add.at(asum[L, h], flat, aL[h][ti[mm], si[mm]])
        prevv = (pt / max(ptn, 1)).reshape(-1)
        dupv = (dup / max(dupn, 1) - dupb / max(dupn, 1)).reshape(-1)
        indv = (ind / max(indn, 1) - indb / max(indn, 1)).reshape(-1)
        frac = bacc / np.maximum(btot, 1e-9)[:, :, None]
        bmean = {b: float(np.mean(frac[:, :, i])) for i, b in enumerate(BUCKETS)}
        # opcode legibility per head (shared operands)
        cend = cen / np.maximum(cencnt, 1)[None, :, None]; gm = gmean / max(gcnt, 1)
        cnt2 = acnt.reshape(nt, nt); supp = (cnt2 >= args.min_count) & offmask
        legz = np.full(nL * H, np.nan)
        for L in range(nL):
            ln_w = tr.h[L].ln_1.weight.detach().numpy().astype(np.float64)
            D = (cend[L] - gm[L]) * ln_w; D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
            W = tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)
            Wq, Wk = W[:, :d], W[:, d:2 * d]
            for h in range(H):
                Mh = Wq[:, h * hd:(h + 1) * hd] @ Wk[:, h * hd:(h + 1) * hd].T / np.sqrt(hd)
                B = D @ Mh @ D.T
                A = asum[L, h].reshape(nt, nt) / np.maximum(cnt2, 1)
                if supp.sum() < 4:
                    continue
                leg = _spearman(B[supp], A[supp])
                null = [_spearman(B[np.ix_(pp, pp)][supp], A[supp]) for pp in perms]
                legz[L * H + h] = (leg - np.nanmean(null)) / (np.nanstd(null) + 1e-9)
        print(f"  [{name}] {len(chunks)} chunks  supported opcode cells {int(supp.sum())}/{nt*(nt-1)}")
        return {"prev": prevv, "dup": dupv, "ind": indv, "buckets": bmean, "legz": legz}

    R = {name: measure(name) for name in corpora}
    a, b = corpora[0], corpora[1]
    heads = [f"{L}.{h}" for L in range(nL) for h in range(H)]

    def topk(v, k):
        return {heads[i] for i in np.argsort(-np.nan_to_num(v, nan=-1e9))[:k]}

    # ---- (1) head-identity stability ----
    print(f"\n[1] HEAD-IDENTITY STABILITY  ({a} vs {b})")
    print(f"  {'signature':>11} {'Spearman':>9} {'top-k overlap':>14}  top-{args.top_k} heads (both)")
    ident = {}
    for sig in ["prev", "dup", "ind"]:
        rho = _spearman(R[a][sig], R[b][sig])
        ta, tb = topk(R[a][sig], args.top_k), topk(R[b][sig], args.top_k)
        ov = sorted(ta & tb)
        ident[sig] = {"spearman": rho, "overlap": len(ov), "shared_top": ov}
        print(f"  {sig:>11} {rho:>9.2f} {len(ov):>10}/{args.top_k}    {ov}")
    leg_rho = _spearman(R[a]["legz"], R[b]["legz"])
    print(f"  {'opcode_legz':>11} {leg_rho:>9.2f}  (per-head QK legibility, shared operands)")

    # ---- (2) coverage-number shift ----
    print(f"\n[2] COVERAGE-NUMBER SHIFT ({a} vs {b}, attention-bucket mean per head)")
    print(f"  {'bucket':>11} {a:>13} {b:>13} {'Δ':>8}")
    for bk in BUCKETS:
        va, vb = R[a]["buckets"][bk], R[b]["buckets"][bk]
        print(f"  {bk:>11} {va:>12.1%} {vb:>12.1%} {vb-va:>+8.1%}")

    out = {"experiment": "corpus robustness (shakespeare vs wikitext)", "model": args.pretrained,
           "corpora": corpora, "identity_stability": ident, "opcode_legz_spearman": leg_rho,
           "buckets": {name: R[name]["buckets"] for name in corpora}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    rho_id = float(np.nanmean([ident[s]["spearman"] for s in ["prev", "dup", "ind"]] + [leg_rho]))
    lr_shift = abs(R[a]["buckets"]["long_range"] - R[b]["buckets"]["long_range"])
    st_shift = abs(R[a]["buckets"]["structural"] - R[b]["buckets"]["structural"])
    print(f"\n[verdict] head identities {'CORPUS-INVARIANT' if rho_id > 0.6 else 'corpus-sensitive'} "
          f"(mean Spearman {rho_id:.2f}); coverage numbers "
          f"{'CORPUS-CONDITIONED' if max(lr_shift, st_shift) > 0.03 else 'stable'} "
          f"(long-range Δ{R[b]['buckets']['long_range']-R[a]['buckets']['long_range']:+.1%}, "
          f"structural Δ{R[b]['buckets']['structural']-R[a]['buckets']['structural']:+.1%})")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
