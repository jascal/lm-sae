"""MLP neuron catalog — the COMPUTE ops, and is the compute vocabulary low-rank too?

Attention MOVES operands; the MLP COMPUTES on them. We've catalogued attention (QK/OV); this reads the
other instruction class. Each GPT-2 MLP neuron i is a read-features -> GELU gate -> write-features
instruction. In the shared token-feature basis D (centroids at a ref layer):
  read   in_i[X] = (ln2-folded d_X) . W_in[:,i]    -- which features FIRE neuron i
  write  out_i[Z] = W_out[i,:] . d_Z               -- which features it EMITS
So neuron i implements "when features {X..} present, write features {Z..}". We (1) catalogue the most
salient neurons (top read->write feature bindings), and (2) test the decisive question: does the COMPUTE
vocabulary compress to a few reused templates (like attention's ~5), or is it high-rank/idiosyncratic?
— via the neuron-mode participation ratio of the transform tensor T_i = out_i (x) in_i vs a random null,
plus the read/write feature-mode effective dimensionality. GPT-2; weights + one forward for the basis.
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


def _participation(s):
    e = np.asarray(s, float) ** 2
    return float(e.sum() ** 2 / (e ** 2).sum())


def _energy_rank(s, frac=0.90):
    e = np.asarray(s, float) ** 2
    return int(np.searchsorted(np.cumsum(e) / e.sum(), frac) + 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--ref-layer", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=48)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--n-catalog", type=int, default=20, help="# salient neurons to name")
    p.add_argument("--n-rank", type=int, default=6000, help="# top-salient neurons for the rank test")
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/mlp_catalog_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained).eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; nL = cfg.n_layer; dff = cfg.n_inner or 4 * d
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
    cand += [t for t, _ in cnt.most_common(250)]
    seen, toks = set(), []
    for t in cand:
        if t not in seen and cnt[t] >= args.min_pos:
            seen.add(t); toks.append(t)
        if len(toks) >= args.n_tokens:
            break
    nt = len(toks); tok2i = {t: i for i, t in enumerate(toks)}
    nm = [tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t in toks]

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
    print(f"{args.pretrained}: {nL} layers x {dff} neurons = {nL*dff}; basis nt={nt} @ layer {args.ref_layer}")

    # ---- read/write feature patterns per neuron, all layers ----
    READ, WRITE, who = [], [], []
    for L in range(nL):
        blk = tr.h[L]
        ln2 = blk.ln_2.weight.detach().numpy().astype(np.float64)
        Win = blk.mlp.c_fc.weight.detach().numpy().astype(np.float64)        # (d, dff)
        Wout = blk.mlp.c_proj.weight.detach().numpy().astype(np.float64)     # (dff, d)
        inm = (D * ln2) @ Win                                               # (nt, dff) read
        outm = Wout @ D.T                                                   # (dff, nt) write
        READ.append(inm.T); WRITE.append(outm)                              # (dff, nt) each
        who += [(L, i) for i in range(dff)]
    R = np.concatenate(READ, 0); W = np.concatenate(WRITE, 0)               # (nL*dff, nt)
    who = np.array(who)

    # ---- catalogue the most salient (peak-read x peak-write) neurons ----
    sal = np.abs(R).max(1) * np.abs(W).max(1)
    out = {"experiment": "MLP neuron catalog", "model": args.pretrained, "n_neurons": int(R.shape[0]),
           "n_tokens": nt, "catalog": []}
    print(f"\n[catalog] top {args.n_catalog} salient compute neurons (read-features -> write-features):")
    for idx in np.argsort(-sal)[: args.n_catalog]:
        L, i = who[idx]
        rd = [nm[x] for x in np.argsort(-np.abs(R[idx]))[:3]]
        wr = [nm[x] for x in np.argsort(-np.abs(W[idx]))[:3]]
        out["catalog"].append({"neuron": [int(L), int(i)], "reads": rd, "writes": wr,
                               "salience": float(sal[idx])})
        print(f"  L{L}.n{i:<4} reads {{{', '.join(repr(x) for x in rd)}}} -> writes {{{', '.join(repr(x) for x in wr)}}}")

    # ---- low-rank test: does the compute vocabulary compress? ----
    top = np.argsort(-sal)[: args.n_rank]
    Rt = R[top] / (np.linalg.norm(R[top], axis=1, keepdims=True) + 1e-12)
    Wt = W[top] / (np.linalg.norm(W[top], axis=1, keepdims=True) + 1e-12)
    # per-neuron transform T_i = out_i (x) in_i, unit-Frobenius, stacked -> neuron-mode rank
    T = (Wt[:, :, None] * Rt[:, None, :]).reshape(len(top), -1)             # (n, nt*nt), already ~unit
    sT = np.linalg.svd(T, full_matrices=False)[1]
    rng = np.random.default_rng(0)
    Rr = rng.standard_normal(Rt.shape); Rr /= np.linalg.norm(Rr, axis=1, keepdims=True)
    Wr = rng.standard_normal(Wt.shape); Wr /= np.linalg.norm(Wr, axis=1, keepdims=True)
    Tr = (Wr[:, :, None] * Rr[:, None, :]).reshape(len(top), -1)
    sTr = np.linalg.svd(Tr, full_matrices=False)[1]
    # read/write feature-mode effective dimensionality
    sR = np.linalg.svd(Rt, full_matrices=False)[1]
    sW = np.linalg.svd(Wt, full_matrices=False)[1]
    out["transform_participation"] = _participation(sT)
    out["transform_participation_random"] = _participation(sTr)
    out["transform_rank90"] = _energy_rank(sT)
    out["read_participation"] = _participation(sR)
    out["write_participation"] = _participation(sW)
    out["n_rank_test"] = int(len(top))

    print(f"\n[low-rank test] {len(top)} salient neurons, transform tensor T_i = out_i (x) in_i:")
    print(f"  COMPUTE participation ratio: REAL {out['transform_participation']:.1f} vs RANDOM "
          f"{out['transform_participation_random']:.1f}  (90%-rank {out['transform_rank90']})")
    print(f"  read-feature participation {out['read_participation']:.1f} / {nt}; "
          f"write-feature participation {out['write_participation']:.1f} / {nt}")
    ratio = out["transform_participation"] / max(out["transform_participation_random"], 1e-9)
    print(f"\n[verdict] MLP compute vocabulary participation {out['transform_participation']:.1f} vs random "
          f"{out['transform_participation_random']:.1f} ({ratio:.0%}) -> "
          f"{'LOW-RANK like attention (compute compresses to few reused templates)' if ratio < 0.5 else 'HIGHER-RANK than attention — compute is more idiosyncratic than routing'}")
    print(f"   (attention QK was ~5 templates vs ~132 random; compare the participation numbers)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
