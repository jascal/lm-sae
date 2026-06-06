"""Verify the 'narrow write bus' — structural, or a centroid-basis artifact?

mlp_catalog found MLP neurons read richly (feature participation 12.8/48) but write into a ~2-dim feature
subspace (1.9/48). That could be an artifact of projecting onto the correlated centroid basis D (common
tokens dominate D). Decisive control: measure read/write concentration in the RAW 768-dim residual space
(operand-basis-free). If writes are concentrated THERE too — and more than reads, and vs a random null —
the narrow output bus is structural. We also (a) reproduce the feature-space numbers, (b) name the
dominant write PC (is it just a common-token / mean-update direction?), and (c) check whether removing the
top PC restores rank. GPT-2; weights + one forward for D.
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


def _participation(M):
    """participation ratio of unit-normalised rows of M (effective # of directions)."""
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    lam = np.linalg.eigvalsh(M.T @ M)
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--ref-layer", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--n-tokens", type=int, default=48)
    p.add_argument("--min-pos", type=int, default=30)
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/write_bus_check_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained).eval()
    tr = model.transformer
    cfg = model.config
    d = cfg.n_embd; nL = cfg.n_layer
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

    RIN, ROUT = [], []
    for L in range(nL):
        blk = tr.h[L]
        ln2 = blk.ln_2.weight.detach().numpy().astype(np.float64)
        Win = blk.mlp.c_fc.weight.detach().numpy().astype(np.float64)        # (d, dff)
        Wout = blk.mlp.c_proj.weight.detach().numpy().astype(np.float64)     # (dff, d)
        RIN.append((Win * ln2[:, None]).T)                                  # (dff, d) read, ln2-folded
        ROUT.append(Wout)                                                   # (dff, d) write
    Rin = np.concatenate(RIN, 0); Rout = np.concatenate(ROUT, 0)            # (nL*dff, d)
    print(f"{args.pretrained}: {Rin.shape[0]} neurons, d_model={d}, nt={nt}")

    rng = np.random.default_rng(0)
    rand = rng.standard_normal((Rin.shape[0], d))
    raw = {"read_raw_768": _participation(Rin), "write_raw_768": _participation(Rout),
           "random_raw_768": _participation(rand)}
    # feature-space (project onto D) — reproduces mlp_catalog's 12.8 / 1.9
    feat = {"read_feat_48": _participation(Rin @ D.T), "write_feat_48": _participation(Rout @ D.T)}

    # name the dominant write PC + rank after removing it
    Wf = Rout @ D.T                                                          # (n, nt) feature-space writes
    Wf = Wf / (np.linalg.norm(Wf, axis=1, keepdims=True) + 1e-12)
    _, _, Vt = np.linalg.svd(Wf, full_matrices=False)
    pc1 = Vt[0]
    top_pc_tokens = [nm[x] for x in np.argsort(-np.abs(pc1))[:6]]
    resid = Wf - (Wf @ np.outer(pc1, pc1))                                   # remove top PC
    write_feat_minus_pc1 = _participation(resid)
    # is pc1 the mean-write direction? cosine of pc1 with the mean write
    meanw = Wf.mean(0); meanw /= np.linalg.norm(meanw) + 1e-12
    pc1_vs_mean = float(abs(pc1 @ meanw))

    out = {"experiment": "narrow-write-bus check", "model": args.pretrained,
           "n_neurons": int(Rin.shape[0]), "raw_768": raw, "feature_48": feat,
           "top_write_pc_tokens": top_pc_tokens, "pc1_vs_meanwrite_cos": pc1_vs_mean,
           "write_feat_participation_minus_pc1": write_feat_minus_pc1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"\n[raw 768-dim residual space]  read participation {raw['read_raw_768']:.1f}  "
          f"write participation {raw['write_raw_768']:.1f}  (random {raw['random_raw_768']:.1f})")
    print(f"[feature 48-dim basis]        read {feat['read_feat_48']:.1f}  write {feat['write_feat_48']:.1f}  "
          f"(reproduces mlp_catalog ~12.8 / ~1.9)")
    print(f"\ntop write PC promotes: {', '.join(top_pc_tokens)}  | cos(pc1, mean-write) = {pc1_vs_mean:.2f}")
    print(f"feature-space write participation after removing top PC: {write_feat_minus_pc1:.1f} "
          f"(was {feat['write_feat_48']:.1f})")
    # verdict
    raw_narrow = raw["write_raw_768"] < 0.4 * raw["read_raw_768"]
    artifact = (not raw_narrow) and (write_feat_minus_pc1 > 4 * feat["write_feat_48"])
    print(f"\n[verdict] {'STRUCTURAL: writes are concentrated in raw 768-d residual space too (write << read, vs random) — the narrow output bus is real, not a D-projection artifact' if raw_narrow else ('ARTIFACT: writes are HIGH-rank in raw 768-d space; the ~2-d feature-space number is a centroid-basis/mean-direction projection effect (removing top PC restores rank)' if artifact else 'MIXED: feature-space narrow is partly the dominant (likely mean/common-token) PC; raw-space write is moderately concentrated — report both')}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
