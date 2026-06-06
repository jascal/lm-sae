"""Gemma-2 QK content-opcode legibility across DEPTH — sweep layers with their Gemma Scope SAEs.

disassemble_gemma.py decoded one layer (12). This sweeps a spread of layers {0,3,6,9,12,18,21,24}, each with
its own Gemma Scope SAE (width-16k JumpReLU), and asks: is the QK content-binding legible in SAE-feature
coords at EVERY depth, or only some? Per layer: diverse content operands (stoplist + top-token dedupe),
B_h = D @ (W_Q^h^T W_K^{kv}/sqrt(qscale)) @ D.T (GQA + RMSNorm-fold + unrotated content-QK), validated vs
empirical attention over operand pairs (label-permuted null). Two model passes total (hidden states for all
layers; then attentions). Runs on cuda (bf16). Writes the legibility-vs-depth profile.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

SCOPE_GLOB = ("/home/allans/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res/"
              "**/layer_{L}/width_16k/average_l0_*/params.npz")
STOP = {"the", "of", "and", "a", "to", "in", "that", "is", "was", "for", "it", "as", "with", "on", "by",
        "at", "an", "be", "or", "are", "from", "this", "his", "her", "he", "she", "they", "i", "you",
        "we", "but", "not", "have", "had", "s", "t", ""}


def _struct(s):
    s = s.replace("Ġ", "").replace("▁", "").replace("Ċ", "\n").strip()
    return s == "" or s in {"<0x0A>"} or all(not ch.isalnum() for ch in s)


def _spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--layers", default="0,3,6,9,12,18,21,24")
    p.add_argument("--n-operands", type=int, default=48)
    p.add_argument("--n-cand", type=int, default=400, help="top-mass features scanned for content operands")
    p.add_argument("--max-tokens", type=int, default=5000)
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--min-count", type=int, default=8)
    p.add_argument("--n-perm", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--corpus", default="wikitext")
    p.add_argument("--output", type=Path, default=Path("runs/gemma_layer_sweep_summary.json"))
    p.add_argument("--txt", type=Path, default=Path("runs/gemma_layer_sweep.txt"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    cfg = model.config
    H, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    qscale = float(getattr(cfg, "query_pre_attn_scalar", hd)) ** 0.5
    layers = [int(x) for x in args.layers.split(",")]
    import urllib.request
    CORPORA = {"shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
               "wikitext": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"}
    txt = urllib.request.urlopen(urllib.request.Request(CORPORA.get(args.corpus, args.corpus),
                                 headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:400000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    off = np.cumsum([0] + [len(c) for c in chunks]); all_ids = np.array([j for c in chunks for j in c]); P = len(all_ids)
    print(f"{args.model}: sweep layers {layers}; {len(chunks)} chunks, {P} tokens")

    # ---- pass 1: hidden_states for all sweep layers ----
    Hs = {L: np.zeros((P, cfg.hidden_size), np.float32) for L in layers}
    with torch.no_grad():
        for ci, c in enumerate(chunks):
            hsall = model(input_ids=torch.tensor([c], device=dev), output_hidden_states=True).hidden_states
            for L in layers:
                Hs[L][off[ci]:off[ci + 1]] = hsall[L][0].float().cpu().numpy()

    # ---- per layer: SAE -> diverse content operands -> pos_op + RMSNorm-folded directions ----
    operands = {}
    for L in layers:
        sae = np.load(glob.glob(SCOPE_GLOB.format(L=L), recursive=True)[0])
        Wenc = torch.tensor(sae["W_enc"], device=dev, dtype=torch.float32); benc = torch.tensor(sae["b_enc"], device=dev, dtype=torch.float32)
        bdec = torch.tensor(sae["b_dec"], device=dev, dtype=torch.float32); thr = torch.tensor(sae["threshold"], device=dev, dtype=torch.float32)
        Wdec = sae["W_dec"].astype(np.float64)
        Xc = torch.tensor(Hs[L], device=dev) - bdec
        mass = torch.zeros(Wenc.shape[1], device=dev)
        for i in range(0, P, 4096):
            pre = Xc[i:i + 4096] @ Wenc + benc
            mass += torch.where(pre > thr, pre, torch.zeros_like(pre)).sum(0)
        cand = mass.argsort(descending=True)[:args.n_cand].cpu().numpy()
        preC = Xc @ Wenc[:, cand] + benc[cand]
        actC = torch.where(preC > thr[cand], preC, torch.zeros_like(preC)).cpu().numpy()      # (P, n_cand)
        feats, gloss, seen, colidx = [], [], set(), []
        for j, f in enumerate(cand):
            if len(feats) >= args.n_operands:
                break
            active = actC[:, j] > 0
            if active.sum() < args.min_count:
                continue
            g = [tok.convert_ids_to_tokens(t).replace("Ġ", "_").replace("▁", "_") for t, _ in Counter(all_ids[active].tolist()).most_common(3)]
            if _struct(g[0]) or g[0].lstrip("_").lower() in STOP or g[0] in seen:
                continue
            seen.add(g[0]); feats.append(int(f)); gloss.append(g); colidx.append(j)
        nt = len(feats)
        ln_gain = 1.0 + model.model.layers[L].input_layernorm.weight.detach().float().cpu().numpy().astype(np.float64)
        Dm = Wdec[feats] * ln_gain; Dm = Dm / (np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-9)
        actO = actC[:, colidx]; mx = actO.max(1); am = actO.argmax(1)
        pos_op = np.where(mx > 0, am, -1)
        operands[L] = {"D": Dm, "gloss": gloss, "pos_op": pos_op, "nt": nt}
        del sae, Wenc, Wdec

    # ---- pass 2: empirical attention per (layer, head) over that layer's operand pairs ----
    asum = {L: np.zeros((H, operands[L]["nt"] ** 2)) for L in layers}
    acnt = {L: np.zeros(operands[L]["nt"] ** 2) for L in layers}
    with torch.no_grad():
        for ci, c in enumerate(chunks):
            atts = model(input_ids=torch.tensor([c], device=dev), output_attentions=True).attentions
            ti, si = np.tril_indices(len(c), k=-1)
            for L in layers:
                nt = operands[L]["nt"]; poc = operands[L]["pos_op"][off[ci]:off[ci + 1]]
                m = (poc[ti] >= 0) & (poc[si] >= 0); flat = poc[ti[m]] * nt + poc[si[m]]
                np.add.at(acnt[L], flat, 1.0)
                aL = atts[L][0].float().cpu().numpy()
                for h in range(H):
                    np.add.at(asum[L][h], flat, aL[h][ti[m], si[m]])

    # ---- per-layer legibility + best binding ----
    rng = np.random.default_rng(0)
    rows = []
    for L in layers:
        nt = operands[L]["nt"]; D = operands[L]["D"]; gl = operands[L]["gloss"]
        cnt2 = acnt[L].reshape(nt, nt); offm = ~np.eye(nt, dtype=bool); supp = (cnt2 >= args.min_count) & offm
        perms = [rng.permutation(nt) for _ in range(args.n_perm)]
        Wq = model.model.layers[L].self_attn.q_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        Wk = model.model.layers[L].self_attn.k_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        zs, best = [], (-9, None)
        for h in range(H):
            kv = h // (H // n_kv)
            Mh = (Wq[h * hd:(h + 1) * hd].T @ Wk[kv * hd:(kv + 1) * hd]) / qscale
            B = D @ Mh @ D.T; A = asum[L][h].reshape(nt, nt) / np.maximum(cnt2, 1)
            if supp.sum() < 4:
                zs.append(float("nan")); continue
            leg = _spear(B[supp], A[supp])
            null = [_spear(B[np.ix_(pp, pp)][supp], A[supp]) for pp in perms]
            z = (leg - np.nanmean(null)) / (np.nanstd(null) + 1e-9)
            zs.append(z)
            if z > best[0]:
                Bo = B.copy(); np.fill_diagonal(Bo, -np.inf); qi, ki = np.unravel_index(int(np.argmax(Bo)), Bo.shape)
                best = (z, (f"{L}.{h}", "/".join(gl[qi]), "/".join(gl[ki])))
        nlz = sum(1 for z in zs if np.isfinite(z) and z > 2)
        rows.append({"layer": L, "n_operands": nt, "supported_cells": int(supp.sum()),
                     "n_legible_z2": nlz, "mean_z": float(np.nanmean(zs)), "best": best[1]})
        b = best[1]
        print(f"  layer {L:>2}: ops {nt:>2} cells {int(supp.sum()):>4}  legible(z>2) {nlz}/{H}  "
              f"mean_z {np.nanmean(zs):+.2f}  best {b[0]+': '+repr(b[1])+'->'+repr(b[2]) if b else 'n/a'}")

    out = {"experiment": "Gemma-2 QK content-opcode legibility across depth", "model": args.model,
           "layers": layers, "per_layer": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    lines = [f"GEMMA-2-2B QK CONTENT-OPCODE LEGIBILITY ACROSS DEPTH ({args.model}, {args.corpus}, {P} tok)", "",
             f"{'layer':>6} {'ops':>4} {'cells':>6} {'legible(z>2)':>13} {'mean_z':>7}  best content binding"]
    for r in rows:
        b = r["best"]; bs = f"{b[0]} {b[1]}->{b[2]}" if b else "n/a"
        lines.append(f"{r['layer']:>6} {r['n_operands']:>4} {r['supported_cells']:>6} "
                     f"{str(r['n_legible_z2'])+'/'+str(H):>13} {r['mean_z']:>+7.2f}  {bs}")
    tot = sum(r["n_legible_z2"] for r in rows)
    lines += ["", f"total legible heads across swept layers: {tot}/{len(layers)*H}"]
    args.txt.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[2:]))
    print(f"\n[done] {args.txt}  +  {args.output}")
    return out


if __name__ == "__main__":
    main()
