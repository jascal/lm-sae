"""Gemma-2-2B disassembly — the unified per-head decode (the recent-model analog of disassemble_gpt2.py).

Combines every channel into one listing, at GPT-2-disassembly parity. For ALL 208 heads (26L x 8H):
  - ADDRESSING profile (attention-bucket split self/sink/prev/structural/local/long_range -> dominant mode);
  - IDIOM tags (prev-token / duplicate / induction, z>1.5);
  - QK token-operand BIND + OV WRITE (copy vs transform), computed in a *universal* per-layer token-centroid
    operand basis (GQA + RMSNorm-fold + unrotated content-QK), exactly as disassemble_gpt2.py — this is the
    detail that previously existed only at the SAE layer, now at every layer.
Plus a per-layer GeGLU MLP catalog (top neurons: read-tokens -> write-tokens, weight-only / first-order).
For the Gemma Scope SAE layer (12): the additional feature-native CONTENT opcode (QK B_h legibility + OV
write in Gemma Scope decoder coords) — the analog of GPT-2's separate sae_opcode_table. Reads
gemma_causal_summary.json if present to annotate which idiom heads are causally load-bearing.

Writes a human-readable listing to runs/gemma/gemma2_disassembly.txt + a JSON. Runs on cuda (bf16). The headline:
the GPT-2 disassembly framework, ported whole to a RoPE/GQA/RMSNorm model, at matched detail.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scope_loader import scope_npz  # noqa: E402

BUCKETS = ["self", "sink", "prev", "structural", "local", "long_range"]
STOP = {"the", "of", "and", "a", "to", "in", "that", "is", "was", "for", "it", "as", "with", "on", "by",
        "at", "an", "be", "or", "are", "from", "this", "his", "her", "he", "she", "they", "i", "you",
        "we", "but", "not", "have", "had", "s", "t", ""}


def _struct(s):
    # structural / special: empty, byte-fallback or special markers (<0x0A>, <unk>, <bos>, <eos>, <pad>), or pure punctuation
    s = s.replace("Ġ", "").replace("▁", "").replace("Ċ", "\n").strip()
    return s == "" or (s.startswith("<") and s.endswith(">")) or all(not ch.isalnum() for ch in s)


def _z(a):
    a = np.asarray(a, float)
    return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)


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
    p.add_argument("--sae-layer", type=int, default=12)
    p.add_argument("--hidden-index", type=int, default=12)
    p.add_argument("--n-operands", type=int, default=48)
    p.add_argument("--n-tokens", type=int, default=40, help="token-centroid operand basis size (all-layer QK/OV bind, GPT-2 parity)")
    p.add_argument("--min-pos", type=int, default=30, help="min corpus count for a token operand")
    p.add_argument("--mlp-per-layer", type=int, default=3, help="top GeGLU neurons catalogued per layer")
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--local-max", type=int, default=8)
    p.add_argument("--min-count", type=int, default=8)
    p.add_argument("--n-perm", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--corpus", default="wikitext")
    p.add_argument("--scope-path", default=None, help="local Gemma Scope dir/glob; else the HF cache, else download")
    p.add_argument("--causal", type=Path, default=Path("runs/gemma/gemma_causal_summary.json"))
    p.add_argument("--output", type=Path, default=Path("runs/gemma/gemma2_disassembly.json"))
    p.add_argument("--txt", type=Path, default=Path("runs/gemma/gemma2_disassembly.txt"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    cfg = model.config
    nL, H, n_kv, hd = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    qscale = float(getattr(cfg, "query_pre_attn_scalar", hd)) ** 0.5
    SL = args.sae_layer
    import urllib.request
    CORPORA = {"shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
               "wikitext": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"}
    txt = urllib.request.urlopen(urllib.request.Request(CORPORA.get(args.corpus, args.corpus),
                                 headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:400000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    off = np.cumsum([0] + [len(c) for c in chunks]); all_ids = np.array([j for c in chunks for j in c]); P = len(all_ids)
    print(f"{args.model}: {nL}L x {H}H (n_kv={n_kv}, head_dim={hd}); {len(chunks)} chunks, {P} tokens")

    # ---- frequent-token operand basis (universal across layers; GPT-2 parity) ----
    cnt = Counter(all_ids.tolist()); op_toks, seen_t = [], set()
    for t, c_ in cnt.most_common(400):
        s = tok.convert_ids_to_tokens(int(t)).replace("Ġ", "_").replace("▁", "_")
        if t in seen_t or c_ < args.min_pos or (s.startswith("<") and s.endswith(">")):
            continue
        seen_t.add(t); op_toks.append(int(t))
        if len(op_toks) >= args.n_tokens:
            break
    n_op = len(op_toks); tok2i = {t: i for i, t in enumerate(op_toks)}
    op_nm = [tok.convert_ids_to_tokens(t).replace("Ġ", "_").replace("▁", "_") for t in op_toks]

    # ---- pass 1: behavioral + coverage (all heads) + per-layer token centroids + resid@SL for the SAE ----
    cen = np.zeros((nL + 1, n_op, cfg.hidden_size)); gm = np.zeros((nL + 1, cfg.hidden_size)); ccnt = np.zeros(n_op); gN = 0
    pt = np.zeros((nL, H)); ptn = 0; dup = np.zeros((nL, H)); dupn = 0; dupb = 0.0
    ind = np.zeros((nL, H)); indn = 0; indb = 0.0
    bacc = np.zeros((nL, H, len(BUCKETS))); btot = np.zeros((nL, H)); Xh = np.zeros((P, cfg.hidden_size), np.float32)
    with torch.no_grad():
        for ci, c in enumerate(chunks):
            o = model(input_ids=torch.tensor([c], device=dev), output_attentions=True, output_hidden_states=True)
            Xh[off[ci]:off[ci + 1]] = o.hidden_states[args.hidden_index][0].float().cpu().numpy()
            Lc = len(c); ca = np.array(c); qi = np.arange(Lc)
            pid = np.array([tok2i.get(int(t), -1) for t in c]); mo = pid >= 0
            ccnt += np.bincount(pid[mo], minlength=n_op); gN += Lc
            for Lh in range(nL + 1):
                hs = o.hidden_states[Lh][0].float().cpu().numpy()
                np.add.at(cen[Lh], pid[mo], hs[mo]); gm[Lh] += hs.sum(0)
            toks = tok.convert_ids_to_tokens(c); struct = np.array([_struct(t) for t in toks])
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]
            DM = (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])
            IM = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
            ptn += Lc - 1; dupn += int(DM.any(1).sum()); indn += int(IM.any(1).sum())
            dq = DM.any(1); iq = IM.any(1)
            if dq.any():
                dupb += float((DM.sum(1)[dq] / np.maximum(qi[dq], 1)).sum())
            if iq.any():
                indb += float((IM.sum(1)[iq] / np.maximum(qi[iq], 1)).sum())
            S, T = np.meshgrid(qi, qi); delta = T - S
            bid = np.full((Lc, Lc), 5, dtype=int)
            bid[(delta >= 2) & (delta <= args.local_max)] = 4
            bid[struct[None, :].repeat(Lc, 0)] = 3
            bid[delta == 1] = 2; bid[S == 0] = 1; bid[S == T] = 0; bid[S > T] = -1
            for L in range(nL):
                a = o.attentions[L][0].float().cpu().numpy()
                pt[L] += np.diagonal(a, offset=-1, axis1=1, axis2=2).sum(1)
                dup[L] += (a * DM[None]).sum((1, 2)); ind[L] += (a * IM[None]).sum((1, 2))
                btot[L] += a.sum((1, 2))
                for b in range(len(BUCKETS)):
                    bacc[L, :, b] += (a * (bid == b)[None]).sum((1, 2))
    cen = cen / np.maximum(ccnt, 1)[None, :, None]; gm = gm / max(gN, 1)
    prevv = pt / max(ptn, 1); dupv = dup / max(dupn, 1) - dupb / max(dupn, 1); indv = ind / max(indn, 1) - indb / max(indn, 1)
    frac = bacc / np.maximum(btot, 1e-9)[:, :, None]
    zP, zD, zI = _z(prevv.reshape(-1)), _z(dupv.reshape(-1)), _z(indv.reshape(-1))

    # ---- Gemma Scope operands (diverse content features) for the SAE layer ----
    sae = np.load(scope_npz(SL, scope_path=args.scope_path))
    Wenc = torch.tensor(sae["W_enc"], device=dev, dtype=torch.float32); benc = torch.tensor(sae["b_enc"], device=dev, dtype=torch.float32)
    bdec = torch.tensor(sae["b_dec"], device=dev, dtype=torch.float32); thr = torch.tensor(sae["threshold"], device=dev, dtype=torch.float32)
    Wdec = sae["W_dec"].astype(np.float64)
    Xc = torch.tensor(Xh, device=dev) - bdec
    mass = torch.zeros(Wenc.shape[1], device=dev)
    for i in range(0, P, 2048):
        pre = Xc[i:i + 2048] @ Wenc + benc
        mass += torch.where(pre > thr, pre, torch.zeros_like(pre)).sum(0)
    feats, gloss, seen = [], [], set()
    for f in mass.argsort(descending=True).cpu().numpy():
        if len(feats) >= args.n_operands:
            break
        active = ((Xc @ Wenc[:, f] + benc[f]) > thr[f]).cpu().numpy()
        if active.sum() < args.min_count:
            continue
        cand = [tok.convert_ids_to_tokens(t).replace("Ġ", "_").replace("▁", "_")
                for t, _ in Counter(all_ids[active].tolist()).most_common(12)]
        g = [c for c in cand if not _struct(c)][:3]  # drop special/byte-fallback tokens (e.g. <unk>) from the label
        if not g or g[0].lstrip("_").lower() in STOP or g[0] in seen:
            continue
        seen.add(g[0]); feats.append(int(f)); gloss.append(g)
    nt = len(feats)
    ln_gain = 1.0 + model.model.layers[SL].input_layernorm.weight.detach().float().cpu().numpy().astype(np.float64)
    D = Wdec[feats] * ln_gain; D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    preO = (Xc @ Wenc[:, feats] + benc[feats])
    actO = torch.where(preO > thr[feats], preO, torch.zeros_like(preO)); mx, am = actO.max(1)
    pos_op = torch.where(mx > 0, am, torch.full_like(am, -1)).cpu().numpy()

    # ---- pass 2: empirical content-attn at the SAE layer over operand pairs ----
    asum = np.zeros((H, nt * nt)); acnt = np.zeros(nt * nt)
    with torch.no_grad():
        for ci, c in enumerate(chunks):
            aL = model(input_ids=torch.tensor([c], device=dev), output_attentions=True).attentions[SL][0].float().cpu().numpy()
            poc = pos_op[off[ci]:off[ci + 1]]; ti, si = np.tril_indices(len(c), k=-1)
            m = (poc[ti] >= 0) & (poc[si] >= 0); flat = poc[ti[m]] * nt + poc[si[m]]
            np.add.at(acnt, flat, 1.0)
            for h in range(H):
                np.add.at(asum[h], flat, aL[h][ti[m], si[m]])
    cnt2 = acnt.reshape(nt, nt); offm = ~np.eye(nt, dtype=bool); supp = (cnt2 >= args.min_count) & offm
    rng = np.random.default_rng(0); perms = [rng.permutation(nt) for _ in range(args.n_perm)]
    Wq = model.model.layers[SL].self_attn.q_proj.weight.detach().float().cpu().numpy().astype(np.float64)
    Wk = model.model.layers[SL].self_attn.k_proj.weight.detach().float().cpu().numpy().astype(np.float64)
    Wv = model.model.layers[SL].self_attn.v_proj.weight.detach().float().cpu().numpy().astype(np.float64)
    Wo = model.model.layers[SL].self_attn.o_proj.weight.detach().float().cpu().numpy().astype(np.float64)
    sae_op = {}
    for h in range(H):
        kv = h // (H // n_kv)
        Mh = (Wq[h * hd:(h + 1) * hd].T @ Wk[kv * hd:(kv + 1) * hd]) / qscale
        B = D @ Mh @ D.T; A = asum[h].reshape(nt, nt) / np.maximum(cnt2, 1)
        leg = _spear(B[supp], A[supp]) if supp.sum() >= 4 else float("nan")
        null = [_spear(B[np.ix_(pp, pp)][supp], A[supp]) for pp in perms] if supp.sum() >= 4 else [0]
        z = (leg - np.nanmean(null)) / (np.nanstd(null) + 1e-9) if np.isfinite(leg) else float("nan")
        Bo = B.copy(); np.fill_diagonal(Bo, -np.inf); qi_, ki_ = np.unravel_index(int(np.argmax(Bo)), Bo.shape)
        OV = Wo[:, h * hd:(h + 1) * hd] @ Wv[kv * hd:(kv + 1) * hd]            # (d,d) W_O^h W_V^kv
        V = D @ OV @ D.T; Vo = V.copy(); np.fill_diagonal(Vo, -np.inf)
        wz, wy = np.unravel_index(int(np.argmax(Vo)), Vo.shape)                # write-feature, source-feature
        sae_op[h] = {"qk_z": float(z), "qk_bind": [gloss[qi_], gloss[ki_]],
                     "ov_write": [gloss[wy], gloss[wz]]}                       # source -> written

    # ---- all-layer token-operand QK bind + OV copy/transform (GPT-2 parity: universal token basis) ----
    # operand x->y bilinears computed in head space (low-rank) so all 26x8 heads are cheap.
    tokop = {}; ovdiags = []; offm_t = ~np.eye(n_op, dtype=bool)
    for L in range(nL):
        sa = model.model.layers[L].self_attn
        lg = 1.0 + model.model.layers[L].input_layernorm.weight.detach().float().cpu().numpy().astype(np.float64)
        Dl = (cen[L] - gm[L]) * lg; Dl = Dl / (np.linalg.norm(Dl, axis=1, keepdims=True) + 1e-9)   # (n_op, d) RMSNorm-folded
        WQ = sa.q_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        WK = sa.k_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        WV = sa.v_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        WO = sa.o_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        for h in range(H):
            kv = h // (H // n_kv)
            Q = Dl @ WQ[h * hd:(h + 1) * hd].T; K = Dl @ WK[kv * hd:(kv + 1) * hd].T   # (n_op, hd) unrotated content
            B = (Q @ K.T) / qscale                                                     # QK opcode over token operands
            V = (Dl @ WO[:, h * hd:(h + 1) * hd]) @ (WV[kv * hd:(kv + 1) * hd] @ Dl.T)  # OV map x->head-out, in operand coords
            ovd = float(np.diag(V).mean() - V[offm_t].mean())                          # diagonal dominance -> copy vs transform
            Bo = B.copy(); np.fill_diagonal(Bo, -np.inf); qb, kb = np.unravel_index(int(np.argmax(Bo)), Bo.shape)
            tokop[(L, h)] = {"bind": [op_nm[qb], op_nm[kb]], "ov_diag": ovd}
            ovdiags.append(ovd)
    ov_hi = float(np.quantile(ovdiags, 0.75))   # copy = top-quartile OV diagonal dominance (like GPT-2)

    # ---- MLP catalog (Gemma GeGLU: reads via gate_proj, writes via down_proj; first-order, weight-only) ----
    mlp = {}
    for L in range(nL):
        layer = model.model.layers[L]
        lg = 1.0 + getattr(layer, "pre_feedforward_layernorm", layer.post_attention_layernorm).weight.detach().float().cpu().numpy().astype(np.float64)
        Dl = (cen[L] - gm[L]) * lg; Dl = Dl / (np.linalg.norm(Dl, axis=1, keepdims=True) + 1e-9)
        Wg = layer.mlp.gate_proj.weight.detach().float().cpu().numpy().astype(np.float64)   # (inter, d)
        Wd = layer.mlp.down_proj.weight.detach().float().cpu().numpy().astype(np.float64)   # (d, inter)
        inm = Dl @ Wg.T; outm = Dl @ Wd                                                     # (n_op, inter) read / write profiles
        sal = np.abs(inm).max(0) * np.abs(outm).max(0)
        mlp[str(L)] = []
        for ni in np.argsort(-sal)[: args.mlp_per_layer]:
            rd = [op_nm[x] for x in np.argsort(-np.abs(inm[:, ni]))[:3]]
            wr = [op_nm[x] for x in np.argsort(-np.abs(outm[:, ni]))[:3]]
            mlp[str(L)].append({"n": int(ni), "reads": rd, "writes": wr})

    # ---- causal annotation (optional) ----
    causal = {}
    if args.causal.exists():
        cj = json.loads(args.causal.read_text())
        for r in cj.get("sets", []):
            if r.get("load_bearing"):
                for L, h in r["heads"]:
                    causal.setdefault(f"{L}.{h}", []).append(r["set"])

    # ---- assemble the per-head disassembly ----
    rows = []
    for L in range(nL):
        for h in range(H):
            i = L * H + h; f = frac[L, h]
            tags = []
            if zP[i] > 1.5:
                tags.append("prev-token")
            if zD[i] > 1.5:
                tags.append("duplicate")
            if zI[i] > 1.5:
                tags.append("induction")
            tp = tokop[(L, h)]
            rec = {"head": f"{L}.{h}", "dominant": BUCKETS[int(np.argmax(f))],
                   "buckets": {b: float(f[k]) for k, b in enumerate(BUCKETS)}, "idioms": tags,
                   "bind": tp["bind"], "write": "copy" if tp["ov_diag"] >= ov_hi else "transform",
                   "ov_diag": tp["ov_diag"], "causal_load_bearing": causal.get(f"{L}.{h}", [])}
            if L == SL:
                rec["sae_opcode"] = sae_op[h]
            rows.append(rec)
    budget = {b: float(np.mean(frac[:, :, k])) for k, b in enumerate(BUCKETS)}
    write_hist = dict(Counter(r["write"] for r in rows))
    out = {"experiment": "Gemma-2-2B disassembly", "model": args.model, "n_layers": nL, "n_heads": H,
           "sae_layer": SL, "n_operands": nt, "n_token_operands": n_op, "attention_budget": budget,
           "write_hist": write_hist, "heads": rows, "mlp": mlp}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    # ---- human-readable listing ----
    lines = [f"GEMMA-2-2B DISASSEMBLY  ({nL} layers x {H} heads + GeGLU MLP; GQA n_kv={n_kv}, RoPE, RMSNorm)",
             f"corpus={args.corpus}  tokens={P}  token-operands={n_op}  SAE-layer={SL} (Gemma Scope, {nt} content operands)",
             "; addr=where-to-read (attn bucket)  WRITE=copy/transform (OV diag)  bind=top QK token binding  "
             "idioms=behavioral role  QK/OV[...]=SAE-feature opcode (SAE layer only)", ""]
    lines.append("attention budget (mean per-head mass): " + "  ".join(f"{b} {budget[b]:.0%}" for b in BUCKETS))
    plumb = sum(budget[b] for b in BUCKETS if b != "long_range")
    lines.append(f"  plumbing {plumb:.0%} | content (long-range) {budget['long_range']:.0%}")
    if causal:
        lines.append("causally load-bearing (induction-NLL ablation): "
                     + ", ".join(sorted(causal, key=lambda k: (int(k.split('.')[0]), int(k.split('.')[1])))))
    lines.append("")
    for L in range(nL):
        lines.append(f"--- layer {L} ---")
        for h in range(H):
            r = rows[L * H + h]
            tag = (" idioms[" + ",".join(r["idioms"]) + "]") if r["idioms"] else ""
            cz = " *CAUSAL*" if r["causal_load_bearing"] else ""
            extra = ""
            if "sae_opcode" in r:
                op = r["sae_opcode"]; q, k = "/".join(op["qk_bind"][0]), "/".join(op["qk_bind"][1])
                s, w = "/".join(op["ov_write"][0]), "/".join(op["ov_write"][1])
                extra = f"  QK[{q}->{k} z{op['qk_z']:.1f}]  OV[{s}=>{w}]"
            qb, kb = r["bind"]
            lines.append(f"  {r['head']:>5}  addr={r['dominant']:<11} WRITE={r['write']:<9} bind {qb!r}->{kb!r}{tag}{cz}{extra}")
        for nrec in mlp[str(L)]:
            lines.append(f"  {L}.MLP.n{nrec['n']:<5} reads {{{','.join(nrec['reads'])}}} -> writes {{{','.join(nrec['writes'])}}}")
    args.txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6]))
    n_idiom = sum(1 for r in rows if r["idioms"]); n_leg = sum(1 for r in rows if r.get("sae_opcode", {}).get("qk_z", 0) > 2)
    print(f"\n[summary] {n_idiom}/{len(rows)} heads carry an idiom tag; WRITE {write_hist}; "
          f"token operands {n_op}; layer-{SL} content opcodes legible (z>2): {n_leg}/{H}")
    print(f"[done] {args.txt}  +  {args.output}")
    return out


if __name__ == "__main__":
    main()
