"""Hub probe — is the early high-fanout head an addressing-frame 'prologue'?

The composition graph found 0.11 (a layer-0 head) K-composing into many prev-token heads' keys. Hypothesis:
it's a one-time SETUP that broadcasts the positional/structural frame the addressing heads then use — a
prologue, not a lambda (no closure) and not feedback (a forward pass is a DAG, no cycles). Two tests:

(1) CONTENT vs POSITION — decompose the hub's output variance into between-position vs between-token
    (ANOVA R^2). Positional >> content => it broadcasts a positional signal.
(2) CAUSAL — path-patch the hub out of a prev-token head's KEY computation (queries untouched) and measure
    whether the mover's Δ=1 attention collapses. If removing the hub disables the mover, the hub IS the
    addressing-frame setup the mover keys on. Controls: a non-hub early writer, and the hub -> a non-mover.
Hub picked by K-composition fanout (weights); GPT-2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _ln(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * w + b


def _causal_softmax(s):
    seq = s.shape[0]; s = s.copy()
    s[np.triu(np.ones((seq, seq), bool), 1)] = -1e30
    s -= s.max(1, keepdims=True); e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=9000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/hub_probe_summary.json"))
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
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) == args.ctx]

    # weight tables
    ln1w = [tr.h[L].ln_1.weight.detach().numpy().astype(np.float64) for L in range(nL)]
    ln1b = [tr.h[L].ln_1.bias.detach().numpy().astype(np.float64) for L in range(nL)]
    Wq = [tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)[:, :d] for L in range(nL)]
    Wk = [tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)[:, d:2 * d] for L in range(nL)]
    Wv = [tr.h[L].attn.c_attn.weight.detach().numpy().astype(np.float64)[:, 2 * d:3 * d] for L in range(nL)]
    bk = [tr.h[L].attn.c_attn.bias.detach().numpy().astype(np.float64)[d:2 * d] for L in range(nL)]
    bv = [tr.h[L].attn.c_attn.bias.detach().numpy().astype(np.float64)[2 * d:3 * d] for L in range(nL)]
    Wo = [tr.h[L].attn.c_proj.weight.detach().numpy().astype(np.float64) for L in range(nL)]

    def OV(L, h):
        sl = slice(h * hd, (h + 1) * hd); return Wv[L][:, sl] @ Wo[L][sl, :]

    # ---- pick the hub by K-composition fanout (weights): early head feeding many later keys ----
    ovn = {}
    for L in range(3):
        for h in range(H):
            ovn[(L, h)] = OV(L, h) / (np.linalg.norm(OV(L, h)) + 1e-9)
    fan = {}
    for (La, ha), ov in ovn.items():
        s = 0.0
        for Lb in range(La + 1, nL):
            for hb in range(H):
                sl = slice(hb * hd, (hb + 1) * hd)
                s += np.linalg.norm(ov @ Wk[Lb][:, sl]) / (np.linalg.norm(Wk[Lb][:, sl]) + 1e-9)
        fan[(La, ha)] = s
    hub = max(fan, key=fan.get)
    ctrl_writer = min(fan, key=fan.get)
    print(f"{args.pretrained}: hub = {hub} (K-comp fanout {fan[hub]:.1f}); control writer = {ctrl_writer} (fanout {fan[ctrl_writer]:.1f})")

    # ---- pass 1: prev-token attn per head + hub-output content/position variance ----
    pt = np.zeros((nL, H)); ptn = 0
    Lh, hh = hub
    Osum_pos = np.zeros((args.ctx, d)); pos_cnt = np.zeros(args.ctx)
    Osum_tok = {}; tot_sum = np.zeros(d); tot_sq = 0.0; tot_n = 0
    with torch.no_grad():
        for c in chunks:
            o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
            Lc = len(c); ptn += Lc - 1
            for L in range(nL):
                pt[L] += np.diagonal(o.attentions[L][0].float().numpy(), offset=-1, axis1=1, axis2=2).sum(1)
            hs = o.hidden_states[Lh][0].float().numpy()
            aH = o.attentions[Lh][0][hh].float().numpy()
            Vh = (_ln(hs, ln1w[Lh], ln1b[Lh]) @ Wv[Lh][:, hh * hd:(hh + 1) * hd] + bv[Lh][hh * hd:(hh + 1) * hd])
            Oh = (aH @ Vh) @ Wo[Lh][hh * hd:(hh + 1) * hd, :]      # (seq, d) hub output per position
            Osum_pos += Oh; pos_cnt += 1
            tot_sum += Oh.sum(0); tot_sq += float((Oh ** 2).sum()); tot_n += Lc
            for i, t in enumerate(c):
                if t not in Osum_tok:
                    Osum_tok[t] = [np.zeros(d), 0]
                Osum_tok[t][0] += Oh[i]; Osum_tok[t][1] += 1
    prevtok = pt / max(ptn, 1)
    pt_head = np.unravel_index(int(prevtok.argmax()), prevtok.shape)
    gmean = tot_sum / tot_n
    total_var = tot_sq / tot_n - float(gmean @ gmean)
    pos_means = Osum_pos / np.maximum(pos_cnt, 1)[:, None]
    between_pos = float(((pos_means - gmean) ** 2).sum(1) @ pos_cnt) / tot_n
    between_tok = sum(cnt * float(((s / cnt) - gmean) @ ((s / cnt) - gmean)) for s, cnt in Osum_tok.values()) / tot_n
    pos_R2 = between_pos / (total_var + 1e-9); tok_R2 = between_tok / (total_var + 1e-9)
    print(f"prev-token head = {pt_head[0]}.{pt_head[1]} (Δ=1 attn {prevtok[pt_head]:.2f})")
    print(f"\n[1: content vs position] hub output variance — POSITION R² {pos_R2:.2f}  vs  CONTENT R² {tok_R2:.2f} "
          f"-> {'POSITIONAL broadcast (addressing-frame signal)' if pos_R2 > 2 * tok_R2 else 'content-driven' if tok_R2 > 2 * pos_R2 else 'mixed'}")

    # ---- pass 2: path-patch hub OUT of the prev-token head's keys; measure Δ=1 attn collapse ----
    Lb, hb = int(pt_head[0]), int(pt_head[1])
    Lc2, hc2 = ctrl_writer

    def delta1_attn(writer, reader):
        Lw, hw = writer; Lr, hr = reader
        slr = slice(hr * hd, (hr + 1) * hd)
        clean_s = patch_s = n = 0.0
        with torch.no_grad():
            for c in chunks:
                o = tr(input_ids=torch.tensor([c]), output_hidden_states=True, output_attentions=True)
                Lc = len(c)
                hsr = o.hidden_states[Lr][0].float().numpy()
                hsw = o.hidden_states[Lw][0].float().numpy()
                aW = o.attentions[Lw][0][hw].float().numpy()
                Vw = (_ln(hsw, ln1w[Lw], ln1b[Lw]) @ Wv[Lw][:, hw * hd:(hw + 1) * hd] + bv[Lw][hw * hd:(hw + 1) * hd])
                Ow = (aW @ Vw) @ Wo[Lw][hw * hd:(hw + 1) * hd, :]      # writer output at each position
                lnc = _ln(hsr, ln1w[Lr], ln1b[Lr]); lnp = _ln(hsr - Ow, ln1w[Lr], ln1b[Lr])
                Q = lnc @ Wq[Lr][:, slr]
                Kc = lnc @ Wk[Lr][:, slr] + bk[Lr][slr]
                Kp = lnp @ Wk[Lr][:, slr] + bk[Lr][slr]
                Pc = _causal_softmax(Q @ Kc.T / np.sqrt(hd)); Pp = _causal_softmax(Q @ Kp.T / np.sqrt(hd))
                clean_s += float(np.diagonal(Pc, offset=-1).sum())
                patch_s += float(np.diagonal(Pp, offset=-1).sum())
                n += Lc - 1
        return clean_s / n, patch_s / n

    hc, hp = delta1_attn(hub, pt_head)                       # hub -> prev-tok keys
    cc, cp = delta1_attn(ctrl_writer, pt_head)               # control writer -> prev-tok keys
    # hub -> a non-mover reader (a late content head, e.g. the strongest induction head's layer)
    nonmover = (nL - 1, 0)
    nc, npp = delta1_attn(hub, nonmover)
    drop_hub = (hc - hp) / max(hc, 1e-9)
    drop_ctrl = (cc - cp) / max(cc, 1e-9)
    print(f"\n[2: causal path-patch] Δ=1 attention of mover {pt_head[0]}.{pt_head[1]} when a writer is removed from its KEYS:")
    print(f"   remove HUB {hub}:          {hc:.3f} -> {hp:.3f}  ({drop_hub:+.0%})")
    print(f"   remove control {ctrl_writer}: {cc:.3f} -> {cp:.3f}  ({drop_ctrl:+.0%})")
    print(f"   HUB -> non-mover {nonmover} Δ=1: {nc:.3f} -> {npp:.3f}  ({(nc-npp)/max(nc,1e-9):+.0%}) (sanity)")
    confirmed = drop_hub > 0.1 and drop_hub > 2 * abs(drop_ctrl)
    print(f"\n[verdict] {'ADDRESSING-FRAME PROLOGUE confirmed: removing the early hub from the prev-token mover''s keys collapses its Δ=1 attention ('+f'{drop_hub:.0%}'+'), the control writer does not — the hub sets up the positional frame the mover keys on' if confirmed and pos_R2 > tok_R2 else 'hub is not the decisive addressing-frame setup for this mover (drop '+f'{drop_hub:+.0%}'+')'}")

    out = {"experiment": "hub probe", "model": args.pretrained, "hub": list(hub),
           "control_writer": list(ctrl_writer), "prev_token_head": [Lb, hb],
           "hub_output_position_R2": pos_R2, "hub_output_content_R2": tok_R2,
           "delta1_clean": hc, "delta1_patched_hub": hp, "drop_hub": drop_hub,
           "delta1_patched_ctrl": cp, "drop_ctrl": drop_ctrl}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
