"""Tier-B pylm composition kernel — run GPT-2 on a CPU with pure numpy + flat weight arrays. No torch, no GPU.

pylm's flat retrieval (`lm.py`) reproduces ~half a model's behaviour at ~0 matmul. The other half is the *composition*
(the forge tax) — proven dense genuine computation, not a flat lookup, so it cannot be tabled. But it is a TC⁰ circuit, so
it runs on a CPU as plain numpy matmuls over flat weights. This is that kernel: load `weights_gpt2.npz` (from
`export_weights.py`) and run the GPT-2 forward pass in ~60 lines of numpy. The only runtime dependency is numpy.

This is the "minimum to RUN" compute half made literal: flat weight arrays (Θ model size) + a tiny numpy interpreter
(O(1) code). Together with the flat retrieval store, it is the whole model — on a CPU, with no deep-learning framework.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def gelu(x):                                                            # GPT-2 uses the tanh approximation
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


class NumpyGPT2:
    """A GPT-2 forward pass in pure numpy over flat weight arrays — the composition kernel."""

    def __init__(self, npz_path, route_frac=0.0):
        raw = dict(np.load(npz_path))                                  # weights may be fp32 / fp16 / int8 (+ "__scale")
        self.nL, self.H, self.d, self.ctx, self.V = (int(x) for x in raw["config"])
        self.W = {}
        for k, v in raw.items():
            if k == "config" or k.endswith("__scale"):
                continue
            sc = raw.get(k + "__scale")                               # int8 weights carry a per-column dequant scale
            self.W[k] = (v.astype(np.float32) * sc.astype(np.float32)) if sc is not None else v.astype(np.float32)
        self.route_frac = route_frac                                   # >0: compute only top (route_frac) of MLP neurons/token (Tier C)

    def logits(self, ids):
        W = self.W; seq = len(ids); hd = self.d // self.H
        x = W["wte"][np.asarray(ids)] + W["wpe"][np.arange(seq)]
        cmask = np.triu(np.full((seq, seq), -1e10, np.float32), 1)
        for L in range(self.nL):
            p = f"h{L}."
            a = layernorm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
            qkv = a @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(seq, self.H, hd).transpose(1, 0, 2)
            k = k.reshape(seq, self.H, hd).transpose(1, 0, 2)
            v = v.reshape(seq, self.H, hd).transpose(1, 0, 2)
            att = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(hd) + cmask)
            o = (att @ v).transpose(1, 0, 2).reshape(seq, self.d)
            x = x + (o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"])
            a2 = layernorm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
            h = gelu(a2 @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
            if self.route_frac > 0:                                    # Tier C: keep only the top-fraction active neurons/token
                kk = max(1, int(self.route_frac * h.shape[-1]))
                thr = np.partition(np.abs(h), -kk, axis=-1)[:, -kk:-kk + 1]
                h = np.where(np.abs(h) >= thr, h, 0.0)
            x = x + (h @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"])
        x = layernorm(x, W["ln_f.weight"], W["ln_f.bias"])
        return x @ W["wte"].T

    def predict(self, ids):
        return int(self.logits(ids)[-1].argmax())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=Path("pylm/weights_gpt2.npz"))
    p.add_argument("--ids", type=Path, default=Path("pylm/holdout_gpt2.json"), help="held-out token ids to score")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=500)
    p.add_argument("--route-frac", type=float, default=0.0, help="Tier C: compute only this fraction of MLP neurons/token")
    p.add_argument("--out", type=Path, default=Path("runs/pylm/numpy_lm_summary.json"))
    args = p.parse_args(argv)

    lm = NumpyGPT2(args.weights, route_frac=args.route_frac)
    hold = json.loads(args.ids.read_text())["holdout_ids"]
    positions = list(range(args.ctx, min(len(hold), args.ctx + args.n_eval)))
    correct = 0
    for i in positions:
        ctx = hold[max(0, i - args.ctx):i]
        correct += (lm.predict(ctx) == hold[i])
    acc = correct / max(len(positions), 1)
    out = {"weights": str(args.weights), "n_layers": lm.nL, "d_model": lm.d, "vocab": lm.V,
           "route_frac": args.route_frac, "n_eval": len(positions), "numpy_next_token_top1": acc,
           "weights_MB": args.weights.stat().st_size / 1e6}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"[numpy-lm] pure-numpy GPT-2 ({lm.nL}L, d={lm.d}) over {out['weights_MB']:.0f} MB flat weights · "
          f"no torch, no GPU{f' · route {args.route_frac:.0%} MLP/token' if args.route_frac else ''}")
    print(f"[numpy-lm] next-token top-1 accuracy on held-out: {acc:.1%}  ({len(positions)} positions)")
    print(f"[done] → {args.out}")
    return out


if __name__ == "__main__":
    main()
