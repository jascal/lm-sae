"""Tier-B pylm composition kernel for the RoPE family (Llama-3.2 / Qwen2.5) — pure numpy on a CPU, no torch.

The modern-architecture counterpart of `numpy_lm.py` (GPT-2): RMSNorm + rotary position embedding + grouped-query
attention + SwiGLU MLP, ~80 lines of numpy over flat `.npz` weights (`export_weights_rope.py`). Same runtime story — flat
weight arrays + numpy matmuls + a tiny interpreter — extended to the models people actually want to run on a laptop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


class NumpyRoPE:
    """A Llama/Qwen forward pass in pure numpy over flat weight arrays."""

    def __init__(self, npz_path, route_frac=0.0):
        raw = dict(np.load(npz_path))
        self.nL, self.H, self.nkv, self.hd, self.d, self.ffn, self.V, self.tie = (int(x) for x in raw["cfg_i"])
        self.theta, self.eps = (float(x) for x in raw["cfg_f"])
        self.route_frac = route_frac
        self.W = {}                                                    # dequantise int8 / upcast to fp32 on load
        for k, v in raw.items():
            if k in ("cfg_i", "cfg_f") or k.endswith("__scale") or k.endswith("__rowscale"):
                continue
            if k + "__scale" in raw:
                self.W[k] = v.astype(np.float32) * raw[k + "__scale"].astype(np.float32)
            elif k + "__rowscale" in raw:
                self.W[k] = v.astype(np.float32) * raw[k + "__rowscale"].astype(np.float32)[:, None]
            else:
                self.W[k] = v.astype(np.float32)
        inv = 1.0 / (self.theta ** (np.arange(0, self.hd, 2) / self.hd))   # rotary frequencies
        self._inv = inv

    def _rope(self, x, pos):                                            # x: (seq, heads, hd)
        f = pos[:, None] * self._inv[None, :]; emb = np.concatenate([f, f], -1)
        cos = np.cos(emb)[:, None, :]; sin = np.sin(emb)[:, None, :]
        half = self.hd // 2
        rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
        return x * cos + rot * sin

    def logits(self, ids, capture=None):
        # capture: optional dict — records per-layer last-position attention (H, seq) and MLP activations (ffn) for
        # explain.py, exactly as numpy_lm.py does for GPT-2 (the head/feature explanation is kernel-agnostic).
        W = self.W; seq = len(ids); H, nkv, hd = self.H, self.nkv, self.hd
        x = W["embed"][np.asarray(ids)]                                # (seq, d)
        pos = np.arange(seq); rep = H // nkv
        cmask = np.triu(np.full((seq, seq), -1e30, np.float32), 1)
        if capture is not None:
            capture["att_last"] = []; capture["mlp_h"] = []
        for L in range(self.nL):
            p = f"l{L}."
            a = rmsnorm(x, W[p + "in_ln"], self.eps)
            q = a @ W[p + "self_attn.q_proj"] + W.get(p + "self_attn.q_proj.bias", 0.0)
            k = a @ W[p + "self_attn.k_proj"] + W.get(p + "self_attn.k_proj.bias", 0.0)
            v = a @ W[p + "self_attn.v_proj"] + W.get(p + "self_attn.v_proj.bias", 0.0)
            q = self._rope(q.reshape(seq, H, hd), pos); k = self._rope(k.reshape(seq, nkv, hd), pos)
            v = v.reshape(seq, nkv, hd)
            k = np.repeat(k, rep, axis=1); v = np.repeat(v, rep, axis=1)   # GQA: expand kv heads to all query heads
            q = q.transpose(1, 0, 2); k = k.transpose(1, 0, 2); v = v.transpose(1, 0, 2)
            att = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(hd) + cmask)
            o = (att @ v).transpose(1, 0, 2).reshape(seq, H * hd)
            x = x + o @ W[p + "self_attn.o_proj"]
            a2 = rmsnorm(x, W[p + "post_ln"], self.eps)
            h = silu(a2 @ W[p + "mlp.gate_proj"]) * (a2 @ W[p + "mlp.up_proj"])
            if self.route_frac > 0:
                kk = max(1, int(self.route_frac * h.shape[-1]))
                thr = np.partition(np.abs(h), -kk, axis=-1)[:, -kk:-kk + 1]
                h = np.where(np.abs(h) >= thr, h, 0.0)
            if capture is not None:
                capture["att_last"].append(att[:, -1, :].copy()); capture["mlp_h"].append(h[-1].copy())
            x = x + h @ W[p + "mlp.down_proj"]
        x = rmsnorm(x, W["norm"], self.eps)
        head = W["embed"].T if self.tie else W["lm_head"]
        return x @ head

    def write_mat(self, L):                                            # MLP write weight (ffn, d) — the SwiGLU down-proj
        return self.W[f"l{L}.mlp.down_proj"]

    @property
    def unembed(self):                                                 # (V, d) token directions (tied embedding / lm_head)
        return self.W["embed"] if self.tie else self.W["lm_head"].T

    def predict(self, ids):
        return int(self.logits(ids)[-1].argmax())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=Path("pylm/weights_qwen05b.npz"))
    p.add_argument("--ids", type=Path, default=Path("pylm/holdout_pythia-160m.json"), help="token-id stream to score")
    p.add_argument("--ctx", type=int, default=48)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--route-frac", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=Path("runs/pylm/numpy_rope_summary.json"))
    args = p.parse_args(argv)
    lm = NumpyRoPE(args.weights, route_frac=args.route_frac)
    hold = json.loads(args.ids.read_text())["holdout_ids"]
    positions = list(range(args.ctx, min(len(hold), args.ctx + args.n_eval)))
    correct = sum(lm.predict(hold[max(0, i - args.ctx):i]) == hold[i] for i in positions)
    acc = correct / max(len(positions), 1)
    out = {"weights": str(args.weights), "n_layers": lm.nL, "d": lm.d, "heads": f"{lm.H}/{lm.nkv}", "vocab": lm.V,
           "route_frac": args.route_frac, "n_eval": len(positions), "numpy_next_token_top1": acc,
           "weights_MB": args.weights.stat().st_size / 1e6}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"[numpy-rope] {lm.nL}L d={lm.d} GQA {lm.H}/{lm.nkv} · {out['weights_MB']:.0f} MB flat weights · no torch")
    print(f"[numpy-rope] next-token top-1: {acc:.1%}  ({len(positions)} positions)")
    return out


if __name__ == "__main__":
    main()
