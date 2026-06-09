"""Tier-B pylm composition kernel for Gemma-2 — pure numpy on a CPU, no torch.

The hardest of the laptop architectures. On top of the RoPE family (`numpy_rope.py`) Gemma-2 adds: an embedding scale
of √d, a four-norm 'sandwich' per layer (input / post-attention / pre-feedforward / post-feedforward, the post-norms
applied to the sub-layer output before the residual add), attention-logit and final-logit soft-capping (tanh), GeGLU
(gelu-tanh gating), a head_dim that is not hidden/heads, alternating sliding-window / full attention, and RMSNorm as
x·(1+w) — the (1+w) offset is baked into the exported weights so `rmsnorm` here is the plain one.

Memory: Gemma-2-2b's 256k vocab makes fp32 ~10 GB, so this kernel keeps weights **low-precision in RAM and upcasts per
matmul** (and chunks the unembed) — the exact strategy the Rust runtime will use for every model, previewed in Python.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


class NumpyGemma:
    """A Gemma-2 forward pass in pure numpy over flat weight arrays — weights stay fp16/int8 in RAM, upcast per op."""

    def __init__(self, npz_path, route_frac=0.0, window=4096):
        raw = dict(np.load(npz_path))
        self.nL, self.H, self.nkv, self.hd, self.d, self.ffn, self.V, self.tie = (int(x) for x in raw["cfg_i"])
        self.theta, self.eps, self.attn_cap, self.final_cap, self.qscalar, self.escale = \
            (float(x) for x in raw["cfg_f"])
        self.route_frac = route_frac; self.window = window
        self.W = {}; self.S = {}                                        # keep weights as-loaded (fp16/int8) — no fp32 copy
        for k, v in raw.items():
            if k.endswith("__scale"):
                self.S[k[:-len("__scale")]] = v
            elif k != "cfg_i" and k != "cfg_f":
                self.W[k] = v
        self._inv = 1.0 / (self.theta ** (np.arange(0, self.hd, 2) / self.hd))

    def _wf(self, name):                                               # one weight upcast to fp32 on demand (per matmul)
        w = self.W[name].astype(np.float32)
        return w * self.S[name].astype(np.float32) if name in self.S else w

    def _rope(self, x, pos):
        f = pos[:, None] * self._inv[None, :]; emb = np.concatenate([f, f], -1)
        cos = np.cos(emb)[:, None, :]; sin = np.sin(emb)[:, None, :]
        half = self.hd // 2
        rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
        return x * cos + rot * sin

    def _unembed(self, x):                                            # chunk over the 256k vocab to avoid a huge fp32 temp
        head = self.W["embed"] if self.tie else self.W["lm_head"].T   # (V, d)
        out = np.empty((x.shape[0], head.shape[0]), np.float32); step = 32768
        for c in range(0, head.shape[0], step):
            out[:, c:c + step] = x @ head[c:c + step].astype(np.float32).T
        return out

    def logits(self, ids, capture=None):
        seq = len(ids); H, nkv, hd = self.H, self.nkv, self.hd
        x = self.W["embed"][np.asarray(ids)].astype(np.float32) * self.escale   # Gemma scales the input embedding by √d
        pos = np.arange(seq); rep = H // nkv; scale = self.qscalar ** -0.5
        causal = np.triu(np.full((seq, seq), -1e30, np.float32), 1)
        band = np.tril(np.full((seq, seq), -1e30, np.float32), -self.window)     # sliding-window: mask keys > window back
        if capture is not None:
            capture["att_last"] = []; capture["mlp_h"] = []
        for L in range(self.nL):
            p = f"l{L}."
            a = rmsnorm(x, self._wf(p + "input_layernorm"), self.eps)
            q = a @ self._wf(p + "self_attn.q_proj"); k = a @ self._wf(p + "self_attn.k_proj")
            v = a @ self._wf(p + "self_attn.v_proj")
            q = self._rope(q.reshape(seq, H, hd), pos); k = self._rope(k.reshape(seq, nkv, hd), pos)
            v = v.reshape(seq, nkv, hd)
            k = np.repeat(k, rep, axis=1); v = np.repeat(v, rep, axis=1)
            q = q.transpose(1, 0, 2); k = k.transpose(1, 0, 2); v = v.transpose(1, 0, 2)
            s = (q @ k.transpose(0, 2, 1)) * scale
            if self.attn_cap > 0:                                      # attention-logit soft-cap (tanh), before the mask
                s = self.attn_cap * np.tanh(s / self.attn_cap)
            s = s + causal + (band if (L % 2 == 0) else 0.0)           # even layers: sliding window; odd: full
            o = (softmax(s) @ v).transpose(1, 0, 2).reshape(seq, H * hd)
            o = o @ self._wf(p + "self_attn.o_proj")
            x = x + rmsnorm(o, self._wf(p + "post_attention_layernorm"), self.eps)   # post-norm on sub-layer output
            a2 = rmsnorm(x, self._wf(p + "pre_feedforward_layernorm"), self.eps)
            h = gelu_tanh(a2 @ self._wf(p + "mlp.gate_proj")) * (a2 @ self._wf(p + "mlp.up_proj"))
            if self.route_frac > 0:
                kk = max(1, int(self.route_frac * h.shape[-1]))
                thr = np.partition(np.abs(h), -kk, axis=-1)[:, -kk:-kk + 1]
                h = np.where(np.abs(h) >= thr, h, 0.0)
            if capture is not None:
                capture["att_last"].append(softmax(s)[:, -1, :].copy()); capture["mlp_h"].append(h[-1].copy())
            mlp = h @ self._wf(p + "mlp.down_proj")
            x = x + rmsnorm(mlp, self._wf(p + "post_feedforward_layernorm"), self.eps)
        x = rmsnorm(x, self._wf("norm"), self.eps)
        z = self._unembed(x)
        return self.final_cap * np.tanh(z / self.final_cap) if self.final_cap > 0 else z   # final-logit soft-cap

    def write_mat(self, L):
        return self._wf(f"l{L}.mlp.down_proj")

    @property
    def unembed(self):
        return self.W["embed"].astype(np.float32) if self.tie else self._wf("lm_head").T

    def predict(self, ids):
        return int(self.logits(ids)[-1].argmax())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=Path("pylm/weights_gemma2_2b.npz"))
    p.add_argument("--ids", type=Path, default=Path("pylm/holdout_pythia-160m.json"))
    p.add_argument("--ctx", type=int, default=48)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--route-frac", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=Path("runs/pylm/numpy_gemma_summary.json"))
    args = p.parse_args(argv)
    lm = NumpyGemma(args.weights, route_frac=args.route_frac)
    hold = json.loads(args.ids.read_text())["holdout_ids"]
    positions = list(range(args.ctx, min(len(hold), args.ctx + args.n_eval)))
    correct = sum(lm.predict(hold[max(0, i - args.ctx):i]) == hold[i] for i in positions)
    acc = correct / max(len(positions), 1)
    out = {"weights": str(args.weights), "n_layers": lm.nL, "d": lm.d, "heads": f"{lm.H}/{lm.nkv}", "vocab": lm.V,
           "route_frac": args.route_frac, "n_eval": len(positions), "numpy_next_token_top1": acc,
           "weights_MB": args.weights.stat().st_size / 1e6}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"[numpy-gemma] {lm.nL}L d={lm.d} GQA {lm.H}/{lm.nkv} · {out['weights_MB']:.0f} MB flat weights · no torch")
    print(f"[numpy-gemma] next-token top-1: {acc:.1%}  ({len(positions)} positions)")
    return out


if __name__ == "__main__":
    main()
