"""Tier-B pylm composition kernel for the Mixture-of-Experts family (Qwen3-MoE) — pure numpy on a CPU, no torch.

The frontier-architecture counterpart of `numpy_rope.py`. On the RoPE backbone (RMSNorm + rotary + grouped-query
attention) Qwen3-MoE adds the two things that define the modern MoE block:
  - **QK-norm** — a per-head RMSNorm over head_dim on q and k, *after* the projection and *before* RoPE.
  - a per-layer **MoE-or-dense FFN** — a plain-gate router (softmax over all experts → top-k → optional renorm) that
    routes each token to its top-k SwiGLU experts; layers not selected by `decoder_sparse_step` stay dense SwiGLU.

This is the research-arm reference for the question the dense kernels can't ask: **is MoE routing retrieval or
composition?** — the router's per-token expert pick is exposed (`capture["router"]`) so the symbolic tier can test
whether a flat token→expert table reproduces it. Same runtime story as the rest of pylm: flat `.npz` weight arrays
(`export_weights_moe.py`) + numpy matmuls + a tiny interpreter, no torch at runtime. A faithful mirror of fieldrun's
`qwen3moe.rs` (validated top-1 against torch), so the math is the same on both sides.
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


class NumpyMoE:
    """A Qwen3-MoE forward pass in pure numpy over flat weight arrays (dense + MoE layers, QK-norm, optional window)."""

    def __init__(self, npz_path, route_frac=0.0):
        raw = dict(np.load(npz_path))
        (self.nL, self.H, self.nkv, self.hd, self.d, self.ffn, self.V, self.tie,
         self.n_exp, self.topk, self.moe_inter, self.norm_topk, self.window) = (int(x) for x in raw["cfg_i"])
        self.theta, self.eps = (float(x) for x in raw["cfg_f"])
        self.moe = [bool(x) for x in raw["moe_flags"]]               # per-layer: True = MoE FFN, False = dense SwiGLU
        self.route_frac = route_frac                                  # (dense-layer Tier-C knob; MoE is already routed)
        self.W = {}                                                   # dequantise int8 / upcast to fp32 on load
        for k, v in raw.items():
            if k in ("cfg_i", "cfg_f", "moe_flags") or k.endswith("__scale") or k.endswith("__rowscale"):
                continue
            if k + "__scale" in raw:
                self.W[k] = v.astype(np.float32) * raw[k + "__scale"].astype(np.float32)
            elif k + "__rowscale" in raw:
                self.W[k] = v.astype(np.float32) * raw[k + "__rowscale"].astype(np.float32)[:, None]
            else:
                self.W[k] = v.astype(np.float32)
        self._inv = 1.0 / (self.theta ** (np.arange(0, self.hd, 2) / self.hd))   # rotary frequencies
        self._top_expert = [0] * self.nL                             # dominant expert per MoE layer at the last forward

    def _rope(self, x, pos):                                          # x: (seq, heads, hd)
        f = pos[:, None] * self._inv[None, :]; emb = np.concatenate([f, f], -1)
        cos = np.cos(emb)[:, None, :]; sin = np.sin(emb)[:, None, :]
        half = self.hd // 2
        rot = np.concatenate([-x[..., half:], x[..., :half]], -1)
        return x * cos + rot * sin

    def _moe_ffn(self, p, a2, capture):
        """Route each token to its top-k experts (softmax over all → top-k → optional renorm), run each as SwiGLU,
        weighted-sum. Returns the layer output (seq, d); records the per-token expert pick + the dominant expert's
        hidden (for explain) when `capture` is requested."""
        W = self.W; seq = a2.shape[0]
        probs = softmax(a2 @ W[p + "gate"])                          # (seq, n_exp) — softmax over ALL experts first
        out = np.zeros((seq, self.d), np.float32)
        picks = np.zeros((seq, self.topk), np.int64)
        mlp_h_last, top_e = None, 0
        for t in range(seq):
            idx = np.argsort(-probs[t])[: self.topk]                 # top-k experts for this token
            denom = probs[t, idx].sum() if self.norm_topk else 1.0
            picks[t] = idx
            for e in idx:
                h = silu(a2[t] @ W[f"{p}experts.{e}.gate"]) * (a2[t] @ W[f"{p}experts.{e}.up"])
                out[t] += (probs[t, e] / denom) * (h @ W[f"{p}experts.{e}.down"])
            if t == seq - 1:
                top_e = int(idx[0])
                mlp_h_last = silu(a2[t] @ W[f"{p}experts.{top_e}.gate"]) * (a2[t] @ W[f"{p}experts.{top_e}.up"])
        if capture is not None:
            capture["router"].append(picks)                         # the per-token expert choice — for the retrieval test
        return out, mlp_h_last, top_e

    def _dense_ffn(self, p, a2):
        W = self.W
        h = silu(a2 @ W[p + "mlp.gate_proj"]) * (a2 @ W[p + "mlp.up_proj"])
        if self.route_frac > 0:                                       # Tier-C: keep only the top-frac neurons (dense only)
            kk = max(1, int(self.route_frac * h.shape[-1]))
            thr = np.partition(np.abs(h), -kk, axis=-1)[:, -kk:-kk + 1]
            h = np.where(np.abs(h) >= thr, h, 0.0)
        return h @ W[p + "mlp.down_proj"], h[-1]

    def logits(self, ids, capture=None):
        # capture (for explain.py, kernel-agnostic): per-layer last-position attention (H, seq) + MLP activations; for a
        # MoE layer mlp_h is the DOMINANT expert's hidden and `capture["router"]` holds the per-token expert pick.
        W = self.W; seq = len(ids); H, nkv, hd = self.H, self.nkv, self.hd
        x = W["embed"][np.asarray(ids)]                              # (seq, d)
        pos = np.arange(seq); rep = H // nkv
        cmask = np.triu(np.full((seq, seq), -1e30, np.float32), 1)   # causal
        if self.window > 0:                                          # sliding window: also mask keys older than `window`
            i = np.arange(seq)[:, None]; j = np.arange(seq)[None, :]
            cmask = np.where(j + self.window <= i, -1e30, cmask)
        if capture is not None:
            capture["att_last"] = []; capture["mlp_h"] = []; capture["router"] = []
        for L in range(self.nL):
            p = f"l{L}."
            a = rmsnorm(x, W[p + "in_ln"], self.eps)
            q = (a @ W[p + "self_attn.q_proj"]).reshape(seq, H, hd)
            k = (a @ W[p + "self_attn.k_proj"]).reshape(seq, nkv, hd)
            v = (a @ W[p + "self_attn.v_proj"]).reshape(seq, nkv, hd)
            q = rmsnorm(q, W[p + "q_norm"], self.eps)                # QK-norm (per head, over hd) before RoPE
            k = rmsnorm(k, W[p + "k_norm"], self.eps)
            q = self._rope(q, pos); k = self._rope(k, pos)
            k = np.repeat(k, rep, axis=1); v = np.repeat(v, rep, axis=1)   # GQA
            q = q.transpose(1, 0, 2); k = k.transpose(1, 0, 2); v = v.transpose(1, 0, 2)
            att = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(hd) + cmask)
            o = (att @ v).transpose(1, 0, 2).reshape(seq, H * hd)
            x = x + o @ W[p + "self_attn.o_proj"]
            a2 = rmsnorm(x, W[p + "post_ln"], self.eps)
            if self.moe[L]:
                mlp, mlp_h, top_e = self._moe_ffn(p, a2, capture); self._top_expert[L] = top_e
            else:
                mlp, mlp_h = self._dense_ffn(p, a2)
                if capture is not None:
                    capture["router"].append(None)
            if capture is not None:
                capture["att_last"].append(att[:, -1, :].copy()); capture["mlp_h"].append(mlp_h.copy())
            x = x + mlp
        x = rmsnorm(x, W["norm"], self.eps)
        head = W["embed"].T if self.tie else W["lm_head"]
        return x @ head

    def write_mat(self, L):
        """The MLP write weight for explain's neuron labels: a MoE layer names its dominant expert's down-proj (set by
        the last `logits(capture=...)`), a dense layer its SwiGLU down-proj."""
        if self.moe[L]:
            return self.W[f"l{L}.experts.{self._top_expert[L]}.down"]
        return self.W[f"l{L}.mlp.down_proj"]

    @property
    def unembed(self):
        return self.W["embed"] if self.tie else self.W["lm_head"].T

    def predict(self, ids):
        return int(self.logits(ids)[-1].argmax())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=Path("pylm/weights_qwen3moe.npz"))
    p.add_argument("--ids", type=Path, default=Path("pylm/holdout_qwen3moe.json"), help="token-id stream to score")
    p.add_argument("--ctx", type=int, default=48)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--route-frac", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=Path("runs/pylm/numpy_moe_summary.json"))
    args = p.parse_args(argv)
    lm = NumpyMoE(args.weights, route_frac=args.route_frac)
    hold = json.loads(args.ids.read_text())["holdout_ids"]
    positions = list(range(args.ctx, min(len(hold), args.ctx + args.n_eval)))
    correct = sum(lm.predict(hold[max(0, i - args.ctx):i]) == hold[i] for i in positions)
    acc = correct / max(len(positions), 1)
    out = {"weights": str(args.weights), "n_layers": lm.nL, "d": lm.d, "heads": f"{lm.H}/{lm.nkv}", "vocab": lm.V,
           "moe_layers": int(sum(lm.moe)), "n_experts": lm.n_exp, "topk": lm.topk,
           "n_eval": len(positions), "numpy_next_token_top1": acc, "weights_MB": args.weights.stat().st_size / 1e6}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"[numpy-moe] {lm.nL}L d={lm.d} GQA {lm.H}/{lm.nkv} · {sum(lm.moe)} MoE layers ({lm.n_exp} experts, top-{lm.topk}) "
          f"· {out['weights_MB']:.0f} MB flat weights · no torch")
    print(f"[numpy-moe] next-token top-1: {acc:.1%}  ({len(positions)} positions)")
    return out


if __name__ == "__main__":
    main()
