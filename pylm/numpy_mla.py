"""Tier-B pylm composition kernel for Multi-head Latent Attention (DeepSeek-V3 / R1 / Kimi-K2) — pure numpy, no torch.

The other frontier attention class (the counterpart of `numpy_moe.py`). MLA compresses attention through low-rank
latents instead of grouped-query sharing:
  - q goes d → q_lora → (nh · qk_head_dim) via q_a/q_b with an RMSNorm on the latent (or d → q directly if q_lora=0);
  - kv goes d → (kv_lora ‖ qk_rope) via kv_a_proj_with_mqa; the kv_lora part is RMSNorm'd then expanded by kv_b to
    per-head (qk_nope ‖ v_head); each head's key is [no-RoPE part ‖ a SINGLE shared RoPE key broadcast to all heads];
  - v_head_dim ≠ qk_head_dim; softmax scale = qk_head_dim^-0.5 (× mscale² under YaRN).
The FFN is DeepSeek MoE: group-limited sigmoid routing (bias-corrected *choice*, sigmoid *weight*) + an always-on
shared expert; the first `first_k_dense` layers are plain SwiGLU. RoPE matches transformers' interleaved path (the
DeepSeek default): the rope slice is permuted (evens‖odds) before split-half rotation, and the exact rotary
frequencies + YaRN attention factor + softmax scale are baked into the export, so the kernel needs no YaRN math.

A faithful mirror of fieldrun's `mla.rs` (validated top-1 vs torch), exposing the router pick (`capture["router"]`) for
the is-routing-retrieval-or-composition question. Pure numpy over flat `.npz` weights (`export_weights_mla.py`).
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


class NumpyMLA:
    """A DeepSeek-V3 forward pass in pure numpy over flat weight arrays (MLA + DeepSeek MoE + YaRN-aware rope)."""

    def __init__(self, npz_path, route_frac=0.0):
        raw = dict(np.load(npz_path))
        (self.nL, self.nh, self.d, self.q_lora, self.kv_lora, self.qk_nope, self.qk_rope, self.v_head, self.V,
         self.tie, self.n_routed, self.topk, self.moe_inter, self.n_group, self.topk_group, self.norm_topk,
         self.first_k) = (int(x) for x in raw["cfg_i"])
        self.eps, self.routed_scaling, self.scale, self.att_factor = (float(x) for x in raw["cfg_f"])
        self.qkh = self.qk_nope + self.qk_rope
        self.route_frac = route_frac
        self._inv = raw["inv_rope"].astype(np.float32)               # rotary freqs (qk_rope//2,), default or YaRN
        self.W = {}
        for k, v in raw.items():
            if k in ("cfg_i", "cfg_f", "inv_rope") or k.endswith("__scale") or k.endswith("__rowscale"):
                continue
            if k + "__scale" in raw:
                self.W[k] = v.astype(np.float32) * raw[k + "__scale"].astype(np.float32)
            elif k + "__rowscale" in raw:
                self.W[k] = v.astype(np.float32) * raw[k + "__rowscale"].astype(np.float32)[:, None]
            else:
                self.W[k] = v.astype(np.float32)

    def _rope(self, x, pos):                                          # x: (seq, heads, qk_rope) — interleaved (DeepSeek)
        d = self.qk_rope
        xp = x.reshape(*x.shape[:-1], d // 2, 2).swapaxes(-1, -2).reshape(x.shape)   # permute evens‖odds, then split-half
        f = pos[:, None] * self._inv[None, :]; emb = np.concatenate([f, f], -1)      # (seq, qk_rope)
        cos = (np.cos(emb) * self.att_factor)[:, None, :]; sin = (np.sin(emb) * self.att_factor)[:, None, :]
        rot = np.concatenate([-xp[..., d // 2:], xp[..., :d // 2]], -1)
        return xp * cos + rot * sin

    def _moe(self, p, a2, capture):
        """DeepSeek MoE: group-limited sigmoid routing (choice = sigmoid+bias; weight = sigmoid, renormed, scaled) over
        routed experts + an always-on shared expert. Records the per-token routed pick for the retrieval test."""
        W = self.W; seq = a2.shape[0]; nr, ng = self.n_routed, self.n_group; gsz = nr // ng
        logits = a2 @ W[p + "gate"]                                   # (seq, n_routed)
        bias = W[p + "gate_bias"]
        out = np.zeros((seq, self.d), np.float32)
        picks = np.zeros((seq, self.topk), np.int64)
        for t in range(seq):
            scores = sigmoid(logits[t])                              # weight uses the bare sigmoid (no bias)
            choice = scores + bias                                   # selection uses sigmoid+bias
            gscore = [np.sort(choice[g * gsz:(g + 1) * gsz])[::-1][:2].sum() for g in range(ng)]
            keep = set(np.argsort(gscore)[::-1][: self.topk_group].tolist())
            cand = [e for e in range(nr) if (e // gsz) in keep]
            cand = sorted(cand, key=lambda e: -choice[e])[: self.topk]
            denom = sum(scores[e] for e in cand) + 1e-20 if self.norm_topk else 1.0
            picks[t] = cand
            for e in cand:
                h = silu(a2[t] @ W[f"{p}experts.{e}.gate"]) * (a2[t] @ W[f"{p}experts.{e}.up"])
                out[t] += (scores[e] / denom * self.routed_scaling) * (h @ W[f"{p}experts.{e}.down"])
        if capture is not None:
            capture["router"].append(picks)
        sh = silu(a2 @ W[p + "shared.gate"]) * (a2 @ W[p + "shared.up"])   # shared expert, always on
        return out + sh @ W[p + "shared.down"], sh[-1]

    def _dense(self, p, a2):
        W = self.W
        h = silu(a2 @ W[p + "mlp.gate_proj"]) * (a2 @ W[p + "mlp.up_proj"])
        return h @ W[p + "mlp.down_proj"], h[-1]

    def logits(self, ids, capture=None):
        W = self.W; seq = len(ids); nh, qkh, qk_nope, qk_rope, vh = self.nh, self.qkh, self.qk_nope, self.qk_rope, self.v_head
        x = W["embed"][np.asarray(ids)]
        pos = np.arange(seq)
        cmask = np.triu(np.full((seq, seq), -1e30, np.float32), 1)
        if capture is not None:
            capture["att_last"] = []; capture["mlp_h"] = []; capture["router"] = []
        for L in range(self.nL):
            p = f"l{L}."
            a = rmsnorm(x, W[p + "in_ln"], self.eps)
            # --- q: latent down/up (or direct) ---
            if self.q_lora > 0:
                q = rmsnorm(a @ W[p + "q_a"], W[p + "q_a_ln"], self.eps) @ W[p + "q_b"]
            else:
                q = a @ W[p + "q"]
            q = q.reshape(seq, nh, qkh)
            q_pass, q_rot = q[..., :qk_nope], q[..., qk_nope:]
            # --- kv: compressed latent (kv_lora) + shared rope key (qk_rope) ---
            ckv = a @ W[p + "kv_a"]                                   # (seq, kv_lora + qk_rope)
            k_lat, k_rot = ckv[:, :self.kv_lora], ckv[:, self.kv_lora:]
            kpv = (rmsnorm(k_lat, W[p + "kv_a_ln"], self.eps) @ W[p + "kv_b"]).reshape(seq, nh, qk_nope + vh)
            k_pass, value = kpv[..., :qk_nope], kpv[..., qk_nope:]
            # --- rope: q_rot per head, k_rot shared (1 head) then broadcast ---
            q_rot = self._rope(q_rot, pos)
            k_rot = self._rope(k_rot[:, None, :], pos)               # (seq, 1, qk_rope)
            K = np.concatenate([k_pass, np.broadcast_to(k_rot, (seq, nh, qk_rope))], -1)   # (seq, nh, qkh)
            Q = np.concatenate([q_pass, q_rot], -1)
            # --- attention (per head) ---
            Qt = Q.transpose(1, 0, 2); Kt = K.transpose(1, 0, 2); Vt = value.transpose(1, 0, 2)
            att = softmax(Qt @ Kt.transpose(0, 2, 1) * self.scale + cmask)
            o = (att @ Vt).transpose(1, 0, 2).reshape(seq, nh * vh)
            x = x + o @ W[p + "o_proj"]
            a2 = rmsnorm(x, W[p + "post_ln"], self.eps)
            if L < self.first_k:
                mlp, mlp_h = self._dense(p, a2)
                if capture is not None:
                    capture["router"].append(None)
            else:
                mlp, mlp_h = self._moe(p, a2, capture)
            if capture is not None:
                capture["att_last"].append(att[:, -1, :].copy()); capture["mlp_h"].append(mlp_h.copy())
            x = x + mlp
        x = rmsnorm(x, W["norm"], self.eps)
        head = W["embed"].T if self.tie else W["lm_head"]
        return x @ head

    def write_mat(self, L):
        """Explain's neuron-label write weight: a dense layer's SwiGLU down-proj, a MoE layer's always-on SHARED
        expert down-proj (mlp_h captures the shared expert's hidden — the dense, always-present FFN feature)."""
        return self.W[f"l{L}.mlp.down_proj"] if L < self.first_k else self.W[f"l{L}.shared.down"]

    @property
    def unembed(self):
        return self.W["embed"] if self.tie else self.W["lm_head"].T

    def predict(self, ids):
        return int(self.logits(ids)[-1].argmax())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=Path("pylm/weights_mla.npz"))
    p.add_argument("--ids", type=Path, default=Path("pylm/holdout_mla.json"))
    p.add_argument("--ctx", type=int, default=48)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("runs/pylm/numpy_mla_summary.json"))
    args = p.parse_args(argv)
    lm = NumpyMLA(args.weights)
    hold = json.loads(args.ids.read_text())["holdout_ids"]
    positions = list(range(args.ctx, min(len(hold), args.ctx + args.n_eval)))
    correct = sum(lm.predict(hold[max(0, i - args.ctx):i]) == hold[i] for i in positions)
    acc = correct / max(len(positions), 1)
    out = {"weights": str(args.weights), "n_layers": lm.nL, "d": lm.d, "heads": lm.nh,
           "qk_nope/qk_rope/v": f"{lm.qk_nope}/{lm.qk_rope}/{lm.v_head}", "n_routed": lm.n_routed, "topk": lm.topk,
           "n_eval": len(positions), "numpy_next_token_top1": acc, "weights_MB": args.weights.stat().st_size / 1e6}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"[numpy-mla] {lm.nL}L d={lm.d} nh={lm.nh} MLA(nope{lm.qk_nope}/rope{lm.qk_rope}/v{lm.v_head}) "
          f"· DeepSeek MoE ({lm.n_routed} routed + shared, top-{lm.topk}) · {out['weights_MB']:.0f} MB · no torch")
    print(f"[numpy-mla] next-token top-1: {acc:.1%}  ({len(positions)} positions)")
    return out


if __name__ == "__main__":
    main()
