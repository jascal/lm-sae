"""Export a DeepSeek-V3 / R1 / Kimi-K2 (MLA) model to flat .npz for the numpy kernel `numpy_mla.py`.

The MLA counterpart of `export_weights_moe.py`. MLA's low-rank latents are stored as the (in,out) projections the kernel
multiplies directly (q_a/q_b or q, kv_a_proj_with_mqa, kv_b, o_proj + the two latent RMSNorms). The FFN is DeepSeek MoE:
the routed experts are packed in memory (`mlp.experts.gate_up_proj` (E,2·mi,d) / `down_proj` (E,d,mi)) and unpacked to one
(in,out) gate/up/down per expert; plus the router (`mlp.gate` + `e_score_correction_bias`) and the always-on shared
expert; the first `first_k_dense_replace` layers are plain SwiGLU. To avoid re-deriving YaRN in numpy, the EXACT rotary
frequencies (`inv_freq`), the YaRN attention factor (`attention_scaling`), and the softmax scale (incl. the mscale²
correction) are pulled straight from the loaded model and baked into the export.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv=None):
    import torch
    from transformers import AutoModelForCausalLM
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V3")
    p.add_argument("--out", type=Path, default=Path("pylm/weights_mla.npz"))
    p.add_argument("--dtype", default="float32", help="float32 · float16 · int8 (per-column quant)")
    args = p.parse_args(argv)

    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    c = m.config; sd = m.state_dict()
    q_lora = getattr(c, "q_lora_rank", None) or 0
    kv_lora = c.kv_lora_rank
    qk_nope, qk_rope, v_head = c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
    n_routed = getattr(c, "n_routed_experts", 0)
    topk = getattr(c, "num_experts_per_tok", 0)
    moe_inter = getattr(c, "moe_intermediate_size", 0)
    n_group = getattr(c, "n_group", 1)
    topk_group = getattr(c, "topk_group", 1)
    norm_topk = int(bool(getattr(c, "norm_topk_prob", False)))
    first_k = getattr(c, "first_k_dense_replace", 0)
    routed_scaling = float(getattr(c, "routed_scaling_factor", 1.0))

    # pull the EXACT rope params from the loaded model (default or YaRN — no re-derivation in numpy)
    base = m.model if hasattr(m, "model") else m
    rot = base.rotary_emb
    inv_rope = rot.inv_freq.detach().float().numpy().astype(np.float32)        # (qk_rope//2,)
    att_factor = float(getattr(rot, "attention_scaling", 1.0))
    scale = float(base.layers[0].self_attn.scaling)                            # qk_head_dim^-0.5 × mscale²

    def col_q(a):
        s = (np.abs(a).max(0) / 127.0); s[s == 0] = 1e-8
        return np.round(a / s).clip(-127, 127).astype(np.int8), s.astype(np.float16)

    def lowp(a):
        return a.astype(np.float16 if args.dtype in ("int8", "float16") else args.dtype)

    def put(name, a, quant):
        if args.dtype == "int8" and quant and a.ndim == 2:
            q, s = col_q(a); W[name] = q; W[name + "__scale"] = s
        else:
            W[name] = lowp(a)

    def npw(key):
        return sd[key].detach().float().numpy()

    W = {"cfg_i": np.array([c.num_hidden_layers, c.num_attention_heads, c.hidden_size, q_lora, kv_lora, qk_nope,
                            qk_rope, v_head, c.vocab_size, int(c.tie_word_embeddings), n_routed, topk, moe_inter,
                            n_group, topk_group, norm_topk, first_k], dtype=np.int64),
         "cfg_f": np.array([float(c.rms_norm_eps), routed_scaling, scale, att_factor], dtype=np.float64),
         "inv_rope": inv_rope}

    W["embed"] = lowp(npw("model.embed_tokens.weight"))
    put("norm", npw("model.norm.weight"), False)
    if not c.tie_word_embeddings:
        W["lm_head"] = lowp(npw("lm_head.weight").T)

    for L in range(c.num_hidden_layers):
        pre = f"model.layers.{L}."
        a = pre + "self_attn."
        put(f"l{L}.in_ln", npw(pre + "input_layernorm.weight"), False)
        put(f"l{L}.post_ln", npw(pre + "post_attention_layernorm.weight"), False)
        if q_lora > 0:
            put(f"l{L}.q_a", npw(a + "q_a_proj.weight").T, True)
            put(f"l{L}.q_a_ln", npw(a + "q_a_layernorm.weight"), False)
            put(f"l{L}.q_b", npw(a + "q_b_proj.weight").T, True)
        else:
            put(f"l{L}.q", npw(a + "q_proj.weight").T, True)
        put(f"l{L}.kv_a", npw(a + "kv_a_proj_with_mqa.weight").T, True)
        put(f"l{L}.kv_a_ln", npw(a + "kv_a_layernorm.weight"), False)
        put(f"l{L}.kv_b", npw(a + "kv_b_proj.weight").T, True)
        put(f"l{L}.o_proj", npw(a + "o_proj.weight").T, True)
        if L < first_k:
            for k in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                put(f"l{L}.{k}", npw(pre + k + ".weight").T, True)
        else:
            put(f"l{L}.gate", npw(pre + "mlp.gate.weight").T, True)            # router (n_routed, d) → (d, n_routed)
            put(f"l{L}.gate_bias", npw(pre + "mlp.gate.e_score_correction_bias"), False)
            gu = npw(pre + "mlp.experts.gate_up_proj")                         # (E, 2*mi, d)
            dn = npw(pre + "mlp.experts.down_proj")                            # (E, d, mi)
            for e in range(n_routed):
                t = gu[e].T                                                   # (d, 2*mi)
                put(f"l{L}.experts.{e}.gate", t[:, :moe_inter], True)
                put(f"l{L}.experts.{e}.up", t[:, moe_inter:], True)
                put(f"l{L}.experts.{e}.down", dn[e].T, True)
            for fr, hf in (("gate", "gate_proj"), ("up", "up_proj"), ("down", "down_proj")):
                put(f"l{L}.shared.{fr}", npw(pre + f"mlp.shared_experts.{hf}.weight").T, True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **W)
    print(f"[export-mla] {args.model}: {len(W)} flat arrays · MLA(nope{qk_nope}/rope{qk_rope}/v{v_head}) · "
          f"{c.num_hidden_layers - first_k}/{c.num_hidden_layers} MoE layers ({n_routed} routed + shared) · "
          f"{args.dtype} → {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
