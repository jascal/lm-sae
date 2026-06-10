"""Export a Qwen3-MoE model to flat .npz for the numpy MoE kernel `numpy_moe.py`.

The MoE counterpart of `export_weights_rope.py`. On the RoPE backbone it adds QK-norm (per-head RMSNorm weights on q/k)
and a per-layer MoE-or-dense FFN: a router (`mlp.gate`, Linear d→E) over experts, each a SwiGLU. transformers stores the
experts PACKED in memory (`mlp.experts.gate_up_proj` (E, 2·mi, d), `mlp.experts.down_proj` (E, d, mi)); this unpacks them
to one (in,out) gate/up/down per expert so the kernel does plain `x @ W`. Layers not selected by `decoder_sparse_step`
(or in `mlp_only_layers`) stay dense SwiGLU and are exported like the RoPE family. The build step is the only torch use.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]


def main(argv=None):
    import torch
    from transformers import AutoModelForCausalLM
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--out", type=Path, default=Path("pylm/weights_qwen3moe.npz"))
    p.add_argument("--dtype", default="float32", help="float32 · float16 · int8 (per-column quant)")
    args = p.parse_args(argv)

    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    c = m.config; sd = m.state_dict()
    nkv = getattr(c, "num_key_value_heads", c.num_attention_heads)
    hd = getattr(c, "head_dim", None) or (c.hidden_size // c.num_attention_heads)
    theta = getattr(c, "rope_theta", None)
    if theta is None:
        theta = (getattr(c, "rope_parameters", None) or {}).get("rope_theta", 1e6)  # Qwen3 default 1e6
    window = int(getattr(c, "sliding_window", 0) or 0) if getattr(c, "use_sliding_window", False) else 0
    n_exp = getattr(c, "num_experts", 0)
    topk = getattr(c, "num_experts_per_tok", 0)
    moe_inter = getattr(c, "moe_intermediate_size", 0)
    norm_topk = int(bool(getattr(c, "norm_topk_prob", False)))

    def col_q(a):                                                     # (in,out) → per-output-column int8 + fp16 scale
        s = (np.abs(a).max(0) / 127.0); s[s == 0] = 1e-8
        return np.round(a / s).clip(-127, 127).astype(np.int8), s.astype(np.float16)

    def lowp(a):                                                     # to the on-disk low precision (fp16 under int8/fp16)
        return a.astype(np.float16 if args.dtype in ("int8", "float16") else args.dtype)

    def put(name, a, quant):                                         # a already in (in,out) layout for matmuls
        if args.dtype == "int8" and quant and a.ndim == 2:
            q, s = col_q(a); W[name] = q; W[name + "__scale"] = s
        else:
            W[name] = lowp(a)

    def npw(key):                                                    # an HF (out,in) Linear weight as float numpy
        return sd[key].detach().float().numpy()

    # which layers are MoE? detect by the packed-experts key (else dense SwiGLU). decoder_sparse_step / mlp_only_layers
    # are baked into which module each layer got, so the state-dict keys are the ground truth.
    moe_flags = [1 if f"model.layers.{L}.mlp.experts.gate_up_proj" in sd else 0 for L in range(c.num_hidden_layers)]

    W = {"cfg_i": np.array([c.num_hidden_layers, c.num_attention_heads, nkv, hd, c.hidden_size,
                            getattr(c, "intermediate_size", 0), c.vocab_size, int(c.tie_word_embeddings),
                            n_exp, topk, moe_inter, norm_topk, window], dtype=np.int64),
         "cfg_f": np.array([float(theta), float(c.rms_norm_eps)], dtype=np.float64),
         "moe_flags": np.array(moe_flags, dtype=np.int64)}

    W["embed"] = lowp(npw("model.embed_tokens.weight"))               # (vocab, d) — fp16 even under int8 (quant-sensitive)
    put("norm", npw("model.norm.weight"), False)
    if not c.tie_word_embeddings:
        W["lm_head"] = lowp(npw("lm_head.weight").T)

    for L in range(c.num_hidden_layers):
        pre = f"model.layers.{L}."
        put(f"l{L}.in_ln", npw(pre + "input_layernorm.weight"), False)
        put(f"l{L}.post_ln", npw(pre + "post_attention_layernorm.weight"), False)
        put(f"l{L}.q_norm", npw(pre + "self_attn.q_norm.weight"), False)   # per-head RMSNorm (hd,)
        put(f"l{L}.k_norm", npw(pre + "self_attn.k_norm.weight"), False)
        for k in PROJ:
            put(f"l{L}.{k}", npw(pre + k + ".weight").T, True)            # (out,in)→(in,out)
        if moe_flags[L]:
            put(f"l{L}.gate", npw(pre + "mlp.gate.weight").T, True)        # router Linear d→E, (E,d)→(d,E)
            gu = npw(pre + "mlp.experts.gate_up_proj")                     # (E, 2*mi, d) = (E, out, in)
            dn = npw(pre + "mlp.experts.down_proj")                        # (E, d, mi)  = (E, out, in)
            for e in range(n_exp):
                t = gu[e].T                                               # (d, 2*mi); gate=first half of OUTPUT, up=second
                put(f"l{L}.experts.{e}.gate", t[:, :moe_inter], True)     # (d, mi)
                put(f"l{L}.experts.{e}.up", t[:, moe_inter:], True)       # (d, mi)
                put(f"l{L}.experts.{e}.down", dn[e].T, True)              # (mi, d)
        else:
            for k in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                put(f"l{L}.{k}", npw(pre + k + ".weight").T, True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **W)
    print(f"[export-moe] {args.model}: {len(W)} flat arrays · {sum(moe_flags)}/{c.num_hidden_layers} MoE layers "
          f"({n_exp} experts, top-{topk}) · {args.dtype} → {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
