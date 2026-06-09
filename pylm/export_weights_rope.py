"""Export a RoPE-family model (Llama / Qwen2.5) to flat .npz for the numpy kernel `numpy_rope.py`.

Same idea as `export_weights.py` (GPT-2) for the modern architecture: RMSNorm + rotary position + grouped-query
attention + SwiGLU MLP. The *build* step (torch, one-time); the runtime kernel needs only numpy. Linear weights are
stored transposed to (in, out) so the kernel does plain `x @ W` (and int8-quantises per output column, like the GPT-2
export). Tied embeddings (Llama-3.2 / Qwen2.5) → the embedding doubles as the unembedding.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def main(argv=None):
    import torch
    from transformers import AutoModelForCausalLM
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--out", type=Path, default=Path("pylm/weights_qwen05b.npz"))
    p.add_argument("--dtype", default="int8", help="float32 · float16 · int8 (per-column quant)")
    args = p.parse_args(argv)

    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    c = m.config; sd = m.state_dict()
    nkv = getattr(c, "num_key_value_heads", c.num_attention_heads)
    hd = getattr(c, "head_dim", None) or (c.hidden_size // c.num_attention_heads)
    theta = getattr(c, "rope_theta", None) or getattr(c, "rope_parameters", {}).get("rope_theta", 1e4)

    def col_q(a):                                                     # (in,out) → per-output-column int8 + fp16 scale
        s = (np.abs(a).max(0) / 127.0); s[s == 0] = 1e-8
        return np.round(a / s).clip(-127, 127).astype(np.int8), s.astype(np.float16)

    W = {"cfg_i": np.array([c.num_hidden_layers, c.num_attention_heads, nkv, hd, c.hidden_size,
                            c.intermediate_size, c.vocab_size, int(c.tie_word_embeddings)], dtype=np.int64),
         "cfg_f": np.array([float(theta), float(c.rms_norm_eps)], dtype=np.float64)}

    def put(name, a, quant):                                         # a already in (in,out) layout for matmuls
        if args.dtype == "int8" and quant and a.ndim == 2:
            q, s = col_q(a); W[name] = q; W[name + "__scale"] = s
        else:
            W[name] = a.astype(np.float16 if args.dtype in ("int8", "float16") else args.dtype)

    emb = sd["model.embed_tokens.weight"].detach().float().numpy()    # (vocab, d) — used as lookup AND (tied) unembed
    if args.dtype == "int8":                                          # per-row (vocab) quant for the dual-use embedding
        s = (np.abs(emb).max(1) / 127.0); s[s == 0] = 1e-8
        W["embed"] = np.round(emb / s[:, None]).clip(-127, 127).astype(np.int8); W["embed__rowscale"] = s.astype(np.float16)
    else:
        W["embed"] = emb.astype(np.float16 if args.dtype == "float16" else args.dtype)
    put("norm", sd["model.norm.weight"].detach().float().numpy(), False)
    if not c.tie_word_embeddings:
        put("lm_head", sd["lm_head.weight"].detach().float().numpy().T, True)
    for L in range(c.num_hidden_layers):
        pre = f"model.layers.{L}."
        put(f"l{L}.in_ln", sd[pre + "input_layernorm.weight"].detach().float().numpy(), False)
        put(f"l{L}.post_ln", sd[pre + "post_attention_layernorm.weight"].detach().float().numpy(), False)
        for k in PROJ:
            put(f"l{L}.{k}", sd[pre + k + ".weight"].detach().float().numpy().T, True)    # (out,in)→(in,out)
        for b in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
            if pre + b + ".bias" in sd:
                put(f"l{L}.{b}.bias", sd[pre + b + ".bias"].detach().float().numpy(), False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **W)
    print(f"[export] {args.model}: {len(W)} flat arrays · {args.dtype} → {args.out} "
          f"({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
