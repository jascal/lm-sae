"""Export a Gemma-2 model to flat .npz for the numpy kernel `numpy_gemma.py`.

Gemma-2 is the hardest of the laptop architectures: on top of the RoPE family's RMSNorm + rotary + grouped-query +
gated MLP it adds (a) an embedding scale of √d, (b) a four-norm 'sandwich' per layer (input / post-attention /
pre-feedforward / post-feedforward), (c) attention-logit and final-logit soft-capping (tanh), (d) GeGLU (gelu-tanh,
not silu), (e) head_dim that is NOT hidden/heads (8×256=2048 ≠ 2304), and (f) RMSNorm computed as x·(1+w).

This export bakes the (1+w) offset into the stored norm weights so the kernel's `rmsnorm` is the plain RoPE one, and
records the soft-caps / scale / query scalar in cfg_f. Default dtype is fp16 — Gemma-2-2b's 256k vocab makes fp32
~10 GB, so the kernel keeps weights low-precision in RAM and upcasts per matmul (the memory strategy the Rust runtime
will use for every model). Linear weights are stored transposed to (in, out) for `x @ W`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
NORMS = ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]


def main(argv=None):
    import torch
    from transformers import AutoModelForCausalLM
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--out", type=Path, default=Path("pylm/weights_gemma2_2b.npz"))
    p.add_argument("--dtype", default="float16", help="float16 (default — fits RAM) · float32 · int8 (per-column quant)")
    args = p.parse_args(argv)

    ld = torch.float16 if args.dtype in ("float16", "int8") else torch.float32   # load fp16 to fit RAM (256k vocab)
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=ld).eval()
    c = m.config; sd = m.state_dict()
    nkv = getattr(c, "num_key_value_heads", c.num_attention_heads)
    hd = getattr(c, "head_dim", None) or (c.hidden_size // c.num_attention_heads)
    theta = getattr(c, "rope_theta", None) or getattr(c, "rope_parameters", {}).get("rope_theta", 1e4)
    qscalar = float(getattr(c, "query_pre_attn_scalar", hd))
    attn_cap = float(getattr(c, "attn_logit_softcapping", 0.0) or 0.0)
    final_cap = float(getattr(c, "final_logit_softcapping", 0.0) or 0.0)

    def col_q(a):
        a = a.astype(np.float32)                                      # quantise in fp32 even if the model is fp16
        s = (np.abs(a).max(0) / 127.0); s[s == 0] = 1e-8
        return np.round(a / s).clip(-127, 127).astype(np.int8), s.astype(np.float16)

    W = {"cfg_i": np.array([c.num_hidden_layers, c.num_attention_heads, nkv, hd, c.hidden_size,
                            c.intermediate_size, c.vocab_size, int(c.tie_word_embeddings)], dtype=np.int64),
         "cfg_f": np.array([float(theta), float(c.rms_norm_eps), attn_cap, final_cap, qscalar,
                            float(c.hidden_size) ** 0.5], dtype=np.float64)}

    fp = np.float16 if args.dtype in ("int8", "float16") else args.dtype

    def put(name, a, quant):
        if args.dtype == "int8" and quant and a.ndim == 2:
            q, s = col_q(a); W[name] = q; W[name + "__scale"] = s
        else:
            W[name] = a.astype(fp)

    def norm(name):                                                   # bake Gemma's (1+w) offset into the stored weight
        return (1.0 + sd[name].detach().float().numpy()).astype(fp)

    W["embed"] = sd["model.embed_tokens.weight"].detach().numpy().astype(fp)   # (vocab,d) keep model dtype — no fp32 temp
    W["norm"] = norm("model.norm.weight")
    if not c.tie_word_embeddings:
        put("lm_head", sd["lm_head.weight"].detach().numpy().T, True)
    for L in range(c.num_hidden_layers):
        pre = f"model.layers.{L}."
        for nm in NORMS:
            W[f"l{L}.{nm}"] = norm(pre + nm + ".weight")
        for k in PROJ:
            put(f"l{L}.{k}", sd[pre + k + ".weight"].detach().numpy().T, True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **W)
    print(f"[export] {args.model}: {len(W)} flat arrays · {args.dtype} → {args.out} "
          f"({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
