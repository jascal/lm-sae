"""Per-architecture knobs for the weight-space disassembler — so one tool runs Gemma-2, Llama-3, etc.

The disassembly math (QK/OV bilinears, MLP read/write, addressing, causal ablation) is
architecture-generic across the RoPE / GQA / RMSNorm / gated-MLP family. Only a few constants differ:

  - **RMSNorm gain.** Gemma uses a *zero-centered* RMSNorm, so the effective gain is `1 + weight`;
    Llama / Qwen / Mistral use a plain RMSNorm (`weight`). We read this off the model type.
  - **QK scale.** Gemma-2 sets `query_pre_attn_scalar`; everyone else uses `sqrt(head_dim)`.
  - **Feature-native SAE opcode.** Only run where a per-layer residual SAE dictionary exists
    (Gemma Scope today). For other models the disassembler still produces the universal
    token-centroid QK/OV bind + MLP catalog — just without the SAE-feature opcode extra.

Layer / attention / MLP submodule access (`model.model.layers[L].self_attn.{q,k,v,o}_proj`,
`.mlp.{gate,up,down}_proj`, `.input_layernorm`) is shared by the whole family, so no config needed.
"""
from __future__ import annotations


def is_gemma(model_type) -> bool:
    return str(model_type).lower().startswith("gemma")


def norm_gain(weight_np, model_type):
    """Effective RMSNorm gain vector — folds Gemma's `1 + weight` zero-centering; plain `weight` else."""
    return (1.0 + weight_np) if is_gemma(model_type) else weight_np


def qk_scale(cfg) -> float:
    """QK softmax scale: sqrt(query_pre_attn_scalar) if the model sets it (Gemma-2), else sqrt(head_dim)."""
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    return float(getattr(cfg, "query_pre_attn_scalar", hd)) ** 0.5


def has_feature_sae(model_type) -> bool:
    """Whether a per-layer residual SAE dictionary is available for the feature-native opcode.

    Gemma Scope (`google/gemma-scope-2b-pt-res`) today; extend as more open SAE suites land
    (e.g. Llama Scope for Llama-3.1-8B)."""
    return is_gemma(model_type)
