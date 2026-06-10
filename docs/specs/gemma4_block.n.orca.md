# architecture Gemma4Block

> Gemma-4's block (dense text path): the Gemma-3 backbone (sandwich norms, QK-norm, dual-base RoPE, 5:1
> sliding:full) plus the changes that define Gemma 4 — a **value-norm** (per-head RMS on v, no learnable weight)
> alongside the q/k norms; attention **scaling = 1.0** (QK-norm makes 1/√d unnecessary); a **different head_dim on
> global layers** (512 vs 256 — q/k/v/o shapes differ per layer type); **partial-rotary** "proportional" RoPE on
> global layers (only the first ¼ of frequency pairs rotate); RMSNorm weights used **directly** (no (1+w) bake);
> and the **Per-Layer-Embedding (PLE) gated-residual block** — a per-layer token-identity embedding gated by the
> post-FFN hidden and added back through its own projection + norm. The MoE variant (26B-A4B) adds a sparse top-k
> expert branch summed with the dense MLP. Dims are the Gemma-4 reference config. Verified + rendered with n-orca.

## hyperparameters

| Name     | Type | Default |
|----------|------|---------|
| d_model  | int  | 2304 |
| n_heads  | int  | 8 |
| n_kv     | int  | 4 |
| head_dim | int  | 256 |
| d_ff     | int  | 9216 |
| d_ple    | int  | 256 |
| window   | int  | 512 |

## tensors

| Name   | Shape           | Dtype   |
|--------|-----------------|---------|
| x      | (B, S, d_model) | float32 |
| ple_in | (B, S, d_ple)   | float32 |
| y      | (B, S, d_model) | float32 |

## layer x [input]
> Residual stream in

## layer ple_in [input]
> Per-layer embedding input: √d_ple-scaled token-identity rows + a 1/√d context projection of the input embedding,
> combined and normed once for all layers

## layer pre_attn_norm
> Pre-attention RMSNorm (weight direct, no (1+w))
- op: RMSNorm(d_model)

## layer attn
> MOVE — GQA with QK-norm AND value-norm (weightless RMS on v), scaling=1.0, partial-rotary dual-base RoPE,
> per-layer-type head_dim (256 sliding / 512 global), sliding-window mask on 5 of 6 layers
- op: Gemma4Attention(n_heads, n_kv, head_dim, window)

## layer post_attn_norm
> Post-attention RMSNorm
- op: RMSNorm(d_model)

## layer add_1
> Attention residual
- op: Add

## layer pre_ff_norm
> Pre-MLP RMSNorm
- op: RMSNorm(d_model)

## layer mlp
> COMPUTE — GeGLU gated MLP (the MoE variant sums a routed top-k expert branch with this dense path)
- op: GeGLU(d_model, d_ff)

## layer post_ff_norm
> Post-MLP RMSNorm
- op: RMSNorm(d_model)

## layer add_2
> MLP residual
- op: Add

## layer ple_gate
> PLE gate: project the post-FFN hidden to d_ple, GELU
- op: Linear(d_model, d_ple)

## layer gate_mul
> Gate the per-layer embedding with the hidden (elementwise)
- op: ElementwiseMul

## layer ple_proj
> Project the gated PLE back to the residual width
- op: Linear(d_ple, d_model)

## layer ple_norm
> Post-PLE RMSNorm
- op: RMSNorm(d_model)

## layer add_3
> PLE gated residual
- op: Add

## layer y [output]
> Residual stream out

## flow

| Source         | Target         | Tensor    |
|----------------|----------------|-----------|
| x              | pre_attn_norm  | x         |
| pre_attn_norm  | attn           | x_normed  |
| attn           | post_attn_norm | attn_pre  |
| post_attn_norm | add_1          | attn_out  |
| x              | add_1          | x_skip    |
| add_1          | pre_ff_norm    | h         |
| pre_ff_norm    | mlp            | h_normed  |
| mlp            | post_ff_norm   | mlp_pre   |
| post_ff_norm   | add_2          | mlp_out   |
| add_1          | add_2          | h_skip    |
| add_2          | ple_gate       | h2        |
| ple_gate       | gate_mul       | g         |
| ple_in         | gate_mul       | ple_emb   |
| gate_mul       | ple_proj       | g_ple     |
| ple_proj       | ple_norm       | ple_pre   |
| ple_norm       | add_3          | ple_out   |
| add_2          | add_3          | h2_skip   |
| add_3          | y              | y_out     |

## invariants
- output_shape: (B, S, d_model)
