# architecture Gemma3Block

> Gemma-3's block: the Gemma-2 sandwich-norm skeleton (pre+post RMSNorm around each sublayer, GeGLU MLP) with the
> two changes that define Gemma 3 — **QK-norm** (a per-head RMSNorm on the query and key projections *before* RoPE,
> replacing Gemma-2's attention-logit soft-cap) and **dual-base RoPE** (sliding/local layers rotate at θ≈10k, full/
> global layers at θ≈1M) over a **5:1 sliding:full layer pattern**. No soft-capping anywhere. The attention is
> expanded here to show where QK-norm sits; dims are the Gemma-3 reference config (head_dim 256 ≠ d_model/n_heads).
> Verified + rendered with n-orca.

## hyperparameters

| Name     | Type | Default |
|----------|------|---------|
| d_model  | int  | 2304 |
| n_heads  | int  | 8 |
| n_kv     | int  | 4 |
| head_dim | int  | 256 |
| d_ff     | int  | 9216 |
| window   | int  | 4096 |

## tensors

| Name | Shape           | Dtype   |
|------|-----------------|---------|
| x    | (B, S, d_model) | float32 |
| y    | (B, S, d_model) | float32 |

## layer x [input]
> Residual stream in

## layer pre_attn_norm
> Pre-attention RMSNorm
- op: RMSNorm(d_model)

## layer q_proj
> Query projection (n_heads · head_dim)
- op: Linear(d_model, n_heads*head_dim)

## layer k_proj
> Key projection (GQA: n_kv heads)
- op: Linear(d_model, n_kv*head_dim)

## layer v_proj
> Value projection (GQA: n_kv heads)
- op: Linear(d_model, n_kv*head_dim)

## layer q_norm
> QK-norm — per-head RMSNorm over head_dim, before RoPE (replaces Gemma-2's logit soft-cap)
- op: PerHeadRMSNorm(head_dim)

## layer k_norm
> QK-norm on the keys
- op: PerHeadRMSNorm(head_dim)

## layer rope_q
> Dual-base rotary: θ_local≈10k on sliding layers, θ_global≈1M on full layers
- op: RoPE(head_dim)

## layer rope_k
> Same per-layer base as rope_q
- op: RoPE(head_dim)

## layer attn_core
> MOVE — masked softmax attention; sliding-window (5 of every 6 layers) or full causal
- op: SlidingCausalAttention(n_heads, n_kv, head_dim, window)

## layer o_proj
> Output projection back to the residual width
- op: Linear(n_heads*head_dim, d_model)

## layer post_attn_norm
> Post-attention RMSNorm (the sandwich norm)
- op: RMSNorm(d_model)

## layer add_1
> Attention residual
- op: Add

## layer pre_ff_norm
> Pre-MLP RMSNorm
- op: RMSNorm(d_model)

## layer mlp
> COMPUTE — GeGLU gated MLP
- op: GeGLU(d_model, d_ff)

## layer post_ff_norm
> Post-MLP RMSNorm
- op: RMSNorm(d_model)

## layer add_2
> MLP residual
- op: Add

## layer y [output]
> Residual stream out

## flow

| Source         | Target         | Tensor    |
|----------------|----------------|-----------|
| x              | pre_attn_norm  | x         |
| pre_attn_norm  | q_proj         | x_normed  |
| pre_attn_norm  | k_proj         | x_normed  |
| pre_attn_norm  | v_proj         | x_normed  |
| q_proj         | q_norm         | q         |
| k_proj         | k_norm         | k         |
| q_norm         | rope_q         | q_n       |
| k_norm         | rope_k         | k_n       |
| rope_q         | attn_core      | q_rot     |
| rope_k         | attn_core      | k_rot     |
| v_proj         | attn_core      | v         |
| attn_core      | o_proj         | attn_mix  |
| o_proj         | post_attn_norm | attn_pre  |
| post_attn_norm | add_1          | attn_out  |
| x              | add_1          | x_skip    |
| add_1          | pre_ff_norm    | h         |
| pre_ff_norm    | mlp            | h_normed  |
| mlp            | post_ff_norm   | mlp_pre   |
| post_ff_norm   | add_2          | mlp_out   |
| add_1          | add_2          | h_skip    |
| add_2          | y              | y_out     |

## invariants
- output_shape: (B, S, d_model)
