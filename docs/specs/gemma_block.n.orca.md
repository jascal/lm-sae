# architecture GemmaBlock

> Gemma-2-2B's distinctive block: RoPE + grouped-query attention like Llama/Qwen, but with **both pre- and
> post-RMSNorm** around each sublayer (a sandwich norm) and a **GeGLU** MLP. Gemma is the architectural outlier in
> the catalog — it has **no attention sink** (0 sink heads vs 117–553 in the others) and **distributes COMPUTE**
> across MLP layers rather than concentrating it in an early detokenizer. Dims for Gemma-2-2B. Verified + rendered
> with n-orca.

## hyperparameters

| Name   | Type | Default |
|--------|------|---------|
| d_model | int | 2304 |
| n_heads | int | 8 |
| n_kv    | int | 4 |
| d_ff    | int | 9216 |

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

## layer attn
> MOVE — grouped-query attention with rotary positions (alternating local/global across layers)
- op: GroupedQueryAttention(d_model, n_heads, n_kv)

## layer post_attn_norm
> Post-attention RMSNorm (Gemma-2's sandwich norm)
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

| Source         | Target         | Tensor       |
|----------------|----------------|--------------|
| x              | pre_attn_norm  | x            |
| pre_attn_norm  | attn           | x_normed     |
| attn           | post_attn_norm | attn_pre     |
| post_attn_norm | add_1          | attn_out     |
| x              | add_1          | x_skip       |
| add_1          | pre_ff_norm    | h            |
| pre_ff_norm    | mlp            | h_normed     |
| mlp            | post_ff_norm   | mlp_pre      |
| post_ff_norm   | add_2          | mlp_out      |
| add_1          | add_2          | h_skip       |
| add_2          | y              | y_out        |

## invariants
- output_shape: (B, S, d_model)
