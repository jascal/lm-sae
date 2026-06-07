# architecture RoPEBlock

> The canonical RoPE-family transformer block (Llama-3.2-1B / Qwen-2.5-1.5B): pre-RMSNorm, grouped-query
> self-attention with rotary positions, SwiGLU gated MLP, no biases. Same MOVE (attention) + COMPUTE (MLP)
> structure as GPT-2 — but position lives in the rotation (RoPE), so there is no learned absolute-position
> register and (as the catalog finds) no positional-broadcast circuit. Dims shown for Llama-3.2-1B
> (Qwen-2.5-1.5B: d_model 1536, n_heads 12, n_kv 2, d_ff 8960). Verified + rendered with n-orca.

## hyperparameters

| Name   | Type | Default |
|--------|------|---------|
| d_model | int | 2048 |
| n_heads | int | 32 |
| n_kv    | int | 8 |
| d_ff    | int | 8192 |

## tensors

| Name | Shape           | Dtype   |
|------|-----------------|---------|
| x    | (B, S, d_model) | float32 |
| y    | (B, S, d_model) | float32 |

## layer x [input]
> Residual stream in

## layer attn_norm
> Pre-attention RMSNorm (no mean-subtraction)
- op: RMSNorm(d_model)

## layer attn
> MOVE — grouped-query self-attention with rotary positions (n_kv shared key/value heads; position in the rotation)
- op: GroupedQueryAttention(d_model, n_heads, n_kv)

## layer add_1
> Attention residual
- op: Add

## layer ff_norm
> Pre-MLP RMSNorm
- op: RMSNorm(d_model)

## layer mlp
> COMPUTE — SwiGLU gated MLP (gate ⊙ up, then down)
- op: SwiGLU(d_model, d_ff)

## layer add_2
> MLP residual
- op: Add

## layer y [output]
> Residual stream out

## flow

| Source    | Target    | Tensor    |
|-----------|-----------|-----------|
| x         | attn_norm | x         |
| attn_norm | attn      | x_normed  |
| attn      | add_1     | attn_out  |
| x         | add_1     | x_skip    |
| add_1     | ff_norm   | h         |
| ff_norm   | mlp       | h_normed  |
| mlp       | add_2     | mlp_out   |
| add_1     | add_2     | h_skip    |
| add_2     | y         | y_out     |

## invariants
- output_shape: (B, S, d_model)
