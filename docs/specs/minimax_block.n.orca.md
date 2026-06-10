# architecture MiniMaxM2Block

> MiniMax-M2's block: the RoPE backbone (pre-RMSNorm, rotary, GQA, SwiGLU experts) with **full-width q/k-norm** —
> a single RMSNorm over the *whole* concatenated projection (n_heads·head_dim for q, n_kv·head_dim for k), not
> per-head like Qwen3/Gemma — and an **all-MoE FFN on every layer** with a **sigmoid router**: sigmoid scores + a
> learned bias pick the top-k, the un-biased sigmoid scores renormed over the top-k weight them. No group
> limiting, no shared expert (unlike DeepSeek), no dense layers. Expert weights ship Mixtral-style
> (`block_sparse_moe.experts.{e}.w1/w2/w3`; mmap-paged per token in fieldrun). Dims are the MiniMax-M2 reference
> config (230B-A10B-class: 256 experts, top-8). Verified + rendered with n-orca.

## hyperparameters

| Name      | Type | Default |
|-----------|------|---------|
| d_model   | int  | 3072 |
| n_heads   | int  | 48 |
| n_kv      | int  | 8 |
| head_dim  | int  | 128 |
| n_experts | int  | 256 |
| top_k     | int  | 8 |
| d_expert  | int  | 1536 |

## tensors

| Name | Shape           | Dtype   |
|------|-----------------|---------|
| x    | (B, S, d_model) | float32 |
| y    | (B, S, d_model) | float32 |

## layer x [input]
> Residual stream in

## layer in_ln
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
> FULL-WIDTH q-norm: one RMSNorm over the whole n_heads·head_dim projection (not per-head)
- op: RMSNorm(n_heads*head_dim)

## layer k_norm
> FULL-WIDTH k-norm over n_kv·head_dim
- op: RMSNorm(n_kv*head_dim)

## layer rope_q
> Single-base rotary on the queries
- op: RoPE(head_dim)

## layer rope_k
> Single-base rotary on the keys
- op: RoPE(head_dim)

## layer attn_core
> MOVE — causal GQA softmax attention
- op: GroupedQueryAttention(n_heads, n_kv, head_dim)

## layer o_proj
> Output projection back to the residual width
- op: Linear(n_heads*head_dim, d_model)

## layer add_1
> Attention residual
- op: Add

## layer post_ln
> Pre-FFN RMSNorm
- op: RMSNorm(d_model)

## layer router
> Sigmoid router: sigmoid scores + learned bias pick the top-k; un-biased sigmoid scores renormed weight them
- op: SigmoidTopKRouter(d_model, n_experts, top_k)

## layer experts
> COMPUTE — n_experts SwiGLU experts (w1=gate, w3=up, w2=down), every layer MoE; top-k run per token
- op: ExpertSwiGLU(d_model, d_expert, n_experts)

## layer moe_sum
> Weighted sum of the top-k expert outputs
- op: WeightedSum

## layer add_2
> FFN residual
- op: Add

## layer y [output]
> Residual stream out

## flow

| Source  | Target    | Tensor       |
|---------|-----------|--------------|
| x       | in_ln     | x            |
| in_ln   | q_proj    | x_normed     |
| in_ln   | k_proj    | x_normed     |
| in_ln   | v_proj    | x_normed     |
| q_proj  | q_norm    | q            |
| k_proj  | k_norm    | k            |
| q_norm  | rope_q    | q_n          |
| k_norm  | rope_k    | k_n          |
| rope_q  | attn_core | q_rot        |
| rope_k  | attn_core | k_rot        |
| v_proj  | attn_core | v            |
| attn_core | o_proj  | attn_mix     |
| o_proj  | add_1     | attn_out     |
| x       | add_1     | x_skip       |
| add_1   | post_ln   | h            |
| post_ln | router    | h_normed     |
| post_ln | experts   | h_tokens     |
| router  | moe_sum   | topk_weights |
| experts | moe_sum   | expert_out   |
| moe_sum | add_2     | moe_out      |
| add_1   | add_2     | h_skip       |
| add_2   | y         | y_out        |

## invariants
- output_shape: (B, S, d_model)
