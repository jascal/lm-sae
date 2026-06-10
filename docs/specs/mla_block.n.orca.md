# architecture MlaBlock

> DeepSeek-V3 / Kimi-K2's block: **MLA (multi-head latent attention)** + DeepSeek MoE. MLA compresses attention
> through low-rank latents: q goes d → q_lora → per-head [no-RoPE 128 ‖ RoPE 64] via q_a/q_b (RMSNorm on the
> latent); kv goes d → (kv_lora ‖ 64-dim rope slice) via one kv_a projection — the latent half is normed and
> expanded by kv_b to per-head [k_nope 128 ‖ v 128], while the rope slice becomes a **single shared key** (MQA-
> style, broadcast to all 128 heads). v_head_dim (128) ≠ qk_head_dim (192). Rotary uses **YaRN** long-context
> scaling (ramp-blended inv_freq, mscale attention factor, mscale² softmax-scale correction) and DeepSeek's
> **interleaved** rotary layout. The MoE is an always-on **shared expert** plus **group-limited sigmoid routing**
> (sigmoid scores + a learned bias pick the experts; the un-biased sigmoid scores, renormed and scaled, weight
> them); the first `first_k_dense_replace` layers are dense. Dims are the DeepSeek-V3 reference config (671B-
> A37B-class: 256 routed experts, top-8 from 4 of 8 groups). Verified + rendered with n-orca.

## hyperparameters

| Name      | Type | Default |
|-----------|------|---------|
| d_model   | int  | 7168 |
| n_heads   | int  | 128 |
| q_lora    | int  | 1536 |
| kv_lora   | int  | 512 |
| qk_nope   | int  | 128 |
| qk_rope   | int  | 64 |
| v_head    | int  | 128 |
| n_experts | int  | 256 |
| top_k     | int  | 8 |
| d_expert  | int  | 2048 |

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

## layer q_a
> Query down-projection to the q latent
- op: Linear(d_model, q_lora)

## layer q_a_ln
> RMSNorm on the q latent
- op: RMSNorm(q_lora)

## layer q_b
> Query up-projection: per head [qk_nope ‖ qk_rope]
- op: Linear(q_lora, n_heads*(qk_nope+qk_rope))

## layer rope_q
> RoPE on each head's 64-dim rope slice (YaRN-scaled, interleaved layout)
- op: YarnRoPE(qk_rope)

## layer kv_a
> One kv down-projection: [kv latent (kv_lora) ‖ shared rope key (qk_rope)]
- op: Linear(d_model, kv_lora+qk_rope)

## layer kv_a_ln
> RMSNorm on the kv latent half
- op: RMSNorm(kv_lora)

## layer kv_b
> KV up-projection: per head [k_nope ‖ v]
- op: Linear(kv_lora, n_heads*(qk_nope+v_head))

## layer rope_k
> RoPE on the SINGLE shared rope key (MQA-style, broadcast to all heads)
- op: YarnRoPE(qk_rope)

## layer attn_core
> MOVE — softmax attention at qk_head_dim = qk_nope+qk_rope (192), values at v_head (128);
> scale = qk_head_dim^-0.5 · mscale² (YaRN)
- op: MlaAttention(n_heads, qk_nope, qk_rope, v_head)

## layer o_proj
> Output projection back to the residual width
- op: Linear(n_heads*v_head, d_model)

## layer add_1
> Attention residual
- op: Add

## layer post_ln
> Pre-FFN RMSNorm (shared expert, router, and routed experts all read this)
- op: RMSNorm(d_model)

## layer shared_expert
> Always-on shared SwiGLU expert
- op: SwiGLU(d_model, d_expert)

## layer router
> Group-limited sigmoid router: sigmoid scores + learned bias pick (8 groups → top-4 groups → top-8 experts);
> the un-biased sigmoid scores, renormed over the top-k and ×routed_scaling, are the weights
- op: GroupSigmoidRouter(d_model, n_experts, top_k)

## layer experts
> COMPUTE — n_experts routed SwiGLU experts; only each token's top-k run (mmap-paged in fieldrun)
- op: ExpertSwiGLU(d_model, d_expert, n_experts)

## layer moe_sum
> Weighted sum of the routed expert outputs
- op: WeightedSum

## layer moe_add
> Routed + shared expert
- op: Add

## layer add_2
> FFN residual
- op: Add

## layer y [output]
> Residual stream out

## flow

| Source        | Target        | Tensor       |
|---------------|---------------|--------------|
| x             | in_ln         | x            |
| in_ln         | q_a           | x_normed     |
| q_a           | q_a_ln        | q_lat        |
| q_a_ln        | q_b           | q_lat_n      |
| q_b           | rope_q        | q_heads      |
| in_ln         | kv_a          | x_normed_kv  |
| kv_a          | kv_a_ln       | kv_lat       |
| kv_a          | rope_k        | k_rope_shared |
| kv_a_ln       | kv_b          | kv_lat_n     |
| kv_b          | attn_core     | k_nope_v     |
| rope_q        | attn_core     | q_rot        |
| rope_k        | attn_core     | k_rot        |
| attn_core     | o_proj        | attn_mix     |
| o_proj        | add_1         | attn_out     |
| x             | add_1         | x_skip       |
| add_1         | post_ln       | h            |
| post_ln       | shared_expert | h_normed_s   |
| post_ln       | router        | h_normed_r   |
| post_ln       | experts       | h_tokens     |
| router        | moe_sum       | topk_weights |
| experts       | moe_sum       | expert_out   |
| moe_sum       | moe_add       | routed_out   |
| shared_expert | moe_add       | shared_out   |
| moe_add       | add_2         | moe_out      |
| add_1         | add_2         | h_skip       |
| add_2         | y             | y_out        |

## invariants
- output_shape: (B, S, d_model)
