# architecture DeepSeekV4Block

> DeepSeek-V4's block — a **new attention class**, not MLA. The residual is `hc_mult` parallel streams kept throughout
> the block and mixed by two **manifold-constrained hyper-connections (mHC)**: each `attn_hc`/`ffn_hc` collapses the
> streams into one sequence (a `pre`-weighted sum) for the sublayer, and the update re-places the output with a `post`
> gate plus a **Sinkhorn-projected doubly-stochastic** `comb` mix back across the streams. Attention is **shared-KV MQA**
> (one KV head; `kv_proj → head_dim`; the same tensor is read as K and V), with **q-LoRA** queries (`q_a → q_a_norm →
> q_b → per-head unweighted q_b_norm`), **partial interleaved RoPE** on each head's trailing `rope_dim`, a **per-head
> attention sink** (gpt-oss; an extra softmax logit dropped from the output), an **undo-RoPE** conjugate rotation on the
> attention output (because K==V), and a **grouped low-rank o_proj** (block-diagonal `o_a` per group → `o_b`). The FFN is
> a **sqrtsoftplus** MoE — `softplus(logits).sqrt()` scores, a learned bias picks the top-k, the un-biased scores
> (renormed × routed_scaling) weight them — plus one always-on **shared expert**; both routed and shared experts apply
> gpt-oss SwiGLU clamps. (No group-limited routing, no first_k_dense.) Dims are the V4-Flash reference config. Verified +
> rendered with n-orca; fieldrun's `dsv4` kernel matches `DeepseekV4ForCausalLM` 60/60 top-1 on a tiny random instance
> (Stage 1: the sliding-only backbone — the CSA/HCA compressors + Lightning Indexer are separate regimes).

## hyperparameters

| Name      | Type | Default |
|-----------|------|---------|
| d_model   | int  | 4096 |
| n_heads   | int  | 64 |
| head_dim  | int  | 512 |
| rope_dim  | int  | 64 |
| q_lora    | int  | 1024 |
| o_groups  | int  | 8 |
| o_lora    | int  | 1024 |
| n_experts | int  | 256 |
| top_k     | int  | 6 |
| d_expert  | int  | 2048 |
| hc_mult   | int  | 4 |

## tensors

| Name | Shape           | Dtype   |
|------|-----------------|---------|
| x    | (B, S, d_model) | float32 |
| y    | (B, S, d_model) | float32 |

## layer x [input]
> Residual stream in (carried as hc_mult parallel streams)

## layer attn_hc
> mHC: collapse the hc_mult streams to one sequence (pre-weighted sum) + emit the post/comb update controls
- op: HyperConnection(hc_mult, d_model)

## layer in_ln
> Pre-attention RMSNorm on the collapsed stream
- op: RMSNorm(d_model)

## layer q_a
> Query down-projection to the q-LoRA latent
- op: Linear(d_model, q_lora)

## layer q_a_ln
> Weighted RMSNorm on the q latent
- op: RMSNorm(q_lora)

## layer q_b
> Query up-projection (n_heads · head_dim)
- op: Linear(q_lora, n_heads*head_dim)

## layer q_b_norm
> Per-head UNWEIGHTED RMSNorm over head_dim
- op: RMSNorm(head_dim)

## layer rope_q
> Partial interleaved RoPE on each head's trailing rope_dim
- op: RoPE(rope_dim)

## layer kv
> Single shared-KV-head projection (read as both K and V)
- op: Linear(d_model, head_dim)

## layer kv_ln
> Weighted RMSNorm on the shared KV head
- op: RMSNorm(head_dim)

## layer rope_kv
> Partial interleaved RoPE on the shared KV head's rope_dim
- op: RoPE(rope_dim)

## layer attn_core
> MOVE — shared-KV MQA softmax attention (K==V, one head broadcast to all), sliding-window causal, per-head sink logit
- op: SharedKVSinkAttention(n_heads, head_dim)

## layer undo_rope
> Conjugate RoPE (-sin) on the attention output's rope slice — undoes the RoPE the value carried (K==V)
- op: RoPE(rope_dim)

## layer o_a
> Grouped low-rank output projection: block-diagonal, o_groups independent blocks
- op: GroupedLinear(n_heads*head_dim, o_groups*o_lora)

## layer o_b
> Output mix back to the residual width
- op: Linear(o_groups*o_lora, d_model)

## layer attn_update
> mHC update: post ⊙ attn_out + Sinkhorn(comb)ᵀ · streams (re-place the sublayer output into the hc_mult streams)
- op: HyperConnectionUpdate(hc_mult, d_model)

## layer ffn_hc
> mHC: collapse the streams for the FFN site + emit its post/comb controls
- op: HyperConnection(hc_mult, d_model)

## layer post_ln
> Pre-FFN RMSNorm on the collapsed stream
- op: RMSNorm(d_model)

## layer router
> sqrtsoftplus router: softplus(logits).sqrt() scores; a learned bias picks the top-k; un-biased scores (renorm × scale) weight them
- op: SqrtSoftplusTopKRouter(d_model, n_experts, top_k)

## layer experts
> COMPUTE — n_experts SwiGLU experts (gpt-oss clamps); top-k run per token
- op: ExpertSwiGLU(d_model, d_expert, n_experts)

## layer moe_sum
> Weighted sum of the top-k expert outputs
- op: WeightedSum

## layer shared
> Always-on shared SwiGLU expert (gpt-oss clamps)
- op: FeedForward(d_model, d_expert)

## layer moe_add
> routed + shared
- op: Add

## layer ffn_update
> mHC update: post ⊙ moe_out + Sinkhorn(comb)ᵀ · streams
- op: HyperConnectionUpdate(hc_mult, d_model)

## layer y [output]
> Residual streams out

## flow

| Source      | Target      | Tensor        |
|-------------|-------------|---------------|
| x           | attn_hc     | streams       |
| attn_hc     | in_ln       | collapsed     |
| in_ln       | q_a         | x_normed      |
| q_a         | q_a_ln      | q_lat         |
| q_a_ln      | q_b         | q_lat_n       |
| q_b         | q_b_norm    | q             |
| q_b_norm    | rope_q      | q_n           |
| rope_q      | attn_core   | q_rot         |
| in_ln       | kv          | x_normed      |
| kv          | kv_ln       | kv_raw        |
| kv_ln       | rope_kv     | kv_n          |
| rope_kv     | attn_core   | kv_rot        |
| attn_core   | undo_rope   | attn_mix      |
| undo_rope   | o_a         | attn_unroped  |
| o_a         | o_b         | grouped       |
| o_b         | attn_update | attn_out      |
| x           | attn_update | streams_skip  |
| attn_hc     | attn_update | attn_ctl      |
| attn_update | ffn_hc      | streams_1     |
| ffn_hc      | post_ln     | collapsed_2   |
| post_ln     | router      | h_normed      |
| post_ln     | experts     | h_tokens      |
| post_ln     | shared      | h_shared      |
| router      | moe_sum     | topk_weights  |
| experts     | moe_sum     | expert_out    |
| moe_sum     | moe_add     | routed        |
| shared      | moe_add     | shared_out    |
| moe_add     | ffn_update  | moe_out       |
| attn_update | ffn_update  | streams_skip2 |
| ffn_hc      | ffn_update  | ffn_ctl       |
| ffn_update  | y           | y_out         |

## invariants
- output_shape: (B, S, d_model)
