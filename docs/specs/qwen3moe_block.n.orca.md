# architecture Qwen3MoeBlock

> Qwen3-MoE's block: the RoPE backbone (pre-RMSNorm, single-base rotary, GQA, no attention bias) plus **QK-norm**
> (per-head RMSNorm on q/k before RoPE) and a **sparse MoE FFN** — a plain-gate router (softmax over all experts →
> top-k → renorm) over per-expert SwiGLU MLPs; only the routed top-k experts run per token (in fieldrun they page
> in from an mmap, so resident set ≠ total params). Layers excluded by `decoder_sparse_step`/`mlp_only_layers` use
> a dense SwiGLU instead. Optional **sliding-window attention** applies one window to *every* layer (no per-layer
> pattern, unlike Gemma). Dims are the Qwen3-MoE reference config (30B-A3B-class: 128 experts, top-8). Verified +
> rendered with n-orca.

## hyperparameters

| Name      | Type | Default |
|-----------|------|---------|
| d_model   | int  | 2048 |
| n_heads   | int  | 32 |
| n_kv      | int  | 4 |
| head_dim  | int  | 128 |
| n_experts | int  | 128 |
| top_k     | int  | 8 |
| d_expert  | int  | 768 |

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

## layer attn
> MOVE — GQA with per-head QK-norm before RoPE; optional all-layer sliding window
- op: GroupedQueryAttention(d_model, n_heads, n_kv)

## layer add_1
> Attention residual
- op: Add

## layer post_ln
> Pre-FFN RMSNorm (the router and experts both read this)
- op: RMSNorm(d_model)

## layer router
> Plain-gate router: Linear(d_model, n_experts) → softmax → top-k → renormalise the kept weights
- op: SoftmaxTopKRouter(d_model, n_experts, top_k)

## layer experts
> COMPUTE — n_experts per-expert SwiGLU MLPs; only each token's top-k run (mmap-paged in fieldrun)
- op: ExpertSwiGLU(d_model, d_expert, n_experts)

## layer moe_sum
> Weighted sum of the top-k expert outputs (router weights)
- op: WeightedSum

## layer add_2
> FFN residual
- op: Add

## layer y [output]
> Residual stream out

## flow

| Source  | Target  | Tensor      |
|---------|---------|-------------|
| x       | in_ln   | x           |
| in_ln   | attn    | x_normed    |
| attn    | add_1   | attn_out    |
| x       | add_1   | x_skip      |
| add_1   | post_ln | h           |
| post_ln | router  | h_normed    |
| post_ln | experts | h_tokens    |
| router  | moe_sum | topk_weights |
| experts | moe_sum | expert_out  |
| moe_sum | add_2   | moe_out     |
| add_1   | add_2   | h_skip      |
| add_2   | y       | y_out       |

## invariants
- output_shape: (B, S, d_model)
