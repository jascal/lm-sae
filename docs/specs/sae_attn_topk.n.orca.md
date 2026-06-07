# architecture AttnTopKSae

> Attention-prefixed Top-k sparse autoencoder (k=8, n_heads=2, attn_dropout=0.0). A MultiHeadAttention block (with residual + LayerNorm) precedes the SAE encoder so each position can see cross-sequence context before being encoded. Sparsity is enforced by TopK; no L1 / L0 penalty needed.

## hyperparameters

| Name         | Type  | Default |
|--------------|-------|---------|
| input_dim    | int   | 128     |
| n_features   | int   | 512     |
| k            | int   | 8       |
| n_heads      | int   | 2       |
| attn_dropout | float | 0.0     |

## tensors

| Name  | Shape             | Dtype   |
|-------|-------------------|---------|
| x     | (B, T, input_dim) | float32 |
| x_hat | (B, T, input_dim) | float32 |

## layer x [input]
> LLM / world-model activation to reconstruct

## layer x_hat [output]
> Reconstructed activation

## layer attn
> Self-attention across the sequence dimension
- op: MultiHeadAttention(input_dim, n_heads, attn_dropout)

## layer add_attn
> Residual add: attention output + original x
- op: Add()

## layer ln
- op: LayerNorm(input_dim)

## layer encoder
- op: Linear(input_dim, n_features)

## layer relu
- op: ReLU()

## layer topk
> Keep only the k largest features per position
- op: TopK(k)

## layer decoder
- op: Linear(n_features, input_dim)

## flow

| Source   | Target   | Tensor   |
|----------|----------|----------|
| x        | attn     | x        |
| attn     | add_attn | attn_out |
| x        | add_attn | x_skip   |
| add_attn | ln       | r        |
| ln       | encoder  | r_n      |
| encoder  | relu     | z_pre    |
| relu     | topk     | z_relu   |
| topk     | decoder  | z_sparse |
| decoder  | x_hat    | x_hat    |

## invariants
- output_shape: (B, T, input_dim)

## verification rules
- reconstruction-shape: x_hat must match x's shape exactly
