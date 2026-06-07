# architecture GPT2Block

> The host this project disassembles: one GPT-2-small pre-norm transformer block.
> Attention is the MOVE class (a QK addressing-mode × an OV write-op); the MLP is
> the COMPUTE class (key–value memories). Concrete GPT-2-small dimensions.
> Authored for lm-sae; verified + rendered with n-orca (github.com/jascal/n-orca).

## hyperparameters

| Name    | Type  | Default |
|---------|-------|---------|
| d_model | int   | 768     |
| n_heads | int   | 12      |
| d_ff    | int   | 3072    |
| dropout | float | 0.1     |

## tensors

| Name | Shape            | Dtype   |
|------|------------------|---------|
| x    | (B, S, d_model)  | float32 |
| y    | (B, S, d_model)  | float32 |

## layer x [input]
> Residual stream in (the bus)

## layer ln_1
> Pre-attention LayerNorm
- op: LayerNorm(d_model)

## layer attn
> MOVE — multi-head self-attention (12 heads; QK addressing × OV write). The operator catalog reads these heads.
- op: MultiHeadAttention(d_model, n_heads, dropout)

## layer add_1
> Attention residual add (write-back to the bus)
- op: Add

## layer ln_2
> Pre-MLP LayerNorm
- op: LayerNorm(d_model)

## layer mlp
> COMPUTE — position-wise MLP (the key–value memories; MLP0 = the detokenizer). The MLP/COMPUTE catalog reads these.
- op: FeedForward(d_model, d_ff, dropout)

## layer add_2
> MLP residual add
- op: Add

## layer y [output]
> Residual stream out

## flow

| Source | Target | Tensor    |
|--------|--------|-----------|
| x      | ln_1   | x         |
| ln_1   | attn   | x_normed  |
| attn   | add_1  | attn_out  |
| x      | add_1  | x_skip    |
| add_1  | ln_2   | h         |
| ln_2   | mlp    | h_normed  |
| mlp    | add_2  | mlp_out   |
| add_1  | add_2  | h_skip    |
| add_2  | y      | y_out     |

## invariants
- output_shape: (B, S, d_model)
