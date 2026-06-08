# architecture GPTNeoXBlock

> The GPT-NeoX block — the Pythia controlled scaling ladder (14m → 1.4b: one architecture, the same training data,
> six sizes, used for the controlled scaling laws). Like GPT-2 it keeps **LayerNorm** (not RMSNorm) and a **dense
> GELU** MLP, but takes position from a **rotary** embedding (like the RoPE family) and standard multi-head attention
> (no GQA). Its distinctive feature is the **parallel residual**: attention and MLP both read the *block input* x
> (through their own LayerNorm) and are summed into the residual together — `y = x + attn(ln_a x) + mlp(ln_m x)` —
> rather than the serial attention-then-MLP of GPT-2 / RoPE. Same MOVE (attention) + COMPUTE (MLP) instruction split.
> Dims shown for Pythia-410m (the ladder scales d_model / n_layers: 14m d128/6L · 70m d512/6L · 160m d768/12L ·
> 410m d1024/24L · 1b d2048/16L · 1.4b d2048/24L). Verified + rendered with n-orca.

## hyperparameters

| Name   | Type | Default |
|--------|------|---------|
| d_model | int | 1024 |
| n_heads | int | 16 |
| d_ff    | int | 4096 |

## tensors

| Name | Shape           | Dtype   |
|------|-----------------|---------|
| x    | (B, S, d_model) | float32 |
| y    | (B, S, d_model) | float32 |

## layer x [input]
> Residual stream in

## layer attn_norm
> Pre-attention LayerNorm (input_layernorm)
- op: LayerNorm(d_model)

## layer attn
> MOVE — multi-head self-attention with rotary positions (standard MHA, no GQA)
- op: MultiHeadAttention(d_model, n_heads)

## layer ff_norm
> Pre-MLP LayerNorm — reads the BLOCK INPUT x (the parallel residual)
- op: LayerNorm(d_model)

## layer mlp
> COMPUTE — dense GELU MLP
- op: FeedForward(d_model, d_ff)

## layer add_attn
> Add attention to the residual
- op: Add

## layer add_mlp
> Add the (parallel) MLP output to the residual
- op: Add

## layer y [output]
> Residual stream out

## flow

| Source    | Target    | Tensor       |
|-----------|-----------|--------------|
| x         | attn_norm | x            |
| x         | ff_norm   | x_par        |
| attn_norm | attn      | x_normed     |
| ff_norm   | mlp       | x_par_normed |
| attn      | add_attn  | attn_out     |
| x         | add_attn  | x_skip       |
| add_attn  | add_mlp   | h            |
| mlp       | add_mlp   | mlp_out      |
| add_mlp   | y         | y_out        |

## invariants
- output_shape: (B, S, d_model)
