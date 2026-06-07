# architecture MambaBlock

> A Mamba (state-space) block — the no-attention control in the catalog. There is **no attention and no separate
> MLP**: the whole layer is a single selective **state-space mixer** (a learned linear recurrence / scan). The
> catalog's SSM result: the in-context-copy *capability* (induction) survives this loss of attention (gain
> +12.1…+12.5, like the transformers) — but with no heads, there is no head-resolved operator to point at, only a
> layer. Dims for Mamba-130m (d_inner = 2×d_model). Verified + rendered with n-orca.

## hyperparameters

| Name    | Type | Default |
|---------|------|---------|
| d_model | int | 768 |
| d_inner | int | 1536 |
| d_state | int | 16 |
| d_conv  | int | 4 |

## tensors

| Name | Shape           | Dtype   |
|------|-----------------|---------|
| x    | (B, S, d_model) | float32 |
| y    | (B, S, d_model) | float32 |

## layer x [input]
> Residual stream in

## layer norm
> Pre-mixer RMSNorm
- op: RMSNorm(d_model)

## layer in_proj
> Project up + split into the scan branch and the gate branch
- op: Linear(d_model, d_inner)

## layer conv
> Depthwise causal 1-D convolution over the sequence (short-range mixing)
- op: Conv1d(d_inner, d_conv)

## layer act
> SiLU activation
- op: SiLU

## layer ssm
> MIX — the selective state-space scan (the sequence mixer; replaces attention)
- op: SelectiveSSM(d_inner, d_state)

## layer gate
> Gated multiplicative interaction (scan output ⊙ SiLU(gate branch))
- op: ElementwiseMul

## layer out_proj
> Project back down to the residual stream
- op: Linear(d_inner, d_model)

## layer add
> Residual
- op: Add

## layer y [output]
> Residual stream out

## flow

| Source   | Target   | Tensor   |
|----------|----------|----------|
| x        | norm     | x        |
| norm     | in_proj  | x_normed |
| in_proj  | conv     | xz       |
| conv     | act      | xc       |
| act      | ssm      | xa       |
| ssm      | gate     | scan     |
| in_proj  | gate     | z        |
| gate     | out_proj | gated    |
| out_proj | add      | mix_out  |
| x        | add      | x_skip   |
| add      | y        | y_out    |

## invariants
- output_shape: (B, S, d_model)
