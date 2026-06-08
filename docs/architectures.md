---
title: Architecture references
---

# Architecture references (n-orca diagrams)

The architectures this project studies, declared as typed-DAG specs in
[**n-orca**](https://github.com/jascal/n-orca) (a Markdown DSL for neural-net architectures that *verifies*
shapes/types and compiles to Mermaid / runnable PyTorch) and rendered here. These are *reference* diagrams — the
hosts we disassemble, plus the SAE the forge-tax sister track acts on — not results. One block per architecture
**family** (models within a family differ only in dims):

- **GPT-2** (small / medium / large) — absolute position, LayerNorm, dense MLP.
- **RoPE family** (Llama-3.2-1B, Qwen-2.5-1.5B) — RoPE, grouped-query attention, RMSNorm, SwiGLU.
- **Gemma-2-2B** — the RoPE outlier: sandwich (pre+post) norm, GeGLU; no sink, distributed COMPUTE.
- **GPT-NeoX** (Pythia 14m → 1.4b) — rotary position, LayerNorm, dense GELU, **parallel residual**; the controlled
  scaling ladder (one architecture, same data, six sizes).
- **Mamba** (130m / 370m / 790m) — state-space mixer, no attention, no separate MLP.

## GPT-2 block — the host the catalog disassembles

One pre-norm GPT-2-small block. **Attention is the MOVE class** (a QK addressing-mode × an OV write-op — the heads
the [operator catalog](operators/README.md) reads); **the MLP is the COMPUTE class** (key–value memories — the
[MLP / COMPUTE catalog](operators/mlp_compute.md); `mlp` at layer 0 is the detokenizer). Verified by n-orca:
**VALID, 7.09M params/block, depth 7.** Spec:
[`docs/specs/gpt2_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/gpt2_block.n.orca.md).

```mermaid
%% architecture GPT2Block
flowchart TD
    x(("x<br/>[input]"))
    ln_1["ln_1<br/>LayerNorm(d_model)"]
    attn["attn — MOVE<br/>MultiHeadAttention(d_model, n_heads, dropout)"]
    add_1["add_1<br/>Add()"]
    ln_2["ln_2<br/>LayerNorm(d_model)"]
    mlp["mlp — COMPUTE<br/>FeedForward(d_model, d_ff, dropout)"]
    add_2["add_2<br/>Add()"]
    y(("y<br/>[output]"))
    x -- "x : (B,S,d_model)" --> ln_1
    ln_1 -- "x_normed : (B,S,d_model)" --> attn
    attn -- "attn_out : (B,S,d_model)" --> add_1
    x -- "x_skip : (B,S,d_model)" --> add_1
    add_1 -- "h : (B,S,d_model)" --> ln_2
    ln_2 -- "h_normed : (B,S,d_model)" --> mlp
    mlp -- "mlp_out : (B,S,d_model)" --> add_2
    add_1 -- "h_skip : (B,S,d_model)" --> add_2
    add_2 -- "y_out : (B,S,d_model)" --> y
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;
    class x input;
    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;
    class y output;
```

The residual stream `x → … → y` is the **bus**; each block reads it (LayerNorm), MOVES (attention) and COMPUTES
(MLP), and writes back (Add). The disassembly reads the operators *inside* the `attn` and `mlp` nodes.
(GPT-2 small/medium/large share this block; they differ only in dims — 768/1024/1280 d_model, 12/16/20 heads.)

## RoPE block — the Llama / Qwen family

Same MOVE+COMPUTE skeleton, but pre-**RMSNorm**, **grouped-query** attention with **rotary positions**, and a
**SwiGLU** gated MLP (no biases). Position lives in the *rotation*, so there is no learned absolute-position
register — and (as the [circuit catalog](circuits/README.md) finds) no positional-broadcast circuit and no
attention sink dependence. Dims: Llama-3.2-1B (2048 / 32 heads / 8 kv / 8192); Qwen-2.5-1.5B (1536 / 12 / 2 / 8960).
Spec: [`docs/specs/rope_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/rope_block.n.orca.md).

```mermaid
%% architecture RoPEBlock
flowchart TD
    x(("x<br/>[input]"))
    attn_norm["attn_norm — RMSNorm(d_model)"]
    attn["attn — MOVE<br/>GroupedQueryAttention(d_model, n_heads, n_kv) + RoPE"]
    add_1["add_1<br/>Add()"]
    ff_norm["ff_norm — RMSNorm(d_model)"]
    mlp["mlp — COMPUTE<br/>SwiGLU(d_model, d_ff)"]
    add_2["add_2<br/>Add()"]
    y(("y<br/>[output]"))
    x -- "x" --> attn_norm
    attn_norm -- "x_normed" --> attn
    attn -- "attn_out" --> add_1
    x -- "x_skip" --> add_1
    add_1 -- "h" --> ff_norm
    ff_norm -- "h_normed" --> mlp
    mlp -- "mlp_out" --> add_2
    add_1 -- "h_skip" --> add_2
    add_2 -- "y_out" --> y
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;
    class x input;
    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;
    class y output;
```

## Gemma-2 block — the architectural outlier

Gemma-2-2B is RoPE+GQA like Llama/Qwen but wraps **each** sublayer in **both** a pre- and a post-RMSNorm (a
sandwich norm) and uses a **GeGLU** MLP. It is the outlier in the catalog: **no attention sink** (0 sink heads vs
117–553 elsewhere) and **distributed COMPUTE** (no single dominant detokenizer MLP). Dims for Gemma-2-2B
(2304 / 8 heads / 4 kv / 9216). Spec:
[`docs/specs/gemma_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/gemma_block.n.orca.md).

```mermaid
%% architecture GemmaBlock
flowchart TD
    x(("x<br/>[input]"))
    pre_attn_norm["pre_attn_norm — RMSNorm"]
    attn["attn — MOVE<br/>GroupedQueryAttention + RoPE"]
    post_attn_norm["post_attn_norm — RMSNorm"]
    add_1["add_1<br/>Add()"]
    pre_ff_norm["pre_ff_norm — RMSNorm"]
    mlp["mlp — COMPUTE<br/>GeGLU(d_model, d_ff)"]
    post_ff_norm["post_ff_norm — RMSNorm"]
    add_2["add_2<br/>Add()"]
    y(("y<br/>[output]"))
    x -- "x" --> pre_attn_norm
    pre_attn_norm -- "x_normed" --> attn
    attn -- "attn_pre" --> post_attn_norm
    post_attn_norm -- "attn_out" --> add_1
    x -- "x_skip" --> add_1
    add_1 -- "h" --> pre_ff_norm
    pre_ff_norm -- "h_normed" --> mlp
    mlp -- "mlp_pre" --> post_ff_norm
    post_ff_norm -- "mlp_out" --> add_2
    add_1 -- "h_skip" --> add_2
    add_2 -- "y_out" --> y
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;
    class x input;
    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;
    class y output;
```

## GPT-NeoX block — the controlled scaling ladder (Pythia)

The **Pythia** ladder (EleutherAI, 14m → 1.4b) is **one GPT-NeoX architecture at six sizes trained on the same data**
— the clean control behind the [scaling laws](scaling.md) (architecture held fixed). The block keeps GPT-2's
**LayerNorm** and **dense GELU** MLP but takes position from a **rotary** embedding (like the RoPE family), with
standard multi-head attention (no GQA). Its one distinctive feature is the **parallel residual**: attention and MLP
both read the *block input* `x` (each through its own LayerNorm) and are summed into the residual *together* —
`y = x + attn(ln_a x) + mlp(ln_m x)` — rather than the serial attention-*then*-MLP of GPT-2 / RoPE / Gemma. Same
MOVE (attention) + COMPUTE (MLP) split, so the arch-generic disassembly (logit-lens read-out, block ablation,
knowledge READ/WRITE) runs on it directly. Verified by n-orca: **VALID, 12.60M params/block (Pythia-410m dims),
depth 5.** Spec: [`specs/gpt_neox_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/specs/gpt_neox_block.n.orca.md).

```mermaid
%% architecture GPTNeoXBlock
flowchart TD
    x(("x<br/>[input]"))
    attn_norm["attn_norm<br/>LayerNorm(d_model)"]
    attn["attn — MOVE<br/>MultiHeadAttention(d_model, n_heads) + RoPE"]
    ff_norm["ff_norm<br/>LayerNorm(d_model)"]
    mlp["mlp — COMPUTE<br/>FeedForward(d_model, d_ff) — dense GELU"]
    add_attn["add_attn<br/>Add()"]
    add_mlp["add_mlp<br/>Add()"]
    y(("y<br/>[output]"))
    x -- "x : (B,S,d_model)" --> attn_norm
    x -- "x_par : (B,S,d_model)" --> ff_norm
    attn_norm -- "x_normed : (B,S,d_model)" --> attn
    ff_norm -- "x_par_normed : (B,S,d_model)" --> mlp
    attn -- "attn_out : (B,S,d_model)" --> add_attn
    x -- "x_skip : (B,S,d_model)" --> add_attn
    add_attn -- "h : (B,S,d_model)" --> add_mlp
    mlp -- "mlp_out : (B,S,d_model)" --> add_mlp
    add_mlp -- "y_out : (B,S,d_model)" --> y
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;
    class x input;
    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;
    class y output;
```

Note `x` fans out to **both** `attn_norm` and `ff_norm` (the parallel residual): the MLP reads the block input, not
the post-attention residual. The disassembly reads the operators inside `attn` and `mlp` exactly as in the other
families. (Pythia sizes scale `d_model`/`n_layers`: 14m d128/6L · 70m d512/6L · 160m d768/12L · 410m d1024/24L ·
1b d2048/16L · 1.4b d2048/24L.)

## Mamba block — the no-attention control (SSM)

Mamba (130m/370m/790m) has **no attention and no separate MLP**: the whole layer is one selective **state-space
mixer** (a learned linear recurrence / scan). The catalog's SSM result: the in-context-copy *capability*
(induction) survives this loss of attention (gain +12.1…+12.5, like the transformers) — but with no heads there is
no head-resolved operator, only a layer. Dims for Mamba-130m (d_model 768, d_inner 1536, d_state 16). Spec:
[`docs/specs/mamba_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/mamba_block.n.orca.md).

```mermaid
%% architecture MambaBlock
flowchart TD
    x(("x<br/>[input]"))
    norm["norm — RMSNorm(d_model)"]
    in_proj["in_proj — Linear(d_model, d_inner)"]
    conv["conv — Conv1d(d_inner, d_conv)"]
    act["act — SiLU()"]
    ssm["ssm — MIX<br/>SelectiveSSM(d_inner, d_state)"]
    gate["gate — ElementwiseMul()"]
    out_proj["out_proj — Linear(d_inner, d_model)"]
    add["add<br/>Add()"]
    y(("y<br/>[output]"))
    x -- "x" --> norm
    norm -- "x_normed" --> in_proj
    in_proj -- "x-branch" --> conv
    conv -- "xc" --> act
    act -- "xa" --> ssm
    ssm -- "scan" --> gate
    in_proj -- "z-gate" --> gate
    gate -- "gated" --> out_proj
    out_proj -- "mix_out" --> add
    x -- "x_skip" --> add
    add -- "y_out" --> y
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;
    class x input;
    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;
    class y output;
```

## Sparse autoencoder — the forge-tax tool (sister track)

A top-K SAE (with an attention pre-mixer): encode the residual into sparse `n_features`, keep the top-K, decode
back. This is the dictionary the [forge-tax track](FORGE_TAX_TRACK.md) measures — *what an SAE feature basis
preserves vs destroys* (it preserves content/mAUC but collapses monosemanticity/cov95). Spec:
[`docs/specs/sae_attn_topk.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/sae_attn_topk.n.orca.md).

```mermaid
%% architecture AttnTopKSae
flowchart TD
    x(("x<br/>[input]"))
    x_hat(("x_hat<br/>[output]"))
    attn["attn<br/>MultiHeadAttention(input_dim, n_heads, attn_dropout)"]
    add_attn["add_attn<br/>Add()"]
    ln["ln<br/>LayerNorm(input_dim)"]
    encoder["encoder<br/>Linear(input_dim, n_features)"]
    relu["relu<br/>ReLU()"]
    topk["topk<br/>TopK(k)"]
    decoder["decoder<br/>Linear(n_features, input_dim)"]
    x -- "x : (B,T,input_dim)" --> attn
    attn -- "attn_out : (B,T,input_dim)" --> add_attn
    x -- "x_skip : (B,T,input_dim)" --> add_attn
    add_attn -- "r : (B,T,input_dim)" --> ln
    ln -- "r_n : (B,T,input_dim)" --> encoder
    encoder -- "z_pre : (B,T,n_features)" --> relu
    relu -- "z_relu : (B,T,n_features)" --> topk
    topk -- "z_sparse : (B,T,n_features)" --> decoder
    decoder -- "x_hat : (B,T,input_dim)" --> x_hat
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;
    class x input;
    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;
    class x_hat output;
```

---

_Diagrams compiled from the committed `.n.orca.md` specs with
[n-orca](https://github.com/jascal/n-orca): `n-orca compile mermaid docs/specs/<spec>.n.orca.md`. Rendered on the
site via mermaid.js._
