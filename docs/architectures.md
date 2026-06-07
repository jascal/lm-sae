---
title: Architecture references
---

# Architecture references (n-orca diagrams)

The architectures this project studies, declared as typed-DAG specs in
[**n-orca**](https://github.com/jascal/n-orca) (a Markdown DSL for neural-net architectures that *verifies*
shapes/types and compiles to Mermaid / runnable PyTorch) and rendered here. These are *reference* diagrams — the
host we disassemble, and the SAE the forge-tax sister track acts on — not results.

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
