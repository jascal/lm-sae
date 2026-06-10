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

Plus the **frontier families the [fieldrun](https://github.com/jascal/fieldrun) runtime executes** (the
distribution form of the pylm decompilation; each kernel validated top-1 vs a torch reference):

- **Gemma-3** — Gemma-2's sandwich skeleton + QK-norm (replacing the soft-cap) + dual-base RoPE, 5:1 sliding:full.
- **Gemma-4** — + value-norm, per-layer-type head_dim, partial-rotary global RoPE, and the Per-Layer-Embedding
  gated residual (dense and MoE variants).
- **Qwen3-MoE** — the RoPE backbone + per-head QK-norm + a softmax-routed sparse-expert FFN (optional all-layer
  sliding window).
- **MLA** (DeepSeek-V3/R1, Kimi-K2) — multi-head *latent* attention (low-rank q/kv, shared rope key, YaRN) +
  shared-expert / group-limited sigmoid MoE.
- **MiniMax-M2** — full-width q/k-norm + an all-MoE sigmoid-routed FFN on every layer.

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

## Gemma-3 block — QK-norm + dual-base RoPE

The Gemma-2 sandwich-norm skeleton with the two changes that define Gemma 3: **QK-norm** (a per-head RMSNorm on
q/k *before* RoPE — it replaces Gemma-2's attention-logit soft-cap, so the attention is expanded here to show where
it sits) and **dual-base RoPE** (sliding/local layers rotate at θ≈10k, full/global at θ≈1M) over a **5:1
sliding:full** layer pattern. No soft-capping anywhere. Same MOVE (attention) + COMPUTE (GeGLU MLP) split. Dims for
the Gemma-3 reference config (d_model 2304, 8 heads / 4 kv, head_dim 256 ≠ d_model/n_heads, d_ff 9216). Verified by
n-orca: **VALID, depth 13.** Spec:
[`docs/specs/gemma3_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/gemma3_block.n.orca.md).

```mermaid
%% architecture Gemma3Block
flowchart TD
    x(("x<br/>[input]"))
    pre_attn_norm["pre_attn_norm<br/>RMSNorm(d_model)"]
    q_proj["q_proj<br/>Linear(d_model, n_heads*head_dim)"]
    k_proj["k_proj<br/>Linear(d_model, n_kv*head_dim)"]
    v_proj["v_proj<br/>Linear(d_model, n_kv*head_dim)"]
    q_norm["q_norm<br/>PerHeadRMSNorm(head_dim)"]
    k_norm["k_norm<br/>PerHeadRMSNorm(head_dim)"]
    rope_q["rope_q<br/>RoPE(head_dim)"]
    rope_k["rope_k<br/>RoPE(head_dim)"]
    attn_core["attn_core<br/>SlidingCausalAttention(n_heads, n_kv, head_dim, window)"]
    o_proj["o_proj<br/>Linear(n_heads*head_dim, d_model)"]
    post_attn_norm["post_attn_norm<br/>RMSNorm(d_model)"]
    add_1["add_1<br/>Add()"]
    pre_ff_norm["pre_ff_norm<br/>RMSNorm(d_model)"]
    mlp["mlp<br/>GeGLU(d_model, d_ff)"]
    post_ff_norm["post_ff_norm<br/>RMSNorm(d_model)"]
    add_2["add_2<br/>Add()"]
    y(("y<br/>[output]"))
    x -- "x : (B,S,d_model)" --> pre_attn_norm
    pre_attn_norm -- "x_normed" --> q_proj
    pre_attn_norm -- "x_normed" --> k_proj
    pre_attn_norm -- "x_normed" --> v_proj
    q_proj -- "q" --> q_norm
    k_proj -- "k" --> k_norm
    q_norm -- "q_n" --> rope_q
    k_norm -- "k_n" --> rope_k
    rope_q -- "q_rot" --> attn_core
    rope_k -- "k_rot" --> attn_core
    v_proj -- "v" --> attn_core
    attn_core -- "attn_mix" --> o_proj
    o_proj -- "attn_pre" --> post_attn_norm
    post_attn_norm -- "attn_out" --> add_1
    x -- "x_skip : (B,S,d_model)" --> add_1
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

## Gemma-4 block — value-norm + Per-Layer Embeddings

The Gemma-3 backbone plus Gemma 4's changes: a **value-norm** (per-head RMS on v, no learnable weight) beside the
q/k norms, attention **scaling = 1.0**, a **different head_dim on global layers** (512 vs 256), **partial-rotary**
RoPE on global layers (only the first ¼ of frequency pairs rotate), and — the structural novelty — the
**Per-Layer-Embedding (PLE) gated-residual block**: a per-layer token-identity embedding, gated by the post-FFN
hidden (GELU of a d→d_ple projection), projected back to the residual through its own norm. The MoE variant
(26B-A4B) sums a routed top-k expert branch with the dense MLP. Dims for the Gemma-4 reference config (d_model 2304,
8 heads / 4 kv, head_dim 256/512, d_ple 256). Verified by n-orca: **VALID, depth 14.** Spec:
[`docs/specs/gemma4_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/gemma4_block.n.orca.md).

```mermaid
%% architecture Gemma4Block
flowchart TD
    x(("x<br/>[input]"))
    ple_in(("ple_in<br/>[input]"))
    pre_attn_norm["pre_attn_norm<br/>RMSNorm(d_model)"]
    attn["attn<br/>Gemma4Attention(n_heads, n_kv, head_dim, window)"]
    post_attn_norm["post_attn_norm<br/>RMSNorm(d_model)"]
    add_1["add_1<br/>Add()"]
    pre_ff_norm["pre_ff_norm<br/>RMSNorm(d_model)"]
    mlp["mlp<br/>GeGLU(d_model, d_ff)"]
    post_ff_norm["post_ff_norm<br/>RMSNorm(d_model)"]
    add_2["add_2<br/>Add()"]
    ple_gate["ple_gate<br/>Linear(d_model, d_ple)"]
    gate_mul["gate_mul<br/>ElementwiseMul()"]
    ple_proj["ple_proj<br/>Linear(d_ple, d_model)"]
    ple_norm["ple_norm<br/>RMSNorm(d_model)"]
    add_3["add_3<br/>Add()"]
    y(("y<br/>[output]"))
    x -- "x : (B,S,d_model)" --> pre_attn_norm
    pre_attn_norm -- "x_normed" --> attn
    attn -- "attn_pre" --> post_attn_norm
    post_attn_norm -- "attn_out" --> add_1
    x -- "x_skip : (B,S,d_model)" --> add_1
    add_1 -- "h" --> pre_ff_norm
    pre_ff_norm -- "h_normed" --> mlp
    mlp -- "mlp_pre" --> post_ff_norm
    post_ff_norm -- "mlp_out" --> add_2
    add_1 -- "h_skip" --> add_2
    add_2 -- "h2" --> ple_gate
    ple_gate -- "g" --> gate_mul
    ple_in -- "ple_emb : (B,S,d_ple)" --> gate_mul
    gate_mul -- "g_ple" --> ple_proj
    ple_proj -- "ple_pre" --> ple_norm
    ple_norm -- "ple_out" --> add_3
    add_2 -- "h2_skip" --> add_3
    add_3 -- "y_out" --> y
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;
    class x,ple_in input;
    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;
    class y output;
```

## Qwen3-MoE block — softmax-routed sparse experts

The RoPE backbone (pre-RMSNorm, GQA, single-base rotary, no attention bias) + per-head QK-norm, with the FFN
replaced by a **sparse MoE**: a plain-gate router (softmax over all experts → top-k → renorm) over per-expert SwiGLU
MLPs — only each token's top-k experts run (in fieldrun they page in from an mmap, so the resident set is the shared
layers + hot experts, not the whole model). Optional **sliding window** applies one window to *every* layer (no
per-layer pattern, unlike Gemma). The MOVE class is unchanged; the COMPUTE class becomes *conditional* — which
expert computes is input-dependent. Dims for the Qwen3-MoE reference config (30B-A3B-class: d_model 2048, 32 heads /
4 kv, 128 experts, top-8, d_expert 768). Verified by n-orca: **VALID, depth 8.** Spec:
[`docs/specs/qwen3moe_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/qwen3moe_block.n.orca.md).

```mermaid
%% architecture Qwen3MoeBlock
flowchart TD
    x(("x<br/>[input]"))
    in_ln["in_ln<br/>RMSNorm(d_model)"]
    attn["attn<br/>GroupedQueryAttention(d_model, n_heads, n_kv)"]
    add_1["add_1<br/>Add()"]
    post_ln["post_ln<br/>RMSNorm(d_model)"]
    router["router<br/>SoftmaxTopKRouter(d_model, n_experts, top_k)"]
    experts["experts<br/>ExpertSwiGLU(d_model, d_expert, n_experts)"]
    moe_sum["moe_sum<br/>WeightedSum()"]
    add_2["add_2<br/>Add()"]
    y(("y<br/>[output]"))
    x -- "x : (B,S,d_model)" --> in_ln
    in_ln -- "x_normed" --> attn
    attn -- "attn_out" --> add_1
    x -- "x_skip : (B,S,d_model)" --> add_1
    add_1 -- "h" --> post_ln
    post_ln -- "h_normed" --> router
    post_ln -- "h_tokens" --> experts
    router -- "topk_weights" --> moe_sum
    experts -- "expert_out" --> moe_sum
    moe_sum -- "moe_out" --> add_2
    add_1 -- "h_skip" --> add_2
    add_2 -- "y_out" --> y
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;
    class x input;
    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;
    class y output;
```

## MLA block — DeepSeek-V3 / Kimi-K2 latent attention

**Multi-head latent attention**, the last new attention class in the supported set: q and kv are compressed
through low-rank latents (q: d → 1536 → per-head [no-RoPE 128 ‖ RoPE 64]; kv: one projection to [512-dim latent ‖
64-dim rope slice], the latent expanded per head to [k_nope ‖ v]). The rope slice of the key is a **single shared
vector** (MQA-style) broadcast to all 128 heads; v_head_dim (128) ≠ qk_head_dim (192). Rotary is **YaRN**-scaled
(ramp-blended inv_freq, mscale attention factor, mscale² softmax correction) in DeepSeek's **interleaved** layout.
The MoE adds an always-on **shared expert** to **group-limited sigmoid routing** (bias-corrected scores *choose* the
experts, un-biased scores *weight* them); the first `first_k_dense_replace` layers are dense. Dims for the
DeepSeek-V3 reference config (671B-A37B-class: d_model 7168, 128 heads, 256 routed experts top-8 from 4 of 8
groups). Verified by n-orca: **VALID, depth 14.** Spec:
[`docs/specs/mla_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/mla_block.n.orca.md).

```mermaid
%% architecture MlaBlock
flowchart TD
    x(("x<br/>[input]"))
    in_ln["in_ln<br/>RMSNorm(d_model)"]
    q_a["q_a<br/>Linear(d_model, q_lora)"]
    q_a_ln["q_a_ln<br/>RMSNorm(q_lora)"]
    q_b["q_b<br/>Linear(q_lora, n_heads*(qk_nope+qk_rope))"]
    rope_q["rope_q<br/>YarnRoPE(qk_rope)"]
    kv_a["kv_a<br/>Linear(d_model, kv_lora+qk_rope)"]
    kv_a_ln["kv_a_ln<br/>RMSNorm(kv_lora)"]
    kv_b["kv_b<br/>Linear(kv_lora, n_heads*(qk_nope+v_head))"]
    rope_k["rope_k<br/>YarnRoPE(qk_rope)"]
    attn_core["attn_core<br/>MlaAttention(n_heads, qk_nope, qk_rope, v_head)"]
    o_proj["o_proj<br/>Linear(n_heads*v_head, d_model)"]
    add_1["add_1<br/>Add()"]
    post_ln["post_ln<br/>RMSNorm(d_model)"]
    shared_expert["shared_expert<br/>SwiGLU(d_model, d_expert)"]
    router["router<br/>GroupSigmoidRouter(d_model, n_experts, top_k)"]
    experts["experts<br/>ExpertSwiGLU(d_model, d_expert, n_experts)"]
    moe_sum["moe_sum<br/>WeightedSum()"]
    moe_add["moe_add<br/>Add()"]
    add_2["add_2<br/>Add()"]
    y(("y<br/>[output]"))
    x -- "x : (B,S,d_model)" --> in_ln
    in_ln -- "x_normed" --> q_a
    q_a -- "q_lat" --> q_a_ln
    q_a_ln -- "q_lat_n" --> q_b
    q_b -- "q_heads" --> rope_q
    in_ln -- "x_normed_kv" --> kv_a
    kv_a -- "kv_lat" --> kv_a_ln
    kv_a -- "k_rope_shared" --> rope_k
    kv_a_ln -- "kv_lat_n" --> kv_b
    kv_b -- "k_nope_v" --> attn_core
    rope_q -- "q_rot" --> attn_core
    rope_k -- "k_rot" --> attn_core
    attn_core -- "attn_mix" --> o_proj
    o_proj -- "attn_out" --> add_1
    x -- "x_skip : (B,S,d_model)" --> add_1
    add_1 -- "h" --> post_ln
    post_ln -- "h_normed_s" --> shared_expert
    post_ln -- "h_normed_r" --> router
    post_ln -- "h_tokens" --> experts
    router -- "topk_weights" --> moe_sum
    experts -- "expert_out" --> moe_sum
    moe_sum -- "routed_out" --> moe_add
    shared_expert -- "shared_out" --> moe_add
    moe_add -- "moe_out" --> add_2
    add_1 -- "h_skip" --> add_2
    add_2 -- "y_out" --> y
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;
    class x input;
    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;
    class y output;
```

## MiniMax-M2 block — full-width q/k-norm, all-MoE

The RoPE backbone with two distinctive choices: **full-width q/k-norm** — one RMSNorm over the *whole*
concatenated projection (n_heads·head_dim for q, n_kv·head_dim for k), not per-head like Qwen3/Gemma — and an
**all-MoE FFN on every layer** with a **sigmoid router** (sigmoid scores + a learned bias choose the top-k; the
un-biased scores, renormed, weight them). No group limiting, no shared expert, no dense layers — the leanest of the
frontier-MoE recipes. Dims for the MiniMax-M2 reference config (230B-A10B-class: d_model 3072, 48 heads / 8 kv, 256
experts, top-8, d_expert 1536). Verified by n-orca: **VALID, depth 12.** Spec:
[`docs/specs/minimax_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/minimax_block.n.orca.md).

```mermaid
%% architecture MiniMaxM2Block
flowchart TD
    x(("x<br/>[input]"))
    in_ln["in_ln<br/>RMSNorm(d_model)"]
    q_proj["q_proj<br/>Linear(d_model, n_heads*head_dim)"]
    k_proj["k_proj<br/>Linear(d_model, n_kv*head_dim)"]
    v_proj["v_proj<br/>Linear(d_model, n_kv*head_dim)"]
    q_norm["q_norm<br/>RMSNorm(n_heads*head_dim)"]
    k_norm["k_norm<br/>RMSNorm(n_kv*head_dim)"]
    rope_q["rope_q<br/>RoPE(head_dim)"]
    rope_k["rope_k<br/>RoPE(head_dim)"]
    attn_core["attn_core<br/>GroupedQueryAttention(n_heads, n_kv, head_dim)"]
    o_proj["o_proj<br/>Linear(n_heads*head_dim, d_model)"]
    add_1["add_1<br/>Add()"]
    post_ln["post_ln<br/>RMSNorm(d_model)"]
    router["router<br/>SigmoidTopKRouter(d_model, n_experts, top_k)"]
    experts["experts<br/>ExpertSwiGLU(d_model, d_expert, n_experts)"]
    moe_sum["moe_sum<br/>WeightedSum()"]
    add_2["add_2<br/>Add()"]
    y(("y<br/>[output]"))
    x -- "x : (B,S,d_model)" --> in_ln
    in_ln -- "x_normed" --> q_proj
    in_ln -- "x_normed" --> k_proj
    in_ln -- "x_normed" --> v_proj
    q_proj -- "q" --> q_norm
    k_proj -- "k" --> k_norm
    q_norm -- "q_n" --> rope_q
    k_norm -- "k_n" --> rope_k
    rope_q -- "q_rot" --> attn_core
    rope_k -- "k_rot" --> attn_core
    v_proj -- "v" --> attn_core
    attn_core -- "attn_mix" --> o_proj
    o_proj -- "attn_out" --> add_1
    x -- "x_skip : (B,S,d_model)" --> add_1
    add_1 -- "h" --> post_ln
    post_ln -- "h_normed" --> router
    post_ln -- "h_tokens" --> experts
    router -- "topk_weights" --> moe_sum
    experts -- "expert_out" --> moe_sum
    moe_sum -- "moe_out" --> add_2
    add_1 -- "h_skip" --> add_2
    add_2 -- "y_out" --> y
    classDef input fill:#dbeafe,stroke:#1e40af,color:#1e3a8a;
    class x input;
    classDef output fill:#dcfce7,stroke:#166534,color:#14532d;
    class y output;
```

## DeepSeek-V4 block — mHC hyper-connections + shared-KV-MQA + sink

A **new attention class**, not MLA. The residual is `hc_mult` parallel streams kept through the whole block and mixed by
two **manifold-constrained hyper-connections (mHC)**: each `attn_hc`/`ffn_hc` collapses the streams into one sequence (a
`pre`-weighted sum) for the sublayer, and the update re-places the output with a `post` gate plus a **Sinkhorn-projected
doubly-stochastic** `comb` mix back across the streams. Attention is **shared-KV MQA** (one KV head, read as both K and
V) with **q-LoRA** queries, **partial interleaved RoPE**, a per-head **attention sink** (an extra softmax logit dropped
from the output), an **undo-RoPE** conjugate rotation on the output (because K==V), and a **grouped low-rank o_proj**. The
FFN is a **sqrtsoftplus** MoE (`softplus(logits).sqrt()` scores; a learned bias picks the top-k; un-biased scores renorm ×
scale weight them) plus an always-on **shared expert**, both with gpt-oss SwiGLU clamps. Dims for the V4-Flash reference
(d_model 4096, 64 heads, head_dim 512, q_lora 1024, 256 experts, top-6, d_expert 2048, hc_mult 4). Verified by n-orca:
**VALID, depth 19.** fieldrun's `dsv4` kernel matches `DeepseekV4ForCausalLM` **60/60 top-1** (Stage 1: the sliding-only
backbone; the CSA/HCA compressors + Lightning Indexer are separate regimes). Spec:
[`docs/specs/dsv4_block.n.orca.md`](https://github.com/jascal/lm-sae/blob/main/docs/specs/dsv4_block.n.orca.md).

```mermaid
%% architecture DeepSeekV4Block
flowchart TD
    x(("x<br/>[input]"))
    attn_hc["attn_hc<br/>HyperConnection(hc_mult, d_model)"]
    in_ln["in_ln<br/>RMSNorm(d_model)"]
    q_a["q_a<br/>Linear(d_model, q_lora)"]
    q_a_ln["q_a_ln<br/>RMSNorm(q_lora)"]
    q_b["q_b<br/>Linear(q_lora, n_heads*head_dim)"]
    q_b_norm["q_b_norm<br/>RMSNorm(head_dim)"]
    rope_q["rope_q<br/>RoPE(rope_dim)"]
    kv["kv<br/>Linear(d_model, head_dim)"]
    kv_ln["kv_ln<br/>RMSNorm(head_dim)"]
    rope_kv["rope_kv<br/>RoPE(rope_dim)"]
    attn_core["attn_core<br/>SharedKVSinkAttention(n_heads, head_dim)"]
    undo_rope["undo_rope<br/>RoPE(rope_dim)"]
    o_a["o_a<br/>GroupedLinear(n_heads*head_dim, o_groups*o_lora)"]
    o_b["o_b<br/>Linear(o_groups*o_lora, d_model)"]
    attn_update["attn_update<br/>HyperConnectionUpdate(hc_mult, d_model)"]
    ffn_hc["ffn_hc<br/>HyperConnection(hc_mult, d_model)"]
    post_ln["post_ln<br/>RMSNorm(d_model)"]
    router["router<br/>SqrtSoftplusTopKRouter(d_model, n_experts, top_k)"]
    experts["experts<br/>ExpertSwiGLU(d_model, d_expert, n_experts)"]
    moe_sum["moe_sum<br/>WeightedSum()"]
    shared["shared<br/>FeedForward(d_model, d_expert)"]
    moe_add["moe_add<br/>Add()"]
    ffn_update["ffn_update<br/>HyperConnectionUpdate(hc_mult, d_model)"]
    y(("y<br/>[output]"))
    x -- "streams : (B,S,d_model)" --> attn_hc
    attn_hc -- "collapsed" --> in_ln
    in_ln -- "x_normed" --> q_a
    q_a -- "q_lat" --> q_a_ln
    q_a_ln -- "q_lat_n" --> q_b
    q_b -- "q" --> q_b_norm
    q_b_norm -- "q_n" --> rope_q
    rope_q -- "q_rot" --> attn_core
    in_ln -- "x_normed" --> kv
    kv -- "kv_raw" --> kv_ln
    kv_ln -- "kv_n" --> rope_kv
    rope_kv -- "kv_rot" --> attn_core
    attn_core -- "attn_mix" --> undo_rope
    undo_rope -- "attn_unroped" --> o_a
    o_a -- "grouped" --> o_b
    o_b -- "attn_out" --> attn_update
    x -- "streams_skip : (B,S,d_model)" --> attn_update
    attn_hc -- "attn_ctl" --> attn_update
    attn_update -- "streams_1" --> ffn_hc
    ffn_hc -- "collapsed_2" --> post_ln
    post_ln -- "h_normed" --> router
    post_ln -- "h_tokens" --> experts
    post_ln -- "h_shared" --> shared
    router -- "topk_weights" --> moe_sum
    experts -- "expert_out" --> moe_sum
    moe_sum -- "routed" --> moe_add
    shared -- "shared_out" --> moe_add
    moe_add -- "moe_out" --> ffn_update
    attn_update -- "streams_skip2" --> ffn_update
    ffn_hc -- "ffn_ctl" --> ffn_update
    ffn_update -- "y_out" --> y
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
