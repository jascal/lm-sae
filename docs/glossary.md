---
title: Glossary
---

# Glossary — the terms this project uses, and where they come from

Each entry: a short **definition** and *how/why we use it here*. Established terms link to a **canonical source**;
working terms with no canonical reference link to the page that defines them here. This is the field-guide's
vocabulary — the natural-history reference for the [operator catalog](operators/README.md) and
[circuit catalog](circuits/README.md).

## Framing & working terms

- **Operator** — a single attention head **class** or MLP class that realizes one operation. The unit of
  the [operator catalog](operators/README.md). *Here:* found behaviourally (an attention mask) or from a
  literature head-set; an operator is a *family of heads* (a class), not one head — see the membership matrix.
- **Circuit** — a **composition** of operators: a writer-op feeding a reader-op's K/Q/V port, chained into a DAG.
  *Here:* the [circuit catalog](circuits/README.md); the primitive is the *edge* (writer → reader via a port).
  Canonical framing: [Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021)](https://transformer-circuits.pub/2021/framework/index.html).
- **`op:` vs `circuit:` (a naming convention)** — a circuit is keyed by its **reader operator**, so a few names
  (`induction`, `duplicate`) name *both* an operator and a circuit. They are different objects at different
  levels: `op:induction` is the head *class*; `circuit:induction` is the *composition* (prev-token → induction)
  that feeds it. When a name could mean either, qualify it `op:<name>` / `circuit:<name>`; the colliding catalog
  pages cross-link to their namesake.
- **MOVE vs COMPUTE** — the two instruction classes: attention **MOVES** operands between positions (a QK
  *addressing mode* × an OV *write op*); the MLP **COMPUTES** on them (key–value memories). *Here:* the operator
  catalog is MOVE; the [MLP catalog](operators/mlp_compute.md) is COMPUTE.
- **Disassembly → decompilation** — *disassembly* = naming components/idioms in isolation; *decompilation*
  = reconstructing the **computation** faithfully (recompile the op-graph, KL ≈ host). *Here:* the program's arc;
  see [DECOMPILATION.md](DECOMPILATION.md).
- **Reconstruction coverage** — `1 − KL(host ‖ kept) / KL(host ‖ all-ablated)`: how much of the forward
  pass an op-set reconstructs. *Here:* the decompilation metric with teeth ([DECOMPILATION.md](DECOMPILATION.md)).
- **Dossier** — the deep per-operator battery (identity · causal×tasks · K/V channels · composition ·
  redundancy · cross-model). *Here:* `operator_dossier.py`; one page per op-class in the catalog.
- **Catalog (not atlas)** — a growing, causally-tested *record* of what we've observed, **not** a claim of
  completeness — the natural-history stance. (We avoid "atlas," which overclaims a complete map.)

## Named attention operators & circuits (from the literature)

- **QK / OV circuits** — QK decides *where* a head attends (addressing); OV decides *what* it writes (the moved
  content). *Here:* the opcode tables read QK/OV in an operand basis. [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html).
- **K- / Q- / V-composition** — one head's output feeding a later head's **key / query / value**. *Here:* the
  edges scored by `composition_dag.py` / `vcomposition.py`. [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html).
- **Previous-token head** — attends to position *q−1*. *Here:* the writer that feeds induction; the canonical
  GPT-2 one is 4.11. [Elhage et al. (2021)](https://transformer-circuits.pub/2021/framework/index.html).
- **Induction head** — attends to the token *after* a previous occurrence of the current token, and copies it (the
  in-context-copy macro). *Here:* the keystone op — universal across our models; the one genuinely reused
  instruction. [Olsson et al., *In-context Learning and Induction Heads* (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
- **Duplicate-token head** — attends to an earlier occurrence of the *same* token. *Here:* the IOI initiator.
  [Wang et al., *Interpretability in the Wild* / IOI (2022)](https://arxiv.org/abs/2211.00593).
- **Name-mover · S-inhibition · backup · negative (copy-suppression) movers** — the **IOI circuit** (indirect-object
  identification): duplicate → S-inhibition → name-mover, with backups (self-repair) and negative movers that write
  *against* the answer. *Here:* the GPT-2-only circuit ops (no published head-set off GPT-2). [Wang et al. (2022)](https://arxiv.org/abs/2211.00593);
  copy-suppression: [McDougall et al. (2023)](https://arxiv.org/abs/2310.04625).
- **Attention sink** — a head that parks most of its attention on position 0 / BOS as a *no-op* / idle register.
  *Here:* common but **absent in Gemma-2**, and *present ≠ depended-on*. [Xiao et al., *StreamingLLM* (2023)](https://arxiv.org/abs/2309.17453);
  cf. [Darcet et al., *Vision Transformers Need Registers* (2023)](https://arxiv.org/abs/2309.16588).
- **Successor / greater-than** — increment / numeric-comparison behaviour; **MLP-dominated**, no clean attention
  head. *Here:* a documented gap (carried by the copy ops). [Hanna et al., *How does GPT-2 compute greater-than?* (2023)](https://arxiv.org/abs/2305.00586).

## Methods

- **Mean-ablation** — replace a component's output with its corpus **mean** and measure the damage to a metric
  (necessary, norm-preserving causal test). *Here:* the arch-generic harness behind every causal column.
- **Path patching / activation patching** — a surgical causal intervention that isolates one **edge** (e.g. remove
  a writer from a reader's key, recompute) rather than a whole component. *Here:* the liveness gate in the circuit
  catalog + the channel patches. [Goldowsky-Dill et al., *Localizing Model Behavior with Path Patching* (2023)](https://arxiv.org/abs/2304.05969);
  causal mediation: [Vig et al. (2020)](https://arxiv.org/abs/2004.12265).
- **Logit lens** — project an intermediate residual through the unembedding to read the model's *running* next-token
  prediction at each layer. *Here:* the debugger's `logit_lens_step` ("where is the answer decided?").
  [nostalgebraist, *interpreting GPT: the logit lens* (2020)](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens).
- **Self-repair / backup heads** — redundant components that *wake up* and compensate when primaries are ablated,
  masking their importance under single-set ablation. *Here:* why name-movers read ~0; quantified in
  `self_repair.py`. [McGrath et al., *The Hydra Effect* (2023)](https://arxiv.org/abs/2307.15771).
- **cov95 / mAUC** — `cov95` = fraction of known features a *single* SAE latent detects at
  AUC ≥ 0.95 (monosemanticity); `mAUC` = mean best-detector AUC (content recoverability). *Here:* the
  [forge-tax track](FORGE_TAX_TRACK.md)'s legibility meters.

## Architecture features (the "species" we compare)

- **Absolute vs RoPE position** — GPT-2 uses learned **absolute** position embeddings; most modern models use
  **RoPE** (rotary, relative). *Here:* the recurring split — the positional register is absolute-position-specific.
  [Su et al., *RoPE / RoFormer* (2021)](https://arxiv.org/abs/2104.09864).
- **GQA** (grouped-query attention) — query heads share key/value heads. *Here:* handled by `h → h//(H/n_kv)` in
  the cross-model reader. [Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245).
- **RMSNorm** — a normalization without mean-subtraction. *Here:* RoPE models use it (easier — no mean confound).
  [Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467).
- **Gated MLP (SwiGLU / GeGLU)** — the gated MLP variants in RoPE-family models. *Here:* the COMPUTE block whose
  per-layer profile the MLP catalog measures. [Shazeer, *GLU Variants* (2020)](https://arxiv.org/abs/2002.05202).
- **GPT-NeoX / parallel residual** — the architecture of the **Pythia** ladder: rotary position + LayerNorm + dense
  GELU MLP, with attention and MLP computed in **parallel** (both read the block input, summed together:
  `y = x + attn(ln x) + mlp(ln x)`) rather than serially. *Here:* the controlled scale ladder (one architecture, same
  data, 14m→1.4b) behind the [scaling laws](scaling.md); see [architectures](architectures.md).
  [Black et al., *GPT-NeoX* (2022)](https://arxiv.org/abs/2204.06745); [Biderman et al., *Pythia* (2023)](https://arxiv.org/abs/2304.01373).
- **SSM / Mamba** — a state-space sequence mixer with **no attention** (a learned linear recurrence). *Here:* the
  no-attention control — does the in-context-copy capability survive losing attention? [Gu & Dao, *Mamba* (2023)](https://arxiv.org/abs/2312.00752).

## The sister track (SAEs / the forge tax)

- **SAE** (sparse autoencoder) — a dictionary that decomposes activations into sparse, ideally monosemantic
  features. *Here:* the thing that *misses* the composition (the motivation for disassembly). [Cunningham et al. (2023)](https://arxiv.org/abs/2309.08600);
  [Bricken et al., *Towards Monosemanticity* (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html).
- **Forge / forge tax** — re-expressing a model's weights so its residual is written in a fixed SAE
  feature basis; the "tax" = forging preserves accuracy (mAUC) but collapses monosemanticity (cov95). *Here:* the
  [forge-tax track](FORGE_TAX_TRACK.md).

## Acronyms & symbols (quick reference)

The bare abbreviations and symbols used throughout the catalog, in one place.

- **MLP** (multi-layer perceptron) — the per-layer feed-forward block; the **COMPUTE** instruction class. *Here:*
  `MLP0` = the layer-0 MLP (the detokenizer / [extended embedding](operators/mlp_detokenizer.md)); the
  [MLP catalog](operators/mlp_compute.md), and MLPs as circuit nodes in the [circuit catalog](circuits/README.md).
- **NLL** (negative log-likelihood) — `−log p(correct token)`, the prediction loss/metric. *Here:* **induction-NLL**
  (on repeated-random sequences) and **generic-NLL** (on prose) are the two behaviours nearly every causal test scores.
- **KL** (Kullback–Leibler divergence) — directed distance between two distributions. *Here:* `KL(host ‖ recompiled)`
  is the faithfulness oracle behind [reconstruction coverage](DECOMPILATION.md).
- **logit-diff** — `logit(IO) − logit(S)` at the last position, the **IOI** metric (how much the model prefers the
  indirect object over the subject).
- **DLA** (direct logit attribution) — a component's *direct* contribution to a token's logit through the unembedding.
  *Here:* the literature IOI head-sets are DLA-derived.
- **η² (eta-squared)** — the fraction of a quantity's variance explained by a categorical label (one-way ANOVA).
  *Here:* MLP0 **token-determinism** (η² of the MLP0 output explained by the current token,
  [extended-embedding test](operators/mlp_detokenizer.md)) and the domain-tiling test.
- **read-out depth** — the relative depth (`layer / n_layers`) at which the **logit lens** first decodes the answer —
  *where a relation resolves* in the forward pass ([scaling synthesis](scaling.md), [findings](FINDINGS.md)).
- **entity-leakage** — when editing one fact of a subject (its capital) also flips another (its language): evidence the
  editable unit is the **entity, not the fact** ([findings](FINDINGS.md)).
- **effective-N / Hill number** — `(Σxᵢ)² / Σxᵢ²`, the *effective* count of contributors in a distribution (1 = one
  dominant, N = N equal). *Here:* how many heads/layers share induction; how many features carry a contribution.
- **LRE** (linear relational embedding) — the hypothesis that a relation is ≈ a linear map from the subject
  representation to the object. [Hernandez et al. (2023)](https://arxiv.org/abs/2308.09124).
- **ROME** (rank-one model editing) — editing a stored fact by a rank-1 weight update to an MLP at the subject token.
  [Meng et al. (2022)](https://arxiv.org/abs/2202.05262).
- **IOI** (indirect-object identification) — the templated *"When Mary and John …, John gave a drink to → Mary"* task
  and its circuit (duplicate → S-inhibition → name-mover). [Wang et al. (2022)](https://arxiv.org/abs/2211.00593).
- **RoPE** (rotary position embedding) — relative position applied as a rotation of the query/key at attention time
  (no learned absolute-position vector). *Here:* the Llama/Qwen/Gemma/Pythia families; the recurring absolute-vs-RoPE
  split (see [Absolute vs RoPE position](#architecture-features-the-species-we-compare)). [Su et al. (2021)](https://arxiv.org/abs/2104.09864).
- **MHA / GQA** — multi-head attention (each head has its own K/V) vs grouped-query attention (query heads **share**
  K/V heads). *Here:* GPT-2/Pythia are MHA; the RoPE family is GQA (handled by `h → h//(H/n_kv)` in the reader).
- **ResidualVM** — this project's steppable debugger: an arch-generic intervention layer (ablate / patch / trace /
  attribution / find-operators / edit-SAE-features) over any HF causal LM ([DECOMPILATION.md](DECOMPILATION.md)).
- **BOS** — the beginning-of-sequence token (position 0), where the attention **sink** parks.
- **KV-cache** — the cached keys/values of past tokens; the autoregressive read/write "tape."
- **ΔTV / ΔV-out** — the composition-gate readouts: **ΔTV** = total-variation change in a reader's *attention* when a
  writer is removed (K/Q-composition); **ΔV-out** = relative change in a reader's *output* (V-composition).
- **resid / residual stream** — the running per-token vector each block reads from and adds to (the shared "bus").
- **`L.H` notation** — `layer.head`; e.g. `4.11` is layer 4, head 11 (GPT-2's prev-token head).
- **W_E / W_U** — token embedding / unembedding matrices. **W_Q / W_K / W_V / W_O** — attention query / key / value /
  output projections. **W_in / W_out** — the MLP's in (gate/up) / out (down) projections.
- **d / d_model** — the residual-stream (hidden) width; **n_layers** (depth), **n_heads / H**, **d_head / hd**.
- **bf16 / fp32** — bfloat16 / 32-bit float weight precision (bf16 used to fit the larger models on a 7.5 GB GPU).
- **OOD** (out-of-distribution) — *Here:* the repeated-random induction probe is mildly OOD for chat-tuned models.
- **TC⁰** — the circuit-complexity class a single bounded-depth forward pass sits in (the dataflow-circuit-per-pass
  vs VM-at-the-loop framing in [DECOMPILATION.md](DECOMPILATION.md)).

---

_Missing a term, or want a canonical link added? It belongs here — open an issue on
[GitHub](https://github.com/jascal/lm-sae)._
