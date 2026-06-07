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
- **SSM / Mamba** — a state-space sequence mixer with **no attention** (a learned linear recurrence). *Here:* the
  no-attention control — does the in-context-copy capability survive losing attention? [Gu & Dao, *Mamba* (2023)](https://arxiv.org/abs/2312.00752).

## The sister track (SAEs / the forge tax)

- **SAE** (sparse autoencoder) — a dictionary that decomposes activations into sparse, ideally monosemantic
  features. *Here:* the thing that *misses* the composition (the motivation for disassembly). [Cunningham et al. (2023)](https://arxiv.org/abs/2309.08600);
  [Bricken et al., *Towards Monosemanticity* (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html).
- **Forge / forge tax** — re-expressing a model's weights so its residual is written in a fixed SAE
  feature basis; the "tax" = forging preserves accuracy (mAUC) but collapses monosemanticity (cov95). *Here:* the
  [forge-tax track](FORGE_TAX_TRACK.md).

---

_Missing a term, or want a canonical link added? It belongs here — open an issue on
[GitHub](https://github.com/jascal/lm-sae)._
