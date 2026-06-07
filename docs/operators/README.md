# Operator catalog — the attention instruction set, surveyed across models

This is the **catalog** of GPT-2-family attention operators, measured exhaustively. Two axes:

> **Taxonomy — classes, instances, variants (read this first).** Each row below is an operator **CLASS**, *not* a
> single operator: it is a *family* of heads that realize the same operation. The **membership matrix** gives the
> head-count per class per model (e.g. GPT-2 has ~22 induction heads, ~31 prev-token heads, 117 heads with
> appreciable sink mass). Three levels of granularity:
> 1. **class** (these 7 universal rows + 5 GPT-2 circuit classes) — the operation;
> 2. **instance** — an individual head realizing the class (the per-head listing is `disassemble_gpt2.py` →
>    `runs/disassembly/gpt2_disassembly.txt`; each dossier's section A lists the class's member heads);
> 3. **variant / sub-class** — structured differences *within* a class (e.g. induction's writer-branching, or
>    token- vs subword-name-completion inductors; the sink "class" is largely *content heads in their idle state* —
>    see `sink.md`). The dossiers (sections C/D/E) expose this intra-class structure.
>
> So the answer to "is it N operators or N classes?" is **classes** — the head counts are in the membership matrix.


- **Universal / addressing operators** (a position-or-token attention mask → measurable in *any* architecture):
  `prevtok, induction, duplicate, sink, self, local, structural`. The **catalog matrix** (6 models) below is their cross-model survey.
- **GPT-2 circuit operators** (literature direct-logit-attribution head-sets, **no published head-set outside
  GPT-2**): `name_mover, backup_name_mover, negative_mover, s_inhibition, coreference`. Catalogued by their per-op dossiers (GPT-2), not the cross-model matrix.

Each operator has a **page** (cross-model catalog row + the deep GPT-2 dossier: identity / causal×tasks / K-V
channels / composition / redundancy / cross-model). Per-op data lives under `runs/disassembly/operators/`.

## Catalog — behavioural signal (max head mass on the op's pattern; *is the op present?*)

| operator class | kind | gpt2 | gpt2-medium | gpt2-large | gemma-2-2b | Llama-3.2-1B | Qwen2.5-1.5B |
|---|---|---|---|---|---|---|---|
| **prevtok** | positional | 0.96 | 0.99 | 0.96 | 0.84 | 0.68 | 0.77 |
| **induction** | content | 0.92 | 0.91 | 0.96 | 0.94 | 0.94 | 0.99 |
| **duplicate** | content | 0.62 | 0.86 | 0.96 | 0.85 | 0.74 | 0.97 |
| **sink** | addressing | 0.95 | 0.96 | 0.96 | 0.07 | 1.00 | 1.00 |
| **self** | addressing | 0.83 | 0.45 | 0.55 | 0.95 | 0.96 | 1.00 |
| **local** | positional | 0.34 | 0.34 | 0.34 | 0.31 | 0.27 | 0.28 |
| **structural** | structural | 0.19 | 0.25 | 0.38 | 0.23 | 0.36 | 0.16 |

## Catalog — membership (# heads carrying the op, mass > 0.15; *how many heads in the class?*)

| operator class | kind | gpt2 | gpt2-medium | gpt2-large | gemma-2-2b | Llama-3.2-1B | Qwen2.5-1.5B |
|---|---|---|---|---|---|---|---|
| **prevtok** | positional | 31 | 53 | 81 | 107 | 46 | 56 |
| **induction** | content | 22 | 62 | 76 | 26 | 40 | 54 |
| **duplicate** | content | 4 | 6 | 23 | 11 | 17 | 16 |
| **sink** | addressing | 117 | 334 | 553 | 0 | 472 | 292 |
| **self** | addressing | 10 | 21 | 74 | 149 | 58 | 43 |
| **local** | positional | 16 | 28 | 42 | 48 | 8 | 22 |
| **structural** | structural | 2 | 4 | 6 | 7 | 3 | 1 |

## Catalog — causal ΔNLL (mean-ablate top-3 heads, generic-prose NLL; *load-bearing on prose?*)

Note: this is **generic-prose** ΔNLL, so *task-specific* ops (induction, duplicate) read low here even though
they are load-bearing on their *own* task — see each op's dossier (section B) for the task-specific causal.

| operator class | kind | gpt2 | gpt2-medium | gpt2-large | gemma-2-2b | Llama-3.2-1B | Qwen2.5-1.5B |
|---|---|---|---|---|---|---|---|
| **prevtok** | positional | +0.02 | +0.00 | +0.02 | +0.09 | -0.00 | +0.20 |
| **induction** | content | +0.01 | -0.01 | +0.01 | -0.29 | +0.00 | +0.00 |
| **duplicate** | content | +0.11 | -0.01 | -0.00 | -0.96 | +0.03 | -0.00 |
| **sink** | addressing | +0.02 | +0.01 | -0.00 | +0.00 | +0.00 | -0.00 |
| **self** | addressing | +0.05 | +0.03 | +0.00 | +0.51 | +0.10 | +3.77 |
| **local** | positional | +0.00 | +0.00 | +0.02 | +0.51 | -0.00 | +0.20 |
| **structural** | structural | -0.05 | -0.05 | +0.00 | -0.12 | +1.47 | -0.06 |

## Catalog index

- [`prevtok`](prevtok.md) — positional
- [`induction`](induction.md) — content
- [`duplicate`](duplicate.md) — content
- [`sink`](sink.md) — addressing
- [`self`](self.md) — addressing
- [`local`](local.md) — positional
- [`structural`](structural.md) — structural
- [`name_mover`](name_mover.md) — circuit
- [`backup_name_mover`](backup_name_mover.md) — circuit
- [`negative_mover`](negative_mover.md) — circuit
- [`s_inhibition`](s_inhibition.md) — circuit
- [`coreference`](coreference.md) — circuit

## The other instruction class: COMPUTE (MLP)

Attention is the **MOVE** class; the **[MLP / COMPUTE catalog](mlp_compute.md)** is the other half of the
instruction set (cross-model per-layer MLP causal profile + the GPT-2 neuron read→write idioms). The ResidualVM
discovery engine found **MLP0 (the detokenizer) is the single most load-bearing component for every behaviour**.

## Gaps (documented, not skipped)

- **succession / greater-than** — MLP-dominated; no clean attention head, so no catalog row (the OV probe sees only
  the attention-side shadow). It is carried by the *copy* ops (see `instruction_reuse.py`: successor ← induction/duplicate).
- **SSM (Mamba)** — no attention heads, so the head-resolved catalog does not apply; induction is present
  *behaviourally* (NLL gain) — see `ssm_induction.py`.

## How this was made

`operator_atlas.py` (the cross-model matrix) + `operator_dossier.py --op <name>` (the deep per-op dossiers) →
`operator_catalog_doc.py` regenerates these docs from the JSON artifacts. See [`../DECOMPILATION.md`](../DECOMPILATION.md).
