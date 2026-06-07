# Operator catalog — attention operators, surveyed across models

A **working catalog** of attention operators — amateur, exploratory home-science: provisional, descriptive, and
*not* a definitive reference (one of many catalogs one could draw).

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

**Discovered-candidate dossiers** (UNNAMED load-bearing heads from the [discovery sweep](discovered.md), given the full battery): [`discovered_7.6`](discovered_7.6.md).

## How to read this catalog

Two axes:

> **Taxonomy — classes, instances, variants (read this first).** Each row below is an operator **CLASS**, *not* a
> single operator: it is a *family* of heads that realize the same operation. The **membership matrix** gives the
> head-count per class per model (e.g. GPT-2 has ~22 induction heads, ~31 prev-token heads, 117 heads with
> appreciable sink mass). Three levels of granularity:
> 1. **class** (these 7 universal rows + 5 GPT-2 circuit classes) — the operation;
> 2. **instance** — an individual head realizing the class (the per-head listing is `disassemble_gpt2.py` →
>    [runs/disassembly/gpt2_disassembly.txt](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/gpt2_disassembly.txt); each dossier's section A lists the class's member heads);
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
| **prevtok** | positional | 0.96 | 0.99 | 0.96 | 0.88 | 0.68 | 0.77 |
| **induction** | content | 0.93 | 0.91 | 0.97 | 0.94 | 0.95 | 0.99 |
| **duplicate** | content | 0.62 | 0.86 | 0.96 | 0.86 | 0.74 | 0.97 |
| **sink** | addressing | 0.96 | 0.97 | 0.97 | 0.06 | 1.00 | 0.99 |
| **self** | addressing | 0.84 | 0.45 | 0.55 | 0.97 | 0.96 | 1.00 |
| **local** | positional | 0.34 | 0.34 | 0.34 | 0.32 | 0.27 | 0.29 |
| **structural** | structural | 0.20 | 0.25 | 0.39 | 0.24 | 0.35 | 0.17 |

## Catalog — membership (# heads carrying the op, mass > 0.15; *how many heads in the class?*)

| operator class | kind | gpt2 | gpt2-medium | gpt2-large | gemma-2-2b | Llama-3.2-1B | Qwen2.5-1.5B |
|---|---|---|---|---|---|---|---|
| **prevtok** | positional | 31 | 53 | 80 | 106 | 47 | 57 |
| **induction** | content | 22 | 61 | 75 | 23 | 40 | 54 |
| **duplicate** | content | 4 | 6 | 23 | 11 | 16 | 16 |
| **sink** | addressing | 117 | 335 | 555 | 0 | 446 | 292 |
| **self** | addressing | 10 | 21 | 74 | 154 | 60 | 43 |
| **local** | positional | 15 | 29 | 44 | 49 | 8 | 22 |
| **structural** | structural | 2 | 4 | 9 | 11 | 19 | 4 |

## Catalog — causal ΔNLL (mean-ablate top-3 heads, generic-prose NLL; *load-bearing on prose?*)

Note: this is **generic-prose** ΔNLL, so *task-specific* ops (induction, duplicate) read low here even though
they are load-bearing on their *own* task — see each op's dossier (section B) for the task-specific causal.

| operator class | kind | gpt2 | gpt2-medium | gpt2-large | gemma-2-2b | Llama-3.2-1B | Qwen2.5-1.5B |
|---|---|---|---|---|---|---|---|
| **prevtok** | positional | +0.01 | +0.03 | +0.01 | -0.01 | +0.01 | +0.22 |
| **induction** | content | +0.01 | +0.00 | +0.01 | -0.28 | +0.00 | -0.00 |
| **duplicate** | content | +0.11 | -0.01 | +0.00 | -1.07 | +0.02 | +0.00 |
| **sink** | addressing | +0.02 | +0.00 | +0.00 | -0.02 | -0.00 | -0.00 |
| **self** | addressing | +0.02 | +0.03 | +0.01 | +0.48 | +0.10 | +3.73 |
| **local** | positional | -0.01 | +0.03 | +0.01 | +0.45 | +0.01 | +0.22 |
| **structural** | structural | -0.01 | -0.03 | -0.00 | -0.12 | +1.38 | -0.05 |

## The other instruction class: COMPUTE (MLP)

Attention is the **MOVE** class; the **[MLP / COMPUTE catalog](mlp_compute.md)** is the other half of the
instruction set (cross-model per-layer MLP causal profile + the GPT-2 neuron read→write idioms). In the discovery
sweeps the early-MLP detokenizer had the largest single-component causal effect of anything measured.

## Growing the catalog: discovered components

The [**discovered components**](discovered.md) page is the discovery engine run across *every* model — every head +
MLP ranked by causal effect (multi-seed), flagged named-vs-**UNNAMED**. The UNNAMED load-bearing components are
candidate operators not yet catalogued (e.g. Llama heads 0.31/1.31/1.29) — the leads to dossier next. The strongest
RoPE candidates are profiled (causal + channel) on [**discovered candidates (cross-model)**](discovered_xmodel.md);
Llama **0.31** is induction-load-bearing **+7.99** (an early induction-enabling head, uncatalogued).

## Gaps (documented, not skipped)

- **succession / greater-than** — MLP-dominated; no clean attention head, so no catalog row (the OV probe sees only
  the attention-side shadow). It is carried by the *copy* ops (see `instruction_reuse.py`: successor ← induction/duplicate).
- **SSM (Mamba)** — no attention heads, so the head-resolved catalog does not apply; induction is present
  *behaviourally* (NLL gain) — see `ssm_induction.py`.

## How this was made

`operator_atlas.py` (the cross-model matrix) + `operator_dossier.py --op <name>` (the deep per-op dossiers) →
`operator_catalog_doc.py` regenerates these docs from the JSON artifacts. See [`../DECOMPILATION.md`](../DECOMPILATION.md).
