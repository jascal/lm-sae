# Operator class `MLP` / COMPUTE

Attention **MOVES** operands; the MLP **COMPUTES** on them. The [operator catalog](README.md) is attention-only — but in the ResidualVM discovery sweeps **MLP0 had the largest single-component causal effect of anything measured** (induction / IOI / generic). A working catalog of the COMPUTE class across architectures — provisional and descriptive.

## Cross-model — per-layer MLP causal ΔNLL (mean-ablate the whole MLP block)

Top MLP layers by causal damage when ablated (generic prose NLL; depth = layer/(L−1)):

| model | arch | L | all-MLP ΔNLL (generic) | top generic-MLP (depth, ΔNLL) | top induction-MLP (depth, ΔNLL) |
|---|---|---|---|---|---|
| gpt2 | GPT-2/absolute | 12 | +2.09 | L0 (d0.0, +1.70) | L0 (d0.0, +11.72) |
| gpt2-medium | GPT-2/absolute | 24 | +2.69 | L0 (d0.0, +7.32) | L0 (d0.0, +20.94) |
| gpt2-large | GPT-2/absolute | 36 | +5.28 | L0 (d0.0, +3.67) | L0 (d0.0, +13.57) |
| gemma-2-2b | RoPE | 26 | +10.74 | L25 (d1.0, +0.84) | L0 (d0.0, +4.25) |
| Llama-3.2-1B | RoPE | 16 | +4.18 | L1 (d0.07, +7.35) | L1 (d0.07, +12.65) |
| Qwen2.5-1.5B | RoPE | 28 | +4.29 | L1 (d0.04, +7.64) | L2 (d0.07, +13.91) |

**Reading it:** COMPUTE is **depth-organized** — an early MLP (commonly called the *detokenizer*, low depth — a label, not a verified mechanism) is the biggest single COMPUTE op for induction in the GPT-2 family, and late MLPs carry generic-LM output. The whole-MLP-stack ablation ΔNLL is large in every model (COMPUTE is load-bearing everywhere, unlike any single attention-op class).

## Named MLP operators (GPT-2) — listed even where the mechanism is unverified

In the natural-history spirit we **list the load-bearing MLP specimens with what we measured**, even where the mechanism is not established. ("Detokenizer" is the common *label* for L0; we record the behaviour and operands, not a mechanism claim.)

| MLP | depth | causal ΔNLL (generic / induction) | recon-importance | top read→write neuron idioms | mechanism |
|---|---|---|---|---|---|
| **MLP0** | 0.0 | +1.70 / +11.72 | 0.77 | `.+;→Ċ+I`; `_you+_and→_to+First` | partial — the **"detokenizer"** label: writes sentence-initial / common tokens |
| MLP11 | 1.0 | +0.05 / +0.32 | 0.08 | `Ċ+_the→.+?`; `Ċ+_the→First+_to` | partial — punctuation / output writes |
| MLP1 | 0.09 | +1.05 / +1.00 | 0.07 | `_Citizen+.→First+_to`; `_the+:→First+_to` | **unverified** (listed by measured effect) |
| MLP2 | 0.18 | +0.11 / +0.04 | 0.05 | `_Citizen+I→First+_to`; `_Citizen+_to→First+_to` | **unverified** (listed by measured effect) |

_MLP0 is GPT-2's single most load-bearing component (recon-importance 0.77 of all MLPs; ablating it costs induction +11.7 NLL) — listed here as a catalog entry even though *how* it works is only partially characterized. The other models' early load-bearing MLP (the detokenizer-analog) is in the cross-model table above; its per-neuron idioms are not yet run (no per-neuron basis off GPT-2 — a documented gap)._

## GPT-2 deep characterization (harvested)

- **COMPUTE vocabulary is low-rank** (`mlp_catalog.py`): transform participation 22 vs random 1666 — a small reused set of compute templates (heavier-tailed than attention's ~5: rank-90 ≈ 186).
- **top neuron read→write idioms:** `The+And→_for+First`, `And+_be→_for+First`, `And+_for→_for+First`, `MAR+_to→_for+First`, `_a+Ċ→_a+MAR`, `_for+And→_for+First`
- **MLPs carry the reconstruction coverage** (`mlp_ops.py`): MLP-only coverage **+0.46** vs attention-only -0.02 (they interact — neither alone reaches the full pass); load-bearing MLPs concentrate in L0, L11, L1, L2 (L0 = the detokenizer).
- **head↔MLP composition edges** exist in weight space (top head→MLP ['2.1', 'L2'], MLP→head ['L1', '2.2']) — the COMPUTE nodes the attention-only DAG missed.

## Gaps

- **Mamba / SSM** has **no separate MLP block** (the state-space mixer is the whole layer) — excluded, the COMPUTE analog of "no attention heads".
- Per-**neuron** read→write idioms are catalogued for **GPT-2 only** (the cheap token-unembedding basis); the cross-model rows are per-**layer** causal profiles. RoPE neuron-idioms need the per-layer SAE / token-centroid basis (the `disassemble_gemma.py` route).

_Data: `runs/disassembly/operators/mlp_compute_summary.json`. Regenerate: `mlp_atlas.py`._