---
title: SAE-feature operands
---

# SAE-feature operands per operator

The token-operand catalog says which **tokens** an operator binds; this says which monosemantic **SAE features** it reads, and whether its OV **copies** that content (+) or **suppresses** it (−) — the dossier's *section-G* layer. **READ** = the attention-weighted dominant key-feature (the SAE feature most present where the head attends; content-filtered, glossed by top tokens). **copy-score** = the OV→unembed diagonal on that feature's own tokens. Provisional, single corpus (Shakespeare prose); `_` = leading space.

> Only **content / circuit** operators bind on content; **positional / addressing** ops (prev-token, local, sink, self) attend by position, so their read-feature is incidental — the copy-score column is load-bearing.

## gpt2 — jbloom/GPT2-Small-SAEs-Reformatted

| operator | head | kind | reads (SAE feature) | copy-score (OV) |
|---|---|---|---|---|
| `duplicate` | 0.5 | content | **_you**; **_the**; **US** | +0.09 (copies) |
| `self` | 0.1 | addressing | **_you**; **US**; **_the** | +0.07 (copies) |
| `structural` | 3.1 | structural | **_your/Your/_Your**; **The**; **_it/it** | -0.03 (≈neutral) |
| `prevtok` | 4.11 | positional | **US**; **cius/ius**; **I** | +0.10 (copies) |
| `local` | 4.11 | positional | **US**; **cius/ius**; **I** | +0.10 (copies) |
| `induction` | 5.5 | content | **First**; **What/_What/_what**; **US** | +0.09 (copies) |
| `sink` | 7.2 | addressing | **US/us**; **_Citizen/_citizens**; **I** | +0.00 (≈neutral) |
| `s_inhibition` | 7.3 | circuit | **US/us**; **MAR/MEN/CI**; **_you/you/You** | +0.01 (≈neutral) |
| `name_mover` | 9.6 | circuit | **_Citizen/_citizens**; **MEN/_men/_Men**; **_it/'d/_are** | +0.04 (copies) |
| `backup_name_mover` | 9.0 | circuit | **_it/'d/_are**; **_Citizen/_citizens**; **US/us** | +0.03 (copies) |
| `coreference` | 9.0 | circuit | **_it/'d/_are**; **_Citizen/_citizens**; **US/us** | +0.03 (copies) |
| `negative_mover` | 10.7 | circuit | **'d/_not/_and**; **And/That/_and**; **_Citizen/_citizens** | -0.01 (≈neutral) |

## gemma-2-2b — Gemma Scope (gemma-scope-2b-pt-res, JumpReLU)

| operator | head | kind | reads (SAE feature) | copy-score (OV) |
|---|---|---|---|---|
| `duplicate` | 1.4 _(SAE L0, head L1)_ | content | **cius/▁belly/VIR**; **▁the**; **UMN/GIL/▁Cai** | -0.15 (suppresses) |
| `sink` | 0.3 | addressing | **cius/▁belly/VIR**; **UMN/GIL/▁Cai**; **▁Citizen** | +0.10 (copies) |
| `local` | 0.0 | positional | **cius/▁belly/VIR**; **UMN/GIL/▁Cai**; **▁the** | +0.02 (≈neutral) |
| `induction` | 6.3 | content | **First**; **▁gods/▁run/▁petition**; **Before** | +0.06 (copies) |
| `prevtok` | 21.7 | positional | **▁Citizen/./▁belly**; **▁the/▁a/▁to**; **First/▁first** | -0.10 (suppresses) |
| `self` | 25.7 _(SAE L24, head L25)_ | addressing | **▁the/⏎⏎/.**; **First**; **▁the/▁your/▁own** | -0.21 (suppresses) |
| `structural` | 24.6 | structural | **▁the/⏎⏎/.**; **First**; **▁with/With/▁With** | +0.11 (copies) |

_GPT-2 has all-layer SAEs (exact per-head layer); Gemma Scope is 8 layers so each Gemma op uses its nearest available SAE layer (offset ≤1, annotated). **Gemma's read-features come out noisier** than GPT-2's — its heads put heavy attention on `<bos>`/structural tokens on this non-repetitive prose, so the dominant *content* key-feature is weaker (a corpus + attention-budget effect, not a tooling one); the copy-score still uses the head's exact OV. The cached Qwen SAE is for qwen2-0.5b (a different model). Data: [operator_sae_operands_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/operator_sae_operands_summary.json). Regenerate: [operator_sae_operands.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_sae_operands.py)._