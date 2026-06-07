---
title: SAE-feature operands
---

# SAE-feature operands per operator (gpt2)

The token-operand catalog says which **tokens** an operator binds; this says which monosemantic **SAE features** it reads and writes — the dossier's *section-G* layer. Using the published per-layer GPT-2 SAEs ([jbloom/GPT2-Small-SAEs-Reformatted](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted), resid_pre, 24576 features/layer): for each operator's top head, **READ** = the attention-weighted dominant key-feature (the SAE feature most present where the head attends; content-filtered, glossed by its top tokens). **copy-score** = the OV→unembed diagonal on that feature's own tokens — attending to those tokens, does the head **raise** their own logit (**+ = copies**) or **lower** it (**− = copy-suppression / negative head**)? Provisional, single corpus (Shakespeare prose); `_` = leading space.

> **Read this with care.** Only **content / circuit** operators bind on *content*; **positional / addressing** operators (prev-token, local, sink, self) attend by *position* or to key-0, so their "read-feature" is whatever token happened to sit there — incidental, not a content bind. The copy-score is the load-bearing column.

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

_Read-features by attention mass; copy-score is for the top read-feature. GPT-2 only here (all-layer SAEs); Gemma Scope / Qwen are single-layer (a follow-up). Data: [operator_sae_operands_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/operator_sae_operands_summary.json). Regenerate: [operator_sae_operands.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_sae_operands.py). See the [operator catalog](README.md) and the [token-operand opcode tables](../DISASSEMBLY.md)._