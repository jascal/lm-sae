---
title: Succession (the +1 operator)
---

# Localizing succession — the +1 / greater-than operator

The operator catalog lists **succession / greater-than** as a *gap* ("MLP-dominated; no clean attention head"). This puts data behind it. Task: a run of consecutive single-token numbers (" 3 4 5 6 7") → predict the next; the **succession-NLL** is the metric. With all attention intact, mean-ablate each layer's MLP; with all MLPs intact, each layer's attention — the layers whose ablation most raises succession-NLL are where the increment is computed.

| model | runs | base NLL | **MLP-dominance** (ΣMLP / (ΣMLP+Σattn)) | top succession-MLP (depth, ΔNLL) | top succession-attn (depth, ΔNLL) |
|---|---|---|---|---|---|
| gpt2 | 254 | 0.05 | **99%** | L1 (0.09, +9.7); L0 (0.00, +9.7); L2 (0.18, +8.9) | L11 (1.00, +0.3); L9 (0.82, +0.0); L1 (0.09, +0.0) |
| gpt2-medium | 254 | 0.04 | **100%** | L0 (0.00, +10.8); L3 (0.13, +5.6); L2 (0.09, +5.0) | L21 (0.91, +0.0); L1 (0.04, +0.0); L19 (0.83, +0.0) |
| gpt2-large | 254 | 0.05 | **95%** | L7 (0.20, +6.9); L8 (0.23, +5.7); L9 (0.26, +5.6) | L0 (0.00, +1.4); L23 (0.66, +0.0); L14 (0.40, +0.0) |

_**Finding: succession is overwhelmingly MLP-computed** (95–100% MLP-dominance) and lives in the **early–mid MLPs** (GPT-2-small L0–L2, gpt2-large L7–9) — putting data behind the catalog's "MLP-dominated, no clean attention head" gap. **GPT-2 family only:** the RoPE tokenizers (Gemma, Llama, Qwen) have **no single-token numbers** (they split ` 1` into multiple tokens), so consecutive number runs don't exist — which is itself why succession studies use GPT-2._

_MLP-dominance = the MLP layers' share of the total (positive) ablation damage to succession; **>50% confirms the catalog's MLP-dominated claim**, and the top-MLP layers say *where* the increment lives. ΔNLL = succession-NLL rise when that layer's MLP / attention is mean-ablated. Provisional, single-token number runs (length 5). Data: [succession_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/succession_summary.json). Regenerate: [succession.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/succession.py). See the [operator catalog](../operators/README.md) gaps._