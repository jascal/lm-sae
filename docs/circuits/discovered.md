---
title: Discovered circuits
---

# Discovered circuit edges — the key-patch run over the top content readers in every model

A **working catalog** (amateur, exploratory, provisional) of de-novo composition **edges**: for the top behavioural content readers in each model we path-patch every upstream head out of the reader's **key** and keep the edges that collapse the reader's attention beyond a reader-matched null. writer → reader (K-composition); *live* = robust collapse.

_6 models · top content readers × all upstream · faithful key-only patch._

## gpt2 (GPT-2/absolute) — 6/6 live edges  (prev-token head 4.11)

| reader | pattern | top upstream writer | key-collapse | z | live? |
|---|---|---|---|---|---|
| 5.1 | induction | 4.11 (=prev-tok head) | +46% | 16.6 | **yes** |
| 6.9 | induction | 4.11 (=prev-tok head) | +26% | 19.8 | **yes** |
| 7.2 | induction | 4.11 (=prev-tok head) | +25% | 16.3 | **yes** |
| 5.5 | induction | 4.11 (=prev-tok head) | +22% | 24.5 | **yes** |
| 7.10 | induction | 4.11 (=prev-tok head) | +17% | 30.7 | **yes** |
| 3.0 | duplicate | 1.9 | +15% | 5.0 | **yes** |

## gpt2-medium (GPT-2/absolute) — 2/6 live edges  (prev-token head 5.11)

| reader | pattern | top upstream writer | key-collapse | z | live? |
|---|---|---|---|---|---|
| 7.2 | induction | 4.13 | +17% | 12.9 | **yes** |
| 9.9 | induction | 4.13 | +11% | 15.4 | **yes** |
| 11.1 | induction | 4.13 | +9% | 42.7 | no |
| 7.11 | duplicate | 4.6 | +9% | 21.7 | no |
| 12.1 | induction | 4.13 | +8% | 22.5 | no |
| 18.5 | induction | 1.6 | +1% | 11.1 | no |

## gpt2-large (GPT-2/absolute) — 2/6 live edges  (prev-token head 14.1)

| reader | pattern | top upstream writer | key-collapse | z | live? |
|---|---|---|---|---|---|
| 5.19 | duplicate | 3.3 | +17% | 135.7 | **yes** |
| 5.8 | duplicate | 3.3 | +16% | 191.7 | **yes** |
| 16.9 | induction | 3.14 | +5% | 13.1 | no |
| 15.4 | induction | 3.14 | +3% | 10.4 | no |
| 19.4 | induction | 3.14 | +2% | 11.5 | no |
| 16.0 | induction | 3.14 | +1% | 16.9 | no |

## gemma-2-2b (RoPE) — 0/6 live edges  (prev-token head 21.7)

| reader | pattern | top upstream writer | key-collapse | z | live? |
|---|---|---|---|---|---|
| 6.2 | induction | 5.0 | +5% | 37.2 | no |
| 6.3 | induction | 5.0 | +3% | 35.0 | no |
| 5.4 | duplicate | 4.4 | +2% | 4.7 | no |
| 8.1 | duplicate | 6.1 | +2% | 6.0 | no |
| 3.2 | duplicate | 2.6 | +1% | 2.3 | no |
| 1.4 | duplicate | 0.0 | +1% | 4.9 | no |

## Llama-3.2-1B (RoPE) — 3/6 live edges  (prev-token head 0.2)

| reader | pattern | top upstream writer | key-collapse | z | live? |
|---|---|---|---|---|---|
| 4.16 | duplicate | 1.9 | +30% | 35.2 | **yes** |
| 5.10 | induction | 1.20 | +29% | 40.5 | **yes** |
| 6.8 | duplicate | 1.9 | +20% | 22.9 | **yes** |
| 12.15 | induction | 1.9 | +4% | 10.0 | no |
| 12.0 | duplicate | 1.28 | +2% | 9.4 | no |
| 10.23 | induction | 1.9 | +2% | 12.3 | no |

## Qwen2.5-1.5B (RoPE) — 1/6 live edges  (prev-token head 13.4)

| reader | pattern | top upstream writer | key-collapse | z | live? |
|---|---|---|---|---|---|
| 2.3 | induction | 1.4 | +85% | 51.8 | **yes** |
| 14.4 | induction | 5.5 | +2% | 2.9 | no |
| 19.5 | induction | 1.4 | +1% | 4.6 | no |
| 19.3 | induction | 1.4 | +0% | 9.1 | no |
| 8.3 | duplicate | 0.4 | +0% | 5.4 | no |
| 14.3 | induction | 2.10 | +0% | 5.3 | no |

## How to read this

- A **live** edge = removing that upstream writer from the reader's key collapses the reader's content attention beyond a reader-matched null → a real K-composition edge. For induction readers the top writer is typically the model's **prev-token head** (the canonical induction wiring), recovered de novo here.
- Provisional and descriptive. Value-channel (move) edges and Q-composition are not in this pass (key/match only). See the [circuit catalog](README.md) for the named circuits.

_Data: [runs/disassembly/circuits/discovered_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/discovered_summary.json). Regenerate: [circuit_discovery.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_discovery.py)._