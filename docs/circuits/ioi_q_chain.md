# Circuit `ioi_q_chain` (GPT-2)

**Q-composition chain (GPT-2-only)** — scope: gpt2

The indirect-object-identification circuit: **duplicate-token → S-inhibition → name-mover**, a Q-composition chain (no published head-set outside GPT-2).

- S-inhibition heads: ['10.0', '8.3', '8.10', '6.7']; name-movers: ['9.9', '11.2', '8.11', '10.0']
- Q-composition live edges: **5**; named-edge live-rate 0.42857142857142855
- IOI baseline logit-diff 2.7526195287704467
- causal z (`ioi_causal.py`): name-mover -2.172512152617327, S-inhibition -1.702171401577731, **negative/copy-suppression 62.41397239659889** (writes against IO), duplicate 5.963926080885061, backup name-mover 0.8435903361394431
- **self-repair** (`self_repair.py`): −primaries ΔLD -0.0019234657287596768, −both 1.0389485120773314 → backups are hot spares (idle with primaries present, carry the circuit once they're gone).

## Cross-model IOI dossier (the circuit's operators, found behaviourally — via the ResidualVM)

The Q-composition *edge wiring* below is GPT-2-validated, but the IOI **behaviour** and its **operators** are not GPT-2-only: the unified [`ResidualVM`](../DECOMPILATION.md) locates them in every model (name-movers by END→indirect-object copy-attention; negative-movers + the most load-bearing heads by an ablation sweep of the logit-diff; the duplicate-token initiator behaviourally). Logit-diff = logit(IO) − logit(S) at the end of a templated *"When X and Y went…, Y gave a drink to →"* prompt.

| model | IOI logit-diff | name-movers (copy→IO) | negative-movers | most load-bearing | ablate name-movers | duplicate init |
|---|---|---|---|---|---|---|
| gpt2 | +2.88 | `9.9`, `10.0`, `10.6` | `10.7`, `9.6` | `8.10`, `8.6` | +22% | `3.0` |
| gpt2-medium | +3.11 | `15.14`, `18.5`, `20.6` | `18.9`, `22.14` | `19.1`, `12.3` | +18% | `7.11` |
| gpt2-large | +4.09 | `22.0`, `29.0`, `23.13` | `32.0`, `26.0` | `20.14`, `18.3` | +19% | `7.5` |
| gemma-2-2b | +3.45 | `18.6`, `21.5`, `22.5` | `22.4`, `22.0` | `23.5`, `20.6` | +26% | `8.1` |
| Llama-3.2-1B | +5.85 | `12.13`, `12.2`, `11.14` | `12.15`, `15.12` | `8.19`, `11.4` | +24% | `8.19` |
| Qwen2.5-1.5B | +6.00 | `27.4`, `27.8`, `24.10` | `23.8`, `25.4` | `24.8`, `0.6` | +13% | `8.3` |

- **The IOI circuit is architecture-invariant** — name-movers, negative/copy-suppression movers, and a duplicate-token initiator are present in all six models, and ablating the name-movers collapses the logit-diff (+13% to +26%) everywhere. The behaviour *strengthens* with GPT-2 scale (logit-diff +2.88 → +3.11 → +4.09) and is largest in the RoPE models (Llama +5.85, Qwen +6.00).
- **The backup-name-mover self-repair is cross-model.** The heads that are *most load-bearing* under ablation are **not** the name-movers (which are backed up) but the **S-inhibition**-type heads — in every model the ablation ranking and the copy-attention ranking disagree, the signature of name-mover redundancy (the Hydra effect) generalised beyond GPT-2.

_Data: [runs/disassembly/circuits/ioi_xmodel_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/ioi_xmodel_summary.json) ([ioi_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/ioi_xmodel.py), built on the ResidualVM)._

_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json) + the discovery artifacts. Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._