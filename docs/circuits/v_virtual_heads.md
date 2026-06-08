# Circuit `v_virtual_heads` (GPT-2)

**V-composition (composed-OV 'virtual heads', GPT-2)** — scope: gpt2

Composed-OV **virtual heads**: an induction head's OV output is re-read as the *value* of a later head (the third Elhage edge type — changes what is moved, not where attention points).

- top V-edges: `5.9->6.7` (ΔV-out 1.32), `5.9->6.0` (ΔV-out 0.60), `5.5->6.7` (ΔV-out 0.87), `5.5->6.6` (ΔV-out 0.72), `5.9->7.3` (ΔV-out 0.54)
- median ΔV-out 0.21387540992208515; static-V↔ΔV-out ρ 0.36; V/K 0.7954648723134081

## Cross-model (the value pathway is not GPT-2-only — via the ResidualVM)

Static **V-composition** `‖W_V^B · OV_A‖ / (‖OV_A‖‖W_V^B‖)` — how much of induction head A's OV output lands in downstream head B's *value*-read subspace (a composed-OV virtual head; the same weight basis the catalog scores K/Q composition in, arch-generic incl. GQA). The control is whether induction-A's V-composition into downstream values **exceeds a random non-induction writer's** (specificity).

| model | composed-OV writer (induction) | → reader values | specificity vs random |
|---|---|---|---|
| gpt2 | `5.5` | `6.9`, `6.6`, `7.1`, `7.6` | +0.008 |
| gpt2-medium | `11.1` | `12.1`, `15.14`, `13.12`, `17.12` | +0.031 |
| gpt2-large | `19.4` | `20.14`, `26.0`, `27.11`, `24.8` | +0.032 |
| gemma-2-2b | `22.3` | `23.4`, `23.5`, `23.6`, `23.7` | +0.013 |
| Llama-3.2-1B | `12.15` | `13.20`, `13.21`, `13.22`, `13.23` | +0.035 |
| Qwen2.5-1.5B | `19.3` | `23.0`, `23.1`, `23.2`, `23.3` | +0.038 |

- **V-composition is architecture-invariant** — in every model an induction head's OV output feeds downstream heads' **values** with positive specificity over a random writer, *locally* (the induction head feeds the *next* layers' value heads). Induction content re-read as a value (a 2-hop OV copy) is a universal motif, completing the cross-model K/Q/**V** composition-edge triad.
- **Scope / honesty.** The static signal is *weak* (specificity ≤0.04) — consistent with the GPT-2 finding that these virtual heads are output-*redundant* (`vcomposition.py`; they add ~nothing to the recompile keep-set). In the GQA models the reader values come out as a **contiguous block** of heads (e.g. Llama `13.20`–`13.23`) because query heads in one KV group **share `W_V`**, so their V-composition is identical — a grouping artifact, not four distinct readers. Dynamic ΔV-out confirmation stays GPT-2-validated (ρ(static, ΔV-out) = +0.36).

_Data: [runs/disassembly/circuits/vcomposition_xmodel_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/vcomposition_xmodel_summary.json) ([vcomposition_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/vcomposition_xmodel.py), built on the ResidualVM)._

_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json) + the discovery artifacts. Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._