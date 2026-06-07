# Circuit `v_virtual_heads` (GPT-2)

**V-composition (composed-OV 'virtual heads', GPT-2)** — scope: gpt2

Composed-OV **virtual heads**: an induction head's OV output is re-read as the *value* of a later head (the third Elhage edge type — changes what is moved, not where attention points).

- top V-edges: `5.9->6.7` (ΔV-out 1.32), `5.9->6.0` (ΔV-out 0.60), `5.5->6.7` (ΔV-out 0.87), `5.5->6.6` (ΔV-out 0.72), `5.9->7.3` (ΔV-out 0.54)
- median ΔV-out 0.21387540992208515; static-V↔ΔV-out ρ 0.36; V/K 0.7954648723134081

_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json) + the discovery artifacts. Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._