# Circuit `induction_kchain_weights` (GPT-2)

**K-composition (weight + path-patch, GPT-2)** — scope: gpt2

The induction macro read from the **weights** (K-composition) and path-patch-gated on GPT-2 (the cross-model behavioural view is on the `induction` page).

- canonical writer **4.11**; 4/5 canonical edges live
- K-composition static 0.06873766100185522 vs random 0.03909262899650067; top edge rel-drop 0.5611611045403996

_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json) + the discovery artifacts. Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._