# Circuit `ioi_q_chain` (GPT-2)

**Q-composition chain (GPT-2-only)** — scope: gpt2

The indirect-object-identification circuit: **duplicate-token → S-inhibition → name-mover**, a Q-composition chain (no published head-set outside GPT-2).

- S-inhibition heads: ['10.0', '8.3', '8.10', '6.7']; name-movers: ['9.9', '11.2', '8.11', '10.0']
- Q-composition live edges: **5**; named-edge live-rate 0.42857142857142855
- IOI baseline logit-diff 2.7526195287704467
- causal z (`ioi_causal.py`): name-mover -2.172512152617327, S-inhibition -1.702171401577731, **negative/copy-suppression 62.41397239659889** (writes against IO), duplicate 5.963926080885061, backup name-mover 0.8435903361394431
- **self-repair** (`self_repair.py`): −primaries ΔLD -0.0019234657287596768, −both 1.0389485120773314 → backups are hot spares (idle with primaries present, carry the circuit once they're gone).

_Data: `runs/disassembly/circuits/atlas_summary.json` + the discovery artifacts. Regenerate: `circuit_catalog_doc.py`._