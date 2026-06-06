# `scripts/` — guide to the code

The scripts are grouped by **research thread**, not by file type. Each group is one arc of
the project; read top-to-bottom within a group. The full narrative (what each result means)
is in the top-level [`README.md`](../README.md); the disassembly thread has its own deep-dive
in [`docs/DISASSEMBLY.md`](../docs/DISASSEMBLY.md).

## How to run

lm-sae is **standalone** — its own venv, `sae-forge` from PyPI, no bio-sae path:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
# GPU (RTX 50-series / Blackwell): .venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python scripts/<group>/<script>.py --help
```

Scripts write a machine-readable `runs/<name>_summary.json` (tracked in git) plus larger
artifacts (`.txt`/`.json`/`.npz`/`.pt`, git-ignored). See [`runs/README.md`](../runs/README.md).

Shared modules live in `common/` and are imported by name; the non-`common` scripts insert
`scripts/common/` on `sys.path` at import, so always run them from the repo root.

---

## `common/` — shared substrate + the core instrument
The three modules every other script builds on.

| script | role |
|--------|------|
| `build_lm_bundle.py` | GPT-2 (cached) → layer activations + the **exact-lexical oracle** `Y` → `data/lm_bundle_gpt2.npz`. Defines `COMMON` (the curated common-token operand list). |
| `forge_cov_mechanism.py` | train a TopK SAE → per-tier **cov95 / mAUC** + the **N1** ablations (rank / LayerNorm / TopK). The host-side instrument. |
| `preserve_hybrid_tiny.py` | the **P1 preserve-verbatim** hybrid: keep top-K oracle-reading atoms verbatim + forge the rest; sweep K. |

## `substrate/` — the models under test
| script | role |
|--------|------|
| `train_tiny_gpt.py` | trains the CPU-feasible tiny GPT-2 (`n_embd=128`, 4 layers, 7.2M params) → `runs/tiny_gpt.pt`. The whole-loop host. |
| `sae_lens_eval.py` | swaps the self-trained SAE for a published **SAELens** GPT-2 dictionary and re-scores the oracle. |

## `cov95_forge_tax/` — the forge tax on a language model
The core finding: forging an SAE basis preserves mAUC but collapses cov95 (monosemanticity).
| script | role |
|--------|------|
| `forge_gpt2.py` | the GPT-2 + 24k-SAELens forge (a **negative control** — the over-completeness wall). |
| `whole_loop_tiny.py` | train → SAE → **forge** → forged-cov95 on the tiny GPT. The tax replicates on an LM. |
| `width_sweep_tiny.py` | N1-width sweep (1×–16× over-complete) → the tax is **emergent**, not over-completeness-driven. |
| `forge_aware_train_tiny.py` | train *through* the SAE basis (geometry-forcing) → halves the tax. |
| `residual_selector_tiny.py` | tests whether the preserve-set can be chosen **label-free** → falsified. |
| `pair_cov95_tiny.py` | relational (bilinear pair) oracle → relations are **compiled**, not composed. |
| `substrate_ablation_gpt2.py` | substrate controls for the GPT-2 forge. |
| `chi_ladder.py` | the χ (monosemanticity) ladder over entanglement bands. |

## `entanglement_tower/` — the M0…Mn decomposition
"Harvest the cleanest features, subtract, repeat" → an additive low-χ-core + high-χ-tail tower.
| script | role |
|--------|------|
| `mps_tower_tiny.py` | build the fixed-model tower (taper / dial / convergence). |
| `mps_tower_retrain_tiny.py` | complement-routing retrain → **backfires** (re-entangles). |
| `mps_tower_geoforce_tiny.py` | geometry-forcing retrain → better but a **no-go** (capability ⇒ entanglement). |
| `serve_tower_tiny.py` | serve the tower as a runnable cascade → interpretability dials, capability cliffs. |

## `disassembly/` — GPT-2 attention as an instruction set
Reads GPT-2's attention as a reused **op-catalog**, scores coverage, **causally validates**, and
checks corpus-robustness. Full write-up in [`docs/DISASSEMBLY.md`](../docs/DISASSEMBLY.md).
Run order: idiom library → opcode tables → scorecard → causal validation → corpus robustness.

| script | role |
|--------|------|
| `idiom_library_v2.py` | the **8 literature-validated idioms** + coreference + the composed IOI chain (the canonical version; `idiom_library.py` is v1). |
| `qk_opcode_table.py` / `qk_feature_probe.py` | the QK opcode `B_h[X,Y]=d_X·M_h·d_Y` over token / SAE-feature operands. |
| `ov_write_channel.py` | the OV write channel (copy vs transform). |
| `sae_opcode_table.py` | richer **SAE-feature operands** — resolves dark heads token identity misses. |
| `addressing_split.py` / `absolute_addressing.py` | the addressing taxonomy (content / relative-Δ / absolute-sink / structural). |
| `coverage_scorecard.py` | "% of attention the catalog explains" + the dark-head work-list (`--corpus` for the prose baseline). |
| `causal_validation.py` / `ioi_causal.py` | mean-ablation: are the named heads **load-bearing**? (induction-NLL; IOI logit-diff). |
| `corpus_robustness.py` | which claims are corpus-invariant vs corpus-conditioned. |
| `disassemble_gpt2.py` | **the unified per-head listing** → `runs/disassembly/gpt2_disassembly.txt`. |
| `induction_probe.py`, `induction_graph.py`, `composition_probe.py`, `composition_graph.py`, `path_patch_induction.py`, `rung3_induction_chain.py`, `cross_position_probe.py`, `hub_probe.py`, `mlp_catalog.py`, `write_bus_check.py`, `instruction_{nmf,templates,tensor_rank}.py` | the supporting probes (induction circuit, composition graph, MLP catalog, write-bus, instruction-tensor structure). |

## `two_basis_forge/` — the two-basis forge + its retraction
U_A (assertion → cov95) + U_C (composition → circuits). **The writer-output `U_C`
circuit-preservation claim was RETRACTED** (see README): the `excess` metric is gameable;
compression-controlled re-validation showed writer-OV ≈ random-OV.
| script | role |
|--------|------|
| `two_basis_single_layer.py` / `two_basis_forge_oracle.py` | the two-basis forge on the tiny host (readers vs writers vs attribution). |
| `two_basis_saelens_gpt2.py` | two-basis on a real SAELens GPT-2 dictionary. |
| `forge_writer_specificity.py` | writer-specificity test (honest negative + mechanism controls). |
| `forge_compression_controlled.py` | the **compression-controlled re-validation** that retired the claim. |
| `forge_revalidate_broad.py` | broadened re-validation (layers × seeds) confirming RETIRE. |

## `gemma/` — the cross-model port (Gemma-2-2B)
Ports the disassembly to a recent RoPE / GQA / RMSNorm model, at GPT-2 parity.
| script | role |
|--------|------|
| `disasm_portable.py` | behavioral idioms + coverage on **any** HF model (the arch-agnostic core). |
| `gemma_opcode_table.py` | QK opcode table with **Gemma Scope** SAE operands (RoPE/GQA/RMSNorm-aware). |
| `gemma_causal.py` | induction-NLL causal validation on Gemma. |
| `disassemble_gemma.py` | **the unified per-head listing at GPT-2 parity** (all-layer QK bind + OV WRITE + GeGLU MLP catalog + SAE-feature opcode at the SAE layer) → `runs/gemma/gemma2_disassembly.txt`. |
| `gemma_layer_sweep.py` | QK content-opcode legibility **across depth** (which layers are content-addressing). |
