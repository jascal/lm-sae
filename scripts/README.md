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
| `ceiling_test.py` | **decompilation milestone 4** — forge-basis reconstruction coverage: KL (capability) + cov95 (legibility) + mAUC across SAE width; content/factorability axes decouple (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
| `host_width_sweep.py` | **reachability test** — train tiny GPTs from scratch across host width, with/without an oracle-feature-recovery aux loss (linear-recoverability proxy); supervision lifts native cov95 at ~zero capability cost (defeats the tower retrain no-go). |
| `monosemantic_aux.py` | **aux-mode comparison** (`--seeds`, multi-seed) — none/linear/decorr/dedicated/sparsedict at fixed width; linear-recoverability is the *effective* lever (3/3 seeds beat none; no direct objective beats it; the full in-loop aligned sparse dict is robustly *worst*). |
| `legibility_crosscheck.py` | **circularity test** — scores the supervision cov95 lift in SAE / raw-neuron / PCA bases; SAE-free metrics corroborate (genuine monosemanticity, not the eval-SAE finding planted features). |
| `oracle_supervised_dag.py` | **does feature legibility buy circuit legibility?** — trains tiny GPTs unsup vs oracle-supervised, measures feature legibility (cov95) vs circuit legibility (static→dynamic composition ρ + induction recovery) + prev-token key position/token split on the same models. Result: PARTIAL/substrate-dominated — supervision reshapes the feature substrate (cov95 ↑, prev-token key shifts token-ward 3/3) with only marginal circuit spillover → largely separate axes (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
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
| `residual_vm.py` | **decompilation milestone 1** — reconstruction-coverage interpreter: keep a head-set, mean-ablate the rest, `1 − KL/floor` vs budget + random + named-idiom set (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
| `composition_dag.py` | **decompilation milestone 2** — the composition-DAG extractor: full weight-space K/Q edge scorer + an idiom-agnostic path-patch gate (ΔTV) with a **reader-matched null**; auto-recovers the induction K-chain + IOI Q-chain (live, imposters/random rejected at 0% FP), reports new candidate sub-DAGs. Unifies `composition_graph.py` + `path_patch_induction.py` (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
| `dag_recompile.py` | **M1↔M2 bridge (+ the V value pathway)** — feeds the M2 live sub-DAG into M1's reconstruction-coverage harness: the path-patch-confirmed induction+IOI sub-DAG is a coverage-efficient keep-set (matches/beats equal-size greedy top-importance, ≫ random) — but raw ΔTV-connectivity is *not* a generic output-importance proxy. Also adds the V-composition heads (`--v-summary`): they add only +0.004 coverage (< random) → ΔV-out, like ΔTV, is edge-coupling not output-importance; the keep-set is the confirmed circuit, across all K/Q/V ports. Consumes `composition_dag_summary.json` + `vcomposition_summary.json` (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
| `mlp_ops.py` | **decompilation milestone 3** — the MLP (COMPUTE) class in the DAG + recompile: extends the coverage harness to keep/ablate MLP layers (M1/bridge were heads-only), finds MLPs load-bearing & concentrated in L0 (the detokenizer); adds weight-legible head↔MLP composition edges; names load-bearing MLPs. Reuses `residual_vm.py` + `composition_dag.py` (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
| `vcomposition.py` | **the value pathway (V-composition)** — adds Elhage's third edge type the M2 ΔTV readout was blind to: static `‖OV_A·W_V^B‖` + a value-output gate (remove A from B's value, measure ΔB-output). Clean K/V double dissociation (V-edges move output not attention; K-edges the mirror); static V predicts ΔV-out (ρ +0.36); top V-edges are composed-OV "virtual heads" (induction→layer-6 values). Completes K/Q/V edges (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
| `validate_new_edges.py` | **M2 follow-up** — targeted single-edge path-patch of the 22 new write-hub edges with a behavioral readout: removing each early **sink** writer from a downstream key collapses that head's **prev-token** attention (incl. the canonical prev-token head 4.11) vs a reader-matched null → 13/22 named (9 prevtok). The hubs broadcast the positional signal prev-token addressing reads. Consumes `composition_dag_summary.json` (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
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

## `gemma/` — the cross-model port (Gemma-2 / Llama-3 / Qwen-2.5)
The weight-space disassembler is **arch-generic**: one `--model` flag runs Gemma-2-2B, Llama-3.2-1B,
and Qwen2.5-1.5B (any RoPE / GQA / RMSNorm / gated-MLP HF LM). Per-architecture constants live in
`arch_config.py`; the **Gemma Scope** SAEs (`google/gemma-scope-2b-pt-res`, Gemma-only) auto-download
on first use via `scope_loader.py` (or `--scope-path` for offline). (Despite the `gemma_*` filenames,
only the opcode-table / layer-sweep are Gemma-specific — the disassembler and causal validator are general.)
| script | role |
|--------|------|
| `disasm_portable.py` | behavioral idioms + coverage on **any** HF model (the arch-agnostic core). |
| `disassemble_gemma.py` | **the unified per-head listing at GPT-2 parity, any model** (all-layer QK bind + OV WRITE + gated-MLP catalog; SAE-feature opcode at the SAE layer where a SAE exists) → `runs/gemma/`, committed `.txt` under `docs/listings/`. |
| `gemma_causal.py` | induction-NLL causal validation (arch-general; `--model`). |
| `sink_ablation.py` | block attention to the sink (key-0) and measure ΔNLL (arch-general) — is the sink load-bearing? Finds magnitude ≠ dependence (only GPT-2 depends). |
| `cross_model_positional.py` | **cross-model test of the positional-broadcast circuit** — decomposes each model's prev-token-head **key** variance into position- vs token-explained: GPT-2 is position-dominated (pos/tok 3.3 — keys encode absolute position, what the sink heads broadcast), all RoPE models token-dominated (0.04–0.35 — position in the rotation). Explains why only GPT-2 needs the sink→prev-token broadcast (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
| `key_patch_cross_model.py` | **cross-model causal key-only path-patch** (the causal complement to `cross_model_positional.py`) — removes an upstream head's KEY *content* from the prev-token head (forward hook on k_proj/c_attn so each model applies its own RoPE; zero-patch sanity 0.0). GPT-2: removing **sink head 1.3** collapses 4.11's prev-token −22% (z 119); RoPE models: no upstream head collapses it (−0%) → GPT-2 reads position from the key content, RoPE from the rotation (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
| `cross_model_ceiling.py` | **decompilation milestone 5** — arch-generic reconstruction-coverage ceiling (M1's metric) on 5 models: the named induction circuit beats random everywhere (mechanisms invariant) but the decompilable *fraction* is **not** architecture-invariant — and the **gpt2-medium control disentangles it**: the absolute-position GPT-2 family keeps a ~20% / 4–17× fraction *at any size* (gpt2-medium 384h ≫ the larger RoPE models) while every RoPE model sits at 3–9% / 1.4–2.9× → it's the **absolute-position family, not scale** (RoPE distributes the circuit; absolute concentrates it). Op-selection ceiling; forge-basis stays SAE/GPU-gated (see [`docs/DECOMPILATION.md`](../docs/DECOMPILATION.md)). |
| `multilingual_ops.py` | idiom-head invariance + attention budget across languages/scripts (the ops are language-universal; needs `datasets` to stream Wikipedia). |
| `gemma_opcode_table.py` | QK opcode table with **Gemma Scope** SAE operands (Gemma-only; RoPE/GQA/RMSNorm-aware). |
| `gemma_layer_sweep.py` | QK content-opcode legibility **across depth** (Gemma Scope; Gemma-only). |
| `arch_config.py` | per-architecture knobs (RMSNorm gain offset, QK scale, SAE availability). |
| `scope_loader.py` | portable Gemma Scope SAE resolver (explicit path → HF cache → download). |
