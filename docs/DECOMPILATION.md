# From op-catalog to decompilation — a research design

[`DISASSEMBLY.md`](DISASSEMBLY.md) reads a transformer's attention as a **first-order instruction set**:
per-head QK/OV opcodes, named idioms, a coverage scorecard ("% of attention *legible*"), causal validation,
and cross-model replication. This document scopes the next phase: turning that disassembly into an
**executable decompilation** — reconstructing the *computation*, not just labeling components — and argues
the obstacle is the *same* entanglement the forge tax measures.

This is a design doc (a target + a metric + milestones), not a results writeup. It is the flagship that
ideas (i) cross-model breadth and (ii) multilingual invariance de-risk: you want the disassembler proven on
≥3 architectures before you trust a single decompilation-coverage number.

## The gap: disassembly ≠ decompilation

| | disassembly (have) | decompilation (want) |
|---|---|---|
| unit | one head / one idiom in isolation | the **composition** — which ops chain into which |
| coverage metric | *% of attention legible* (named or weight-binding) | *% of the forward pass faithfully reconstructable* |
| MLP | first-order read→write token catalog | MLP neurons as first-class ops (key–value memories) |
| validation | mean-ablation damages the metric (necessary) | **recompile the extracted program; KL ≈ host** (sufficient) |
| output | a human-readable listing | a runnable reduced model + a symbolic trace |

The disassembly already took the first two rungs of composition: the **induction macro** read as
`OV_A ∘ QK_B` (prev-token head feeds the induction head; weight-composed diagonal vs behavior ρ≈0.78,
path-patch-gated), and the **IOI chain** `duplicate → S-inhibition → name-mover` read as a product of
Q-composition scores. Decompilation is the *general* algorithm those two are instances of.

## The load-bearing idea: reconstruction coverage

Replace "% legible" with a metric that has teeth — **does the extracted op-graph, compiled back into a
runnable model, reproduce the host's next-token distribution?**

```
reconstruction_coverage(budget B) =
    1 − KL( host ‖ recompiled[ ops selected under budget B ] ) / KL( host ‖ mean-ablated-everything )
```

- **Recompile = forge.** sae-forge's `NativeModel.from_projected_weights` already takes a basis + projected
  weights and produces a runnable model (`forward_mode=native_in_basis`). A decompilation is a *structured*
  forge: keep exactly the ops the disassembly named (heads' QK/OV in the operand basis + catalogued MLP
  neurons), mean-ablate the rest, run, measure KL. This is the "recompile" half of larql's
  decompile→query→**recompile** loop, used here as the faithfulness oracle.
- **Sweep the budget** (how many ops kept) → a coverage *curve*, the decompilation analog of the
  coverage scorecard's single number. The area under it is "how much of the model the catalog explains,
  *executably*."
- **Per-task traces.** For a templated task (IOI, induction, greater-than), emit the **symbolic trace**:
  the ordered DAG of ops that fire to produce the output, each edge path-patch-validated. A correct trace
  + a faithful recompile = a decompilation of that behavior.

## Four work-items

### 1. The composition DAG extractor
Generalize rung-2/3 into the standard algorithm: from the weights, score every directed edge
`OV_g → QK_h` (head g's write feeds head h's query/key/value) as a composition bilinear in the operand
basis; threshold + path-patch-validate to keep live edges. Output: the model's circuit DAG (heads + MLP
nodes, typed edges Q/K/V-composition). The induction macro and IOI chain become the first two validated
sub-DAGs; the extractor produces them automatically and finds new ones.

### 2. The MLP gap
The catalog is attention-centric; MLPs carry real computation (greater-than is MLP-dominated — the OV probe
sees only its attention-side shadow). Treat MLP neurons as ops: in-direction (gate/up read) and
out-direction (down write) in the operand basis = a key→value memory; cluster into named MLP idioms
(detokenization, entity attributes, successor arithmetic). Add them as nodes in the DAG (1) and as keepable
ops in the recompile (the metric).

### 3. The operand-basis ceiling
First-order disassembly uses centroid / SAE operands that are imperfect; composition only reads cleanly in a
basis where the ops *compose* cleanly. This is where decompilation meets the rest of the program: the
two-basis forge's **U_C (composition subspace)** is exactly "the basis in which circuits survive." (The
specific *writer-output* U_C construction was retracted — see DISASSEMBLY.md — but the *target* stands: find
the basis that makes reconstruction_coverage high at low budget.) `polygram` models the operand-dictionary
geometry; the decompiler consumes it.

### 4. Executable verification & emission
Wire (1)+(2) into the recompile metric (the load-bearing idea), and emit the DAG in a checkable form:
`n-orca` can express the extracted circuit as a typed neural DAG; `larql` can hold the decompiled model as a
queryable index ("the model IS the database"). A decompilation is *accepted* only if recompile-KL clears a
threshold — no unverified symbolic story counts.

## The unifying claim (why this is one program, not a side quest)

> Composition is not cleanly feature-readable for the **same reason** cov95 collapses under forging:
> the composition is entangled (the entanglement tower's irreducible high-χ core).

So **"how much can we decompile" and "how big is the un-forgeable core" are the same quantity measured from
two directions** — reconstruction_coverage from the disassembly side, `1 − cov95-tax` / the tower's
convergent core from the forge side. The falsifiable prediction:

> The reconstruction-coverage curve **plateaus** at the same fraction the entanglement tower identifies as
> the irreducible entangled core (~the low-χ substrate), and **no op-budget closes the gap** — because the
> residual *is* the entangled composition, which by the tower's no-go is not decompilable into first-order
> ops. Decompilation has a hard ceiling, and that ceiling is the forge tax.

If true, this is the result: a single number — measured independently as (a) the decompilable fraction of
the forward pass and (b) the forgeable/low-χ fraction — that bounds how interpretable a transformer can be
made *without* retraining. If the curve instead reaches ~1.0, the entanglement story is wrong and the model
is fully decompilable (also a publishable result).

### What form is the core? (not an SAE, not a dense slab — a program)
"SAE vs dense" is a false dichotomy for the entangled remainder. The forge tax says the core has **no sparse
feature (SAE) basis** in which composition is monosemantic — that rules out the SAE form. But the χ-ladder
showed composition *is* factorable in **weight/attention-composition coordinates** (induction = `OV_A∘QK_B`;
IOI = a head chain), and the instruction-tensor result found the program is **low-rank (~5–11 templates vs
~132 random) — in operand-pair space, not residual-direction space** (which is exactly why the residual-
direction search, joint-`U_C`, failed). So the core's natural form is a **program**: a compact DAG of reused
weight-space bilinear templates over *now-explicit* operands. Subtracting the clean low-χ substrate is then a
**change of coordinates, not a sparsification** — it makes the operands explicit so the remainder reads as
"bilinears with clean arguments," but it does **not** shrink the dense magnitude (the tower's ~24% irreducible
variance floor; the retrain no-go). Consequence for this program: the lever that "simplifies" the core is the
**composition-DAG extractor in operand coordinates** (work-item 1+2), *not* a better SAE — and the
reconstruction-coverage curve should plateau at the entangled-core fraction precisely because first-order
*feature* ops can't express the composition, while *template* ops over operands can. (Open at scale: whether
the template count stays small and whether the irreducible floor itself grows with capability — the tower
convergence was measured on small hosts.)

## Execution model: an interpreter over the op-graph ("ResidualVM")

The recompile-KL harness is most useful not as a one-shot metric but as a **steppable interpreter** over the
extracted op-graph: run the named ops on the residual bus, in selectable *fidelity modes*, and watch
reconstruction-KL — which makes decompilation **debuggable**, not just measurable. (Execution-model framing
contributed by Grok; integrated + corrected here.)

**The honest mechanics — dataflow per pass, "VM" only at the loop.** It is tempting to cast this as a von
Neumann machine (fetch–decode–execute over a stored program). That is the wrong abstraction for a *single
forward pass*: there is no program counter, no instruction fetched per cycle, no data-dependent control flow,
and the weights are never modified by the data path. One pass is a **fixed-depth dataflow circuit** (closer to
an ASIC / systolic array; Merrill's TC⁰) — the "ISA" (the op-catalog) is **hardwired and applied in
parallel**, a *description* of fixed functional units, not a runtime dispatch. The stored-program /
von-Neumann character appears **only at the autoregressive loop**: the residual stream + KV-cache as a
read/write tape, the decode step as the clock, chain-of-thought as working memory (see the
`llm-as-accreting-vm` framing). So the interpreter executes a **fixed circuit per token** and is a **VM at the
generation level** — not a stored-program CPU per layer.

Formal grounding (why the DAG/loop split is load-bearing, not decorative): the object is a **clocked
sequential machine** = combinational core (the fixed DAG) + state register (KV-cache + the growing token
sequence) + clock (the decode step). The **DAG alone is weak** — one bounded-depth pass sits in **TC⁰**
(Merrill–Sabharwal); the **loop supplies the power** — transformer + decoding + an unbounded scratchpad is
Turing-complete (Pérez et al., CoT-expressivity). Two precisions: (a) the recursion goes through a **discrete
token bottleneck** — high-dim state is sampled to a *token* and re-embedded, so depth-per-step is bounded but
steps are unbounded (hard problems buy back missing within-pass depth with *longer* CoT); (b) the program is
**fixed, not self-modifying** — the same DAG every step, only the data changes (microcode/ASIC-like; Turing
power lives entirely in the outer tape+clock). The decompilation payoff: **the loop is a clean recurrence —
the hard part is reading the DAG**, and `reconstruction_coverage` is exactly the measure of how much of that
fixed high-dim DAG reduces to a compact symbolic program over explicit operands.

What the frame *does* buy, mapped to checked-in results:

| VM concept | what it actually is here | grounded in |
|---|---|---|
| memory hierarchy (registers/L1 vs main memory) | entanglement-tower levels: low-χ monosemantic core (addressable, cov95-high) vs high-χ entangled remainder | `mps_tower_*`, serve-tower cov95-saturation-vs-capability-cliff |
| fidelity modes: full / preserve-hybrid / decompiled | exact host / verbatim-pin ~6–12% of atoms + forge the rest / run only the recompiled op-graph | `preserve_hybrid_tiny.py`, sae-forge `NativeModel` |
| ISA vs model-specific microcode | the op-catalog (idioms) is invariant across architectures **and** languages; the **sink/plumbing policy is model-specific** | the 4-model + multilingual results |
| associative memory bank | MLP neurons as content-addressable key→value stores | work-item 2 (the MLP gap) |
| the recompiler / JIT | sae-forge projects the kept ops into a runnable module | reconstruction-coverage metric above |

**What it adds beyond the metric (the real new lever): an interactive debugger.** Step layer-by-layer;
breakpoint when a named idiom fires; inspect the low-χ "registers" (SAE latents); **ablate / preserve / swap a
single op and watch reconstruction-KL move live**. That turns reconstruction-coverage from a number into a
tool for *localizing* where decompilation fails — i.e. it operationalizes milestones 1+4.

**The hard constraint the frame must not hide.** The entangled core is **preserve-or-pay**, not "approximate
main memory": by the tower's no-go, fidelity modes move you *along* the interpretability↔capability frontier,
never off it. An interpreter that "approximates the core cheaply" is just choosing a point on that frontier
(and paying the capability cliff) — the ceiling is the forge tax, restated.

Demarcation: adopt the execution layer, the memory-hierarchy mapping, the fidelity modes, and the ISA-vs-
microcode framing; **drop the von-Neumann mechanics** (no per-layer instruction fetch) in favor of
dataflow-circuit-per-pass + VM-at-the-loop. `COMPOSE` / `PRESERVE` / `TOWER_TRUNCATE` are not opcodes the model
runs — they are *extracted descriptions* (the DAG) or *execution modes* (knobs on the interpreter).

## Milestones (each a PR, gated on the prior)

1. **Recompile-KL harness, built as the interpreter** (§Execution model) — **DONE (v1, GPT-2): `scripts/disassembly/residual_vm.py`** (see First result). v1 recompiles by *keeping ops at full fidelity and mean-ablating the complement*; v2 = feature-basis recompilation via sae-forge `NativeModel` (the ceiling test, milestone 4).
2. **Composition-DAG extractor** — weight-space edge scorer + path-patch gate; auto-recover induction + IOI;
   report new sub-DAGs. (Generalizes `path_patch_induction.py` / `composition_graph.py`.) **DONE (GPT-2):
   `scripts/disassembly/composition_dag.py`** — static composition predicts dynamic writer specificity
   (ρ=+0.37); the canonical induction K-chain and IOI Q-chain are auto-recovered AND live, imposters/random
   rejected (0% FP); 22 new live edges surfaced (see Composition-DAG section).
3. **MLP ops** — neuron key→value catalog + named MLP idioms; add to the DAG + the recompile. **DONE (GPT-2):
   `scripts/disassembly/mlp_ops.py`** — the recompile now charges for MLPs (M1/bridge kept heads only); MLPs are
   load-bearing, concentrated in L0 (the detokenizer); head↔MLP composition edges are weight-legible (see MLP
   ops section).
4. **The ceiling test** — reconstruction-coverage plateau vs the tower's entangled core, same host; the
   unifying claim stands or falls. **First result DONE (v2, tiny GPT): `scripts/cov95_forge_tax/ceiling_test.py`** — content/factorability axes decouple; the capability plateau is GPU-scale-gated (see Ceiling test section).
5. **Cross-model** — repeat the ceiling on Gemma-2 / Llama-3 / Qwen-2.5 (idea i) to test whether the
   decompilable fraction is architecture-invariant like the mechanisms are.

## First result (milestone 1, GPT-2)

`residual_vm.py` on GPT-2 (Shakespeare; floor = KL(host ‖ all-144-heads-mean-ablated) = 1.92): keep a head-set
at full fidelity, mean-ablate the complement, sweep the budget B by marginal ablation importance vs a random
control + the named induction circuit.

- **Attention is distributed / redundant.** The coverage curve rises *gradually* — **128 of 144 heads are
  needed for 90% coverage**; no small set reconstructs the forward pass. The single most-important head in
  isolation is even net-negative (keep 1, ablate 143 → slightly below the all-ablated floor): heads interact.
  (Same program-wide redundancy seen in the circuit work.)
- **But the named catalog is coverage-efficient.** Top-B beats random-B at every budget except B=1 (largest
  gap mid-range, Δ≈+0.31 at B=24–32), and the **5-head induction circuit (prev-token 4.11 + induction
  5.0/5.5/6.9/7.11) reconstructs +0.164 coverage vs +0.032 for a random 5-head set — ~5×**. The disassembly's
  named ops are disproportionately load-bearing (importance ranks: 7.11 #2, 4.11 #8). The op-catalog buys real
  reconstruction.
- **Scope (honest).** v1 keeps the kept heads at *full fidelity* (exact weights), so coverage → 1 as B → all
  *by construction* — it measures **which/how-many ops matter** (op-selection coverage), **not** the
  entangled-core ceiling. The plateau-below-1 prediction (forge tax as decompilation ceiling) needs the
  **feature-basis recompilation** (sae-forge `NativeModel`, milestone 4), where kept ops must be expressed in a
  clean basis and composition bottlenecks. Milestone 1 delivers the interpreter + metric + the op-selection
  result; the ceiling test is the next build. `runs/disassembly/residual_vm_gpt2_summary.json`.

## Composition-DAG extractor (milestone 2) — first result (GPT-2)

`composition_dag.py` unifies the two precursors into one extractor that reads the **call graph**, not one
idiom. `composition_graph.py` gave the static adjacency (Elhage Q/K-composition on raw weights, mean-write
removed) but only validated the single prev-token→induction K-edge; `path_patch_induction.py` gave the dynamic
gate but measured an *induction-specific* collapse, so it could only confirm induction. M2 generalizes both:
score the full K/Q composition DAG over all causal head pairs, then gate the strongest edges with an
**idiom-agnostic** dynamic metric — the mean total-variation change in the reader's attention pattern when the
writer is removed from that port (**ΔTV**), defined for *any* reader. GPT-2, Shakespeare, weights + two forward
passes; 170 gated edges (105 K / 65 Q); attention recompute is exact (max|Δ| = 9.85e-7 vs the model).

The one methodological move that makes this work: raw ΔTV **grows with reader depth/magnitude**, so a global
random null is confounded (early-layer write-hubs dominate). The fix is a **reader-matched null** — for each
reader, compare the real writer against *random causal writers into the same reader head* (the path-patching
null that isolates writer *specificity* from reader depth). An edge is "live" if its ΔTV beats its reader's
matched 2σ null; **specificity = ΔTV − matched null**.

- **Static composition predicts dynamic liveness across the whole graph.** The headline (cleaner, depth-
  unconfounded) metric is **Spearman(static, reader-matched specificity) = +0.37**; the raw
  Spearman(static, ΔTV) = +0.52 is higher but inflated by the depth co-scaling both quantities share, so the
  specificity correlation is the one to trust. Either way the weight-space score is a graph-wide predictor of
  which writes actually shape which reads — the broad version of `path_patch_induction`'s induction-only ρ,
  not a single-idiom result.
- **The induction K-chain is auto-recovered AND live.** Static prev-token→induction K-composition 0.069 vs
  causal baseline 0.042 (1.6×) vs random 0.039. Dynamically, the **canonical prev-token head 4.11 → inductors
  is 4/5 live** (4.11→5.5/5.0/7.11/6.9 clear their matched nulls; 4.11→5.1 marginal), the strong-edge median
  Δinduction is +0.015, and the **top edge collapses 56% of induction attention** under key-path patching (the
  original strong readout, retained for induction edges).
- **The IOI Q-chain is auto-recovered.** Static S-inhibition→name-mover Q-composition 0.065 vs causal 0.042
  (1.5×); the recovered chain is duplicate-token (3.0) → S-inhibition (8.10/8.3/10.0) → name-mover
  (9.9/10.0/11.2). Dynamically **5/11 S-inhib→name-mover Q-edges are live** — the real S-inhibition heads
  8.10/8.3 reshape the name-movers' query attention; the spurious 6.7 (a Q-composition false-positive) does
  **not** (negative specificity).
- **The gate is selective — it rejects imposters.** Across the named cross-product, **43% of edges are live vs
  0% random false-positives**. The selectivity is the point: the *non-canonical* high-prev-token writers
  (2.2/3.2/3.7, which have high prev-token attention but aren't the prev-token head) → inductors are mostly
  **dead**, and the spurious S-inhib head is dead. The extractor keeps the real sub-circuit and discards the
  cross-product noise.
- **22 new live edges surfaced** above their reader-matched 2σ nulls, not in induction/IOI — dominated by
  early-layer **write-hubs** feeding many readers (consistent with positional / duplicate-token hubs). The
  highest-specificity candidates:

  | port | edge | ΔTV | reader null | specificity |
  |---|---|---|---|---|
  | K | 0.9→2.9 | 0.444 | 0.035 | **+0.410** |
  | K | 0.9→2.5 | 0.449 | 0.044 | **+0.404** |
  | K | 1.8→10.9 | 0.293 | 0.007 | **+0.287** |
  | K | 1.8→9.3 | 0.333 | 0.088 | **+0.245** |
  | Q | 0.9→1.3 | 0.259 | 0.018 | **+0.240** |
  | K | 1.8→3.2 | 0.362 | 0.133 | **+0.230** |
  | Q | 1.10→2.11 | 0.270 | 0.055 | **+0.215** |
  | K | 1.8→10.5 | 0.198 | 0.021 | **+0.177** |

  Two heads recur as hubs: **0.9** (a layer-0 writer dominating several layer-1/2 readers) and **1.8** (a
  layer-1 writer reaching *long-range* into late-layer keys, 9.3/10.5/10.9). These are *candidate* sub-DAGs,
  **not** validated circuits — behavioral labeling (and targeted single-edge path-patching, as for induction)
  is the obvious follow-up; the full ranked list is in the summary JSON's `novel_live_edges`.

**Feeds the recompile harness (milestones 1/4) — now tested, see the next section.** The live edges this
extractor confirms are exactly the *keepable ops* the ResidualVM reconstruction-coverage interpreter
(`residual_vm.py`) should retain: M1 selects heads by marginal ablation importance, but the DAG supplies the
**structured op-set** (which writer→reader wires carry the computation), so a DAG-guided keep-set is the
natural upgrade to M1's flat head-budget — and the recompile-KL then verifies the extracted sub-DAG *executes*.
The M1↔M2 bridge below confirms this. Completing the DAG with V-composition + MLP nodes (milestone 3) is what
makes that op-set whole.

**Scope (honest).** (a) The dynamic gate runs on natural text, so the IOI Q-edges are confirmed by generic
attention reshaping (ΔTV), not the IOI-task logit-difference — that task-specific causal validation already
lives in `ioi_causal.py`; M2's contribution is the *unified static→dynamic* recovery. (b) ΔTV measures whether
an edge *reshapes the reader's attention pattern* (Q/K composition); V-composition (writing values without
moving attention) needs a different readout, not done here. (c) MLP nodes are absent — they are milestone 3.
**Compute:** GPT-2, CPU-feasible — weights + two forward passes over the corpus (one for behavioural labels,
one for the path-patch gate); the full run is **~70 s wall-clock and ~5 GB RAM on CPU** (no GPU), so it scales
to any HF model the box can hold a forward pass of. `runs/disassembly/composition_dag_summary.json` (re-run the
script to regenerate the figure).

### Validating the new write-hub edges (follow-up to M2)

`validate_new_edges.py` takes the 22 new live edges to the next rung: a **targeted single-edge path-patch with
a behavioural readout** (the induction-style strong test, generalized). For each edge A→B it surgically removes
A's output from B's port, recomputes B's attention, and measures the collapse of B's *named* components —
{prev-token, duplicate, induction, sink} — against a **reader-matched random-writer null**. A pattern collapse
beyond the null *names* the edge's function; ΔTV with no named collapse is real-but-unlabeled shaping.

- **The write-hubs are early SINK heads broadcasting a positional signal.** All three hubs (0.11, 0.9, 1.8) and
  the minor ones (1.3, 1.7, 1.9) self-label **sink** (their own attention parks on position-0). **13/22 edges
  resolve to a named function — 9 prev-token, 3 sink, 1 duplicate.** Removing these hubs from a downstream key
  collapses that head's **prev-token attention**: e.g. `0.9→2.5` −56%, `1.8→9.3 / 1.8→10.9` −45%, and — the
  headline — `1.3→4.11` −11% and `1.8→4.11` −8% into the *canonical prev-token head itself*.

  | edge | writer→reader | pattern | rel-collapse (beats null) |
  |---|---|---|---|
  | K | 0.9→2.5 | prev-token | −56% |
  | K | 1.8→9.3 | prev-token | −45% |
  | K | 1.8→10.9 | prev-token | −45% |
  | Q | 0.9→1.3 | sink | −42% |
  | Q | 0.9→1.10 | prev-token | −23% |
  | Q | 0.9→1.2 | prev-token | −23% |
  | K | 1.8→11.8 | prev-token | −14% |
  | K | **1.3→4.11** | **prev-token** | **−11%** |
  | K | 0.11→1.1 | prev-token | −10% |
  | K | **1.8→4.11** | **prev-token** | **−8%** |
- **The prev-token mechanism is not self-contained in 4.11.** It *reads a positional signal piped in from early
  sink heads*; remove that input and 4.11's previous-token addressing degrades. This edge-resolves the
  disassembly's **position/structure register** and reframes the **sink**: a sink head is a no-op in *where it
  reads* (parks on pos-0) but its **OV-write is a load-bearing positional broadcast** — attention-pattern and
  write-content are decoupled. *Hypothesis* (consistent with the sink-ablation result — GPT-2 is the only
  family member that depends on its sink, position-independently = the absolute-positional-embedding signature):
  these hubs propagate GPT-2's learned absolute positions; the exact encoding pathway is left open.
- **Honest scope.** (a) The reader-matched null *includes other early/sink heads* that also carry positional
  signal, so the test is **conservative** — several of the 9 "unlabeled" edges still collapse prev-token (e.g.
  `0.9→2.9` −55%) but don't beat that positional null (those readers' prev-token mass is fragile to *any* key
  perturbation), so they are real shaping not attributable to one writer. (b) The readout is attention-pattern
  collapse (like M2's ΔTV), evidence of a positional-broadcast role, not a task-level loss metric.

  *Next* (sharpen the mechanism): (a) **task-level readout** — re-do the edge patch as a forward-pass
  intervention and read next-token KL / induction-NLL, turning "reshapes the pattern" into "changes the
  output"; (b) **the positional-embedding test** — re-run the patch with `wpe` zeroed/randomized: if these
  hubs propagate GPT-2's *absolute* positions, their prev-token collapse should vanish without `wpe`; (c)
  **cross-model prediction** — RoPE models (Gemma-2 / Llama-3 / Qwen-2.5) do **not** depend on their sink
  (sink-ablation result), so they should show **no** such absolute-positional-broadcast circuit — the
  prev-token signal is carried by RoPE in the QK directly, not piped from a sink head. A clean falsifier of the
  "GPT-2 absolute-positions" hypothesis. `runs/disassembly/validate_new_edges_summary.json` (~40 s on CPU;
  re-run to regenerate the figure).

### Cross-model: the positional broadcast is GPT-2-specific (forward-pointer (c), done)

`scripts/gemma/cross_model_positional.py` runs the cross-model test. A *key-only* causal path-patch that
respects each model's RoPE is non-trivial to do faithfully across architectures, so the test goes to the
**representational** question the broadcast hypothesis really turns on: **is the prev-token head's position
carried in the key *content* (absolute, must be written into the residual) or in the QK *rotation* (RoPE,
applied at attention time)?** For each model it finds the top prev-token head, captures its **pre-rotation**
keys over the corpus, and decomposes the key variance into the fraction explained by absolute **position** vs by
**token** identity. The prediction: GPT-2 position-dominated (its keys must encode *where* — exactly what the
sink heads broadcast), the RoPE models token-dominated (keys encode *what*; position is the rotation).

*Method:* a between-group variance decomposition on the head's key vectors. `position_fraction` = the
between-position-index variance (variance of the mean key at each absolute position) over the total key-
covariance trace; `token_fraction` = the same over token identity (mean key per frequent token). Both are
fractions of the same total, so `pos/tok` is a clean within-head ratio comparable across architectures.

| model | pos. encoding | prev-token head | key var: **position** | key var: token | pos/tok |
|---|---|---|---|---|---|
| **GPT-2** | absolute (`wpe`) | 4.11 | **59%** | 18% | **3.3** |
| Gemma-2-2B | RoPE | 21.7 | 7% | 23% | 0.30 |
| Llama-3.2-1B | RoPE | 0.2 | 2% | 64% | 0.04 |
| Qwen-2.5-1.5B | RoPE | 13.4 | 9% | 25% | 0.35 |

**CONFIRMED — a clean 1-vs-3 split (~10×).** Only GPT-2's prev-token key is position-dominated; every RoPE
model's is token-dominated. This is the cross-model *explanation* that ties the positional thread together: GPT-2
encodes absolute position as key content → its prev-token head must *read that content* → it depends on the early
sink heads that **broadcast** it (the `validate_new_edges` collapse) → and GPT-2 is the only family member that
**depends on its sink** (the sink-ablation result, position-independently = the absolute-positions signature). The
RoPE models need none of this — position rides in the rotation — so they have no sink dependence and no
positional-broadcast circuit. The figure (`cross_model_positional.png`, regenerable) shows the bars flip:
position > token only for GPT-2. Note **Llama-3.2-1B's prev-token head is in *layer 0*** (head 0.2) and is the
most token-pure of all (pos/tok 0.04) — it reads the raw token embedding directly and leans entirely on RoPE for
position, the cleanest case of the RoPE pattern.

**Scope.** This is the *representational* confirmation (the key **is** position-encoded only in GPT-2),
corroborating the *causal* GPT-2 result; a faithful key-only causal path-patch across RoPE models
(forward-pointer (a)) is the heavier next step. *Next:* re-run on an oracle-supervised host (#19/#20) — does
training a more legible model shift the prev-token key's position-vs-token content, i.e. does supervision touch
the positional machinery or only the feature substrate? ~33 s for all four models.
`runs/gemma/cross_model_positional_summary.json`.

## Circuit-structured keep-set selection (M1↔M2 bridge) — first result (GPT-2)

`dag_recompile.py` closes the loop between the two milestones: it feeds the M2-extracted live sub-DAG into M1's
reconstruction-coverage harness (same mean-ablation metric, `coverage = 1 − KL(host‖keep)/KL(host‖all-ablated)`)
and asks whether the **weight-cheap** DAG (weights + 2 forward passes) picks the keep-set as well as M1's
**expensive** marginal-ablation importance ranking (one forward pass per head). GPT-2, Shakespeare, floor KL
1.82; ~95 s on CPU (most of it the 144-head importance ranking the DAG sidesteps).

- **The path-patch-confirmed circuit IS the recompile keep-set — and it beats the greedy importance set at equal
  size.** The 12-head induction+IOI live sub-DAG reconstructs **+0.333** coverage vs **+0.230** for the 12
  individually-most-important heads (greedy top-B) and **+0.038** for random-12 — i.e. **145% of the
  greedy-optimal** at equal budget, +0.295 over random. It *beats* greedy importance because marginal ablation
  is myopic (ranks heads by their *individual* effect) while the DAG selects a *coordinated interacting circuit*
  — exactly the regime M1 flagged ("heads interact; the single most-important head in isolation is net-negative").
  So the auto-extracted circuit is a coverage-efficient keep-set **without** the per-head ablation sweep.
- **But raw connectivity is NOT a generic output-importance proxy** — the honest dissociation. Ranking *all*
  heads by DAG-connectivity (summed incident live-edge specificity) only weakly tracks marginal importance:
  Spearman = +0.24 over all 144 heads but **+0.02 among the heads the DAG actually gated**. And the larger
  keep-set that adds the new write-hubs (`dag_all_live`, 30 heads) reaches only **67%** of the greedy-optimal
  coverage (+0.361 vs +0.541). Reason: **ΔTV measures attention-*reshaping*, not output-*importance*** — the new
  early-layer write-hubs (0.9→2.x, 1.8→{9.3,10.5,10.9}) strongly shape downstream attention yet are
  output-redundant (mean-ablating them barely moves next-token KL). *Mechanistic hypothesis:* this is the
  program-wide redundancy seen throughout the stack (substrate/core redundancy in the tower; the redundant
  prev-token *population* feeding a bottleneck inductor in rung-3) — an early write-hub broadcasts positional /
  duplicate-token signal along *many parallel paths*, so removing any one (or even the hub's whole output, mean-
  ablated) leaves the downstream readers able to recover it elsewhere; high attention-influence, low *marginal*
  output-importance.

  As an *ordering*, DAG-connectivity still beats random at every budget (it front-loads the circuit heads) but
  lags the marginal-importance ordering at small budgets — consistent with "connectivity finds the circuit, not
  the importance rank":

  | budget B | top-importance | DAG-connectivity | random |
  |---|---|---|---|
  | 4 | +0.153 | +0.054 | +0.025 |
  | 12 | +0.230 | +0.186 | +0.082 |
  | 24 | +0.391 | +0.416 | +0.170 |
  | 48 | +0.610 | +0.636 | +0.408 |
  | 64 | +0.715 | +0.690 | +0.426 |

**Takeaway.** The bridge confirms the program's central use of the extractor: a *path-patch-confirmed* sub-DAG
is the structured op-set the recompiler should keep (rivals/beats greedy importance, ≫ random, no ablation
sweep) — but the gate's ΔTV is a *circuit-liveness* signal, not a drop-in importance score, so the keep-set
must come from the confirmed circuit, not from thresholding raw connectivity. This is the M1→M4 hand-off: the
DAG keep-set is what the feature-basis recompilation (milestone 4) should express in a clean basis.

*Next:* (a) feed *validated* new edges (single-edge path-patch of the write-hub candidates from M2) into this
keep-set so the recompiler grows beyond the two textbook circuits; (b) make the gate jointly attention-liveness
*and* output-importance (e.g. an attribution / logit-effect term alongside ΔTV) so connectivity becomes a true
keep-set score; (c) re-run on an oracle-supervised host (#19/#20) — a more legible host should yield a cleaner
live DAG and a higher-coverage keep-set. The script takes `--dag-summary` so any of these DAGs (other corpora,
supervised models) drops straight in. `runs/disassembly/dag_recompile_summary.json` (re-run to regenerate the
figure).

## MLP ops in the DAG + the recompile (milestone 3) — first result (GPT-2)

`mlp_ops.py` adds the **COMPUTE** instruction class. M1 and the bridge kept/ablated attention heads only — MLPs
ran at full fidelity, so the coverage metric never charged for them — yet MLPs carry real computation
(greater-than is MLP-dominated; `mlp_catalog.py` read the neuron key→value vocabulary). M3 extends the
mean-ablation harness to keep/ablate **MLP layers** as well as heads (floor = all heads *and* all MLPs ablated),
adds **head↔MLP composition edges** to the DAG, and names the load-bearing MLPs. GPT-2, Shakespeare, floor KL
3.39.

- **MLPs are load-bearing — and the load is concentrated in L0.** Removing *all* MLPs (heads intact) collapses
  coverage to **−0.019** (≈ floor): attention alone cannot reconstruct the forward pass. The single most
  important op in the whole recompile is the **layer-0 MLP** (marginal importance **+0.772**; next are L11 +0.08,
  L1 +0.07) — GPT-2's **detokenizer**, the same `e_ext = e + MLP0(·)` enrichment the QK/copy disassembly already
  reads through. Its top neurons read sentence-boundary punctuation (`. ; ?`) and write structural/line-start
  tokens (newline, `I`, `First`) — detokenization / boundary formatting.
- **A few MLPs reconstruct most (attention intact).** Sweeping the MLP budget with all heads kept, the
  top-importance MLPs dominate a random-MLP control at every budget (L0 first):

  | MLPs kept | top-importance | random |
  |---|---|---|
  | 1 | +0.118 | −0.002 |
  | 2 | +0.257 | −0.004 |
  | 4 | +0.436 | +0.313 |
  | 8 | +0.819 | +0.595 |

  A **combined sparse op-set** of just the 12 M2 circuit heads + the top-4 MLP layers reaches **+0.286** coverage
  — a tiny MOVE+COMPUTE program (16 ops of 156).
- **The DAG gains MLP nodes.** Head→MLP read edges (`‖OV_a · W_in^L‖`, mean-write removed) and MLP→head write
  edges (`‖W_out^L · W_{Q/K}^B‖`) are weight-legible; top edges e.g. `2.1→L2`, `11.0→L11` (read) and `L1→2.2`,
  `L0→4.11` (write). So the call graph now has typed head↔MLP edges, not just head↔head.

**Scope (honest).** (a) Mean-ablating *all* MLPs is severe (L0 dominates), so "attention-only / MLP-only" are
**necessity** statements, **not** a clean attention-vs-MLP credit split — attention's reconstruction value is
the MLP-intact bridge (#23), where circuit heads reconstruct with MLPs on. (b) Static head↔MLP composition does
**not** rank MLP recompile-importance (Spearman = **−0.43**): the most important MLPs are *early* (L0) and have
the *fewest* incoming head→MLP edges (a depth confound), so the DAG edges give **structure**, importance comes
from the recompile — the same lesson as the bridge's "ΔTV ≠ KL-importance." (c) The full per-neuron catalog +
low-rank/named-idiom analysis lives in `mlp_catalog.py`; M3 reuses its read→write naming for the load-bearing
layers only. (d) V-composition and MLP→MLP dynamic gating remain future work.
`runs/disassembly/mlp_ops_summary.json` (re-run to regenerate the figure).

## Ceiling test (milestone 4) — first result (v2, tiny GPT)

`ceiling_test.py` recompiles the tiny GPT by forcing computation **through the SAE feature basis** (sae-forge
`NativeModel`, `native_in_basis`), sweeping SAE width 1–8×, measuring three things at once: forged-model
output faithfulness (KL vs host; unigram floor 2.17), feature-content retention (mAUC), and monosemantic
factorization (cov95).

| width | forged-model KL | cov95 host→forged | mAUC host→forged |
|---|---|---|---|
| 1× | 43.3 | 0.60→0.00 | 0.92→0.67 |
| 2× | 5.11 | 0.64→0.00 | 0.92→0.78 |
| 4× | 5.08 | 0.64→0.12 | 0.93→0.84 |
| 8× | 5.08 | 0.60→0.16 | 0.92→0.85 |

The single-ceiling prediction is **refined into two axes — and the capability axis is scale-confounded**:
- **Feature content reconstructs; monosemantic factorization does not.** mAUC retention ~86% (rises with
  width) vs cov95 retention ~11% (collapses) — the forge tax in the recompilation frame: the basis carries
  *what* the model represents, not a *monosemantic factorization* of it.
- **The forged model's output is globally broken at every width** (KL 5–43 ≫ unigram 2.17 → negative capability
  coverage; 1× catastrophic, 2–8× plateau at KL≈5.08). This is the known tiny-whole-model-forge artifact, **not**
  the entangled-core ceiling — so this substrate *cannot* isolate the capability plateau-vs-core (the doc's
  central prediction). Settling that needs a **high-quality GPU-scale forge** (SAELens + polygram compression)
  the 8 GB box can't run.

So milestone 4 **builds the harness and settles the content/factorability axis** (decoupled, robust across
widths) and **identifies the capability-ceiling test as GPU-scale-gated** — an honest partial result: the
unifying "one ceiling" claim is wrong as stated (≥2 axes), and the clean capability test is deferred to better
forge hardware. (The tower's ~24% irreducible core is the target a GPU-scale capability curve would be
compared against.) `ceiling_test.py`, `runs/cov95_forge_tax/ceiling_test_summary.json`.

## Reachability — host-width × oracle-supervision (first result)

The ceiling test asks what's *achievable*; this asks what's *reachable by training*. The entanglement-tower
retrain no-go ("you can't train interpretability in without losing capability") used a *reconstruction*
bottleneck — the mAUC axis that already survives forging, not cov95. `host_width_sweep.py` retries on the
right axis: train tiny GPTs from scratch across host widths, with/without an **auxiliary oracle-feature-
recovery loss** (a linear head from the residual must predict the exact per-token oracle labels), and measure
native cov95 + capability.

| host width | params | LM-loss unsup→sup | cov95 unsup→sup | Δcov95 |
|---|---|---|---|---|
| 32 | 1.7M | 6.54→6.42 | 0.48→0.62 | +0.14 |
| 64 | 3.4M | 6.21→6.16 | 0.62→0.79 | +0.17 |
| 128 | 7.2M | 5.99→5.99 | 0.69→0.76 | +0.07 |
| 256 † | 16.0M | 6.04→6.05 | 0.45→0.69 | +0.24 |

† w256 is **undertrained** (16M params on 107k tokens; its unsup LM loss 6.04 is *worse* than w128's 5.99) —
its low unsupervised cov95 is a compute-budget artifact, not scarcity counter-evidence.

- **Reachability — CONFIRMED.** Oracle-supervision lifts native cov95 at *every* width (+0.07…+0.24, mean
  **+0.155**) at **zero/negative capability cost** (mean **−0.037 nats** — it slightly *helps* LM loss). So
  interpretable, equally-capable solutions are not only existent (superposition is linear compression —
  decompressing preserves the function) but **reachable by training pressure**, with the manufactured-oracle
  substrates serving as the *training signal*, not just the grader. This is the constructive counter to the
  tower retrain no-go: the no-go used the wrong (reconstruction) axis; supervising the oracle-feature axis
  lifts monosemanticity for free.
- **Scarcity — partially supported.** Unsupervised cov95 *rises* with host width in the well-trained regime
  (0.48→0.62→0.69 for 32→64→128), consistent with superposition being capacity-driven; it drops at 256 (0.45),
  but that host is **undertrained** (16M params on 107k tokens — its LM loss is also worse than w128), a budget
  artifact, not counter-evidence. So the forge tax is *partly* a capacity-scarcity artifact — relieved by width
  up to the training budget, and relieved more cheaply by supervision.

**Feeds milestone 1:** supervision yields cleaner, more monosemantic low-χ residuals — which are exactly the
"registers" the ResidualVM interpreter reads, so a supervised host should decompile further (higher
reconstruction-coverage at lower op-budget) than an unsupervised one. That's the direct hand-off to the
recompile harness.

### What kind of pressure lifts cov95? (aux-mode comparison)
Is the lift real monosemanticity or just linear recoverability? `monosemantic_aux.py` compares aux modes at a
fixed well-trained width (128), training from scratch, **over 3 seeds** (per-seed model init + batch order +
eval-SAE init):

| aux mode | LM-loss | cov95 (mean ± std, 3 seeds) |
|---|---|---|
| none | 6.02 | 0.68 ± 0.04 |
| linear (recoverability) | 6.01 | **0.76 ± 0.00** |
| decorr (orthogonal read-directions) | 6.00 | 0.76 ± 0.00 |
| dedicated (one-neuron-per-feature) | 6.00 | 0.69 ± 0.00 |
| sparsedict (full in-loop aligned SAE) | 6.16 | **0.48 ± 0.07** |

Paired across seeds: **`linear > none` in 3/3 (+0.080 ± 0.043); `sparsedict < none` in 3/3 (−0.195 ± 0.086);
`linear > sparsedict` in 3/3 (+0.276)** — every signal survives the noise.

**The simple linear-recoverability proxy is robustly the *best*** — and *every* "more direct" monosemanticity
objective fails to beat it: orthogonalizing the probe's read-directions is **inert** (decorr ≡ linear, identical
every seed), dedicated raw neurons sit *below* `none`'s mean, and the **full sparse-dictionary-in-the-loop** (a
jointly-trained TopK SAE on the residual, reconstruction + sparsity, first F latents aligned to the oracle) is
robustly the **worst** — cov95 0.48 (below `none` in 3/3) *and* the only one with a real capability cost
(+0.16 nats). The heavier the direct pressure, the worse it gets. So the cov95 lift comes from making features
**linearly prominent** (so the downstream SAE can isolate them), **not** from forcing axis-alignment /
sparse-coding in the host residual: the SAE does the factorization; an in-loop dictionary's reconstruction
pressure distorts the representation in a way that doesn't transfer to the fresh eval SAE. This answers "is it
just recoverability?" — yes, and recoverability *is* the effective lever for SAE-measured cov95. **Bonus:**
supervision also makes cov95 **variance-free** (linear pins exactly 22/29 features over threshold every seed,
vs `none`'s noisy 0.62–0.72) — it lifts cov95 *and* makes it reproducible. `monosemantic_aux.py`,
`runs/cov95_forge_tax/monosemantic_aux_summary.json`.

### Is the lift real monosemanticity, or the eval-SAE finding what we planted? (non-SAE cross-check)
cov95 fits a TopK SAE — so does the linear aux just make features recoverable in a way a linear-ish SAE
prefers (circular)? `legibility_crosscheck.py` scores the none→linear lift in **three bases** (same
symmetric-AUC scorer, only one involving an SAE), width 128, 3 seeds:

| basis | none → linear | Δ (mean ± std) | up in |
|---|---|---|---|
| sae (fitted TopK dictionary) | 0.68 → 0.76 | +0.08 ± 0.04 | 3/3 |
| neuron (raw residual dims, no fit) | 0.68 → 0.76 | +0.08 ± 0.07 | 2/3 |
| pca (rotated basis, fit *without* labels) | 0.36 → 0.69 | **+0.33 ± 0.09** | 3/3 |

**Corroborated — the lift is genuine monosemanticity, not circular.** Both SAE-free metrics rise too: features
become single-detector-isolable by *raw neurons* (+0.08) and, most strongly, by *PCA components* (+0.33). So
supervision doesn't merely make features recoverable in a way the eval-SAE prefers — it makes them
axis-isolable in the natural and rotated bases as well. (For the unsupervised model, neuron-cov95 *equals*
sae-cov95 exactly — the fitted SAE adds nothing over raw neurons here, so it can't be inflating the result.)
**Mechanism:** the biggest lift is in PCA — unsupervised pca-cov95 is only 0.36 (the oracle features sit *off*
the high-variance axes), so supervision **pushes the features into the residual's principal (high-variance)
directions**, where *any* axis-aligned probe (neuron, PCA, or SAE) isolates them. That's a basis-independent
signature of real monosemanticity, and it explains *how* recoverability becomes monosemanticity: prominence =
high-variance placement. `legibility_crosscheck.py`, `runs/cov95_forge_tax/legibility_crosscheck_summary.json`.

Caveats + scope: the aux loss pressures **linear recoverability** of the oracle (which, per the aux-mode
comparison above, is the *effective* lever — direct decorr/dedicated/sparsedict objectives don't beat it, and
the in-loop sparse dict actively hurts). The scarcity trend is only cleanly visible
in the **well-trained regime** (≤w128 here); confirming it at w256+ needs more tokens/steps (compute scaling),
not more width. Tiny hosts, short training. **Status of the open levers:** the *training-pressure* axis is
settled (decorr/dedicated/sparse-dict all tested, none beats linear-recoverability; multi-seed-confirmed) and
the *measurement* worry is resolved (non-SAE bases — raw-neuron + PCA — corroborate the lift, see cross-check
above → genuine monosemanticity). **Remaining follow-ups:** (1) multi-seed +
adequately-trained wide hosts to clean the scarcity curve; (3) richer oracles (spaCy POS/NER) + curriculum
annealing of the aux weight; (4) polygram geometry penalties. But the direction is clear and the cost is
~zero, so the reachability lever is real. `host_width_sweep.py`,
`runs/cov95_forge_tax/host_width_sweep_summary.json`.

## Boundaries / risks

- **Recompile faithfulness is OOD-sensitive** (partial reconstructions are low-norm inputs to `lm_head`);
  use norm-preserving mean-ablation + a random-op-budget control (the lesson from the entanglement-tower
  ablations) so the curve measures structure, not norm artifacts.
- **Path-patching cost** scales with edges; the weight-space edge scorer must prune before patching.
- **The operand basis is the bottleneck** (the P2 lesson recurs): a wrong basis caps coverage regardless of
  the algorithm. Expect to co-design the basis (U_C / polygram) with the extractor.
- **Single-token-prediction framing** — a faithful decompilation of next-token logits is not a proof of the
  model's *mechanism* off-distribution; scope claims to the measured distribution.

## Dependencies (cross-repo)

`sae-forge` (recompile/verify — the metric), `polygram` (operand-dictionary geometry — the basis), `n-orca`
(emit the circuit DAG as a typed graph), `larql` (decompile→query→recompile as a queryable index). The
disassembly toolkit here (`scripts/disassembly/`, `scripts/gemma/`) is the front-end that produces the ops.
