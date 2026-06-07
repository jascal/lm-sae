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
   report new sub-DAGs. (Generalizes `path_patch_induction.py` / `composition_graph.py`.)
3. **MLP ops** — neuron key→value catalog + named MLP idioms; add to the DAG + the recompile.
4. **The ceiling test** — reconstruction-coverage plateau vs the tower's entangled core, same host; the
   unifying claim stands or falls.
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
