# A Field Guide to Attention

### A cross-architecture, causally-validated catalog of transformer operators and circuits

*`lm-sae` — disassembling a transformer's attention into a catalogued instruction set.*

A frozen language model computes in a basis we didn't choose. **`lm-sae` reads that computation as a small,
reused *instruction set*** — names the operators (single attention heads / MLPs), collects the circuits
(compositions of operators), validates them causally, and surveys both **exhaustively across model families** —
then asks the harder question: can we **decompile** it, i.e. reconstruct the *computation* faithfully, not just
label the components?

The throughline: **a language model is legible in the right basis even where it is *not* legible as single SAE
features.** Most of attention is positional/structural plumbing; the content-carrying minority is a largely
*named*, causally load-bearing, corpus-robust op-catalog — and the catalog ports across architectures with a few
sharp, architecture-specific exceptions.

**Mode — natural history, borrowed from biology.** A trained model is something *grown*, not designed: the job is
to **catalog** the operators it developed, **taxonomize** them (operators → circuits; class → instance → variant),
and derive **generalizable** principles by comparing across model *species* (the absolute-position GPT-2 family vs
RoPE vs a non-attention SSM). "Catalog," not "atlas," is deliberate — like a field biologist's catalog it is an
honest, growing, causally-tested record of what we have observed, **not** a claim of completeness. The advantage
over wet biology: we can also **breed synthetic specimens and intervene at high speed** — train hosts from scratch,
swap the sequence-mixer to a non-attention recurrence, manufacture ground-truth-feature substrates (the
[sister track](docs/FORGE_TAX_TRACK.md)), and ablate / path-patch any component in milliseconds.

> **Amateur, exploratory home-science — read it that way.** Everything here is *a working catalog*, not *the*
> definitive one: findings are **descriptive and provisional** — measurements to be checked and re-run, not settled
> results. Where a mechanism is unknown we still **list the specimen** with what we measured (e.g. MLP0) and say so.

> A sister investigation — the **exact-lexical oracle** and the **cov95 forge tax** (what an SAE *feature basis*
> destroys, and what you must preserve) — has moved to [`docs/FORGE_TAX_TRACK.md`](docs/FORGE_TAX_TRACK.md). It is
> the "what the SAE misses" motivation for this disassembly work.

> **Status.** A CPU/single-GPU research MVP. GPT-2 small/medium/large for the deep catalog; Gemma-2-2B /
> Llama-3.2-1B / Qwen-2.5-1.5B (RoPE/GQA/RMSNorm) + Mamba-130m/370m/790m (SSM) for the cross-architecture sweeps,
> bf16 on one RTX 5050. Every table is backed by a tracked `runs/**/*_summary.json` (see
> [`runs/README.md`](runs/README.md)); the catalog docs are **generated** from those artifacts.

---

## Read next

| you want | go to |
|---|---|
| the **operator catalog** (every op class × model + per-op dossiers) | [`docs/operators/`](docs/operators/README.md) |
| the **circuit catalog** (composed circuits, cross-model + discovered) | [`docs/circuits/`](docs/circuits/README.md) |
| the **decompilation** design + all milestone results | [`docs/DECOMPILATION.md`](docs/DECOMPILATION.md) |
| the GPT-2 + cross-model **disassembly** deep-dive | [`docs/DISASSEMBLY.md`](docs/DISASSEMBLY.md) |
| the **oracle / forge-tax** sister track | [`docs/FORGE_TAX_TRACK.md`](docs/FORGE_TAX_TRACK.md) |

## Results at a glance

| area | finding | key number |
|---|---|---|
| disassembly | GPT-2 attention is a reused op-catalog; **8/8** literature idioms recovered from weights | ~99% of content mass legible, ~2% dark |
| disassembly | named heads are **causally load-bearing** + **corpus-robust** | induction-NLL z=8.6; head identities ρ≈0.84 across corpora |
| operator catalog | **7 universal operator classes × 6 models**; induction is the most universal | induction signal .91–.99 in all 6 |
| operator catalog | the **sink** is common but **absent in Gemma**, and *present ≠ depended-on* | sink 0 heads in Gemma vs 117–553 elsewhere; prose-causal ≈0 everywhere |
| operator catalog | RoPE leans on **self** where GPT-2 leans on the sink | self causal ΔNLL Qwen **+3.77** |
| circuit catalog | the **induction edge** (prev-tok→induction) is live in **all 6** models, *stronger* in RoPE | key-collapse +17% (gpt2) … **+89%** (Qwen) |
| circuit catalog | **positional-broadcast** (sink→prev-tok key) is **GPT-2-small/medium-only** | +22/+32% vs ≈0 elsewhere |
| decompilation | the named induction circuit beats random everywhere, but the **decompilable *fraction* is absolute-position-family-specific** | abs-pos ~20% / 4–17× vs RoPE 3–9% |
| SSM port | the in-context-copy **capability survives a non-attention mixer** | induction gain +12.1…+12.5 across Mamba **and** GPT-2 |
| discovery | the ResidualVM engine finds **MLP0 as the top component for every behaviour** + unnamed candidate ops | MLP0 ΔNLL +7.8/+7.5/+1.8; candidates 7.6, 5.9 |

---

## The idea: disassembly → catalog → decompilation

| | disassembly (have) | decompilation (target) |
|---|---|---|
| unit | one head / MLP / idiom in isolation | the **composition** — which ops chain into which |
| coverage | *% of attention legible* (named or weight-binding) | *% of the forward pass faithfully reconstructable* |
| validation | mean-ablation damages the metric (necessary) | **recompile the extracted program; KL ≈ host** (sufficient) |

The **instruction set** has two classes: **MOVE** = attention (a QK *addressing mode* × an OV *write op*) and
**COMPUTE** = MLP (key–value memories). An **operator** is one head/MLP-class; a **circuit** is a composition —
a writer-op feeding a reader-op's K/Q/V port, chained. **Decompilation coverage** replaces "% legible" with a
metric that has teeth: `1 − KL(host ‖ recompiled[ops kept]) / KL(host ‖ all-ablated)` — does the extracted
op-graph, recompiled, reproduce the host's next-token distribution? Full design + milestones in
[`docs/DECOMPILATION.md`](docs/DECOMPILATION.md).

## The operator catalog

Every behaviourally-maskable **operator class** measured across every architecture — `operator_atlas.py` →
[`docs/operators/`](docs/operators/README.md). Each cell: behavioural **signal** (max head-mass on the op's
pattern), **membership** (# heads — these are *classes*, head families, not single heads), top head + depth, and
a uniform **causal ΔNLL**.

| class | gpt2 | -medium | -large | gemma-2 | llama-3.2 | qwen-2.5 |
|---|---|---|---|---|---|---|
| prevtok | .96 | .99 | .96 | .84 | .68 | .77 |
| induction | .92 | .91 | .96 | .94 | .94 | **.99** |
| duplicate | .62 | .86 | .96 | .85 | .74 | .97 |
| **sink** | .95 | .96 | .96 | **.07** | 1.00 | 1.00 |
| self | .83 | .45 | .55 | .95 | .96 | **1.00** |
| local | .34 | .34 | .34 | .31 | .27 | .28 |
| structural | .19 | .25 | .38 | .23 | .36 | .16 |

- **Induction is the most universal class** (.91–.99 everywhere). **Sink** splits 5-vs-1 (present in GPT-2 family
  + Llama + Qwen, **absent in Gemma**) and is *present-but-not-depended-on* on prose (causal ≈0 everywhere — the
  magnitude-vs-dependence point). **RoPE leans on `self`** (causal ΔNLL Qwen +3.77) where GPT-2 leans on the sink.
- **Deep per-op dossiers** (`operator_dossier.py`) run a 6-section battery on one op — A identity · B causal×5
  tasks · C K/V channel decomposition (match vs move) · D composition in/out edges · E redundancy curve · F
  cross-model. 9 GPT-2 classes catalogued. Surfaced: induction redundancy is **superadditive** (5h +6.39 ≫ sum of
  solos ~2.18); the sink "class" is largely **content heads idling**; copy-suppression ablation *raises* IOI.
- **Taxonomy:** class → instance (an individual head) → variant (intra-class structure). The membership matrix is
  the head count per class.

## The circuit catalog

Compositions of operators, surveyed and **collected** across models — `circuit_atlas.py` →
[`docs/circuits/`](docs/circuits/README.md). Cross-model **edge liveness** (path-patch: remove the writer from the
reader's key → attention collapse %):

| circuit | gpt2 | -medium | -large | gemma-2 | llama-3.2 | qwen-2.5 |
|---|---|---|---|---|---|---|
| **induction** (prev-tok→induction) | +17% | +23% | +8% | +18% | **+70%** | **+89%** |
| **positional_broadcast** (sink→prev-tok key) | +22% | +32% | +0% | +0% | (skip) | +0% |
| duplicate | (skip) | +0% | +4% | +3% | +9% | +13% |

The induction edge is live in **every** model (and *stronger* in RoPE — content matching lives in the key
everywhere); positional-broadcast is **GPT-2-small/medium-only** (RoPE reads position from the rotation). The
GPT-2 discovery artifacts are harvested into the catalog too: the **IOI Q-chain** (duplicate→S-inhibition→
name-mover, + self-repair), the **V-composition "virtual heads"** (induction 5.9 → layer-6 value 6.7), and the
**22 novel-live discovered edges** (13 named — early sink/write-hub → prev-token-key broadcasters).

## Decompilation milestones

`docs/DECOMPILATION.md` carries the full results. Headlines:
- **M1 — reconstruction-coverage** (`residual_vm.py`): keep a head-set, mean-ablate the rest, `1−KL/floor`. The
  5-head induction circuit reconstructs +0.164 vs random-5 +0.032 (~5×) — the named catalog is coverage-efficient.
- **M2 — composition-DAG extractor** (`composition_dag.py`): weight-space K/Q/V edge scorer + path-patch gate
  (ΔTV vs reader-matched null); auto-recovers the induction K-chain + IOI Q-chain, rejects imposters at 0% FP,
  reports novel edges. **M3** adds the MLP/COMPUTE nodes; **V-composition** completes the K/Q/V edge types.
- **M4 — forge-basis ceiling**: content reconstructs (mAUC ~86%) but monosemantic factorability collapses
  (cov95 ~11%) — two decoupled axes.
- **M5 — cross-model ceiling**: the named induction circuit beats random in **all 5** models (mechanisms
  invariant), but the *decompilable fraction* is **not** architecture-invariant — the **absolute-position family**
  (GPT-2, incl. gpt2-medium control) keeps ~20% / 4–17× at any size, every RoPE model sits at 3–9%. RoPE
  *distributes* the circuit; absolute positions *concentrate* it.

## Cross-architecture synthesis

The same disassembler ports across the RoPE/GQA/RMSNorm/gated-MLP family at GPT-2 parity (`gemma/`,
[`docs/DISASSEMBLY.md`](docs/DISASSEMBLY.md)). The recurring shape: **mechanisms are invariant** (idioms,
induction causal in all four at z 8.3–27.3, K/Q/V composition), **but the positional register is
absolute-position-family-specific** — three independent signatures (the sink, the positional-broadcast circuit,
the decompilable fraction) all single out GPT-2's learned absolute positions. The plumbing *fraction* is
model-invariant (~87%). And one level deeper: porting induction to **Mamba** (pure SSM, no heads) shows the
in-context-copy **capability** survives the loss of attention entirely (gain +12.1…+12.5 in both families) — though
the *mechanism* is unverified without head-resolution (`ssm_induction.py`).

## The ResidualVM debugger — the catalog's growth engine

The catalogs survey what we *named*; the **programmatic** debugger (`residual_vm_debugger.py`) finds what we
haven't. It is automation-first (returns structured data; UI-optional): `attribution_sweep` ablates every head +
every MLP and flags strong **un-named** components (candidate new ops); `edge_probe` path-patches upstream writers
(candidate circuit edges); `logit_lens_step` locates where the answer is decided. Driven on GPT-2 it found **MLP0
had the largest single-component causal effect of anything measured** (induction +7.8, IOI +7.5, generic +1.8 — a
direct pointer to the MLP/COMPUTE catalog gap), candidate unnamed ops (induction→**7.6**, IOI→**5.9**), and a
discovered circuit (7.6 fed by induction heads + candidate writers 2.11/5.9). These feed the next dossiers.

The catalog now spans both instruction classes — **MOVE** (the [operator catalog](docs/operators/README.md), 7
attention classes × 6 models) and **COMPUTE** (the [MLP catalog](docs/operators/mlp_compute.md): COMPUTE
concentrates on an early-MLP detokenizer in 5/6 models, distributes in Gemma).

## Future work

- **Per-neuron MLP idioms cross-model** — the COMPUTE catalog has per-layer causal profiles for all models, but
  per-*neuron* read→write idioms only for GPT-2 (the cheap token-unembedding basis); extend via per-layer SAEs.
- **Cross-model circuit discovery** — `composition_dag` de-novo edge discovery is GPT-2-only; port to RoPE.
- **Dossier the discovered candidates** (7.6, 5.9, 8.3) and the discovered edges.
- **Dossier the discovered candidates** (7.6, 5.9, 8.3) and the discovered edges.
- **Per-class SAE-feature operands** (what each op reads/writes in feature space) — needs a SAE.
- **Executable decompilation** — recompile a path-patch-validated circuit and measure KL ≈ host (the M-series target).

---

## Repository map

```
lm-sae/
├── README.md                ← you are here (the disassembly → decompilation program)
├── docs/
│   ├── DECOMPILATION.md      ← decompilation design + all milestone/catalog results
│   ├── DISASSEMBLY.md        ← GPT-2 + cross-model disassembly deep-dive
│   ├── operators/            ← GENERATED operator catalog (survey matrices + per-op pages)
│   ├── circuits/             ← GENERATED circuit catalog (cross-model edges + per-circuit pages)
│   ├── FORGE_TAX_TRACK.md    ← sister track: the oracle & cov95 forge tax
│   └── listings/             ← committed full per-head disassembly listings (GPT-2 + Gemma)
├── scripts/                  ← grouped by research thread — see scripts/README.md
│   └── disassembly/          ← op/circuit catalog builders, dossiers, the ResidualVM debugger, doc generators
└── runs/                     ← result artifacts; *_summary.json tracked — see runs/README.md
```

## How to run

Standalone — its own venv, `sae-forge` from PyPI, no bio-sae path:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
# GPU (RTX 50-series / Blackwell, sm_120): install the cu128 torch wheel
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# the GPT-2 disassembly pipeline (CPU)
.venv/bin/python scripts/disassembly/idiom_library_v2.py
.venv/bin/python scripts/disassembly/coverage_scorecard.py --corpus wikitext
.venv/bin/python scripts/disassembly/causal_validation.py

# the operator catalog (survey across models, then a deep dossier, then the docs)
.venv/bin/python scripts/disassembly/operator_atlas.py            # needs the GPU for the RoPE models
.venv/bin/python scripts/disassembly/operator_dossier.py --op induction
.venv/bin/python scripts/disassembly/operator_catalog_doc.py      # regenerates docs/operators/ from JSON

# the circuit catalog + the discovery engine
.venv/bin/python scripts/disassembly/circuit_atlas.py
.venv/bin/python scripts/disassembly/circuit_catalog_doc.py       # regenerates docs/circuits/ from JSON
.venv/bin/python scripts/disassembly/residual_vm_debugger.py      # programmatic op/circuit discovery
```

The two `*_catalog_doc.py` generators read only the committed `*_summary.json` (no GPU/torch), so the catalog docs
regenerate anywhere. See [`scripts/README.md`](scripts/README.md) for the full per-group guide and run order.

## Honest caveats

1. **First-order + superposition-limited.** Single-component instructions + named idioms + path-patched edges;
   the operand basis (token-centroid / SAE) caps fidelity. Not yet an *executable* decompilation.
2. **Causal claims are metric-specific** (confirmed on the metric each idiom serves) and **coverage magnitudes are
   corpus-conditioned** (use the prose baseline). The catalog's causal column is *generic-prose* ΔNLL — task-specific
   ops read low there (see each dossier's section B).
3. **IOI circuit ops are GPT-2-only** (literature DLA head-sets; no published set off GPT-2). **SSM** has no heads
   (induction present only behaviourally) and no separate MLP block (excluded from the COMPUTE catalog). The
   cross-model **MLP/COMPUTE** rows are per-*layer* causal profiles; per-*neuron* idioms are GPT-2-only.
4. **Small/mid hosts** (GPT-2 family, 1–2B RoPE models, ≤790M Mamba) — not frontier scale.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
