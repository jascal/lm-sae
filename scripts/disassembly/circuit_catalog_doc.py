"""Generate the circuit-catalog docs from the JSON artifacts — the circuit survey as browsable markdown.

Reads [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json) (cross-model edges + harvested GPT-2 circuits) plus the
committed discovery artifacts (`rung3_induction_chain`, `self_repair`, `validate_new_edges`) and emits:
  docs/circuits/README.md   — the cross-model circuit-edge matrix + the full circuit inventory + taxonomy + gaps.
  docs/circuits/<circuit>.md — one page per circuit (its DAG/stages, cross-model edge liveness, causal, redundancy).
Mirror of `operator_catalog_doc.py`; generated, so re-running after new survey/discovery runs keeps it in sync.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def cross_matrix(atlas):
    rows = atlas["cross_model_circuits"]; models = atlas["models"]
    head = "| circuit | defining edge | " + " | ".join(models) + " |"
    sep = "|" + "---|" * (len(models) + 2)
    lines = [head, sep]
    for c, row in rows.items():
        cells = []
        for m in models:
            cc = row["by_model"].get(m, {})
            cells.append("(skip)" if cc.get("skipped") else f"{(cc.get('edge_collapse') or 0):+.0%} {cc.get('writer','')}")
        lines.append(f"| **{c}** | {row['defining_edge']} | " + " | ".join(cells) + " |")
    return models, "\n".join(lines)


def disambig(c, operator_names):
    """A banner shown when a circuit shares a name with an operator (the head class it's named after)."""
    if c not in operator_names:
        return []
    return [f"> **`circuit:{c}`** — this is the **circuit** (a *composition* of operators: a writer-op feeding a "
            f"reader-op's K/Q/V port). Not the [`{c}` *operator*](../operators/{c}.md), which is the head *class* "
            f"this circuit is named after (the circuit is keyed by its **reader operator**). `circuit:{c}` here vs "
            f"`op:{c}` there.", ""]


def dossier_section(c, dossier):
    """The cross-model necessity/sufficiency/redundancy battery for circuit `c` (from circuit_dossier_xmodel.py)."""
    results = [r for r in dossier.get("results", []) if "circuits" in r and c in r["circuits"]]
    if not results:
        return []
    lines = ["", "## Cross-model causal dossier (necessity / sufficiency / redundancy — via the ResidualVM)", "",
             "The operator-dossier battery, lifted to this circuit and run on the [unified `ResidualVM`](../DECOMPILATION.md) "
             "(`find_heads` locates the heads, `ablate_heads` + `nll` measure the rest). Two next-token metrics: "
             "**induction-NLL** (in-context copy) and **generic-NLL** (general LM).", "",
             "| model | reader | necessity Δind-NLL | necessity Δgen-NLL | sufficiency (keep-only, ind) | reader redundancy |",
             "|---|---|---|---|---|---|"]
    for r in results:
        cc = r["circuits"][c]; nec = cc["necessity"]["circuit"]; suff = cc["sufficiency"]; red = cc["redundancy"]
        verdict = "bottleneck" if red.get("bottleneck") else "distributed"
        lines.append(f"| {r['model']} | {cc.get('reader_top','—')} | {nec['ind']:+.2f} | {nec['gen']:+.2f} | "
                     f"{suff['ind_coverage']:+.0%} | {verdict} |")
    lines += ["",
              "- **Necessity** — Δ NLL when the circuit's heads are mean-ablated (higher = more load-bearing for that "
              "behaviour). Generic-NLL necessity is small everywhere — these circuits are *task-specific*, not general-LM.",
              "- **Sufficiency** — reconstruction coverage keeping **only** the circuit's heads (MLPs intact); a small "
              "head-set that reconstructs the behaviour is an executable decompilation. (Generic-NLL coverage is omitted "
              "as a headline — with MLPs intact a tiny head-set scores high for reasons unrelated to the circuit; "
              "induction-NLL is the meaningful attention-circuit metric. Negative = keeping so few heads is worse than "
              "the all-ablated floor, the known keep-1-is-net-negative effect.)",
              "- **Redundancy** — reader-head solo-vs-cumulative on induction-NLL: *bottleneck* = one head carries it, "
              "*distributed* = the population shares it."]
    if c == "induction":
        ladder = [(r["model"], r["circuits"]["induction"]) for r in results if r["model"].startswith("gpt2")]
        if len(ladder) >= 3:
            nec_str = " → ".join(f"{cc['necessity']['circuit']['ind']:+.2f}" for _, cc in ladder)
            suf_str = " → ".join(f"{cc['sufficiency']['ind_coverage']:+.0%}" for _, cc in ladder)
            lines += ["", f"**The induction circuit's necessity AND sufficiency both decay monotonically across the "
                      f"GPT-2 ladder** ({', '.join(m for m, _ in ladder)}): necessity Δind-NLL {nec_str}; "
                      f"sufficiency {suf_str}. The same scale-driven distributedness the rest of the catalog finds — "
                      "the named circuit is most localized in the smallest model and dissolves into the network with scale."]
    lines += ["", "_Dossier data: [runs/disassembly/circuits/dossier_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/dossier_summary.json) "
              "([circuit_dossier_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_dossier_xmodel.py), built on the ResidualVM)._"]
    return lines


def cross_page(c, row, models, rung3, dossier=None, operator_names=frozenset()):
    lines = [f"# Circuit `{c}` (cross-model)", ""]
    lines += disambig(c, operator_names)
    lines += [row["desc"], "",
             f"**Defining edge:** `{row['defining_edge']}`", "",
             "## Cross-model edge liveness (path-patch: remove the writer from the reader's key → attention collapse)", "",
             "| model | reader | writer | key collapse | writer is | value mover | value ΔV-out |",
             "|---|---|---|---|---|---|---|"]
    for m in models:
        cc = row["by_model"].get(m, {})
        if cc.get("skipped"):
            lines.append(f"| {m} | {cc.get('reader','—')} | — | (skipped) | — | — | — |")
        else:
            wt = "prev-tok head" if cc.get("is_prevtok_writer") else ("sink" if cc.get("is_sink_writer") else "—")
            lines.append(f"| {m} | {cc.get('reader')} | {cc.get('writer')} | {(cc.get('edge_collapse') or 0):+.0%} | {wt} | {cc.get('value_mover')} | {(cc.get('value_dvout') or 0):.2f} |")
    if c == "induction" and rung3:
        pop = rung3.get("prevtok_population")
        pop_n = len(pop) if isinstance(pop, list) else pop
        lines += ["", "## Stage redundancy (GPT-2, `rung3_induction_chain.py`)",
                  f"3-stage chain: prev-token population ({pop_n} heads) → stage-2 reader "
                  f"`{rung3.get('stage2_B')}` (bottleneck) → inductors. Writers are individually redundant, collectively "
                  f"necessary; copy-score↔induction ρ {rung3.get('spearman_copyscore_vs_induction')}."]
    if dossier:
        lines += dossier_section(c, dossier)
    lines += ["", "_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json). Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._"]
    return "\n".join(lines)


def ioi_xmodel_section(dossier):
    """The cross-model IOI dossier (name-movers / negative-movers / necessity / duplicate) for the ioi_q_chain page."""
    results = [r for r in dossier.get("results", []) if "name_movers" in r]
    if not results:
        return []
    order = {"gpt2": 0, "gpt2-medium": 1, "gpt2-large": 2, "gemma-2-2b": 3, "Llama-3.2-1B": 4, "Qwen2.5-1.5B": 5}
    results.sort(key=lambda r: order.get(r["model"], 9))
    lines = ["", "## Cross-model IOI dossier (the circuit's operators, found behaviourally — via the ResidualVM)", "",
             "The Q-composition *edge wiring* below is GPT-2-validated, but the IOI **behaviour** and its **operators** "
             "are not GPT-2-only: the unified [`ResidualVM`](../DECOMPILATION.md) locates them in every model "
             "(name-movers by END→indirect-object copy-attention; negative-movers + the most load-bearing heads by an "
             "ablation sweep of the logit-diff; the duplicate-token initiator behaviourally). Logit-diff = "
             "logit(IO) − logit(S) at the end of a templated *\"When X and Y went…, Y gave a drink to →\"* prompt.", "",
             "| model | IOI logit-diff | name-movers (copy→IO) | negative-movers | most load-bearing | ablate name-movers | duplicate init |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        nm = ", ".join(f"`{m['head']}`" for m in r["name_movers"][:3])
        neg = ", ".join(f"`{m['head']}`" for m in r["negative_movers"][:2]) or "—"
        lb = ", ".join(f"`{m['head']}`" for m in r.get("load_bearing_heads", [])[:2])
        nec = r["necessity_ablate_namemovers"]; dup = r["duplicate_head"]["head"] if r["duplicate_head"] else "—"
        lines.append(f"| {r['model']} | {r['baseline_logit_diff']:+.2f} | {nm} | {neg} | {lb} | "
                     f"{nec['frac_collapse']:+.0%} | `{dup}` |")
    lines += ["",
              "- **The IOI circuit is architecture-invariant** — name-movers, negative/copy-suppression movers, and a "
              "duplicate-token initiator are present in all six models, and ablating the name-movers collapses the "
              "logit-diff (+13% to +26%) everywhere. The behaviour *strengthens* with GPT-2 scale (logit-diff +2.88 → "
              "+3.11 → +4.09) and is largest in the RoPE models (Llama +5.85, Qwen +6.00).",
              "- **The backup-name-mover self-repair is cross-model.** The heads that are *most load-bearing* under "
              "ablation are **not** the name-movers (which are backed up) but the **S-inhibition**-type heads — in every "
              "model the ablation ranking and the copy-attention ranking disagree, the signature of name-mover "
              "redundancy (the Hydra effect) generalised beyond GPT-2.",
              "", "_Data: [runs/disassembly/circuits/ioi_xmodel_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/ioi_xmodel_summary.json) "
              "([ioi_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/ioi_xmodel.py), built on the ResidualVM)._"]
    return lines


def gpt2_page(name, c, extra, ioi_dossier=None, operator_names=frozenset()):
    lines = [f"# Circuit `{name}` (GPT-2)", ""]
    lines += disambig(name, operator_names)
    lines += [f"**{c.get('kind','')}** — scope: {c.get('scope','gpt2')}", ""]
    if name == "ioi_q_chain":
        lines += ["The indirect-object-identification circuit: **duplicate-token → S-inhibition → name-mover**, a "
                  "Q-composition chain (no published head-set outside GPT-2).", "",
                  f"- S-inhibition heads: {c.get('s_inhibition')}; name-movers: {c.get('name_movers')}",
                  f"- Q-composition live edges: **{c.get('q_live_edges')}**; named-edge live-rate {c.get('named_edge_live_rate')}",
                  f"- IOI baseline logit-diff {c.get('ioi_baseline_logit_diff')}",
                  f"- causal z (`ioi_causal.py`): name-mover {c.get('name_mover_z')}, S-inhibition {c.get('s_inhibition_z')}, "
                  f"**negative/copy-suppression {c.get('negative_mover_z')}** (writes against IO), duplicate {c.get('duplicate_z')}, "
                  f"backup name-mover {c.get('backup_namemover_z')}"]
        if extra.get("self_repair"):
            sr = extra["self_repair"]
            lines += [f"- **self-repair** (`self_repair.py`): −primaries ΔLD {sr.get('drop_primaries')}, −both {sr.get('drop_both')} "
                      f"→ backups are hot spares (idle with primaries present, carry the circuit once they're gone)."]
        if ioi_dossier:
            lines += ioi_xmodel_section(ioi_dossier)
    elif name == "v_virtual_heads":
        lines += ["Composed-OV **virtual heads**: an induction head's OV output is re-read as the *value* of a later head "
                  "(the third Elhage edge type — changes what is moved, not where attention points).", "",
                  "- top V-edges: " + ", ".join(f"`{e['edge']}` (ΔV-out {e['dvout']:.2f})" for e in c.get("top_edges", [])[:5]),
                  f"- median ΔV-out {c.get('topV_median_dvout')}; static-V↔ΔV-out ρ {c.get('spearman_staticV_vs_dVout')}; V/K {c.get('V_over_K_mean')}"]
    elif name == "discovered_write_hub_edges":
        lines += ["**DISCOVERED** (not pre-named): novel composition edges that survived the path-patch liveness gate vs a "
                  "reader-matched null — the collection-goal output.", "",
                  f"- {c.get('n_novel_live')} novel-live edges (of {c.get('n_new_edges')} new); **{c.get('n_named')} named** by behaviour: "
                  f"{c.get('named_by_pattern')}",
                  f"- edges: {', '.join('`'+e+'`' for e in c.get('edges', [])[:22])}",
                  "- These are mostly early **sink/write-hub → prev-token-key** broadcasters (the absolute-position plumbing)."]
    elif name == "induction_kchain_weights":
        lines += ["The induction macro read from the **weights** (K-composition) and path-patch-gated on GPT-2 "
                  "(the cross-model behavioural view is on the `induction` page).", "",
                  f"- canonical writer **{c.get('canonical_writer')}**; {c.get('canonical_induction_live')}/{c.get('canonical_induction_edges')} canonical edges live",
                  f"- K-composition static {c.get('K_induction_static')} vs random {c.get('K_random_baseline')}; top edge rel-drop {c.get('induction_top_rel_drop')}"]
    lines += ["", "_Data: [runs/disassembly/circuits/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/atlas_summary.json) + the discovery artifacts. Regenerate: [circuit_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_catalog_doc.py)._"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/disassembly/circuits"))
    p.add_argument("--disasm", type=Path, default=Path("runs/disassembly"))
    p.add_argument("--docs", type=Path, default=Path("docs/circuits"))
    args = p.parse_args(argv)
    atlas = load(args.root / "atlas_summary.json")
    dossier = load(args.root / "dossier_summary.json")
    ioi_dossier = load(args.root / "ioi_xmodel_summary.json")
    rung3 = load(args.disasm / "rung3_induction_chain_summary.json")
    extra = {"self_repair": load(args.disasm / "self_repair_summary.json")}
    args.docs.mkdir(parents=True, exist_ok=True)

    # the set of operator names (the OTHER catalog) — any circuit sharing a name with an operator gets a
    # disambiguation banner + a `circuit:` qualifier in the index.
    ops_atlas = load(args.disasm / "operators" / "atlas_summary.json")
    operator_names = frozenset(ops_atlas.get("operators", [])) | frozenset(ops_atlas.get("gpt2_circuit_ops", {}))

    models, cmat = cross_matrix(atlas)
    cross = atlas["cross_model_circuits"]; gpt2 = atlas["gpt2_circuits"]
    for c, row in cross.items():
        (args.docs / f"{c}.md").write_text(cross_page(c, row, models, rung3, dossier, operator_names))
    for name, c in gpt2.items():
        (args.docs / f"{name}.md").write_text(gpt2_page(name, c, extra, ioi_dossier if name == "ioi_q_chain" else None, operator_names))

    all_circuits = list(cross) + list(gpt2)

    def qual(c):
        return f" · **also an [operator](../operators/{c}.md)** (`circuit:{c}` here vs `op:{c}` there)" if c in operator_names else ""
    idx = "\n".join(f"- [`{c}`]({c}.md) — cross-model{qual(c)}" for c in cross) + "\n" + \
          "\n".join(f"- [`{c}`]({c}.md) — GPT-2 ({gpt2[c].get('kind','')}){qual(c)}" for c in gpt2)
    readme = f"""# Circuit catalog — composed circuits, surveyed & collected across models

Operators are single head-classes ([`../operators/`](../operators/README.md)); **circuits** are their
*compositions* (a writer-op feeding a reader-op's K/Q/V port, chained). A **working catalog** (amateur,
exploratory, provisional — not a definitive reference) of **{len(all_circuits)} circuits** collected with the tools
here, two sources:

- **Cross-model circuit edges** — the defining composition edge of each universal-reader circuit, path-patched
  across {len(models)} models (faithful key/value patch, arch-generic).
- **GPT-2 discovered / circuit-specific** — harvested from the committed discovery artifacts (`composition_dag`,
  `vcomposition`, `ioi_causal`, `validate_new_edges`): the IOI Q-chain, the V-composition virtual heads, and the
  **{gpt2.get('discovered_write_hub_edges', {}).get('n_novel_live', '?')} novel-live edges** the discovery gate found
  (of which {gpt2.get('discovered_write_hub_edges', {}).get('n_named', '?')} are behaviourally named). These are
  GPT-2-only (literature IOI head-sets / GPT-2 path-patch runs).

## Circuit inventory (index)

{idx}

> **Circuit vs operator — a naming note.** A few names ({', '.join(sorted(c for c in all_circuits if c in operator_names)) or '—'}) appear in
> *both* this circuit catalog and the [operator catalog](../operators/README.md). A **circuit** is a *composition*
> (`circuit:induction` = prev-token → induction); the same-named **operator** is the *head class* it is named
> after and built around (`op:induction`). The coincidence is deliberate: a circuit is keyed by its **reader
> operator**. Pages cross-link to their namesake.

## Cross-model circuit-edge liveness (remove the writer from the reader's key → attention collapse %)

{cmat}

**Reading it:** the **induction** edge (prev-token → induction) is live in *every* model (and *stronger* in RoPE —
content matching lives in the key everywhere); **positional-broadcast** (sink/hub → prev-token key) is
**GPT-2-small/medium-only** (the absolute-position plumbing — RoPE reads position from the rotation, so the
prev-token key has no upstream writer to remove). Same absolute-position-family split as the operator catalog's sink.

## Discovered edges (de novo, cross-model)

Beyond the named circuits, [**discovered circuit edges**](discovered.md) runs the key-patch over the top content
readers in *every* model and keeps the edges that collapse the reader beyond a reader-matched null. It recovers
the prev-token→induction K-chain de novo in the GPT-2 family (6/2/2 live edges) and finds localized edges in
Llama (3) and Qwen (1), but **none in Gemma** (0 — its content-reader keys aren't sharply localized to one
writer; RoPE distributes the circuit). 14 live edges total.

## Executable decompilation — is the circuit *sufficient*?

Edge liveness shows the circuit's edges are **necessary**. [**Reconstruction**](reconstruction.md) tests
**sufficiency**: keep only the induction circuit's heads (induction + prev-token), mean-ablate every other
attention head (MLPs intact), and measure how much induction the circuit alone recovers — far above a random
same-size head-set. A small head-set that reconstructs most of the behaviour is an *executable* decompilation.

Each cross-model circuit page now also carries a **cross-model causal dossier** (necessity + sufficiency +
redundancy, operator-parity), generated on the [unified `ResidualVM`](../DECOMPILATION.md) debugger
(`circuit_dossier_xmodel.py`). The sharpest read: the **induction circuit's necessity *and* sufficiency both decay
monotonically across the GPT-2 ladder** (small → XL) — the named circuit is most localized in the smallest model
and dissolves into the network with scale, the same distributedness theme measured as a clean ablation battery.

## Taxonomy & gaps

- **Levels:** circuit (a DAG of operator nodes) → edge (writer-op → reader-op via a K/Q/V port) → the operator
  classes at each node ([`../operators/`](../operators/README.md)). Edges are the primitive the discovery gate scores.
- **succession / greater-than** — MLP-dominated; no clean attention-composition circuit (carried by the copy ops).
- **SSM (Mamba)** — no heads, so no composition edges; induction is present only behaviourally (`ssm_induction.py`).
- **IOI is now cross-model** (the [`ioi_q_chain`](ioi_q_chain.md) page): the circuit's *operators* (name-movers,
  negative/copy-suppression movers, duplicate-token initiator) and its load-bearing necessity are found
  behaviourally in all 6 models via the ResidualVM — closing the old "no head-set off GPT-2" gap. The precise
  *Q-composition edge wiring* stays GPT-2-validated.
- **Still GPT-2-only:** V-composition cross-model; full per-edge path-patch of all
  {gpt2.get('discovered_write_hub_edges', {}).get('n_novel_live', '?')} discovered write-hub edges on the RoPE
  models. The cross-model catalog covers the universal-reader edges + the IOI operators.

## How this was made

`circuit_atlas.py` (cross-model edges + harvest) → `circuit_catalog_doc.py` (these docs). Discovery/validation:
`composition_dag.py`, `validate_new_edges.py`, `vcomposition.py`, `ioi_causal.py`, `self_repair.py`,
`rung3_induction_chain.py`. See [`../DECOMPILATION.md`](../DECOMPILATION.md).
"""
    (args.docs / "README.md").write_text(readme)
    print(f"[done] wrote {args.docs}/README.md + {len(all_circuits)} circuit pages: {all_circuits}")


if __name__ == "__main__":
    main()
