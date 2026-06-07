"""Generate the circuit-catalog docs from the JSON artifacts — the circuit survey as browsable markdown.

Reads `runs/disassembly/circuits/atlas_summary.json` (cross-model edges + harvested GPT-2 circuits) plus the
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


def cross_page(c, row, models, rung3):
    lines = [f"# Circuit `{c}` (cross-model)", "", row["desc"], "",
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
    lines += ["", "_Data: `runs/disassembly/circuits/atlas_summary.json`. Regenerate: `circuit_catalog_doc.py`._"]
    return "\n".join(lines)


def gpt2_page(name, c, extra):
    lines = [f"# Circuit `{name}` (GPT-2)", "", f"**{c.get('kind','')}** — scope: {c.get('scope','gpt2')}", ""]
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
    lines += ["", "_Data: `runs/disassembly/circuits/atlas_summary.json` + the discovery artifacts. Regenerate: `circuit_catalog_doc.py`._"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/disassembly/circuits"))
    p.add_argument("--disasm", type=Path, default=Path("runs/disassembly"))
    p.add_argument("--docs", type=Path, default=Path("docs/circuits"))
    args = p.parse_args(argv)
    atlas = load(args.root / "atlas_summary.json")
    rung3 = load(args.disasm / "rung3_induction_chain_summary.json")
    extra = {"self_repair": load(args.disasm / "self_repair_summary.json")}
    args.docs.mkdir(parents=True, exist_ok=True)

    models, cmat = cross_matrix(atlas)
    cross = atlas["cross_model_circuits"]; gpt2 = atlas["gpt2_circuits"]
    for c, row in cross.items():
        (args.docs / f"{c}.md").write_text(cross_page(c, row, models, rung3))
    for name, c in gpt2.items():
        (args.docs / f"{name}.md").write_text(gpt2_page(name, c, extra))

    all_circuits = list(cross) + list(gpt2)
    idx = "\n".join(f"- [`{c}`]({c}.md) — cross-model" for c in cross) + "\n" + \
          "\n".join(f"- [`{c}`]({c}.md) — GPT-2 ({gpt2[c].get('kind','')})" for c in gpt2)
    readme = f"""# Circuit catalog — composed circuits, surveyed & collected across models

Operators are single head-classes ([`../operators/`](../operators/README.md)); **circuits** are their
*compositions* (a writer-op feeding a reader-op's K/Q/V port, chained). This catalogs **{len(all_circuits)} circuits**
collected with the tools here, two sources:

- **Cross-model circuit edges** — the defining composition edge of each universal-reader circuit, path-patched
  across {len(models)} models (faithful key/value patch, arch-generic).
- **GPT-2 discovered / circuit-specific** — harvested from the committed discovery artifacts (`composition_dag`,
  `vcomposition`, `ioi_causal`, `validate_new_edges`): the IOI Q-chain, the V-composition virtual heads, and the
  **{gpt2.get('discovered_write_hub_edges', {}).get('n_novel_live', '?')} novel-live edges** the discovery gate found
  (of which {gpt2.get('discovered_write_hub_edges', {}).get('n_named', '?')} are behaviourally named). These are
  GPT-2-only (literature IOI head-sets / GPT-2 path-patch runs).

## Cross-model circuit-edge liveness (remove the writer from the reader's key → attention collapse %)

{cmat}

**Reading it:** the **induction** edge (prev-token → induction) is live in *every* model (and *stronger* in RoPE —
content matching lives in the key everywhere); **positional-broadcast** (sink/hub → prev-token key) is
**GPT-2-small/medium-only** (the absolute-position plumbing — RoPE reads position from the rotation, so the
prev-token key has no upstream writer to remove). Same absolute-position-family split as the operator catalog's sink.

## Circuit inventory (index)

{idx}

## Taxonomy & gaps

- **Levels:** circuit (a DAG of operator nodes) → edge (writer-op → reader-op via a K/Q/V port) → the operator
  classes at each node ([`../operators/`](../operators/README.md)). Edges are the primitive the discovery gate scores.
- **succession / greater-than** — MLP-dominated; no clean attention-composition circuit (carried by the copy ops).
- **SSM (Mamba)** — no heads, so no composition edges; induction is present only behaviourally (`ssm_induction.py`).
- **Not yet run:** the IOI Q-chain / V-composition cross-model (no published head-sets off GPT-2); full per-edge
  path-patch of all {gpt2.get('discovered_write_hub_edges', {}).get('n_novel_live', '?')} discovered edges on the
  RoPE models. The cross-model catalog covers the universal-reader edges.

## How this was made

`circuit_atlas.py` (cross-model edges + harvest) → `circuit_catalog_doc.py` (these docs). Discovery/validation:
`composition_dag.py`, `validate_new_edges.py`, `vcomposition.py`, `ioi_causal.py`, `self_repair.py`,
`rung3_induction_chain.py`. See [`../DECOMPILATION.md`](../DECOMPILATION.md).
"""
    (args.docs / "README.md").write_text(readme)
    print(f"[done] wrote {args.docs}/README.md + {len(all_circuits)} circuit pages: {all_circuits}")


if __name__ == "__main__":
    main()
