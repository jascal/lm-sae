"""Generate the operator-catalog docs from the JSON artifacts — the survey as browsable markdown.

Reads [runs/disassembly/operators/atlas_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/atlas_summary.json) (the cross-model matrix) + each
[runs/disassembly/operators/dossiers/<op>/<model>.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/dossiers/<op>/<model>.json) (the deep per-op dossiers) and emits:
  docs/operators/README.md   — the master operator x model matrix (signal + causal), the catalog index, the gaps.
  docs/operators/<op>.md     — one page per operator: its cross-model catalog row + its GPT-2 dossier (A-F).
The docs are GENERATED, not hand-written, so re-running after new survey/dossier runs keeps the catalog in sync.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt_signal(c):
    return f"{c['signal']:.2f}" if c.get("signal") is not None else "—"


def atlas_tables(atlas):
    ok = [r for r in atlas["results"] if "cells" in r]
    ops = atlas["operators"]; kinds = atlas["kinds"]
    head = "| operator class | kind | " + " | ".join(r["model"] for r in ok) + " |"
    sep = "|" + "---|" * (len(ok) + 2)
    sig = [head, sep]; cau = [head, sep]; memb = [head, sep]
    for op in ops:
        sig.append(f"| **{op}** | {kinds[op]} | " + " | ".join(f"{r['cells'][op]['signal']:.2f}" for r in ok) + " |")
        cau.append(f"| **{op}** | {kinds[op]} | " + " | ".join(f"{r['cells'][op]['causal_dNLL']:+.2f}" for r in ok) + " |")
        memb.append(f"| **{op}** | {kinds[op]} | " + " | ".join(f"{r['cells'][op]['n_heads']}" for r in ok) + " |")
    return ok, ops, kinds, "\n".join(sig), "\n".join(cau), "\n".join(memb)


def redundancy_verdict(rd):
    """Classify the cumulative-ablation curve: distributed (superadditive population) / bottleneck (one head ≈ op) /
    compensatory (non-monotonic — ablating the full set recovers vs the peak, i.e. self-repair among the heads)."""
    if not rd:
        return "—"
    curve = rd.get("curve") or []; full = rd.get("full", 0.0); ms = rd.get("max_solo", 0.0)
    peak = max((c["effect"] for c in curve), default=full)
    peak_n = next((c["n"] for c in curve if c["effect"] == peak), len(curve))
    if peak > 0.1 and full < 0.7 * peak:
        return f"**compensatory** (peak {peak:+.2f}@{peak_n}h → full {full:+.2f}; non-monotonic)"
    if full <= 1.4 * ms and ms > 0.1:
        return f"**bottleneck** (best 1h {ms:+.2f} ≈ full {full:+.2f})"
    return f"distributed (full {full:+.2f} ≫ best 1h {ms:+.2f})"


def xdossier_section(op, xrows):
    """The arch-generic cross-model deep dossier (identity + causal + channel) for a universal behavioural op."""
    if not xrows:
        return []
    lines = ["## Cross-model deep dossier (arch-generic) — `operator_dossier_xmodel.py`", "",
             "The deep battery's arch-generic core — behavioural head-ID + mean-ablation causal + the faithful "
             "key-only path-patch channel (the model re-applies its own RoPE) — run across **every** model, not just "
             "GPT-2. (The full A–F dossier below stays GPT-2-only: its channel/composition math is written against "
             "GPT-2's fused-QKV layout, and the named *output* ops have no published head-set off GPT-2.)", "",
             "| model | top head | #heads (mass≥thr) | causal induction ΔNLL | causal generic ΔNLL | redundancy (top heads) | KEY top writer (collapse) | VALUE top mover (ΔV-out) |",
             "|---|---|---|---|---|---|---|---|"]
    for r in xrows:
        ch = r["channel"]
        if "key_top" in ch:
            kc = f"{ch['key_top']['head']} ({ch['key_top']['collapse']:+.0%}, conc {ch.get('key_concentration', 0):.0f}×)"
            vc = f"{ch['value_top']['head']} ({ch['value_top']['dvout']:.2f})"
        else:
            kc = vc = "— (addresses by position/key-0)"
        red = redundancy_verdict(r.get("redundancy"))
        lines.append(f"| {r['model']} | {r['top_head']} | {r['n_heads_mass']} | {r['causal_induction_dNLL']:+.2f} | {r['causal_generic_dNLL']:+.2f} | {red} | {kc} | {vc} |")
    lines += ["", "_Mean-ablate the op's top behavioural heads → induction-NLL / generic-NLL damage; **redundancy** "
              "cumulative-ablates the top heads in solo-effect order (bottleneck = one head ≈ the whole op; distributed "
              "= the population far exceeds any single head; **compensatory** cases — which head triggers the recovery "
              "— are dug in [outlier mechanism digs](outlier_digs.md)); channel = remove "
              "each upstream head from the reader's key → top collapser + the value/move channel. "
              "Data: [xmodel_dossiers_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/xmodel_dossiers_summary.json). "
              "Regenerate: [operator_dossier_xmodel.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_dossier_xmodel.py)._", ""]
    return lines


def op_page(op, atlas, dossier, circuit_names=frozenset(), xrows=None):
    kinds = atlas["kinds"]; ok = [r for r in atlas["results"] if "cells" in r]
    lines = [f"# Operator `{op}`", ""]
    if op in circuit_names:
        lines += [f"> **`op:{op}`** — this is the **operator** (a head *class*: the family of heads that realize the "
                  f"operation). Not the [`{op}` *circuit*](../circuits/{op}.md), which is the *composition* "
                  f"(a writer-op feeding a reader-op) named after — and built around — this operator.", ""]
    if dossier and dossier.get("spec"):
        lines += [f"**{dossier['spec']['kind']}** — {dossier['spec']['desc']}", ""]
    elif op in kinds:
        lines += [f"**{kinds[op]}** operator (universal/addressing — measured across all models in the catalog).", ""]
    # cross-model catalog row (universal ops)
    if op in atlas["operators"]:
        nseed = atlas.get("seeds") or (atlas["results"][0].get("seeds") if atlas.get("results") else None)
        lines += ["## Cross-model (catalog row)" + (f" — signal/causal are mean ± σ over {nseed} probe-resample seeds" if nseed else ""), "",
                  "| model | arch | signal (±σ) | #heads | top head | depth | causal ΔNLL (±σ) |",
                  "|---|---|---|---|---|---|---|"]
        for r in ok:
            c = r["cells"][op]
            ss = f" ± {c['signal_std']:.3f}" if "signal_std" in c else ""
            cs = f" ± {c['causal_std']:.3f}" if "causal_std" in c else ""
            lines.append(f"| {r['model']} | {r['arch']} | {c['signal']:.3f}{ss} | {c['n_heads']} | {c['top_head']} | {c['top_depth']:.2f} | {c['causal_dNLL']:+.3f}{cs} |")
        lines.append("")
    else:
        heads = atlas.get("gpt2_circuit_ops", {}).get(op)
        if heads:
            lines += [f"GPT-2-only circuit op (literature DLA head-set): {', '.join(heads)}. No published head-set in the RoPE models — not in the cross-model catalog.", ""]
    # the arch-generic cross-model deep dossier (universal behavioural ops)
    lines += xdossier_section(op, xrows)
    # the deep dossier (GPT-2)
    if dossier:
        A = dossier.get("A_identity", {}); B = dossier.get("B_causal", {}); C = dossier.get("C_channels", {})
        D = dossier.get("D_composition", {}); E = dossier.get("E_redundancy", {}); Fx = dossier.get("F_cross_model")
        lines += [f"## Deep dossier (GPT-2) — `operator_dossier.py --op {op}`", ""]
        if A:
            ranked = ", ".join(f"{r['head']}" + (f" ({r['signal']:.2f})" if r.get("signal") is not None else "") for r in A.get("ranked", [])[:6])
            lines += [f"**A · identity** ({A.get('note','')}): heads {A.get('op_heads')}. ranked: {ranked}", ""]
        if B:
            eff = B.get("op_effects", {}); thr = B.get("threshold", {})
            cells = ", ".join(f"{t} {eff[t]:+.2f}{'*' if eff.get(t,0) > thr.get(t,9) else ''}" for t in B.get("tasks", []))
            lines += [f"**B · causal × tasks** (* = beyond random control): {cells}  → serves **{B.get('serves') or 'none'}**", ""]
        if C and "key_top" in C:
            kt = C["key_top"]; vt = C["value_top"]
            ptag = " (=prev-token head)" if C.get("key_top_is_prevtok_head") else ""
            lines += [f"**C · channels** (reader {C.get('reader')}): KEY/match top {kt['head']}{ptag} collapse {kt['collapse']:+.0%} "
                      f"(concentration {C.get('key_concentration',0):.1f}×); VALUE/move top {vt['head']} ΔV-out {vt['dvout']:.2f} (median {C.get('value_median',0):.2f}).", ""]
        elif C and C.get("note"):
            lines += [f"**C · channels**: {C['note']}", ""]
        elif op in (atlas.get("gpt2_circuit_ops", {})):
            lines += ["**C · channels**: output/circuit op — carried by OV→unembedding, not a key/value match (see composition out-edges).", ""]
        if D:
            ink = ", ".join(f"{n}({s:.3f})" for n, s in D.get("in_key", [])[:4])
            outv = ", ".join(f"{n}({s:.3f})" for n, s in D.get("out_value", [])[:4])
            lines += [f"**D · composition**: IN→key {ink or '—'}; OUT→value {outv or '—'}.", ""]
        if E:
            solo = ", ".join(f"{n}({e:+.2f})" for n, e in E.get("solo", []))
            curve = " → ".join(f"{c['n']}h {c['effect']:+.2f}" for c in E.get("cumulative_curve", []))
            verdict = "BOTTLENECK (one head ≈ whole op)" if E.get("bottleneck") else f"DISTRIBUTED population (full {E.get('full_op_effect',0):+.2f} ≫ best single {E.get('max_single_head_effect',0):+.2f})"
            lines += [f"**E · redundancy** (task `{E.get('primary_task')}`): solo {solo}; cumulative {curve} → {verdict}.", ""]
        if Fx:
            row = "; ".join(f"{c['model']} sig {c['behaviour_signal']:.2f}" + (f"/gain {c['gain']:+.1f}" if c.get('gain') is not None else "") for c in Fx if "error" not in c)
            lines += [f"**F · cross-model**: {row}", ""]
        if dossier.get("G_sae_operands"):
            lines += [f"**G · SAE operands**: {dossier['G_sae_operands']}", ""]
    lines += ["", f"_Data: `runs/disassembly/operators/dossiers/{op}/` + the catalog. Regenerate: [operator_catalog_doc.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_catalog_doc.py)._"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/disassembly/operators"))
    p.add_argument("--docs", type=Path, default=Path("docs/operators"))
    args = p.parse_args(argv)
    atlas = json.loads((args.root / "atlas_summary.json").read_text())
    args.docs.mkdir(parents=True, exist_ok=True)

    # the set of circuit names (the OTHER catalog) — any operator that shares a name with a circuit gets a
    # disambiguation banner + an `op:` qualifier in the index, so a reader never confuses the two.
    cir = json.loads((args.root.parent / "circuits" / "atlas_summary.json").read_text()) if (args.root.parent / "circuits" / "atlas_summary.json").exists() else {}
    circuit_names = frozenset(cir.get("cross_model_circuits", {})) | frozenset(cir.get("gpt2_circuits", {}))

    # arch-generic cross-model deep dossier (operator_dossier_xmodel.py): op -> [per-model record], in run order
    xd = json.loads((args.root / "xmodel_dossiers_summary.json").read_text()) if (args.root / "xmodel_dossiers_summary.json").exists() else {}
    xdoss = {}
    for r in xd.get("results", []):
        for op, rec in r.get("ops", {}).items():
            xdoss.setdefault(op, []).append({**rec, "model": r["model"]})

    def load_dossier(op):
        f = args.root / "dossiers" / op / "gpt2_summary.json"
        return json.loads(f.read_text()) if f.exists() else None

    all_ops = list(atlas["operators"]) + list(atlas.get("gpt2_circuit_ops", {}))
    for op in all_ops:
        (args.docs / f"{op}.md").write_text(op_page(op, atlas, load_dossier(op), circuit_names, xdoss.get(op)))

    # discovered-candidate dossiers: any dossiers/<op>/ not in the registered set (e.g. discovered_7.6)
    dossier_dir = args.root / "dossiers"
    discovered = sorted(d.name for d in dossier_dir.iterdir() if d.is_dir() and d.name not in all_ops and (d / "gpt2_summary.json").exists()) if dossier_dir.exists() else []
    for op in discovered:
        (args.docs / f"{op}.md").write_text(op_page(op, atlas, load_dossier(op), circuit_names, xdoss.get(op)))

    ok, ops, kinds, sig_tbl, cau_tbl, memb_tbl = atlas_tables(atlas)
    circuit = atlas.get("gpt2_circuit_ops", {})
    idx = "\n".join(f"- [`{op}`]({op}.md) — {kinds.get(op, 'circuit')}"
                    + (f" · **also a [circuit](../circuits/{op}.md)** (`op:{op}` here vs `circuit:{op}` there)" if op in circuit_names else "")
                    for op in all_ops)
    disc_idx = ("\n\n**Discovered-candidate dossiers** (UNNAMED load-bearing heads from the [discovery sweep](discovered.md), "
                "given the full battery): " + ", ".join(f"[`{op}`]({op}.md)" for op in discovered) + ".") if discovered else ""
    readme = f"""# Operator catalog — attention operators, surveyed across models

A **working catalog** of attention operators — amateur, exploratory home-science: provisional, descriptive, and
*not* a definitive reference (one of many catalogs one could draw).

## Catalog index

{idx}{disc_idx}

> **Operator vs circuit — a naming note.** A few names ({', '.join(sorted(n for n in all_ops if n in circuit_names)) or '—'}) appear in
> *both* this operator catalog and the [circuit catalog](../circuits/README.md). They are different objects at
> different levels: an **operator** is a *head class* (`op:induction`); the same-named **circuit** is the
> *composition* that feeds/anchors it (`circuit:induction` = prev-token → induction). The circuit is named after
> its **reader operator**, which is why the names coincide. Pages on either side cross-link to their namesake.

## How to read this catalog

Two axes:

> **Taxonomy — classes, instances, variants (read this first).** Each row below is an operator **CLASS**, *not* a
> single operator: it is a *family* of heads that realize the same operation. The **membership matrix** gives the
> head-count per class per model (e.g. GPT-2 has ~22 induction heads, ~31 prev-token heads, 117 heads with
> appreciable sink mass). Three levels of granularity:
> 1. **class** (these {len(ops)} universal rows + {len(circuit)} GPT-2 circuit classes) — the operation;
> 2. **instance** — an individual head realizing the class (the per-head listing is `disassemble_gpt2.py` →
>    [runs/disassembly/gpt2_disassembly.txt](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/gpt2_disassembly.txt); each dossier's section A lists the class's member heads);
> 3. **variant / sub-class** — structured differences *within* a class (e.g. induction's writer-branching, or
>    token- vs subword-name-completion inductors; the sink "class" is largely *content heads in their idle state* —
>    see `sink.md`). The dossiers (sections C/D/E) expose this intra-class structure.
>
> So the answer to "is it N operators or N classes?" is **classes** — the head counts are in the membership matrix.


- **Universal / addressing operators** (a position-or-token attention mask → measurable in *any* architecture):
  `{', '.join(ops)}`. The **catalog matrix** ({len(ok)} models) below is their cross-model survey.
- **GPT-2 circuit operators** (literature direct-logit-attribution head-sets, **no published head-set outside
  GPT-2**): `{', '.join(circuit)}`. Catalogued by their per-op dossiers (GPT-2), not the cross-model matrix.

Each operator has a **page**: the cross-model catalog row, then — for the universal behavioural ops — an
**arch-generic cross-model deep dossier** (behavioural head-ID + mean-ablation causal + key/value channel on
*every* model, via `operator_dossier_xmodel.py`), then the full **GPT-2 deep dossier** (identity / causal×tasks /
K-V channels / composition / redundancy / cross-model). The GPT-2 A–F battery stays GPT-2-only because its
channel/composition math is written against GPT-2's fused-QKV layout and the named *output* ops (name-movers,
S-inhibition) have no published head-set off GPT-2. Per-op data lives under `runs/disassembly/operators/`.

## Catalog — behavioural signal (max head mass on the op's pattern; *is the op present?*)

{sig_tbl}

## Catalog — membership (# heads carrying the op, mass > 0.15; *how many heads in the class?*)

{memb_tbl}

## Catalog — causal ΔNLL (mean-ablate top-3 heads, generic-prose NLL; *load-bearing on prose?*)

Note: this is **generic-prose** ΔNLL, so *task-specific* ops (induction, duplicate) read low here even though
they are load-bearing on their *own* task — see each op's dossier (section B) for the task-specific causal.

{cau_tbl}

## The other instruction class: COMPUTE (MLP)

Attention is the **MOVE** class; the **[MLP / COMPUTE catalog](mlp_compute.md)** is the other half of the
instruction set (cross-model per-layer MLP causal profile + the GPT-2 neuron read→write idioms). In the discovery
sweeps the early-MLP detokenizer had the largest single-component causal effect of anything measured.

## Growing the catalog: discovered components

The [**discovered components**](discovered.md) page is the discovery engine run across *every* model — every head +
MLP ranked by causal effect (multi-seed), flagged named-vs-**UNNAMED**. The UNNAMED load-bearing components are
candidate operators not yet catalogued (e.g. Llama heads 0.31/1.31/1.29) — the leads to dossier next. The strongest
RoPE candidates are profiled (causal + channel) on [**discovered candidates (cross-model)**](discovered_xmodel.md);
Llama **0.31** is induction-load-bearing **+7.99** (an early induction-enabling head, uncatalogued).

## Gaps (documented, not skipped)

- **succession / greater-than** — MLP-dominated; no clean attention head, so no catalog row (the OV probe sees only
  the attention-side shadow). It is carried by the *copy* ops (see `instruction_reuse.py`: successor ← induction/duplicate).
- **SSM (Mamba)** — no attention heads, so the head-resolved catalog does not apply; induction is present
  *behaviourally* (NLL gain) — see `ssm_induction.py`.

## How this was made

`operator_atlas.py` (the cross-model matrix) + `operator_dossier.py --op <name>` (the deep per-op dossiers) →
`operator_catalog_doc.py` regenerates these docs from the JSON artifacts. See [`../DECOMPILATION.md`](../DECOMPILATION.md).
"""
    (args.docs / "README.md").write_text(readme)
    print(f"[done] wrote {args.docs}/README.md + {len(all_ops)} operator pages: {all_ops}")


if __name__ == "__main__":
    main()
