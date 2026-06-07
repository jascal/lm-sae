"""Generate the operator-catalog docs from the JSON artifacts — the survey as browsable markdown.

Reads `runs/disassembly/operators/atlas_summary.json` (the cross-model matrix) + each
`runs/disassembly/operators/dossiers/<op>/<model>.json` (the deep per-op dossiers) and emits:
  docs/operators/README.md   — the master operator x model matrix (signal + causal), the catalog index, the gaps.
  docs/operators/<op>.md     — one page per operator: its cross-model atlas row + its GPT-2 dossier (A-F).
The docs are GENERATED, not hand-written, so re-running after new atlas/dossier runs keeps the catalog in sync.
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


def op_page(op, atlas, dossier):
    kinds = atlas["kinds"]; ok = [r for r in atlas["results"] if "cells" in r]
    lines = [f"# Operator `{op}`", ""]
    if dossier and dossier.get("spec"):
        lines += [f"**{dossier['spec']['kind']}** — {dossier['spec']['desc']}", ""]
    elif op in kinds:
        lines += [f"**{kinds[op]}** operator (universal/addressing — measured across all models in the atlas).", ""]
    # cross-model atlas row (universal ops)
    if op in atlas["operators"]:
        lines += ["## Cross-model (atlas row)", "",
                  "| model | arch | signal | #heads | top head | depth | causal ΔNLL |",
                  "|---|---|---|---|---|---|---|"]
        for r in ok:
            c = r["cells"][op]
            lines.append(f"| {r['model']} | {r['arch']} | {c['signal']:.3f} | {c['n_heads']} | {c['top_head']} | {c['top_depth']:.2f} | {c['causal_dNLL']:+.3f} |")
        lines.append("")
    else:
        heads = atlas.get("gpt2_circuit_ops", {}).get(op)
        if heads:
            lines += [f"GPT-2-only circuit op (literature DLA head-set): {', '.join(heads)}. No published head-set in the RoPE models — not in the cross-model atlas.", ""]
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
    lines += ["", f"_Data: `runs/disassembly/operators/dossiers/{op}/` + the atlas. Regenerate: `operator_catalog_doc.py`._"]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/disassembly/operators"))
    p.add_argument("--docs", type=Path, default=Path("docs/operators"))
    args = p.parse_args(argv)
    atlas = json.loads((args.root / "atlas_summary.json").read_text())
    args.docs.mkdir(parents=True, exist_ok=True)

    def load_dossier(op):
        f = args.root / "dossiers" / op / "gpt2_summary.json"
        return json.loads(f.read_text()) if f.exists() else None

    all_ops = list(atlas["operators"]) + list(atlas.get("gpt2_circuit_ops", {}))
    for op in all_ops:
        (args.docs / f"{op}.md").write_text(op_page(op, atlas, load_dossier(op)))

    ok, ops, kinds, sig_tbl, cau_tbl, memb_tbl = atlas_tables(atlas)
    circuit = atlas.get("gpt2_circuit_ops", {})
    idx = "\n".join(f"- [`{op}`]({op}.md) — {kinds.get(op, 'circuit')}" for op in all_ops)
    readme = f"""# Operator catalog — the attention instruction set, surveyed across models

This is the **catalog** of GPT-2-family attention operators, measured exhaustively. Two axes:

> **Taxonomy — classes, instances, variants (read this first).** Each row below is an operator **CLASS**, *not* a
> single operator: it is a *family* of heads that realize the same operation. The **membership matrix** gives the
> head-count per class per model (e.g. GPT-2 has ~22 induction heads, ~31 prev-token heads, 117 heads with
> appreciable sink mass). Three levels of granularity:
> 1. **class** (these {len(ops)} universal rows + {len(circuit)} GPT-2 circuit classes) — the operation;
> 2. **instance** — an individual head realizing the class (the per-head listing is `disassemble_gpt2.py` →
>    `runs/disassembly/gpt2_disassembly.txt`; each dossier's section A lists the class's member heads);
> 3. **variant / sub-class** — structured differences *within* a class (e.g. induction's writer-branching, or
>    token- vs subword-name-completion inductors; the sink "class" is largely *content heads in their idle state* —
>    see `sink.md`). The dossiers (sections C/D/E) expose this intra-class structure.
>
> So the answer to "is it N operators or N classes?" is **classes** — the head counts are in the membership matrix.


- **Universal / addressing operators** (a position-or-token attention mask → measurable in *any* architecture):
  `{', '.join(ops)}`. The **atlas** ({len(ok)} models) below is their cross-model survey.
- **GPT-2 circuit operators** (literature direct-logit-attribution head-sets, **no published head-set outside
  GPT-2**): `{', '.join(circuit)}`. Catalogued by their per-op dossiers (GPT-2), not the cross-model matrix.

Each operator has a **page** (cross-model atlas row + the deep GPT-2 dossier: identity / causal×tasks / K-V
channels / composition / redundancy / cross-model). Per-op data lives under `runs/disassembly/operators/`.

## Atlas — behavioural signal (max head mass on the op's pattern; *is the op present?*)

{sig_tbl}

## Atlas — membership (# heads carrying the op, mass > 0.15; *how many heads in the class?*)

{memb_tbl}

## Atlas — causal ΔNLL (mean-ablate top-3 heads, generic-prose NLL; *load-bearing on prose?*)

Note: this is **generic-prose** ΔNLL, so *task-specific* ops (induction, duplicate) read low here even though
they are load-bearing on their *own* task — see each op's dossier (section B) for the task-specific causal.

{cau_tbl}

## Catalog index

{idx}

## Gaps (documented, not skipped)

- **succession / greater-than** — MLP-dominated; no clean attention head, so no atlas row (the OV probe sees only
  the attention-side shadow). It is carried by the *copy* ops (see `instruction_reuse.py`: successor ← induction/duplicate).
- **SSM (Mamba)** — no attention heads, so the head-resolved atlas does not apply; induction is present
  *behaviourally* (NLL gain) — see `ssm_induction.py`.

## How this was made

`operator_atlas.py` (the cross-model matrix) + `operator_dossier.py --op <name>` (the deep per-op dossiers) →
`operator_catalog_doc.py` regenerates these docs from the JSON artifacts. See [`../DECOMPILATION.md`](../DECOMPILATION.md).
"""
    (args.docs / "README.md").write_text(readme)
    print(f"[done] wrote {args.docs}/README.md + {len(all_ops)} operator pages: {all_ops}")


if __name__ == "__main__":
    main()
