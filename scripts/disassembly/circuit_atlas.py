"""The circuit ATLAS — composed multi-operator circuits, surveyed/collected across models.

Operators are single head-classes (`operator_atlas.py`); CIRCUITS are their *compositions* — a writer-op feeding a
reader-op's K/Q/V port, chained. This catalogs them the same way: a circuits × models matrix + an inventory that
*collects* every circuit we can identify with the tools here. Two sources, combined (exhaustive, no recompute of
what already exists):

  1. CROSS-MODEL circuit EDGES (recomputed across all models, arch-generic) — the defining composition edge of each
     universal-reader circuit, via the faithful key/value path-patch in `circuit_content_patch.run_model`:
       - INDUCTION circuit         : prev-token head --K--> induction head (does the induction reader's key collapse
                                      when its prev-token writer is removed? top collapser should BE the prev-tok head);
       - POSITIONAL-BROADCAST circuit: early sink/write-hub --K--> prev-token head's key (GPT-2-specific);
       - DUPLICATE circuit          : same-token reader (often layer-0; cross-model where a reader exists).
     Each also reports the VALUE/move channel (what the circuit moves).
  2. GPT-2 DISCOVERED / circuit-specific (HARVESTED from committed discovery artifacts, no recompute):
       - IOI Q-chain  duplicate -> S-inhibition -> name-mover  (`composition_dag` + `ioi_causal` z-scores);
       - V-composition "virtual heads" induction -> layer-6 values (`vcomposition`);
       - copy-suppression / negative movers, backup name-movers (`ioi_causal`);
       - the 22 NOVEL-LIVE write-hub edges discovered + named (`composition_dag`, `validate_new_edges`).
     These are GPT-2-only (literature IOI head-sets / GPT-2 path-patch runs) — flagged, not forced cross-model.

Output: `runs/disassembly/circuits/atlas_summary.json` + `atlas.png`. The companion `circuit_dossier`/doc-gen turn
this into per-circuit pages. Gaps documented: succession/greater-than = MLP-dominated; SSM = no heads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))


def harvest_gpt2(disasm_dir):
    """Pull the committed GPT-2 discovery artifacts into circuit records (no recompute)."""
    def load(name):
        f = disasm_dir / f"{name}_summary.json"
        return json.loads(f.read_text()) if f.exists() else {}
    cd = load("composition_dag"); vc = load("vcomposition"); io = load("ioi_causal"); ve = load("validate_new_edges")
    out = {}
    # IOI Q-chain: duplicate -> S-inhibition -> name-mover (composition_dag liveness + ioi_causal z)
    idioms = io.get("idioms", [])                                                  # list of {idiom, heads, delta_ld, z}
    zmap = {e.get("idiom"): e for e in idioms if isinstance(e, dict)}

    def zof(*names):
        for n in names:
            e = zmap.get(n)
            if isinstance(e, dict) and "z" in e:
                return e["z"]
        return None
    if cd:
        out["ioi_q_chain"] = {
            "kind": "Q-composition chain (GPT-2-only)", "scope": "gpt2",
            "stages": ["duplicate", "s_inhibition", "name_mover"],
            "writers": cd.get("inductors") and cd.get("s_inhibition"),
            "duplicate_heads": None, "s_inhibition": cd.get("s_inhibition"), "name_movers": cd.get("name_movers"),
            "q_live_edges": cd.get("ioi_q_live"), "chain_top": cd.get("ioi_chain_top"),
            "named_edge_live_rate": cd.get("named_edge_live_rate"),
            "ioi_baseline_logit_diff": io.get("baseline_logit_diff"),
            "name_mover_z": zof("copy_namemover", "name_mover"), "duplicate_z": zof("duplicate_token"),
            "s_inhibition_z": zof("s_inhibition"), "negative_mover_z": zof("negative_namemover", "copy_suppression"),
            "backup_namemover_z": zof("backup_namemover")}
        # induction K-chain (canonical writer) — also recomputed cross-model below; here the GPT-2 weight view
        out["induction_kchain_weights"] = {
            "kind": "K-composition (weight + path-patch, GPT-2)", "scope": "gpt2",
            "canonical_writer": cd.get("canonical_writer"), "inductors": cd.get("inductors"),
            "canonical_induction_live": cd.get("canonical_induction_live"),
            "canonical_induction_edges": cd.get("canonical_induction_edges"),
            "K_induction_static": cd.get("K_induction_static"), "K_random_baseline": cd.get("K_random_baseline"),
            "induction_top_rel_drop": cd.get("induction_top_rel_drop")}
        # discovered novel-live edges (the collection goal)
        nle = cd.get("novel_live_edges", [])
        out["discovered_write_hub_edges"] = {
            "kind": "DISCOVERED (novel-live composition edges)", "scope": "gpt2",
            "n_novel_live": cd.get("n_novel_live"), "edges": [e.get("edge") for e in nle][:22],
            "named_by_pattern": ve.get("named_by_pattern"), "n_named": ve.get("n_named"), "n_new_edges": ve.get("n_new_edges")}
    if vc:
        topv = [e for e in vc.get("edges", []) if e.get("kind") == "topV"][:8]
        out["v_virtual_heads"] = {
            "kind": "V-composition (composed-OV 'virtual heads', GPT-2)", "scope": "gpt2",
            "top_edges": [{"edge": f"{e['A']}->{e['B']}", "dvout": e.get("dvout"), "static": e.get("static")} for e in topv],
            "topV_median_dvout": vc.get("topV_median_dVout"), "spearman_staticV_vs_dVout": vc.get("spearman_staticV_vs_dVout"),
            "V_over_K_mean": vc.get("V_over_K_mean")}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--max-upstream", type=int, default=80)
    p.add_argument("--device", default="cuda")
    p.add_argument("--disasm-dir", type=Path, default=Path("runs/disassembly"))
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/circuits"))
    p.add_argument("--reuse", type=Path, default=Path("runs/gemma/circuit_content_patch_summary.json"),
                   help="if present and covers all --models, harvest instead of recomputing the cross-model edges")
    args = p.parse_args(argv)

    import circuit_content_patch as ccp

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    # ---- cross-model circuit edges (harvest the committed run if it covers the models, else recompute) ----
    reuse = json.loads(args.reuse.read_text()) if args.reuse.exists() else {"results": []}
    have = {r["model"]: r for r in reuse.get("results", []) if "circuits" in r}
    cross = []
    for mid in model_ids:
        if mid in have:
            cross.append(have[mid]); print(f"=== {mid} (harvested) ===")
            continue
        print(f"=== {mid} (recompute) ===")
        try:
            cross.append(ccp.run_model(mid, SimpleNamespace(ctx=args.ctx, chunks=args.chunks, max_upstream=args.max_upstream, device=args.device)))
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); cross.append({"model": mid, "error": str(e)})
    ok = [r for r in cross if "circuits" in r]

    # ---- assemble the cross-model circuit rows from the reader-circuit edges ----
    # circuit -> (reader op in ccp, what the defining edge is)
    def cell(r, cc):
        d = r["circuits"].get(cc, {})
        if "top_collapse" not in d:
            return {"reader": d.get("reader"), "skipped": True}
        return {"reader": d.get("reader"), "writer": d.get("key_top_head"), "edge_collapse": d.get("top_collapse"),
                "is_prevtok_writer": d.get("top_is_prevtok_head"), "is_sink_writer": d.get("top_is_sink"),
                "value_mover": d.get("value_top_head"), "value_dvout": d.get("value_top_dvout")}
    crossmodel_circuits = {
        "induction": {"desc": "prev-token head --K--> induction head (the in-context-copy macro)", "reader_cc": "induction",
                      "defining_edge": "prevtok_head -> induction (K)"},
        "positional_broadcast": {"desc": "early sink/write-hub --K--> prev-token head's key (absolute-position broadcast)",
                                 "reader_cc": "prevtok", "defining_edge": "sink-writer -> prevtok key (K)"},
        "duplicate": {"desc": "same-token reader (duplicate-token detection; IOI initiator)", "reader_cc": "duplicate",
                      "defining_edge": "(reader-side; writer often layer-0)"},
    }
    rows = {}
    for cname, spec in crossmodel_circuits.items():
        rows[cname] = {"scope": "cross-model", "desc": spec["desc"], "defining_edge": spec["defining_edge"],
                       "by_model": {r["model"].split("/")[-1]: cell(r, spec["reader_cc"]) for r in ok}}

    gpt2 = harvest_gpt2(args.disasm_dir)
    out = {"experiment": "circuit atlas — composed circuits x models (cross-model edges + GPT-2 discovered)",
           "models": [r["model"].split("/")[-1] for r in ok], "cross_model_circuits": rows, "gpt2_circuits": gpt2,
           "note_gaps": "succession/greater-than = MLP-dominated (no clean attention head); SSM = no heads.",
           "n_circuits_catalogued": len(rows) + len(gpt2)}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "atlas_summary.json").write_text(json.dumps(out, indent=2, default=float))

    # ---- print the survey ----
    print("\n[CIRCUIT ATLAS] cross-model circuit EDGE liveness (key-collapse %; writer head) per circuit x model:")
    mods = [r["model"].split("/")[-1] for r in ok]
    print(f"  {'circuit':>22} | " + " | ".join(f"{m[:11]:>11}" for m in mods))
    for cname, row in rows.items():
        cells = []
        for m in mods:
            c = row["by_model"].get(m, {})
            if c.get("skipped"):
                cells.append(f"{'(skip)':>11}")
            else:
                tag = "P" if c.get("is_prevtok_writer") else ("S" if c.get("is_sink_writer") else "")
                cells.append(f"{c.get('edge_collapse', 0):+.0%} {c.get('writer', '')}{tag}"[:11].rjust(11))
        print(f"  {cname:>22} | " + " | ".join(cells))
    print("  (writer tag in cell; P=prev-tok head, S=sink; value/move channel in the JSON)")
    print("\n[CIRCUIT ATLAS] GPT-2 discovered / circuit-specific (harvested — GPT-2-only):")
    for cname, c in gpt2.items():
        extra = (f"q_live {c.get('q_live_edges')}, name-mover z {c.get('name_mover_z')}" if cname == "ioi_q_chain" else
                 f"{c.get('n_novel_live')} novel-live, {c.get('n_named')} named" if cname == "discovered_write_hub_edges" else
                 f"top {c['top_edges'][0]['edge']} ΔV-out {c['top_edges'][0]['dvout']:.2f}" if cname == "v_virtual_heads" and c.get("top_edges") else
                 f"writer {c.get('canonical_writer')}, {c.get('canonical_induction_live')}/{c.get('canonical_induction_edges')} live" if cname == "induction_kchain_weights" else "")
        print(f"  {cname:>26} [{c['kind']}]: {extra}")
    print(f"\n[done] catalogued {out['n_circuits_catalogued']} circuits → {args.outdir / 'atlas_summary.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(mods) + 4), 3.2))
        M = np.array([[max(rows[c]["by_model"].get(m, {}).get("edge_collapse", 0) or 0, 0) for m in mods] for c in rows])
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=max(M.max(), 0.01))
        ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, fontsize=8, rotation=20, ha="right")
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(list(rows), fontsize=9)
        for i, c in enumerate(rows):
            for j, m in enumerate(mods):
                cc = rows[c]["by_model"].get(m, {})
                txt = "skip" if cc.get("skipped") else f"{(cc.get('edge_collapse') or 0):+.0%}\n{cc.get('writer','')}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="w" if M[i, j] < 0.5 * max(M.max(), .01) else "k")
        fig.colorbar(im, ax=ax, fraction=0.04, label="defining-edge key collapse")
        ax.set_title("Circuit atlas — cross-model composition-edge liveness (GPT-2 discovered circuits in JSON)", fontsize=10)
        fig.tight_layout(); fig.savefig(args.outdir / "atlas.png", dpi=130); print(f"[fig] {args.outdir / 'atlas.png'}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    return out


if __name__ == "__main__":
    main()
