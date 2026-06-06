"""Coverage scorecard — what fraction of GPT-2's attention does the op/idiom catalog actually explain?

The disassembly thread has built channels piecemeal (content-opcode B_h, relative-Δ / absolute-sink /
structural addressing, OV-write, the idiom library). No published work gives a SINGLE number for how
COMPLETE that catalog is. This unifies them into one accounting.

Per head, over a corpus, every attention edge (query t -> key s, s<=t) is assigned by PRIORITY to one bucket:

  self        s == t                      (degenerate / value-carry)
  sink        s == 0                      (BOS attention-sink / no-op)
  prev        t - s == 1                  (previous-token addressing)
  structural  key s is \n / . , ; : ! ?   (sentence-structure addressing)
  local       2 <= t - s <= 8             (local window)
  long_range  everything else             (the content-addressing candidate)

A head's mass is EXPLAINED if it lands in {self, sink, prev, structural, local} (named positional/
structural plumbing), OR it is long_range AND the head carries a content mechanism — a validated idiom
(induction / name-mover / copy-suppression / S-inhibition, from idiom_library_v2_summary.json), a
behaviorally-legible TOKEN-operand binding (QK off-diagonal z>2, from qk_opcode_table_summary.json), or a
legible SAE-FEATURE-operand binding (from sae_opcode_table_summary.json — credits the dark heads that the
token basis missed but SAE features resolved). Because
the SINK dominates the attention budget, a single "% explained" is misleadingly high; the honest target is
the LONG-RANGE (content) mass, split three ways — named-by-idiom / weight-legible-but-unnamed / DARK. We
report a STRICT coverage (plumbing + named idioms only) and a LEGIBLE coverage (+ legible bindings), and
rank the DARK HEADS (highest long_range, no idiom + not legible) as the explicit work-list for new idioms.

GPT-2; one forward pass for the mass decomposition + the two existing channel summaries. CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONTENT_IDIOMS = {"induction", "copy_namemover", "backup_namemover", "negative_namemover",
                  "copy_suppression", "s_inhibition", "coreference"}
STRUCT_STR = {".", ",", ";", ":", "!", "?", "Ċ", "\n", "ĊĊ"}


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--local-max", type=int, default=8, help="upper Δ for the 'local' window")
    p.add_argument("--leg-z", type=float, default=2.0, help="content-legibility z to credit long-range mass")
    p.add_argument("--idioms", type=Path, default=Path("runs/idiom_library_v2_summary.json"))
    p.add_argument("--opcodes", type=Path, default=Path("runs/qk_opcode_table_summary.json"),
                   help="token-operand opcode table (content-legibility credit)")
    p.add_argument("--sae-opcodes", type=Path, default=Path("runs/sae_opcode_table_summary.json"),
                   help="SAE-feature opcode table (additional content credit where it ran)")
    p.add_argument("--output", type=Path, default=Path("runs/coverage_scorecard_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval()
    tr = model.transformer
    cfg = model.config
    H = cfg.n_head; nL = cfg.n_layer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]

    buckets = ["self", "sink", "prev", "structural", "local", "long_range"]
    acc = np.zeros((nL, H, len(buckets)))
    tot = np.zeros((nL, H))
    with torch.no_grad():
        for c in chunks:
            Lc = len(c)
            toks = tok.convert_ids_to_tokens(c)
            struct = np.array([any(s in t for s in STRUCT_STR) or t in STRUCT_STR for t in toks])
            qi = np.arange(Lc)
            S, T = np.meshgrid(qi, qi)                      # S=key index, T=query index; valid where S<=T
            delta = T - S
            # priority-assigned bucket id per (query T, key S)
            bid = np.full((Lc, Lc), 5, dtype=int)           # default long_range
            bid[(delta >= 2) & (delta <= args.local_max)] = 4   # local
            bid[struct[None, :].repeat(Lc, 0).astype(bool)] = 3  # structural (key is a delimiter)
            bid[delta == 1] = 2                             # prev
            bid[S == 0] = 1                                 # sink
            bid[S == T] = 0                                 # self
            bid[S > T] = -1                                 # invalid (upper triangle), excluded
            o = tr(input_ids=torch.tensor([c]), output_attentions=True)
            for L in range(nL):
                a = o.attentions[L][0].float().numpy()       # (H, Lc, Lc)
                tot[L] += a.sum((1, 2))
                for b in range(len(buckets)):
                    acc[L, :, b] += (a * (bid == b)[None]).sum((1, 2))
    frac = acc / np.maximum(tot, 1e-9)[:, :, None]            # (nL, H, 6) per-head mass fractions

    # ---- content-mechanism overlay from the existing channel summaries ----
    idi = _load(args.idioms) or {}
    per_head_idioms = idi.get("per_head_idioms", {})
    opc = _load(args.opcodes) or {}
    leg = {f"{h['layer']}.{h['head']}": float(h.get("leg_z", 0.0)) for h in opc.get("heads", [])}
    # SAE-operand legibility (best of dominant / content-weighted), where the SAE opcode table ran.
    sae = _load(args.sae_opcodes) or {}

    def _z(h, *keys):
        vals = [h.get(k) for k in keys]
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]  # drop None/NaN
        return max(vals) if vals else float("-inf")
    sae_leg = {h["head"]: _z(h, "z_dominant", "z_content", "z_sae") for h in sae.get("heads", [])}

    def has_content(L, h):
        tags = per_head_idioms.get(f"{L}.{h}", [])
        if any(t in CONTENT_IDIOMS for t in tags):
            return "idiom"
        if leg.get(f"{L}.{h}", 0.0) > args.leg_z:
            return "legible-content"
        if sae_leg.get(f"{L}.{h}", float("-inf")) > args.leg_z:
            return "sae-legible"
        return None

    rows = []
    bi = {b: i for i, b in enumerate(buckets)}
    for L in range(nL):
        for h in range(H):
            f = frac[L, h]
            lr = float(f[bi["long_range"]])
            src = has_content(L, h)                          # 'idiom' | 'legible-content' | None
            plumbing = float(f.sum() - lr)                   # self+sink+prev+structural+local
            dom = buckets[int(np.argmax(f))]
            rows.append({"head": f"{L}.{h}", "layer": L, "dominant": dom,
                         "self": float(f[0]), "sink": float(f[1]), "prev": float(f[2]),
                         "structural": float(f[3]), "local": float(f[4]), "long_range": lr,
                         "plumbing": plumbing, "content_src": src,
                         "idioms": per_head_idioms.get(f"{L}.{h}", []), "leg_z": leg.get(f"{L}.{h}", 0.0)})

    budget = {b: float(np.mean(frac[:, :, bi[b]])) for b in buckets}
    plumbing_frac = float(np.mean([r["plumbing"] for r in rows]))
    # split the long-range (content-candidate) mass by how it is credited
    def _lr(src):
        return float(np.mean([r["long_range"] if r["content_src"] == src else 0.0 for r in rows]))
    lr_named = _lr("idiom")
    lr_legible = _lr("legible-content")              # token-operand B_h legible
    lr_sae = _lr("sae-legible")                      # only the SAE-operand B_h made it legible
    lr_dark = float(np.mean([r["long_range"] if r["content_src"] is None else 0.0 for r in rows]))
    lr_total = budget["long_range"]
    cov_named = plumbing_frac + lr_named
    cov_legible = plumbing_frac + lr_named + lr_legible
    cov_sae = cov_legible + lr_sae                   # + heads only the SAE operand basis explains
    dark = [r for r in sorted(rows, key=lambda r: -r["long_range"]) if r["content_src"] is None][:15]

    # per-idiom named-content contribution: footprint (overlapping) + exclusive (named ONLY by this idiom)
    nheads = len(rows)
    named_rows = [r for r in rows if r["content_src"] == "idiom"]
    idiom_fp, idiom_excl = {}, {}
    for r in named_rows:
        ctags = [t for t in r["idioms"] if t in CONTENT_IDIOMS]
        for t in ctags:
            idiom_fp[t] = idiom_fp.get(t, 0.0) + r["long_range"] / nheads
        if len(ctags) == 1:
            idiom_excl[ctags[0]] = idiom_excl.get(ctags[0], 0.0) + r["long_range"] / nheads
    idiom_breakdown = {t: {"footprint": idiom_fp.get(t, 0.0), "exclusive": idiom_excl.get(t, 0.0)}
                       for t in sorted(idiom_fp, key=lambda k: -idiom_fp[k])}

    out = {"experiment": "attention coverage scorecard", "model": args.pretrained, "n_heads": nL * H,
           "attention_budget": budget, "plumbing_frac": plumbing_frac,
           "long_range_total": lr_total, "long_range_named": lr_named,
           "long_range_legible_unnamed": lr_legible, "long_range_sae_only": lr_sae,
           "long_range_dark": lr_dark, "coverage_named": cov_named, "coverage_legible": cov_legible,
           "coverage_with_sae": cov_sae, "named_by_idiom": idiom_breakdown,
           "dark_heads": [{"head": r["head"], "long_range": r["long_range"], "dominant": r["dominant"],
                           "leg_z": r["leg_z"], "idioms": r["idioms"]} for r in dark],
           "heads": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    print(f"{args.pretrained}: attention coverage scorecard over {nL*H} heads")
    print("\n[attention budget] mean per-head mass by bucket:")
    for b in buckets:
        mark = "  <- the no-op SINK" if b == "sink" else ("  <- content candidate" if b == "long_range" else "")
        print(f"  {b:11} {budget[b]:6.1%}{mark}")
    print(f"\n  plumbing (self+sink+prev+structural+local) = {plumbing_frac:.1%} of all attention")
    print(f"\n[long-range content = {lr_total:.1%} of attention] split:")
    print(f"  named by a validated idiom    {lr_named/lr_total:6.1%} of long-range  ({lr_named:.1%} abs)")
    print(f"  token-operand B_h legible     {lr_legible/lr_total:6.1%} of long-range  ({lr_legible:.1%} abs)")
    print(f"  SAE-operand B_h legible only  {lr_sae/lr_total:6.1%} of long-range  ({lr_sae:.1%} abs)")
    print(f"  DARK (no channel explains)    {lr_dark/lr_total:6.1%} of long-range  ({lr_dark:.1%} abs)")
    if idiom_breakdown and lr_named > 0:
        print("\n[named-by-idiom] each idiom's share of long-range content "
              "(footprint = heads it tags; exclusive = mass named ONLY by it):")
        for t, v in idiom_breakdown.items():
            print(f"  {t:18} footprint {v['footprint']/lr_total:5.1%}  exclusive {v['exclusive']/lr_total:5.1%}")
    print(f"\n[COVERAGE]  named-idiom: {cov_named:.1%}  |  +token-legible: {cov_legible:.1%}  "
          f"|  +SAE-operand: {cov_sae:.1%}")
    if lr_sae > 0:
        print(f"  -> SAE operands shrink dark long-range mass by {lr_sae/max(lr_sae+lr_dark,1e-9):.0%} "
              f"(where the SAE opcode table ran)")
    print("  (most 'explained' mass is the positional/structural sink, not named circuits —")
    print("   the honest target is the long-range split above.)")
    print("\n[dark heads] highest long-range mass with NO channel explaining it (work-list):")
    print(f"  {'head':>6} {'long_rng':>8} {'dominant':>11} {'leg_z':>6}")
    for r in dark:
        print(f"  {r['head']:>6} {r['long_range']:>8.1%} {r['dominant']:>11} {r['leg_z']:>6.2f}")
    if not idi:
        print("\n[warn] idiom summary missing — run idiom_library_v2.py first (no idiom credit applied)")
    if not opc:
        print("[warn] opcode summary missing — run qk_opcode_table.py first (no legibility credit)")
    if not sae:
        print("[warn] SAE opcode summary missing — run sae_opcode_table.py for SAE-operand credit")
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
