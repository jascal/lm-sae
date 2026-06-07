"""Render the committed per-head disassembly listings as site pages, with operator tags hyperlinked.

The unified per-head listings (`disassemble_gpt2.py` / `disassemble_gemma.py`) live as committed text under
`docs/listings/`. This turns each into a browsable `docs/disassembly/<model>.md` page: the per-head data is kept
monospace (a code span per line, preserving the addressing / write / binding columns) and every trailing operator
**role tag** (`[induction]`, `[prev-tok→induction-feed]`, `[line-anchor]`, …) is hyperlinked to its
[operator catalog](../operators/README.md) page. CPU-only — reads the committed `.txt`, no model needed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# operator role-tag (as it appears in the listings) -> operator catalog page slug
ROLE_OP = {
    "induction": "induction", "prev-tok→induction-feed": "prevtok", "prev-token": "prevtok", "prevtok": "prevtok",
    "line-anchor": "structural", "structural": "structural", "duplicate": "duplicate", "duplicate-token": "duplicate",
    "name-mover": "name_mover", "name_mover": "name_mover", "copy/name-mover": "name_mover",
    "backup-name-mover": "backup_name_mover", "backup name-mover": "backup_name_mover",
    "negative-name-mover": "negative_mover", "copy-suppression": "negative_mover", "negative": "negative_mover",
    "s-inhibition": "s_inhibition", "s_inhibition": "s_inhibition", "sink": "sink", "self": "self", "local": "local",
    "coreference": "coreference",
}
# (listing file, page slug, title, arch tag, discovery_summary model key)
LISTINGS = [
    ("gpt2_disassembly.txt", "gpt2", "GPT-2 (small)", "GPT-2 / absolute-position", "gpt2"),
    ("gemma2_disassembly.txt", "gemma-2-2b", "Gemma-2-2B", "RoPE / GQA / RMSNorm", "gemma-2-2b"),
    ("llama32_1b_disassembly.txt", "llama-3.2-1b", "Llama-3.2-1B", "RoPE / GQA / RMSNorm", "Llama-3.2-1B"),
    ("qwen25_15b_disassembly.txt", "qwen-2.5-1.5b", "Qwen-2.5-1.5B", "RoPE / GQA / RMSNorm", "Qwen2.5-1.5B"),
]
TAG = re.compile(r"\[([^\]]+)\]")
HEAD_RE = re.compile(r"^\s*(?:L(\d+)\.H(\d+)|(\d+)\.(\d+))\s")   # GPT-2 'L5.H1' or RoPE '5.1'
MLP_RE = re.compile(r"^\s*L?(\d+)\.MLP")


def load_discovery(path):
    """model-key -> {comps: {'L.H'|'mlpL': rec}, base_induction, candidates, n_unnamed} from the discovery sweep."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    by_model = {}
    for r in d.get("results", []):
        comps = {}
        for c in (r.get("top_components", []) + r.get("candidate_unnamed", [])):
            key = f"mlp{c['L']}" if c.get("kind") == "mlp" else c["comp"]
            comps[key] = c
        by_model[r["model"]] = {"comps": comps, "base_induction": r.get("base_induction"),
                                "candidates": r.get("candidate_unnamed", []), "n_unnamed": r.get("n_unnamed_load_bearing"),
                                "seeds": r.get("seeds")}
    return by_model


def causal_badge(rec, base_ind):
    ind = rec.get("induction_dNLL_mean"); std = rec.get("induction_dNLL_std", 0.0); gen = rec.get("generic_dNLL")
    pct = f", {ind / base_ind:.0%} of base" if base_ind else ""
    g = f", generic {gen:+.2f}" if gen is not None else ""
    flag = " ⚠ **UNNAMED-candidate**" if not rec.get("named") else ""
    return f" — ★ causal: induction ΔNLL **{ind:+.2f}**±{std:.2f}{pct}{g}{flag}"


def link_role(tag):
    op = ROLE_OP.get(tag.strip())
    return f"[`{tag}`](../operators/{op}.md)" if op else f"`[{tag}]`"


def render(txt, disc=None):
    lines_in = txt.splitlines()
    intro = [ln[1:].strip() for ln in lines_in if ln.startswith(";")]
    used = set()
    out = []
    comps = (disc or {}).get("comps", {}); base_ind = (disc or {}).get("base_induction")
    matched = set(); mlp_badged = set()
    for ln in lines_in:
        s = ln.rstrip()
        if not s or s.startswith(";"):
            continue
        m = re.match(r"^---\s*layer\s+(\d+)\s*---$", s.strip())
        if m:
            out.append(f"\n### Layer {m.group(1)}\n"); continue
        tags = TAG.findall(s)
        for t in tags:
            if t.strip() in ROLE_OP:
                used.add(ROLE_OP[t.strip()])
        prefix = s.split("[", 1)[0].rstrip() if tags else s
        prefix = re.sub(r"\bidioms\s*$", "", prefix).rstrip()   # RoPE listings write 'idioms[tag]'; drop the dangling label
        linked = " ".join(link_role(t) for t in tags)
        # discovery-pass causal overlay: badge any head/MLP the sweep ranked as load-bearing
        badge = ""
        hm = HEAD_RE.match(s)
        if hm and ".MLP" not in s:
            key = f"{hm.group(1) or hm.group(3)}.{hm.group(2) or hm.group(4)}"
            if key in comps:
                badge = causal_badge(comps[key], base_ind); matched.add(key)
        else:
            mm = MLP_RE.match(s)
            if mm:
                key = f"mlp{mm.group(1)}"
                if key in comps and key not in mlp_badged:
                    badge = causal_badge(comps[key], base_ind); matched.add(key); mlp_badged.add(key)
        out.append(f"- `{prefix.strip()}`" + (f" {linked}" if linked else "") + badge)
    return intro, out, used, matched


def discovery_callout(disc):
    """A top-of-page block surfacing the discovery sweep's causal results for this model."""
    if not disc:
        return []
    cands = disc.get("candidates", []); bi = disc.get("base_induction"); ns = disc.get("seeds")
    clist = ", ".join(f"`{c['comp']}`" for c in cands[:12]) or "—"
    return ["> **Discovery pass (causal overlay).** The ★ badges below are from the cross-model discovery sweep "
            f"([discovered components](../operators/discovered.md), {ns or '?'}-seed): every head/MLP mean-ablated and "
            "ranked by its **induction-NLL** damage" + (f" (base induction NLL {bi:.2f})" if bi else "") + ". A head "
            f"is flagged **⚠ UNNAMED-candidate** when it is load-bearing but matches no catalogued operator — a lead to "
            f"dossier. **{disc.get('n_unnamed', 0)} unnamed load-bearing** here: {clist}. "
            "Only the sweep's top-ranked components carry a badge (most heads are not individually load-bearing).", ""]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listings", type=Path, default=Path("docs/listings"))
    p.add_argument("--docs", type=Path, default=Path("docs/disassembly"))
    p.add_argument("--discovered", type=Path, default=Path("runs/disassembly/operators/discovered_summary.json"))
    args = p.parse_args(argv)
    args.docs.mkdir(parents=True, exist_ok=True)
    disc_by_model = load_discovery(args.discovered)
    index_rows = []
    for fname, slug, title, arch, dkey in LISTINGS:
        f = args.listings / fname
        if not f.exists():
            print(f"[skip] {f} missing"); continue
        disc = disc_by_model.get(dkey)
        intro, body, used, matched = render(f.read_text(), disc)
        legend = " · ".join(f"[{op}](../operators/{op}.md)" for op in sorted(used)) or "—"
        page = [f"---\ntitle: {title} disassembly\n---", "",
                f"# {title} — per-head disassembly", "", f"**{arch}.** " + (" ".join(intro) if intro else ""), "",
                f"Operator roles referenced (hyperlinked inline below): {legend}. "
                f"Full raw listing: [`{fname}`](https://github.com/jascal/lm-sae/blob/main/docs/listings/{fname}). "
                f"See the [operator catalog](../operators/README.md) for what each role means.", ""]
        page += discovery_callout(disc)
        page += ["_First-order, single-component reads (+ the induction idiom); provisional. Each head line: head · "
                "ADDR (where it reads) · WRITE (copy/transform) · top content binding · operator role" +
                (" · ★ discovery-pass causal (when load-bearing)" if matched else "") + ". "
                "Lines like `L.MLP.n####` are **MLP neurons** (the COMPUTE class — `n####` is the neuron's index in "
                "that layer's gated-MLP intermediate dimension, e.g. Gemma-2-2B has 9216/layer), **not** attention "
                "heads; each lists the top read-tokens → write-tokens (the layer's most salient few)._", ""]
        page += body
        page += ["", "_Generated from the committed listing + discovery sweep by `disassembly_pages.py`._"]
        (args.docs / f"{slug}.md").write_text("\n".join(page))
        index_rows.append((slug, title, arch, len(used), len(matched)))
        print(f"  wrote docs/disassembly/{slug}.md ({len(body)} lines, {len(used)} roles linked, {len(matched)} causal badges)")

    idx = ["# Per-head disassemblies", "",  # no front-matter, so GitHub Pages' jekyll-readme-index promotes this to /disassembly/
           "The unified per-head listing for each model — every attention head's **addressing** (where it reads) × "
           "**write** (copy/transform) × top content binding × **operator role**, plus the per-layer MLP read→write "
           "neurons. Operator-role tags are hyperlinked to the [operator catalog](../operators/README.md), and "
           "load-bearing heads/MLPs carry a ★ **discovery-pass causal** badge (induction-NLL damage, from the "
           "[discovered-components sweep](../operators/discovered.md)). "
           "First-order and provisional (single-component reads + the induction idiom).", "",
           "| model | architecture | operator roles linked | causal badges |", "|---|---|---|---|"]
    for slug, title, arch, n, nb in index_rows:
        idx.append(f"| [{title}]({slug}.md) | {arch} | {n} | {nb} |")
    idx += ["", "_The deeper composition / circuit story is in the [circuit catalog](../circuits/README.md) and "
            "[DISASSEMBLY.md](../DISASSEMBLY.md). Generated by `disassembly_pages.py`._"]
    (args.docs / "README.md").write_text("\n".join(idx))
    print(f"[done] {len(index_rows)} disassembly pages + index → {args.docs}/")


if __name__ == "__main__":
    main()
