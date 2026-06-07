"""Render the committed per-head disassembly listings as site pages, with operator tags hyperlinked.

The unified per-head listings (`disassemble_gpt2.py` / `disassemble_gemma.py`) live as committed text under
`docs/listings/`. This turns each into a browsable `docs/disassembly/<model>.md` page: the per-head data is kept
monospace (a code span per line, preserving the addressing / write / binding columns) and every trailing operator
**role tag** (`[induction]`, `[prev-tok→induction-feed]`, `[line-anchor]`, …) is hyperlinked to its
[operator catalog](../operators/README.md) page. CPU-only — reads the committed `.txt`, no model needed.
"""
from __future__ import annotations

import argparse
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
LISTINGS = [
    ("gpt2_disassembly.txt", "gpt2", "GPT-2 (small)", "GPT-2 / absolute-position"),
    ("gemma2_disassembly.txt", "gemma-2-2b", "Gemma-2-2B", "RoPE / GQA / RMSNorm"),
    ("llama32_1b_disassembly.txt", "llama-3.2-1b", "Llama-3.2-1B", "RoPE / GQA / RMSNorm"),
    ("qwen25_15b_disassembly.txt", "qwen-2.5-1.5b", "Qwen-2.5-1.5B", "RoPE / GQA / RMSNorm"),
]
TAG = re.compile(r"\[([^\]]+)\]")


def link_role(tag):
    op = ROLE_OP.get(tag.strip())
    return f"[`{tag}`](../operators/{op}.md)" if op else f"`[{tag}]`"


def render(txt):
    lines_in = txt.splitlines()
    intro = [ln[1:].strip() for ln in lines_in if ln.startswith(";")]
    used = set()
    out = []
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
        linked = " ".join(link_role(t) for t in tags)
        out.append(f"- `{prefix.strip()}`" + (f" {linked}" if linked else ""))
    return intro, out, used


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listings", type=Path, default=Path("docs/listings"))
    p.add_argument("--docs", type=Path, default=Path("docs/disassembly"))
    args = p.parse_args(argv)
    args.docs.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for fname, slug, title, arch in LISTINGS:
        f = args.listings / fname
        if not f.exists():
            print(f"[skip] {f} missing"); continue
        intro, body, used = render(f.read_text())
        legend = " · ".join(f"[{op}](../operators/{op}.md)" for op in sorted(used)) or "—"
        page = [f"---\ntitle: {title} disassembly\n---", "",
                f"# {title} — per-head disassembly", "", f"**{arch}.** " + (" ".join(intro) if intro else ""), "",
                f"Operator roles referenced (hyperlinked inline below): {legend}. "
                f"Full raw listing: [`{fname}`](https://github.com/jascal/lm-sae/blob/main/docs/listings/{fname}). "
                f"See the [operator catalog](../operators/README.md) for what each role means.", "",
                "_First-order, single-component reads (+ the induction idiom); provisional. Each line: head · "
                "ADDR (where it reads) · WRITE (copy/transform) · top content binding · operator role._", ""]
        page += body
        page += ["", "_Generated from the committed listing by `disassembly_pages.py`._"]
        (args.docs / f"{slug}.md").write_text("\n".join(page))
        index_rows.append((slug, title, arch, len(used)))
        print(f"  wrote docs/disassembly/{slug}.md ({len(body)} lines, {len(used)} operator roles linked)")

    idx = ["---\ntitle: Head disassemblies\n---", "",
           "# Per-head disassemblies", "",
           "The unified per-head listing for each model — every attention head's **addressing** (where it reads) × "
           "**write** (copy/transform) × top content binding × **operator role**, plus the per-layer MLP read→write "
           "neurons. Operator-role tags are hyperlinked to the [operator catalog](../operators/README.md). "
           "First-order and provisional (single-component reads + the induction idiom).", "",
           "| model | architecture | operator roles linked |", "|---|---|---|"]
    for slug, title, arch, n in index_rows:
        idx.append(f"| [{title}]({slug}.md) | {arch} | {n} |")
    idx += ["", "_The deeper composition / circuit story is in the [circuit catalog](../circuits/README.md) and "
            "[DISASSEMBLY.md](../DISASSEMBLY.md). Generated by `disassembly_pages.py`._"]
    (args.docs / "README.md").write_text("\n".join(idx))
    print(f"[done] {len(index_rows)} disassembly pages + index → {args.docs}/")


if __name__ == "__main__":
    main()
