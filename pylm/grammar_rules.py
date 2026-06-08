"""Render the flat-file GRAMMAR (store_grammar.json) into a HUMAN-READABLE table — the closed-class lexicon by category
+ example skeleton→successor rules. Decompiler-side (needs the tokenizer to decode ids); emits committed JSON + markdown.

The grammar idiom (`grammar.py`) stores two flat things in `store_grammar.json`: `closed_ids` (the function-word /
punctuation lexicon the scaffold operates over) and `skel` (skeleton → next-token transition rules, content collapsed
to OPEN). Those are token-id-keyed (machine-readable). This decodes them into a readable table grouped by grammatical
category, so the grammar the model uses as its scaffold can be published. Output: runs/pylm/grammar_rules_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CATEGORIES = {
    "determiners": "the a an this that these those my your his her its our their some any no each every either neither",
    "prepositions": "of to in for on with at by from up about into over after under above below between out off down through",
    "conjunctions": "and or but nor so yet because although while if then than as when where since unless whereas",
    "pronouns": "i you he she it we they me him us them mine yours hers ours theirs who whom whose which what",
    "auxiliaries": "is are was were be been being am do does did have has had having will would shall should can could may might must",
    "particles": "not no too very just only also there here now thus hence however therefore",
}
PUNCT = list(".,;:!?'\"()[]{}-—…/")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=Path("pylm/store_grammar.json"))
    p.add_argument("--model", default="gpt2")
    p.add_argument("--examples", type=int, default=18, help="example skeleton rules to render")
    p.add_argument("--out", type=Path, default=Path("runs/pylm/grammar_rules_summary.json"))
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    s = json.loads(args.store.read_text())
    closed = set(s.get("closed_ids", [])); skel = s.get("skel", {})

    def dec(i):
        return tok.decode([int(i)]).strip() or repr(tok.decode([int(i)]))

    # which single-token closed-class words (per category) are actually in the model's lexicon
    lexicon = {}
    for cat, words in CATEGORIES.items():
        present = []
        for w in words.split():
            for f in (" " + w, w, " " + w.capitalize()):
                ids = tok(f, add_special_tokens=False)["input_ids"]
                if len(ids) == 1 and ids[0] in closed:
                    present.append(w); break
        lexicon[cat] = sorted(set(present))
    punct_present = [pp for pp in PUNCT
                     if any(len(tok(f, add_special_tokens=False)["input_ids"]) == 1
                            and tok(f, add_special_tokens=False)["input_ids"][0] in closed for f in (pp, " " + pp))]

    # readable example rules: skeleton (closed tokens shown, content = ·) → top successor
    def render_key(k):
        n, body = k.split(":", 1)
        toks = ["·" if x == "O" else dec(x) for x in body.split("/")]
        return " ".join(toks)
    rules = []
    for k, succ in skel.items():
        if "O" in k.split(":", 1)[1].split("/") and any(x != "O" for x in k.split(":", 1)[1].split("/")):
            rules.append((render_key(k), dec(succ[0])))
        if len(rules) >= args.examples * 4:
            break
    # prefer rules whose successor is itself closed-class (the grammatical transitions), dedup
    seen = set(); picked = []
    for lhs, rhs in rules:
        if lhs in seen:
            continue
        seen.add(lhs); picked.append((lhs, rhs))
        if len(picked) >= args.examples:
            break

    result = {"model": s.get("model"), "n_closed_class": len(closed), "n_skeleton_rules": len(skel),
              "lexicon": lexicon, "punctuation": punct_present, "example_rules": picked,
              "note": ("lexicon/categories are language-universal (English closed class) and the scaffold is "
                       "corpus-invariant + cross-architecture (PR #133); the skeleton TABLE is model/tokenizer-captured")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[grammar-rules] {result['model']}: {len(closed)} closed-class tokens, {len(skel)} skeleton rules")
    for cat, ws in lexicon.items():
        print(f"  {cat:12s} ({len(ws)}): {' '.join(ws)}")
    print(f"  punctuation ({len(punct_present)}): {' '.join(punct_present)}")
    print("  example rules (· = any content word):")
    for lhs, rhs in picked[:8]:
        print(f"    {lhs}  →  {rhs!r}")
    print(f"[done] → {args.out}")
    return result


if __name__ == "__main__":
    main()
