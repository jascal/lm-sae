"""Flat-file GRAMMAR idiom — the model's content-free grammatical scaffold as a flat table (no neural net).

The core-structure analysis (`scripts/disassembly/core_grammar.py`) found the entangled core's *most-shared*
directions form a compact, corpus-invariant, **closed-class scaffold** — a generic grammar (top-16 directions: 22×
chance cross-corpus overlap, 28× the base closed-class rate). This idiom decompiles that scaffold to a flat file the
same way `knowledge.py` decompiles facts: collapse every CONTENT token to a single `OPEN` symbol, keep
function-words / punctuation as themselves, and store the grammatical **skeleton → next-token** table. It is a pure
dict lookup over grammatical categories — it generalises across content (and corpora) exactly as the core's grammar
head does, firing where the lexical n-gram is too sparse. The closed-class id set + skeleton tables live in the flat
store; this file is pure stdlib (the `closed_ids()` builder takes a tokenizer and is used only by the decompiler).
"""
from __future__ import annotations

OPEN = "O"   # every open-class (content) token collapses to one symbol — the grammar is content-free by construction

# closed-class (function-word) lexicon + punctuation — the grammatical categories the core's top directions promote
CLOSED = ("the a an this that these those my your his her its our their some any no each every either neither "
          "of to in for on with at by from up about into over after under above below between out off down through "
          "and or but nor so yet because although while if then than as when where since unless whereas "
          "i you he she it we they me him us them mine yours hers ours theirs who whom whose which what "
          "is are was were be been being am do does did have has had having will would shall should can could may might must "
          "not no too very just only also there here now thus hence however therefore").split()
PUNCT = list(".,;:!?'\"()[]{}-—…/")


def closed_ids(tok):
    """Single-token closed-class id set for a tokenizer (decompiler-side; the tokenizer is passed in, not imported)."""
    ids = set()
    forms = [f for w in CLOSED for f in (w, " " + w, w.capitalize(), " " + w.capitalize())]
    forms += [f for p in PUNCT for f in (p, " " + p)] + ["\n", "\n\n"]
    for f in forms:
        t = tok(f, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ids.add(t[0])
    return ids


def skeleton(ctx, closed, n):
    """The grammatical skeleton of the last n tokens: closed-class tokens kept verbatim, content collapsed to OPEN."""
    return "/".join((str(t) if t in closed else OPEN) for t in ctx[-n:])


class GrammarStore:
    """Runtime (pure-stdlib) flat-file grammar: grammatical skeleton → ranked successor ids; backoff skel-3 → skel-2."""

    def __init__(self, closed, skel):
        self.closed = set(closed)
        self.skel = skel          # {"3:O/265/O": [id, ...], "2:265/O": [id, ...]}

    def lookup(self, ctx):
        for n in (3, 2):
            if len(ctx) >= n:
                hit = self.skel.get(f"{n}:{skeleton(ctx, self.closed, n)}")
                if hit:
                    return hit, f"grammar-{n}"
        return None, None
