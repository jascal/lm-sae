"""pylm — a decompiled language model in pure Python. NO neural-net code or concepts.

This is the program's literal end-goal: reimplement a real small LLM's *behaviour* as a small symbolic Python
program backed by flat-file stores — the "model IS the database" thesis made runnable. There is no matrix, no
attention, no layer; only the catalogued idioms expressed as plain Python over flat-file knowledge:

  INDUCTION (in-context copy) — the keystone reused instruction (an inline-cache macro): if the current local
    context has occurred before *in this sequence*, predict what followed it. Pure list scan, no weights.
  N-GRAM backoff — the statistical knowledge store (a flat file of trigram→bigram→unigram successor tables built
    from a corpus): the "memorised" continuations the model also carries. Pure dict lookup.
  (KNOWLEDGE relations + structural rules are added in later steps.)

The PROGRAM (this file's `PyLM.predict`) is kept deliberately tiny — the bias is toward small *code*; the *data*
(the n-gram store) lives in a flat file (`store.json`). Validation (`validate.py`) measures what fraction of a real
model's next-token predictions this pure-Python decompilation reproduces — the decompilable fraction, made literal.
"""
from __future__ import annotations

import json
from pathlib import Path


class PyLM:
    """The decompiled LM. Operates on token-id sequences; tokenization is a flat-file preprocessing step."""

    def __init__(self, store_path, knowledge_path=None):
        s = json.loads(Path(store_path).read_text())
        self.quad = s.get("quad", {})  # "a,b,c" -> [next_id, ...] (4-gram, the deepest memorised context)
        self.tri = s["tri"]          # "a,b" -> [next_id, ...] (ranked corpus successors)
        self.bi = s["bi"]            # "b"   -> [next_id, ...]
        self.uni = s["uni"]          # [next_id, ...] (most frequent tokens)
        self.min_induction = s.get("min_induction_match", 3)
        self.min_accept = s.get("min_induction_accept", 2)  # ignore 1-token "induction" (it's noise, ~chance)
        self.knowledge = None        # optional flat fact table (the 'database'); a relational lookup, not statistics
        if knowledge_path:
            from knowledge import KnowledgeStore
            self.knowledge = KnowledgeStore(knowledge_path)

    def predict(self, ctx, k=1):
        """Next-token prediction for a token-id context. Returns the top-1 id (or a ranked list if k>1)."""
        ranked, fired = self._candidates(ctx)
        return (ranked[0] if ranked else self.uni[0]) if k == 1 else ranked[:k]

    def predict_explain(self, ctx):
        """(top-1 id, which instruction fired) — for the per-instruction validation breakdown."""
        ranked, fired = self._candidates(ctx)
        return (ranked[0] if ranked else self.uni[0]), fired

    def _candidates(self, ctx):
        if self.knowledge is not None:                      # KNOWLEDGE — a stored fact (relational lookup) beats a guess
            fact, rel = self.knowledge.lookup(ctx)
            if fact is not None:
                return [fact], f"knowledge:{rel}"
        ind, tag = self._induction(ctx)
        if ind is not None:
            return [ind], tag
        return self._ngram(ctx)

    def _induction(self, ctx):
        # INDUCTION — the longest local context (≥ min_accept tokens) that recurs earlier in ctx; predict the token
        # that followed its last earlier occurrence (in-context copy). 1-token "matches" are ~chance noise, so we
        # only accept span ≥ min_accept and let the n-gram store handle the rest.
        for span in range(self.min_induction, self.min_accept - 1, -1):
            if len(ctx) <= span:
                continue
            tail = ctx[-span:]
            for i in range(len(ctx) - span - 1, -1, -1):
                if ctx[i:i + span] == tail:
                    return ctx[i + span], f"induction-{span}"
        return None, None

    def _ngram(self, ctx):
        # N-GRAM backoff — the flat-file store: 4-gram → trigram → bigram → unigram.
        if self.quad and len(ctx) >= 3:
            q = self.quad.get(f"{ctx[-3]},{ctx[-2]},{ctx[-1]}")
            if q:
                return q, "quad"
        if len(ctx) >= 2:
            t = self.tri.get(f"{ctx[-2]},{ctx[-1]}")
            if t:
                return t, "trigram"
        b = self.bi.get(str(ctx[-1]))
        if b:
            return b, "bigram"
        return self.uni, "unigram"

    def ranked(self, ctx, k=8):
        """Ranked candidate ids the LM considers — the induction copy (if any) first, then the n-gram backoff list.
        Gives a *pool* to sample from (the flat store keeps no probabilities, so generation samples over rank)."""
        ind, _ = self._induction(ctx)
        ng, _ = self._ngram(ctx)
        out = ([ind] if ind is not None else []) + [c for c in ng if c != ind]
        return out[:k]


def program_loc():
    """Lines of actual program code in this file (the 'small code' the decompilation compresses to)."""
    src = Path(__file__).read_text().splitlines()
    code = [ln for ln in src if ln.strip() and not ln.strip().startswith("#")]
    in_doc = False; out = []
    for ln in code:
        if ln.lstrip().startswith(('"""', "'''")):
            in_doc = not in_doc if ln.strip().count('"""') == 1 else in_doc
            continue
        if not in_doc:
            out.append(ln)
    return len(out)
