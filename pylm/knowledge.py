"""Capture the model's KNOWLEDGE into a flat fact table — the 'model IS the database', literally.

The n-gram store memorises surface continuations; a **knowledge** store holds *facts* as a flat relation table
(`capital[France] = Paris`) plus the template that triggers a lookup, so pylm can answer a factual prompt — and
*paraphrases of it* — that never appeared in the corpus, by pure-Python relational lookup. The facts are read out of
the model (run it on relation templates, record its answer) — capturing the database the model stores, not the
corpus. Single-token subjects/objects (the flat table is id→id).

`KnowledgeStore.lookup(ctx)` (used by `lm.py` when given a `--knowledge` file): if the context matches a relation
template `prefix · <subject> · suffix`, extract the subject token and return the stored object — else None (fall back
to induction / n-gram). No neural net runs in the lookup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# relation -> (template, {subject: true object}). The fact is read out of the model by argmax over the OBJECT SET
# (the constrained readout `relation_decompile` uses): the knowledge is *in* the model even when a high-frequency
# token ("the") outranks it in full vocab — so we read the database out, not the surface top-1.
RELATIONS = {
    "capital": ("The capital of {S} is", {"France": "Paris", "Italy": "Rome", "Japan": "Tokyo", "Spain": "Madrid",
                "Germany": "Berlin", "Russia": "Moscow", "Egypt": "Cairo", "Greece": "Athens", "Poland": "Warsaw",
                "Norway": "Oslo", "Cuba": "Havana", "Peru": "Lima", "Chile": "Santiago", "Kenya": "Nairobi"}),
    "language": ("The language of {S} is", {"France": "French", "Italy": "Italian", "Japan": "Japanese",
                 "Spain": "Spanish", "Germany": "German", "Russia": "Russian", "Greece": "Greek", "Poland": "Polish",
                 "Norway": "Norwegian"}),
}


class KnowledgeStore:
    def __init__(self, path):
        self.rels = json.loads(Path(path).read_text())["relations"]

    def lookup(self, ctx):
        for r in self.rels.values():
            pre, suf = r["prefix"], r["suffix"]
            n = len(pre) + 1 + len(suf)
            if len(ctx) < n:
                continue
            if list(ctx[-n:-len(suf) - 1]) == pre and list(ctx[-len(suf):]) == suf:
                subj = ctx[-len(suf) - 1]
                obj = r["table"].get(str(subj))
                if obj is not None:
                    return obj, r["name"]
        return None, None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=Path("pylm/knowledge.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(args.model).eval().to(dev)

    def one_tok(w):
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        return ids[0] if len(ids) == 1 else None

    out = {"model": args.model, "relations": {}}; correct = 0; total = 0
    with torch.no_grad():
        for name, (tmpl, facts) in RELATIONS.items():
            probe = tok(tmpl.format(S="France"))["input_ids"]; sid = one_tok("France")
            si = probe.index(sid); pre, suf = probe[:si], probe[si + 1:]
            obj_ids = sorted({oid for o in facts.values() if (oid := one_tok(o)) is not None})  # the candidate object set
            table = {}
            for s, true_o in facts.items():
                stok = one_tok(s); otok = one_tok(true_o)
                if stok is None or otok is None:
                    continue
                ids = pre + [stok] + suf
                lg = m(input_ids=torch.tensor([ids], device=dev)).logits[0, -1]
                pred = obj_ids[int(torch.tensor([float(lg[o]) for o in obj_ids]).argmax())]  # constrained read-out
                table[str(stok)] = pred; total += 1; correct += (pred == otok)
            out["relations"][name] = {"name": name, "prefix": pre, "suffix": suf, "table": table,
                                      "template": tmpl, "n_facts": len(table)}
    out["fact_accuracy"] = correct / max(total, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[knowledge] {args.model}: read {total} facts across {len(out['relations'])} relations "
          f"({out['fact_accuracy']:.0%} correct vs truth) → {args.out}")
    for name, r in out["relations"].items():
        rows = ", ".join(f"{tok.decode([int(s)]).strip()}→{tok.decode([o]).strip()}" for s, o in list(r["table"].items())[:6])
        print(f"  {name}: {rows} ...")


if __name__ == "__main__":
    main()
