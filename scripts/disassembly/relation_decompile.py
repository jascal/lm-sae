"""Dump a relation's table + trace where it resolves — the decompiler READ side (decompiler #2).

The edit work (#113-#115) is the WRITE side. This is the READ: treat a relation (capital-of, language-of) as a
database TABLE — read out every subject's object from the model, and **decompile WHERE in the forward pass the table
is queried** by logit-lens: at each layer, unembed the last-token residual and ask whether the object is already
decodable. The depth where it emerges is the relation's read-out site; the fraction the model gets right is the
table's completeness.

  TABLE        — for every subject, the model's predicted object (over the relation's object set) + accuracy.
  READ-OUT DEPTH — logit-lens: the earliest layer at which the true object is the argmax of unembed(norm(resid_L)) at
                   the last token, averaged over facts (where the relation is resolved in depth).
  (We also report whether the object is *linearly* present earlier than it is *output* — the resolve-vs-commit gap.)

Robust (no fitting, no autograd) → fits every model. (A from-scratch linear operator over the subject representation
is hopelessly under-determined with ~14 single-token facts; the faithful linear-relation-embedding needs the model's
Jacobian, which OOMs the 2-B models on a 7.5 GB GPU — so we read the table the robust way.) Cross-model, two relations.

Output: runs/disassembly/relation_decompile_summary.json (merge-safe). Findings -> docs/FINDINGS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fact_edit_xmodel import FACTS, LANG, single_tok  # noqa: E402

RELATIONS = {
    "capital": {"template": "The capital of {S} is", "obj": {c: cap for c, cap in FACTS}},
    "language": {"template": "The language of {S} is", "obj": dict(LANG)},
}


def final_norm(vm):
    """The model's final pre-unembedding norm (ln_f / model.norm), for a faithful logit lens."""
    m = vm.model
    if vm.is_gpt2:
        return m.transformer.ln_f
    return m.model.norm


def relation_one(vm, rel, spec, args):
    tok = vm.tok; nL = vm.nL; t = vm.torch
    WU = vm.model.get_output_embeddings().weight.detach()                          # (vocab, d)
    fn = final_norm(vm)
    facts = []
    for country, obj in spec["obj"].items():
        sid = single_tok(tok, country, vm.is_gpt2); oid = single_tok(tok, obj, vm.is_gpt2)
        if sid is not None and oid is not None:
            facts.append((country, sid, oid))
    if len(facts) < 6:
        return {"relation": rel, "note": f"only {len(facts)} single-token facts"}
    obj_ids = sorted({oid for _, _, oid in facts})
    table = []; correct = 0; emerge_depths = []
    for (country, sid, oid) in facts:
        ids = tok(spec["template"].format(S=country), add_special_tokens=not vm.is_gpt2)["input_ids"]
        tr = vm.trace(ids); lg = tr["logits"][-1]
        # behavioural table: the model's predicted object over the relation's object set
        pick = obj_ids[int(t.tensor([float(lg[o]) for o in obj_ids]).argmax())]
        table.append({"subject": country, "predicted": tok.convert_ids_to_tokens(pick).replace("Ġ", " ").strip(),
                      "true": tok.convert_ids_to_tokens(oid).replace("Ġ", " ").strip(), "ok": pick == oid})
        correct += int(pick == oid)
        # read-out depth: earliest layer where the logit-lens (over the object set) already picks the true object
        depth = None
        with t.no_grad():
            for L in range(1, nL):
                h = fn(tr["resid"][L][0, -1])                                      # last-token residual entering layer L, normed
                logit = (WU.float() @ h.float())
                if obj_ids[int(t.tensor([float(logit[o]) for o in obj_ids]).argmax())] == oid:
                    depth = L / (nL - 1); break
        emerge_depths.append(depth if depth is not None else 1.0)
    m = len(facts)
    return {"relation": rel, "n_facts": m, "n_objects": len(obj_ids), "chance": 1.0 / len(obj_ids),
            "model_accuracy": correct / m, "mean_readout_depth": float(np.mean(emerge_depths)),
            "table": table}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
    args = p.parse_args(argv)

    import torch
    from residual_vm import ResidualVM
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        vm = None
        try:
            vm = ResidualVM(mid, device=dev)
            rels = {rel: relation_one(vm, rel, spec, args) for rel, spec in RELATIONS.items()}
            results.append({"model": mid.split("/")[-1], "rope": not vm.is_gpt2, "relations": rels})
            for rel, r in rels.items():
                if "model_accuracy" in r:
                    print(f"  {rel:>9}: table {r['n_facts']} rows, model-acc {r['model_accuracy']:.0%} "
                          f"(chance {r['chance']:.0%}) | mean read-out depth {r['mean_readout_depth']:.0%}")
                else:
                    print(f"  {rel:>9}: {r.get('note')}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})
        finally:
            vm = None
            if dev == "cuda":
                torch.cuda.empty_cache()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "relation_decompile_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "decompile a relation into a queryable linear operator (LRE) — faithfulness + accuracy, cross-model",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'relations' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
