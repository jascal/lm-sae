"""A small, locally-runnable CAPABILITY benchmark for the models/cores we build — so "we trained a *better* model" is
measurable even when faithful reproduction falls short.

The min_to_run / distillation metrics (NLL, agreement) measure *fidelity to a teacher*. This measures *capability*:
held-out perplexity + next-token accuracy on diverse text the model was NOT distilled on (distinct Gutenberg books +
Wikipedia), plus a LAMBADA-style **last-word accuracy** (predict the final token of a passage given the rest — the
classic small-model LM test). Model-agnostic (GPT-2 family, Qwen/Llama, and `*4bit*` via bitsandbytes). Establishes a
capability ladder for the full models now; the same harness scores a distilled/compressed core once one is saved.

Output: runs/disassembly/benchmark_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from min_to_run import _fetch_wiki, _gutenberg  # noqa: E402


def run_model(mid, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    if "4bit" in mid.lower():
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
        m = AutoModelForCausalLM.from_pretrained(mid, quantization_config=cfg, device_map=dev).eval()
    else:
        big = any(s in mid for s in ("1b", "1.4b", "1.5b", "2b", "2.8b", "3b", "7b", "8b", "-xl", "-large", "-medium"))
        m = AutoModelForCausalLM.from_pretrained(mid, **({"dtype": torch.bfloat16} if big else {})).eval().to(dev)

    # held-out text: books + wiki NOT used in min_to_run's distillation training set
    held = _gutenberg(2554, args.chars) + "\n\n" + _gutenberg(1727, args.chars) + "\n\n" + _fetch_wiki(args.chars)
    ids = tok(held)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 16][: args.n]

    tot = 0.0; k = 0; acc = 0; lastword_hit = 0; lastword_n = 0
    with torch.no_grad():
        for c in chunks:
            lg = m(input_ids=torch.tensor([c], device=dev)).logits[0].float()
            lp = torch.log_softmax(lg, -1); y = c[1:]
            top1 = lg[:-1].argmax(-1)
            for p in range(len(y)):
                tot += float(-lp[p, y[p]]); k += 1; acc += int(top1[p] == y[p])
            # LAMBADA-style: predict the FINAL token of the passage given all the rest
            lastword_hit += int(top1[-1] == c[-1]); lastword_n += 1
    ppl = float(np.exp(tot / max(k, 1)))
    return {"model": mid.split("/")[-1], "perplexity": ppl, "nll": tot / max(k, 1),
            "next_token_acc": acc / max(k, 1), "lastword_acc": lastword_hit / max(lastword_n, 1),
            "n_chunks": len(chunks), "n_tokens": k}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,gpt2-xl")
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--n", type=int, default=200, help="held-out passages to score")
    p.add_argument("--chars", type=int, default=400000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
    args = p.parse_args(argv)

    import torch
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = run_model(mid, args)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(r)
            print(f"  perplexity {r['perplexity']:.1f} · next-token acc {r['next_token_acc']:.1%} · "
                  f"last-word acc {r['lastword_acc']:.1%}  ({r['n_tokens']} tokens)")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "benchmark_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "local capability benchmark — held-out perplexity + next-token + last-word accuracy", "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'perplexity' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
