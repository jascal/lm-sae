"""Multilingual ops — do the attention idioms change with language? (research idea ii)

On a multilingual model, runs the behavioral disassembly (per-head induction / prev-token / duplicate idiom
scores + the self/sink/prev/structural/local/long_range attention budget) on **the same domain (Wikipedia)
in several languages spanning scripts** (en/fr/de Latin, zh CJK, ru Cyrillic, ar Arabic). Then asks:

  - are the **mechanism heads language-INVARIANT**? (Spearman of each per-head idiom-score vector across
    language pairs + top-k head overlap) — predicted yes: induction/prev/dup are positional/structural, not
    lexical, so the same heads should run them regardless of language.
  - do the **coverage fractions shift with tokenization/script**? (the budget per language) — predicted yes:
    CJK / non-Latin scripts tokenize at different granularity, moving the prev/local/structural split.

Wikipedia text is streamed per language via `datasets` (wikimedia/wikipedia, same source family as the
English WikiText baseline). Arch-generic (any HF causal LM); run on a multilingual model (Gemma-2 / Qwen-2.5).
GPT-2 is English-only so it is not a meaningful host here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BUCKETS = ["self", "sink", "prev", "structural", "local", "long_range"]


def _struct(s):
    s = s.replace("Ġ", "").replace("▁", "").replace("Ċ", "\n").strip()
    return s == "" or (s.startswith("<") and s.endswith(">")) or all(not ch.isalnum() for ch in s)


def _spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def fetch_lang(lang, n_chars):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    buf, tot = [], 0
    for ex in ds:
        t = ex.get("text", "")
        if t:
            buf.append(t); tot += len(t)
        if tot >= n_chars:
            break
    return "\n\n".join(buf)[:n_chars]


def behavioral(model, tok, dev, text, args, nL, H):
    """Per-head idiom scores (induction/prev/duplicate) + attention budget for one corpus."""
    import torch
    ids = tok(text)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
    prev = np.zeros((nL, H)); dup = np.zeros((nL, H)); ind = np.zeros((nL, H))
    ptn = dupn = indn = 0
    bacc = np.zeros((nL, H, len(BUCKETS))); btot = np.zeros((nL, H))
    with torch.no_grad():
        for c in chunks:
            a_all = model(input_ids=torch.tensor([c], device=dev), output_attentions=True).attentions
            Lc = len(c); ca = np.array(c); qi = np.arange(Lc)
            toks = tok.convert_ids_to_tokens(c); struct = np.array([_struct(t) for t in toks])
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]
            DM = (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])
            IM = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
            ptn += Lc - 1; dupn += int(DM.any(1).sum()); indn += int(IM.any(1).sum())
            S, T = np.meshgrid(qi, qi); delta = T - S
            bid = np.full((Lc, Lc), 5, dtype=int)
            bid[(delta >= 2) & (delta <= args.local_max)] = 4
            bid[struct[None, :].repeat(Lc, 0)] = 3
            bid[delta == 1] = 2; bid[S == 0] = 1; bid[S == T] = 0; bid[S > T] = -1
            for L in range(nL):
                a = a_all[L][0].float().cpu().numpy()
                prev[L] += np.diagonal(a, offset=-1, axis1=1, axis2=2).sum(1)
                dup[L] += (a * DM[None]).sum((1, 2)); ind[L] += (a * IM[None]).sum((1, 2))
                btot[L] += a.sum((1, 2))
                for b in range(len(BUCKETS)):
                    bacc[L, :, b] += (a * (bid == b)[None]).sum((1, 2))
    frac = bacc / np.maximum(btot, 1e-9)[:, :, None]
    budget = {b: float(np.mean(frac[:, :, k])) for k, b in enumerate(BUCKETS)}
    return {"n_tokens": len(ids), "n_chunks": len(chunks), "budget": budget,
            "prev": (prev / max(ptn, 1)).reshape(-1), "dup": (dup / max(dupn, 1)).reshape(-1),
            "ind": (ind / max(indn, 1)).reshape(-1)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-2-2b")
    p.add_argument("--langs", default="en,fr,de,zh,ru,ar")
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--local-max", type=int, default=8)
    p.add_argument("--n-chars", type=int, default=300000)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/gemma/multilingual_ops_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    cfg = model.config
    nL, H = cfg.num_hidden_layers, cfg.num_attention_heads
    langs = [x.strip() for x in args.langs.split(",")]
    print(f"{args.model}: {nL}L x {H}H; langs {langs}")

    per = {}
    for lg in langs:
        print(f"[{lg}] streaming Wikipedia ...", flush=True)
        text = fetch_lang(lg, args.n_chars)
        r = behavioral(model, tok, dev, text, args, nL, H)
        per[lg] = r
        heads = [(L, h) for L in range(nL) for h in range(H)]
        topind = [f"{heads[i][0]}.{heads[i][1]}" for i in np.argsort(-r["ind"])[:args.top_k]]
        b = r["budget"]
        print(f"  {lg}: {r['n_tokens']} tok | sink {b['sink']:.0%} prev {b['prev']:.0%} struct {b['structural']:.0%} "
              f"local {b['local']:.0%} content {b['long_range']:.0%} | top-induction {topind}")

    # ---- cross-language invariance: Spearman of per-head idiom-score vectors + top-k overlap ----
    def pairwise(metric):
        sp = {}; vals = []
        for i, a in enumerate(langs):
            for bn in langs[i + 1:]:
                s = _spear(per[a][metric], per[bn][metric]); sp[f"{a}-{bn}"] = s; vals.append(s)
        return sp, (float(np.nanmean(vals)) if vals else float("nan"))

    def topk_overlap(metric):
        tops = {lg: set(np.argsort(-per[lg][metric])[:args.top_k].tolist()) for lg in langs}
        en = tops.get(langs[0])
        ov = {lg: len(tops[lg] & en) / args.top_k for lg in langs[1:]} if en is not None else {}
        return ov

    invariance = {}
    print("\n=== cross-language head-identity invariance (Spearman of per-head idiom scores) ===")
    for metric in ("ind", "prev", "dup"):
        sp, mean = pairwise(metric)
        ov = topk_overlap(metric)
        invariance[metric] = {"pairwise_spearman": sp, "mean_spearman": mean,
                              f"top{args.top_k}_overlap_vs_{langs[0]}": ov}
        print(f"  {metric:>4}: mean pairwise Spearman {mean:+.2f}  | top-{args.top_k} overlap vs {langs[0]}: "
              + " ".join(f"{lg}:{v:.0%}" for lg, v in ov.items()))

    out = {"experiment": f"multilingual ops: {args.model}", "model": args.model, "langs": langs,
           "budgets": {lg: per[lg]["budget"] for lg in langs},
           "n_tokens": {lg: per[lg]["n_tokens"] for lg in langs},
           "top_induction_heads": {lg: [f"{(i // H)}.{i % H}" for i in np.argsort(-per[lg]["ind"])[:args.top_k].tolist()] for lg in langs},
           "invariance": invariance}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[verdict] induction-head identity mean Spearman across {len(langs)} languages: "
          f"{invariance['ind']['mean_spearman']:+.2f} "
          f"({'LANGUAGE-INVARIANT' if invariance['ind']['mean_spearman'] > 0.6 else 'language-dependent'}).")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
