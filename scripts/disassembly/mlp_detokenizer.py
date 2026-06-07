"""Is the early MLP an "extended embedding" / detokenizer? — characterizing the COMPUTE class's load-bearing layer.

The discovery sweep found MLP0/MLP1 the single most load-bearing components in every model, mechanism *unverified*.
The canonical reading (Elhage et al.'s "MLP0 ≈ an extended embedding"; nostalgebraist's detokenizer) is that the
early MLP's output is largely a function of the **current token identity** (enriching the embedding / assembling
sub-word pieces) rather than the broader context. Two arch-generic measurements (reusing `mlp_atlas.mlp_blocks`):

  1. **Token-determinism** (the clean test, no entropy confound): the fraction of an MLP layer's output variance
     explained by the *current token identity* (an η²: 1 − mean within-token variance / total variance, over the
     frequent tokens). ≈1 ⇒ the output is token-determined (embedding-like); ≈0 ⇒ context-determined. The
     extended-embedding claim predicts **high at MLP0, decaying with depth**.
  2. **Category-split ablation** (supporting, confound-shown): mean-ablate each MLP layer and measure next-token-NLL
     damage split by the **target** token's category — word-start (leading space marker), continuation (mid-word
     subword), other. Reported *with the per-category baseline NLL*, because word-starts are inherently
     higher-entropy, so absolute ΔNLL is confounded by baseline difficulty — read it relative to baseline.

Reported per layer (0/1/2/mid/late) per model. Provisional, single corpus (Shakespeare prose).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_atlas import mlp_blocks  # noqa: E402

CATS = ["word-start", "continuation", "other"]
MARKERS = ("Ġ", "▁", " ")   # 'Ġ' (BPE space), '▁' (SentencePiece space), literal space


def categorize(raw):
    """Category of a token from its raw tokenizer piece (convert_ids_to_tokens)."""
    started = raw[:1] in MARKERS
    core = raw.lstrip("Ġ▁ ")
    if core.isalpha():
        return "word-start" if started else "continuation"
    return "other"


def run_model(model_id, args, dev):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    is_gpt2 = "gpt2" in model_id.lower()
    if is_gpt2:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_id, attn_implementation="eager").eval().to(dev)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    mlps = mlp_blocks(model); nL = len(mlps)
    tok = AutoTokenizer.from_pretrained(model_id)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:300000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]

    # category of every vocab id we'll meet (cache raw-piece lookups)
    raw_cache = {}

    def cat_of(tid):
        if tid not in raw_cache:
            raw_cache[tid] = categorize(tok.convert_ids_to_tokens(int(tid)))
        return raw_cache[tid]

    # per-layer mean MLP output (corpus) — the mean-ablation target
    cap = {L: [] for L in range(nL)}

    def grab(L):
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            cap[L].append(t.detach().reshape(-1, t.shape[-1]))
        return hook
    hk = [mlps[L].register_forward_hook(grab(L)) for L in range(nL)]
    with torch.no_grad():
        for c in chunks:
            model(input_ids=torch.tensor([c], device=dev))
    for h in hk:
        h.remove()
    mean_mlp = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    probe_layers = sorted(set([0, 1, 2, nL // 2, nL - 2]) & set(range(nL)))

    # --- (1) token-determinism: fraction of MLP-output variance explained by current-token identity ---
    freq = set(t for t, _ in Counter(t for c in chunks for t in c).most_common(args.n_tok))
    perL = {L: {"tok": defaultdict(lambda: [0, None, 0.0]), "n": 0, "sum": None, "ss": 0.0} for L in probe_layers}
    store = {}
    hk2 = [mlps[L].register_forward_hook((lambda L: lambda m, i, o: store.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))(L)) for L in probe_layers]
    with torch.no_grad():
        for c in chunks:
            store.clear(); model(input_ids=torch.tensor([c], device=dev))
            for L in probe_layers:
                out = store[L][0].float(); nsq = (out * out).sum(-1).cpu().numpy(); outc = out.cpu().numpy(); A = perL[L]
                for pos, tid in enumerate(c):
                    if tid not in freq:
                        continue
                    v = outc[pos]; rec = A["tok"][tid]
                    rec[0] += 1; rec[1] = v.copy() if rec[1] is None else rec[1] + v; rec[2] += float(nsq[pos])
                    A["n"] += 1; A["sum"] = v.copy() if A["sum"] is None else A["sum"] + v; A["ss"] += float(nsq[pos])
    for h in hk2:
        h.remove()

    def determinism(L):
        A = perL[L]; n = A["n"]
        if n == 0:
            return None
        total = A["ss"] / n - float((A["sum"] / n) @ (A["sum"] / n))
        wv = 0.0; wn = 0
        for _tid, (nt, s, ss) in A["tok"].items():
            if nt < args.min_count:
                continue
            wv += (ss / nt - float((s / nt) @ (s / nt))) * nt; wn += nt
        return float(1 - (wv / max(wn, 1)) / total) if total > 1e-9 else None

    def ablate_hook(L):
        def hook(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            rep = mean_mlp[L].to(t.dtype).expand_as(t)
            return (rep,) + tuple(o[1:]) if isinstance(o, tuple) else rep
        return hook

    def nll_by_cat(ablate_L=None):
        h = mlps[ablate_L].register_forward_hook(ablate_hook(ablate_L)) if ablate_L is not None else None
        tot = {c: 0.0 for c in CATS}; cnt = {c: 0 for c in CATS}
        try:
            with torch.no_grad():
                for c in chunks:
                    lp = F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                    y = c[1:]
                    nlls = (-lp[torch.arange(len(y)), torch.tensor(y, device=dev)]).cpu().numpy()
                    for j, tid in enumerate(y):
                        cc = cat_of(tid); tot[cc] += float(nlls[j]); cnt[cc] += 1
        finally:
            if h:
                h.remove()
        return {c: tot[c] / max(cnt[c], 1) for c in CATS}, cnt
    base, counts = nll_by_cat()
    layers = []
    for L in probe_layers:
        ab, _ = nll_by_cat(L)
        d = {c: ab[c] - base[c] for c in CATS}
        det = determinism(L)
        ratio = d["continuation"] / d["word-start"] if d["word-start"] > 1e-6 else None
        layers.append({"layer": L, "depth": L / (nL - 1) if nL > 1 else 0.0, "determinism": det, "dNLL": d, "cont_over_start": ratio})
        print(f"  L{L:>2} (d{L / max(nL - 1, 1):.2f}): token-determinism {('%.2f' % det) if det is not None else 'n/a'} | ΔNLL start {d['word-start']:+.3f} cont {d['continuation']:+.3f} other {d['other']:+.3f}")
    return {"model": model_id.split("/")[-1], "rope": not is_gpt2, "n_layers": nL,
            "token_counts": counts, "baseline_by_cat": base, "layers": layers}


def write_doc(out, docs):
    lines = ["---", "title: MLP extended-embedding test", "---", "",
             "# Is the early MLP an \"extended embedding\" / detokenizer?", "",
             "MLP0/MLP1 are the single most causally load-bearing components in every model "
             "([discovered components](discovered.md)), with **mechanism unverified**. The canonical reading is that "
             "the early MLP's output is largely a function of the **current token identity** (an *extended embedding* "
             "/ detokenizer) rather than the broader context. Two measurements per MLP layer:", "",
             "1. **Token-determinism** (clean, no entropy confound) — the fraction of the layer's output variance "
             "explained by the current token identity (η²: 1 − within-token var / total var, over frequent tokens). "
             "**≈1 = token-determined (embedding-like); ≈0 = context-determined.** The extended-embedding claim "
             "predicts high at MLP0, decaying with depth.", "",
             "2. **Category-split ablation** (supporting) — mean-ablate the layer, next-token-NLL damage split by the "
             "**target** token's category, shown *with the per-category baseline NLL*. Word-starts are inherently "
             "higher-entropy, so read ΔNLL **relative to its baseline**, not in absolute terms.", "",
             "Provisional, single corpus (Shakespeare prose).", ""]
    for r in out["results"]:
        if "layers" not in r:
            continue
        tc = r["token_counts"]; tot = sum(tc.values()) or 1; bl = r["baseline_by_cat"]
        mix = ", ".join(f"{c} {tc[c] / tot:.0%}" for c in CATS)
        det0 = next((L["determinism"] for L in r["layers"] if L["layer"] == 0), None)
        detL = [L["determinism"] for L in r["layers"] if L["determinism"] is not None]
        trend = ("decays with depth" if len(detL) >= 2 and detL[0] == max(detL) else "is not monotonic") if detL else "n/a"
        det_line = f"MLP0 token-determinism **{det0:.2f}**; across probed layers it {trend}." if det0 is not None else ""
        lines += [f"## {r['model']} ({'RoPE' if r['rope'] else 'GPT-2/absolute'}, {r['n_layers']} layers)", "",
                  f"Target-token mix: {mix}. Baseline NLL by category: " + ", ".join(f"{c} {bl[c]:.2f}" for c in CATS) + ". " + det_line, "",
                  "| layer | depth | **token-determinism** | ΔNLL word-start | ΔNLL continuation | ΔNLL other |",
                  "|---|---|---|---|---|---|"]
        for L in r["layers"]:
            d = L["dNLL"]; det = L["determinism"]
            lines.append(f"| {L['layer']} | {L['depth']:.2f} | **{('%.2f' % det) if det is not None else 'n/a'}** | {d['word-start']:+.3f} | {d['continuation']:+.3f} | {d['other']:+.3f} |")
        lines.append("")
    lines += ["_Why Llama-3.2-1B's MLP0 is the context-determined outlier is dug in "
              "[outlier mechanism digs](outlier_digs.md) (it inherits the context-mixing of its layer-0 heads). "
              "Token-determinism = η² of the MLP-layer output on current-token identity (frequent tokens). "
              "Data: [mlp_detokenizer_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/mlp_detokenizer_summary.json). "
              "Regenerate: [mlp_detokenizer.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/mlp_detokenizer.py). "
              "See the [MLP / COMPUTE catalog](mlp_compute.md)._"]
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "mlp_detokenizer.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--chunks", type=int, default=40)
    p.add_argument("--n-tok", type=int, default=300, help="frequent tokens used for the determinism η²")
    p.add_argument("--min-count", type=int, default=5, help="min occurrences for a token to enter the determinism estimate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/operators"))
    p.add_argument("--docs", type=Path, default=Path("docs/operators"))
    p.add_argument("--docs-only", action="store_true", help="re-render the page from the committed summary JSON; no GPU")
    args = p.parse_args(argv)
    if args.docs_only:
        out = json.loads((args.outdir / "mlp_detokenizer_summary.json").read_text())
        write_doc(out, args.docs)
        print(f"[docs-only] re-rendered {args.docs / 'mlp_detokenizer.md'}")
        return out
    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            results.append(run_model(mid, args, dev))
            if dev == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid, "error": str(e)})
    out = {"experiment": "early-MLP detokenizer test (token-category ablation)", "results": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "mlp_detokenizer_summary.json").write_text(json.dumps(out, indent=2, default=float))
    write_doc(out, args.docs)
    print(f"\n[done] {len([r for r in results if 'layers' in r])} models → {args.outdir / 'mlp_detokenizer_summary.json'} + {args.docs / 'mlp_detokenizer.md'}")
    return out


if __name__ == "__main__":
    main()
