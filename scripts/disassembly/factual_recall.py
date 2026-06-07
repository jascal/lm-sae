"""Where do facts live? — per-layer localization of factual recall (attention vs MLP), cross-model.

The catalog so far is about *mechanisms* (induction, copy). The deeper decompiler goal is *knowledge* ("the model
IS the database"). This is the natural-history version of the ROME / causal-tracing question: for a set of factual
completions ("The capital of France is" -> " Paris"), measure the object's NLL, then mean-ablate each layer's
**attention** (all heads) and its **MLP** in turn, and read off **which layers / which substrate** the fact depends
on. The ROME prediction is mid-layer MLPs; this checks whether that holds across architectures. Arch-generic
(reuses `circuit_content_patch._arch` + `mlp_atlas.mlp_blocks`). Facts kept only where the object is single-token.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_content_patch import _arch  # noqa: E402
from mlp_atlas import mlp_blocks  # noqa: E402

FACTS = [
    ("The capital of France is", " Paris"), ("The capital of Japan is", " Tokyo"),
    ("The capital of Italy is", " Rome"), ("The capital of Russia is", " Moscow"),
    ("The capital of China is", " Beijing"), ("The capital of Egypt is", " Cairo"),
    ("The capital of Spain is", " Madrid"), ("The capital of Germany is", " Berlin"),
    ("The capital of Canada is", " Ottawa"), ("The capital of Greece is", " Athens"),
    ("The capital of Cuba is", " Havana"), ("The capital of Peru is", " Lima"),
    ("The capital of Iran is", " Tehran"), ("The capital of Austria is", " Vienna"),
    ("The capital of Poland is", " Warsaw"), ("The capital of Norway is", " Oslo"),
    ("The capital of Kenya is", " Nairobi"), ("The capital of Chile is", " Santiago"),
    ("The Eiffel Tower is in the city of", " Paris"), ("The Colosseum is in the city of", " Rome"),
    ("The author of Romeo and Juliet is William", " Shakespeare"), ("The chemical symbol for gold is", " Au"),
    ("The largest planet in the solar system is", " Jupiter"), ("The sun rises in the", " east"),
]


def run_model(model_id, args, dev):
    import torch
    import torch.nn.functional as F
    is_gpt2 = "gpt2" in model_id.lower()
    from transformers import AutoTokenizer
    if is_gpt2:
        from transformers import GPT2LMHeadModel
        m = GPT2LMHeadModel.from_pretrained(model_id, attn_implementation="eager").eval().to(dev)
    else:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    a = _arch(m); H = a["H"]; hd = a["hd"]; nL = m.config.num_hidden_layers; oproj = a["oproj"]; mlps = mlp_blocks(m)
    tok = AutoTokenizer.from_pretrained(model_id)

    facts = []
    for prompt, obj in FACTS:
        ot = tok(obj, add_special_tokens=False)["input_ids"]
        if len(ot) != 1:
            continue
        ids = tok(prompt, add_special_tokens=not is_gpt2)["input_ids"]
        facts.append((ids, ot[0]))
    if len(facts) < 6:
        raise RuntimeError(f"only {len(facts)} single-token facts for {model_id}")

    # corpus mean for ablation: use the fact prompts themselves (mean over their positions)
    cap = {L: [] for L in range(nL)}; mcap = {L: [] for L in range(nL)}
    hks = [oproj[L].register_forward_pre_hook((lambda L: lambda mod, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    mhks = [mlps[L].register_forward_hook((lambda L: lambda mod, i, o: mcap[L].append((o[0] if isinstance(o, tuple) else o).detach().reshape(-1, (o[0] if isinstance(o, tuple) else o).shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for ids, _ in facts:
            m(input_ids=torch.tensor([ids], device=dev))
    for h in hks + mhks:
        h.remove()
    meanv = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}
    mean_mlp = {L: torch.cat(mcap[L], 0).mean(0) for L in range(nL)}

    def attn_hook(L):
        def hook(mod, inp):
            return (meanv[L].to(inp[0].dtype).expand_as(inp[0]),)
        return hook

    def mlp_hook(L):
        def hook(mod, i, o):
            t = o[0] if isinstance(o, tuple) else o
            rep = mean_mlp[L].to(t.dtype).expand_as(t)
            return (rep,) + tuple(o[1:]) if isinstance(o, tuple) else rep
        return hook

    def fact_nll(attn_L=None, mlp_L=None):
        hs = []
        if attn_L is not None:
            hs.append(oproj[attn_L].register_forward_pre_hook(attn_hook(attn_L)))
        if mlp_L is not None:
            hs.append(mlps[mlp_L].register_forward_hook(mlp_hook(mlp_L)))
        tot = 0.0
        try:
            with torch.no_grad():
                for ids, obj in facts:
                    lp = F.log_softmax(m(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float(), -1)
                    tot += float(-lp[obj])
        finally:
            for x in hs:
                x.remove()
        return tot / len(facts)

    # generic next-token NLL on prose — the CONTROL for the detokenizer confound (early MLPs hurt everything)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:80000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + 64] for i in range(0, len(pids), 64) if len(pids[i:i + 64]) >= 8][:12]

    def gen_nll(attn_L=None, mlp_L=None):
        hs = []
        if attn_L is not None:
            hs.append(oproj[attn_L].register_forward_pre_hook(attn_hook(attn_L)))
        if mlp_L is not None:
            hs.append(mlps[mlp_L].register_forward_hook(mlp_hook(mlp_L)))
        tot = 0.0; k = 0
        try:
            with torch.no_grad():
                for c in chunks:
                    lp = F.log_softmax(m(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                    y = torch.tensor(c[1:], device=dev); tot += float(-lp[torch.arange(len(y)), y].sum()); k += len(y)
        finally:
            for x in hs:
                x.remove()
        return tot / max(k, 1)

    base = fact_nll(); gbase = gen_nll()
    attn_d = [fact_nll(attn_L=L) - base for L in range(nL)]
    mlp_d = [fact_nll(mlp_L=L) - base for L in range(nL)]
    gmlp_d = [gen_nll(mlp_L=L) - gbase for L in range(nL)]
    depth = [L / (nL - 1) if nL > 1 else 0.0 for L in range(nL)]

    def share(v):
        s = sum(max(x, 0.0) for x in v) + 1e-9
        return [max(x, 0.0) / s for x in v]
    fs = share(mlp_d); gs = share(gmlp_d)
    excess = [fs[L] - gs[L] for L in range(nL)]                                   # fact-specific MLP importance (controls detokenizer)
    top_mlp = sorted(range(nL), key=lambda L: -mlp_d[L])[:3]
    top_excess = sorted(range(nL), key=lambda L: -excess[L])[:3]
    top_attn = sorted(range(nL), key=lambda L: -attn_d[L])[:3]
    print(f"  {len(facts)} facts | raw top-MLP {[(L, round(mlp_d[L], 1)) for L in top_mlp]} | "
          f"FACT-SPECIFIC top-MLP (vs generic) {[(L, round(depth[L], 2), round(excess[L], 2)) for L in top_excess]}")
    return {"model": model_id.split("/")[-1], "rope": not is_gpt2, "n_layers": nL, "n_facts": len(facts),
            "base_fact_nll": base, "attn_dNLL": attn_d, "mlp_dNLL": mlp_d, "generic_mlp_dNLL": gmlp_d, "depth": depth,
            "top_mlp_layers": [{"layer": L, "depth": depth[L], "dNLL": mlp_d[L]} for L in top_mlp],
            "top_factspecific_mlp": [{"layer": L, "depth": depth[L], "excess": excess[L], "fact_dNLL": mlp_d[L], "generic_dNLL": gmlp_d[L]} for L in top_excess],
            "top_attn_layers": [{"layer": L, "depth": depth[L], "dNLL": attn_d[L]} for L in top_attn],
            "peak_mlp_depth": depth[int(np.argmax(mlp_d))], "peak_factspecific_depth": depth[int(np.argmax(excess))]}


def write_doc(out, docs):
    L = ["---", "title: Where do facts live?", "---", "", "# Where do facts live? — per-layer localization of factual recall", "",
         "The catalog is about *mechanisms*; this is about *knowledge*. For a set of factual completions "
         "(\"The capital of France is\" → \" Paris\", single-token objects only), measure the object's NLL, then "
         "mean-ablate each layer's **MLP** in turn. **The confound:** raw fact-ΔNLL is dominated by the *early* MLPs — "
         "but those (MLP0 the [detokenizer](../operators/mlp_detokenizer.md)) carry **all** token processing, not "
         "facts specifically. So we control against **generic** prose-NLL ablation and report the **fact-specific "
         "excess** = (a layer's share of fact-importance) − (its share of generic-importance). That is where facts "
         "are hurt *disproportionately* — the natural-history analog of the ROME mid-layer-MLP store.", "",
         "| model | #facts | raw top-MLP (depth, ΔNLL) — *detokenizer-dominated* | **fact-specific** top-MLP (depth, excess) | fact-specific peak depth |",
         "|---|---|---|---|---|"]
    for r in out["results"]:
        if "base_fact_nll" not in r:
            continue
        tm = "; ".join(f"L{x['layer']} ({x['depth']:.2f}, {x['dNLL']:+.1f})" for x in r["top_mlp_layers"])
        fx = "; ".join(f"L{x['layer']} ({x['depth']:.2f}, {x['excess']:+.2f})" for x in r["top_factspecific_mlp"])
        L.append(f"| {r['model']} | {r['n_facts']} | {tm} | **{fx}** | {r['peak_factspecific_depth']:.2f} |")
    L += ["", "_**Finding.** Raw fact-ΔNLL is dominated by the very-early detokenizer MLPs (L0–1) — they carry all "
          "token processing. Once that is controlled for, the **fact-specific** MLP importance concentrates in "
          "**early-mid layers** (depth ≈ 0.1–0.27 — gpt2 L3, gpt2-medium L3, gpt2-large L7–9, Gemma L3, Qwen L3), "
          "broadly consistent with the ROME early-mid MLP knowledge store, recovered here as natural history. The "
          "recurring outliers show again: **Gemma** adds a **late** fact site (L21–22), and **Llama** localizes facts "
          "late (L14–15, depth ≈0.93). The excess magnitudes are small — facts are also distributed — but the "
          "disproportionate fact-specific load sits early-mid._", ""]
    L += ["", "_The **raw** column is the detokenizer confound (early MLPs hurt everything); the **fact-specific** "
          "column controls for it (fact-importance share minus generic-importance share) — that is where the *facts* "
          "live as opposed to general token processing. Proper causal localization is ROME-style subject-corruption "
          "tracing; this is a cheaper ablation-contrast proxy. Provisional, ~24 facts (capitals + a few), single-token "
          "objects. "
          "Data: [factual_recall_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/factual_recall_summary.json). "
          "Regenerate: [factual_recall.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/factual_recall.py)._"]
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "factual_recall.md").write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-ids", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/circuits"))
    p.add_argument("--docs", type=Path, default=Path("docs/circuits"))
    args = p.parse_args(argv)
    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.model_ids.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            results.append(run_model(mid, args, dev))
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})
        if dev == "cuda":
            torch.cuda.empty_cache()
    out = {"experiment": "where do facts live — per-layer factual-recall localization", "results": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "factual_recall_summary.json").write_text(json.dumps(out, indent=2, default=float))
    write_doc(out, args.docs)
    print(f"\n[done] {len([r for r in results if 'base_fact_nll' in r])} models → {args.outdir / 'factual_recall_summary.json'}")
    return out


if __name__ == "__main__":
    main()
