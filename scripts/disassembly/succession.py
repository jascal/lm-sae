"""Localizing succession (the +1 / greater-than operator) — filling a documented catalog gap.

The operator catalog lists **succession / greater-than** as a *gap*: "MLP-dominated; no clean attention head". This
puts data behind that claim. Task: a run of consecutive single-token numbers (" 3 4 5 6 7") → predict the next
(" 8"); the **succession-NLL** is the metric the operator serves. With all attention intact, mean-ablate each
layer's **MLP**, and with all MLPs intact mean-ablate each layer's **attention** — the layers whose ablation most
raises succession-NLL are where the increment is computed. Confirms (or not) the MLP-dominance claim and localizes
it, cross-model. Arch-generic (`circuit_content_patch._arch` + `mlp_atlas.mlp_blocks`).
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
    tok = AutoTokenizer.from_pretrained(model_id); rng = np.random.default_rng(args.seed)

    # single-token numbers " 1".." 250"
    num = {}
    for v in range(1, 260):
        i = tok(f" {v}", add_special_tokens=False)["input_ids"]
        if len(i) == 1:
            num[v] = i[0]
    runlen = args.runlen
    starts = [a0 for a0 in num if all((a0 + j) in num for j in range(runlen + 1))]
    if len(starts) < 8:
        raise RuntimeError(f"only {len(starts)} number runs for {model_id}")
    seqs = []
    for _ in range(args.probes):
        a0 = starts[int(rng.integers(0, len(starts)))]
        ids = [num[a0 + j] for j in range(runlen)]
        if not is_gpt2:
            ids = [tok.bos_token_id] + ids if tok.bos_token_id is not None else ids
        seqs.append((ids, num[a0 + runlen]))

    # corpus mean for ablation (use the succession sequences themselves)
    cap = {L: [] for L in range(nL)}; mcap = {L: [] for L in range(nL)}
    hks = [oproj[L].register_forward_pre_hook((lambda L: lambda mod, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    mhks = [mlps[L].register_forward_hook((lambda L: lambda mod, i, o: mcap[L].append((o[0] if isinstance(o, tuple) else o).detach().reshape(-1, (o[0] if isinstance(o, tuple) else o).shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for ids, _ in seqs:
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

    def succ_nll(attn_L=None, mlp_L=None, all_attn=False):
        hs = []
        if all_attn:
            for L in range(nL):
                hs.append(oproj[L].register_forward_pre_hook(attn_hook(L)))
        if attn_L is not None:
            hs.append(oproj[attn_L].register_forward_pre_hook(attn_hook(attn_L)))
        if mlp_L is not None:
            hs.append(mlps[mlp_L].register_forward_hook(mlp_hook(mlp_L)))
        tot = 0.0
        try:
            with torch.no_grad():
                for ids, target in seqs:
                    lp = F.log_softmax(m(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float(), -1)
                    tot += float(-lp[target])
        finally:
            for x in hs:
                x.remove()
        return tot / len(seqs)

    base = succ_nll()
    attn_d = [succ_nll(attn_L=L) - base for L in range(nL)]
    mlp_d = [succ_nll(mlp_L=L) - base for L in range(nL)]
    all_attn_d = succ_nll(all_attn=True) - base
    depth = [L / (nL - 1) if nL > 1 else 0.0 for L in range(nL)]
    top_mlp = sorted(range(nL), key=lambda L: -mlp_d[L])[:3]
    top_attn = sorted(range(nL), key=lambda L: -attn_d[L])[:3]
    sum_mlp = sum(max(x, 0) for x in mlp_d); sum_attn = sum(max(x, 0) for x in attn_d)
    print(f"  base {base:.2f} | ΣMLP {sum_mlp:.1f} vs Σattn {sum_attn:.1f} | top-MLP {[(L, round(mlp_d[L], 1)) for L in top_mlp]} "
          f"| top-attn {[(L, round(attn_d[L], 1)) for L in top_attn]}")
    return {"model": model_id.split("/")[-1], "rope": not is_gpt2, "n_layers": nL, "n_runs": len(starts),
            "base_succ_nll": base, "attn_dNLL": attn_d, "mlp_dNLL": mlp_d, "all_attn_dNLL": all_attn_d, "depth": depth,
            "sum_mlp": sum_mlp, "sum_attn": sum_attn, "mlp_dominance": sum_mlp / (sum_mlp + sum_attn + 1e-9),
            "top_mlp_layers": [{"layer": L, "depth": depth[L], "dNLL": mlp_d[L]} for L in top_mlp],
            "top_attn_layers": [{"layer": L, "depth": depth[L], "dNLL": attn_d[L]} for L in top_attn]}


def write_doc(out, docs):
    L = ["---", "title: Succession (the +1 operator)", "---", "", "# Localizing succession — the +1 / greater-than operator", "",
         "The operator catalog lists **succession / greater-than** as a *gap* (\"MLP-dominated; no clean attention "
         "head\"). This puts data behind it. Task: a run of consecutive single-token numbers (\" 3 4 5 6 7\") → predict "
         "the next; the **succession-NLL** is the metric. With all attention intact, mean-ablate each layer's MLP; "
         "with all MLPs intact, each layer's attention — the layers whose ablation most raises succession-NLL are "
         "where the increment is computed.", "",
         "| model | runs | base NLL | **MLP-dominance** (ΣMLP / (ΣMLP+Σattn)) | top succession-MLP (depth, ΔNLL) | top succession-attn (depth, ΔNLL) |",
         "|---|---|---|---|---|---|"]
    for r in out["results"]:
        if "base_succ_nll" not in r:
            continue
        tm = "; ".join(f"L{x['layer']} ({x['depth']:.2f}, {x['dNLL']:+.1f})" for x in r["top_mlp_layers"])
        ta = "; ".join(f"L{x['layer']} ({x['depth']:.2f}, {x['dNLL']:+.1f})" for x in r["top_attn_layers"])
        L.append(f"| {r['model']} | {r['n_runs']} | {r['base_succ_nll']:.2f} | **{r['mlp_dominance']:.0%}** | {tm} | {ta} |")
    L += ["", "_**Finding: succession is overwhelmingly MLP-computed** (95–100% MLP-dominance) and lives in the "
          "**early–mid MLPs** (GPT-2-small L0–L2, gpt2-large L7–9) — putting data behind the catalog's "
          "\"MLP-dominated, no clean attention head\" gap. **GPT-2 family only:** the RoPE tokenizers (Gemma, Llama, "
          "Qwen) have **no single-token numbers** (they split ` 1` into multiple tokens), so consecutive number runs "
          "don't exist — which is itself why succession studies use GPT-2._", "",
          "_MLP-dominance = the MLP layers' share of the total (positive) ablation damage to succession; **>50% "
          "confirms the catalog's MLP-dominated claim**, and the top-MLP layers say *where* the increment lives. "
          "ΔNLL = succession-NLL rise when that layer's MLP / attention is mean-ablated. Provisional, single-token "
          "number runs (length " + str(out.get("runlen", "?")) + "). Data: "
          "[succession_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/succession_summary.json). "
          "Regenerate: [succession.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/succession.py). "
          "See the [operator catalog](../operators/README.md) gaps._"]
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "succession.md").write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-ids", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--runlen", type=int, default=5)
    p.add_argument("--probes", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/operators"))
    p.add_argument("--docs", type=Path, default=Path("docs/operators"))
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
    out = {"experiment": "localizing succession (the +1 operator)", "runlen": args.runlen, "results": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "succession_summary.json").write_text(json.dumps(out, indent=2, default=float))
    write_doc(out, args.docs)
    print(f"\n[done] {len([r for r in results if 'base_succ_nll' in r])} models → {args.outdir / 'succession_summary.json'}")
    return out


if __name__ == "__main__":
    main()
