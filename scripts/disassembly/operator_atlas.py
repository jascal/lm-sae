"""The operator CATALOG (survey matrix) — every (behavioural) operator measured across every model.

The per-op `operator_dossier.py` goes DEEP on one instruction in GPT-2. This goes WIDE: the full catalog of
behaviourally-maskable operators × all cached architectures, in a single survey matrix. For each (operator,
model) cell it records, uniformly and arch-generically (output_attentions + a mean-ablation hook on the attention
output projection — GPT-2 `c_proj` / RoPE `o_proj`):

  - SIGNAL    : max over heads of the head's mean attention mass on the operator's pattern (0..1) — is the op present?
  - N_HEADS   : how many heads carry it (mass > 0.15) — sparse vs distributed;
  - TOP/DEPTH : the strongest head and its relative depth — where in the stack the op lives;
  - CAUSAL    : ΔNLL on held-out prose when the op's top-K heads are mean-ablated — is it load-bearing?

Operators are the UNIVERSAL / addressing instructions that have a position-or-token attention mask, so they are
measurable in any architecture: prev-token, induction, duplicate, sink (no-op), self, local, structural (newline
landmark). The IOI CIRCUIT operators (name-mover / backup / negative / S-inhibition / coreference) are
literature-defined by direct-logit-attribution and have **no published head-set outside GPT-2**, so they are NOT
in this cross-model matrix — they are catalogued by `operator_dossier.py --op <name>` on GPT-2 (referenced in the
README). succession/greater-than is MLP-dominated (no clean attention head) — a documented gap, not a row. SSM
models (Mamba) have no heads — induction is present behaviourally (see `ssm_induction.py`) but unmappable here.

Output: `runs/disassembly/operators/atlas_summary.json` + `atlas.png` (the survey, in the operator-catalog tree).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# universal / addressing operators: each maps a token list -> the (query,key) mask it attends along.
UNIVERSAL = ["prevtok", "induction", "duplicate", "sink", "self", "local", "structural"]
KINDS = {"prevtok": "positional", "induction": "content", "duplicate": "content", "sink": "addressing",
         "self": "addressing", "local": "positional", "structural": "structural"}
CONTENT_OPS = {"induction", "duplicate"}                                            # use repeated-random probes
# the IOI circuit operators — GPT-2-only (literature DLA head-sets); catalogued by the per-op dossier, not here.
GPT2_CIRCUIT = {"name_mover": [(9, 6), (9, 9), (10, 0), (10, 10)],
                "backup_name_mover": [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (11, 2)],
                "negative_mover": [(10, 7), (11, 10)], "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
                "coreference": [(9, 0)]}


def masks(toks, newline_id):
    ca = np.array(toks); n = len(ca); qi = np.arange(n); pv = np.full(n, -1); pv[1:] = ca[:-1]
    qq = qi[:, None]; kk = qi[None, :]
    last_nl = np.full(n, -1)                                                        # nearest preceding newline index
    if newline_id is not None:
        seen = -1
        for i in range(n):
            last_nl[i] = seen
            if ca[i] == newline_id:
                seen = i
    return {
        "prevtok": (kk == qq - 1) & (qq >= 1),
        "induction": (pv[None, :] == ca[:, None]) & (kk >= 1) & (kk < qq),
        "duplicate": (ca[None, :] == ca[:, None]) & (kk < qq),
        "sink": (kk == 0) & (qq >= 1),
        "self": (kk == qq) & (qq >= 1),
        "local": (kk < qq) & (kk >= qq - 3) & (qq >= 1),
        "structural": (kk == last_nl[:, None]) & (last_nl[:, None] >= 0),
    }


def run_model(model_id, args, dev):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    cfg = model.config; H = cfg.num_attention_heads; nL = cfg.num_hidden_layers; NH = nL * H
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
    is_gpt2 = hasattr(model, "transformer") and hasattr(model.transformer, "h")
    if is_gpt2:
        oproj = [b.attn.c_proj for b in model.transformer.h]
    else:
        oproj = [ly.self_attn.o_proj for ly in model.model.layers]
    nlids = tok("\n", add_special_tokens=False)["input_ids"]; newline_id = nlids[0] if nlids else None
    V = cfg.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
    rng = np.random.default_rng(args.seed)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    pids = tok(prose)["input_ids"]
    prose_chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]
    rep_seqs = [(lambda s: s + s)([int(x) for x in rng.integers(lo, hi, args.rep_len)]) for _ in range(args.chunks)]

    # ---- behavioural mass per head, accumulated on the appropriate probe set per op ----
    mass = {op: np.zeros(NH) for op in UNIVERSAL}; ntot = {op: 0 for op in UNIVERSAL}
    for op in UNIVERSAL:
        probe = rep_seqs if op in CONTENT_OPS else prose_chunks
        with torch.no_grad():
            for s in probe:
                o = model(input_ids=torch.tensor([s], device=dev), output_attentions=True)
                M = masks(s, newline_id)[op]; ntot[op] += int(M.sum())
                if M.sum() == 0:
                    continue
                for L in range(nL):
                    at = o.attentions[L][0].float().cpu().numpy()
                    mass[op][L * H:(L + 1) * H] += (at * M[None]).sum((1, 2))
        mass[op] /= max(ntot[op], 1)

    # ---- mean-ablation harness (arch-generic: o_proj/c_proj input slice -> corpus mean) ----
    cap = {L: [] for L in range(nL)}
    hk = [oproj[L].register_forward_pre_hook(
        (lambda L: lambda m, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for s in prose_chunks:
            model(input_ids=torch.tensor([s], device=dev))
    for h in hk:
        h.remove()
    meanv = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    def lm_nll(ablate):
        by = {}
        for (L, h) in ablate:
            by.setdefault(L, []).append(h)
        hs = []
        for L, hss in by.items():
            def mk(L, hss):
                def hook(m, inp):
                    x = inp[0].clone()
                    for h in hss:
                        x[..., h * hd:(h + 1) * hd] = meanv[L][h * hd:(h + 1) * hd].to(x.dtype)
                    return (x,)
                return hook
            hs.append(oproj[L].register_forward_pre_hook(mk(L, hss)))
        tot = 0.0; n = 0
        try:
            with torch.no_grad():
                for s in prose_chunks:
                    lp = F.log_softmax(model(input_ids=torch.tensor([s], device=dev)).logits[0, :-1].float(), -1)
                    y = torch.tensor(s[1:], device=dev); tot += float(-lp[torch.arange(len(y)), y].sum()); n += len(y)
        finally:
            for h in hs:
                h.remove()
        return tot / max(n, 1)
    base = lm_nll(set())

    cells = {}
    for op in UNIVERSAL:
        order = np.argsort(-mass[op]); top = int(order[0]); sig = float(mass[op][top])
        nh = int((mass[op] > args.head_thr).sum())
        topk = [(int(i) // H, int(i) % H) for i in order[: args.ablate_k] if mass[op][int(i)] > 0.05]
        dnll = (lm_nll(set(topk)) - base) if topk else 0.0
        cells[op] = {"signal": sig, "n_heads": nh, "top_head": f"{top // H}.{top % H}",
                     "top_depth": round((top // H) / max(nL - 1, 1), 2), "causal_dNLL": float(dnll),
                     "ablated": [f"{L}.{h}" for L, h in topk]}
    return {"model": model_id.split("/")[-1], "arch": "GPT-2/absolute" if is_gpt2 else "RoPE",
            "n_layers": nL, "n_heads": H, "base_nll": base, "cells": cells}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=20)
    p.add_argument("--rep-len", type=int, default=24)
    p.add_argument("--head-thr", type=float, default=0.15, help="mass above which a head 'carries' the op")
    p.add_argument("--ablate-k", type=int, default=3, help="top heads ablated for the causal column")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/operators"))
    args = p.parse_args(argv)

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = run_model(mid, args, dev)
            if dev == "cuda":
                torch.cuda.empty_cache()
            results.append(r)
            print(f"  {r['arch']} {r['n_layers']}L x {r['n_heads']}H  base-NLL {r['base_nll']:.3f}")
            for op in UNIVERSAL:
                c = r["cells"][op]
                print(f"    {op:>11} ({KINDS[op]:>10}): signal {c['signal']:.3f}  heads {c['n_heads']:>2}  top {c['top_head']:>5} (d{c['top_depth']:.2f})  causal ΔNLL {c['causal_dNLL']:+.3f}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid, "error": str(e)})

    out = {"experiment": "operator atlas — behavioural operators x models (universal/addressing catalog)",
           "operators": UNIVERSAL, "kinds": KINDS, "gpt2_circuit_ops": {k: [f"{L}.{h}" for L, h in v] for k, v in GPT2_CIRCUIT.items()},
           "note_circuit": "IOI circuit ops are GPT-2-only (literature DLA head-sets); see operator_dossier.py dossiers.",
           "note_gaps": "succession/greater-than = MLP-dominated (no clean attention head); SSM (Mamba) = no heads (see ssm_induction.py).",
           "results": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "atlas_summary.json").write_text(json.dumps(out, indent=2, default=float))

    ok = [r for r in results if "cells" in r]
    print("\n[ATLAS] signal (max head mass on the op's pattern) per operator x model — the survey:")
    hdr = f"  {'operator':>11} {'kind':>10} | " + " | ".join(f"{r['model'][:10]:>10}" for r in ok)
    print(hdr)
    for op in UNIVERSAL:
        print(f"  {op:>11} {KINDS[op]:>10} | " + " | ".join(f"{r['cells'][op]['signal']:>10.3f}" for r in ok))
    print("\n[ATLAS] causal ΔNLL (mean-ablate top-{}) per operator x model — load-bearing?:".format(args.ablate_k))
    print(hdr)
    for op in UNIVERSAL:
        print(f"  {op:>11} {KINDS[op]:>10} | " + " | ".join(f"{r['cells'][op]['causal_dNLL']:>+10.3f}" for r in ok))
    print("\n[circuit ops — GPT-2-only, see dossiers] " + ", ".join(f"{k} ({len(v)}h)" for k, v in GPT2_CIRCUIT.items()))
    print("[gaps] succession/greater-than = MLP-dominated; SSM = no heads (induction behaviourally present, ssm_induction.py)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (aS, aC) = plt.subplots(1, 2, figsize=(max(11, 2.0 * len(ok) + 5), 6.0))
        S = np.array([[r["cells"][op]["signal"] for r in ok] for op in UNIVERSAL])
        C = np.array([[r["cells"][op]["causal_dNLL"] for r in ok] for op in UNIVERSAL])
        for ax, M, ttl, cmap, fmt in ((aS, S, "behavioural signal (op present?)", "viridis", "{:.2f}"),
                                      (aC, C, "causal ΔNLL (load-bearing?)", "RdBu_r", "{:+.2f}")):
            vmax = np.abs(M).max() or 1.0
            im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=(-vmax if ax is aC else 0), vmax=vmax)
            ax.set_xticks(range(len(ok))); ax.set_xticklabels([f"{r['model'][:10]}\n{r['arch'].split('/')[0]}" for r in ok], fontsize=7, rotation=20, ha="right")
            ax.set_yticks(range(len(UNIVERSAL))); ax.set_yticklabels([f"{op}\n{KINDS[op]}" for op in UNIVERSAL], fontsize=8)
            for i in range(len(UNIVERSAL)):
                for j in range(len(ok)):
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6,
                            color="w" if (ax is aS and M[i, j] < 0.5 * vmax) or (ax is aC and abs(M[i, j]) > 0.5 * vmax) else "k")
            fig.colorbar(im, ax=ax, fraction=0.046); ax.set_title(ttl, fontsize=10)
        fig.suptitle("Operator atlas — universal/addressing operators across architectures (signal + causal)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(args.outdir / "atlas.png", dpi=130)
        print(f"[fig] {args.outdir / 'atlas.png'}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.outdir / 'atlas_summary.json'}")
    return out


if __name__ == "__main__":
    main()
