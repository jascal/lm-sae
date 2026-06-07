"""Key-only causal path-patch, cross-model — the CAUSAL complement to #26's representational test.

`validate_new_edges.py` showed (causally, GPT-2) that removing an early SINK head from the prev-token head's KEY
collapses its prev-token attention — the absolute-position signal the key reads is *piped in* via the residual.
`cross_model_positional.py` (#26) showed *representationally* that only GPT-2's prev-token key is position-encoded
(RoPE keys are token-encoded; position rides in the rotation), and deferred the *causal* cross-model patch as the
heavier step because a key-only patch must respect each model's RoPE. This is that patch.

Done as a forward-pass intervention so each model applies its OWN RoPE/GQA/RMSNorm: for the top prev-token head B,
remove each upstream head A's direct contribution from B's KEY *content* (pre-rotation) and re-read B's attention.
  RoPE family:  hook B.self_attn.k_proj — feed it `input_layernorm(resid_B − A_out)` (q untouched → key-only).
  GPT-2:        hook B.attn.c_attn — overwrite the key slice of its output with `c_attn(ln_1(resid_B − A_out))[k]`.
The model rotates the patched key, so the relative-position match is preserved; only the key *content* changes.
Prediction: removing a SINK head collapses GPT-2's prev-token attention (its key carries absolute position), but
NO upstream head collapses a RoPE model's (the rotation, untouched, still aligns q−1) — a standout collapser in
GPT-2, flat in Gemma/Llama/Qwen. A_out is each head's exact additive write (`o_proj(only-A) − o_proj(0)`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _arch(model):
    cfg = model.config; H = cfg.num_attention_heads; nkv = getattr(cfg, "num_key_value_heads", None) or H
    if hasattr(model, "model") and hasattr(model.model, "layers"):                 # Gemma / Llama / Qwen
        hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
        L = list(model.model.layers)
        return dict(is_gpt2=False, layers=L, oproj=[ly.self_attn.o_proj for ly in L],
                    norm=[ly.input_layernorm for ly in L], kproj=[ly.self_attn.k_proj for ly in L], hd=hd, H=H, nkv=nkv)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):           # GPT-2 (combined c_attn, no RoPE)
        L = list(model.transformer.h)
        return dict(is_gpt2=True, layers=L, oproj=[b.attn.c_proj for b in L], norm=[b.ln_1 for b in L],
                    cattn=[b.attn.c_attn for b in L], hd=cfg.n_embd // H, H=H, nkv=H, d=cfg.n_embd)
    raise SystemExit("unknown architecture")


def run_model(model_id, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    a = _arch(model); H = a["H"]; hd = a["hd"]; nL = model.config.num_hidden_layers; NH = nL * H
    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8][: args.chunks]

    def pmask(Lc):
        qi = np.arange(Lc); return (qi[None, :] == (qi[:, None] - 1)) & (qi[:, None] >= 1)

    def skmask(Lc):
        qi = np.arange(Lc); return (qi[None, :] == 0) & (qi[:, None] >= 1)

    # ---- pass 1: prev-token head B + per-head sink mass ----
    pt = np.zeros(NH); sk = np.zeros(NH); ntok = 0
    with torch.no_grad():
        for c in chunks:
            o = model(input_ids=torch.tensor([c], device=dev), output_attentions=True)
            Lc = len(c); ntok += Lc; PT = pmask(Lc); SK = skmask(Lc)
            for L in range(nL):
                at = o.attentions[L][0].float().cpu().numpy()
                pt[L * H:(L + 1) * H] += (at * PT[None]).sum((1, 2)); sk[L * H:(L + 1) * H] += (at * SK[None]).sum((1, 2))
    pt /= max(ntok, 1); sk /= max(ntok, 1)
    B = int(np.argmax(pt)); LB, hB = B // H, B % H
    print(f"  prev-token head {LB}.{hB} (Δ=1 mass {pt[B]:.3f})")
    if LB == 0:
        return {"model": model_id, "note": "prev-token head in layer 0 — no upstream heads", "prevtok_head": f"{LB}.{hB}"}
    upstream = [(L, h) for L in range(LB) for h in range(H)][: args.max_upstream]

    oprojs = a["oproj"]; norm_mods = a["norm"]

    def head_contrib(L, captured, h):                                          # head h's exact additive residual write
        x = torch.zeros_like(captured); x[..., h * hd:(h + 1) * hd] = captured[..., h * hd:(h + 1) * hd]
        return oprojs[L](x) - oprojs[L](torch.zeros_like(captured[..., :1, :]))

    # ---- pass 2: per chunk, capture block-LB input + o_proj inputs; patch each upstream head's KEY contribution ----
    cap = {}                                                                   # cap['resid']=block-LB input; cap[L]=o_proj input
    hooks = []
    hooks.append(a["layers"][LB].register_forward_pre_hook(lambda m, inp: cap.__setitem__("resid", inp[0].detach())))
    for L in range(LB):
        hooks.append(oprojs[L].register_forward_pre_hook((lambda L: lambda m, inp: cap.__setitem__(L, inp[0].detach()))(L)))
    clean_pt = 0.0; patched_pt = {u: 0.0 for u in upstream}; tot = 0; sane = None
    with torch.no_grad():
        for c in chunks:
            cap.clear()
            o = model(input_ids=torch.tensor([c], device=dev), output_attentions=True)
            Lc = len(c); tot += Lc; PT = pmask(Lc)
            clean_pt += float((o.attentions[LB][0, hB].float().cpu().numpy() * PT).sum())
            resid = cap["resid"]
            for (La, ha) in upstream:
                dvec = head_contrib(La, cap[La], ha)                           # (1,seq,d) head A's residual write
                key_in = norm_mods[LB](resid - dvec)                           # normed (resid − A_out): key-only content
                if a["is_gpt2"]:
                    d = a["d"]; kslice = a["cattn"][LB](key_in)[..., d:2 * d]   # patched key projection (no RoPE)

                    def hook(m, inp, out, _k=kslice, _d=d):
                        out = out.clone(); out[..., _d:2 * _d] = _k; return out
                    hk = a["cattn"][LB].register_forward_hook(hook)
                else:
                    def pre(m, inp, _ki=key_in):                               # feed k_proj the patched content; model rotates it
                        return (_ki,) + inp[1:]
                    hk = a["kproj"][LB].register_forward_pre_hook(pre)
                try:
                    op = model(input_ids=torch.tensor([c], device=dev), output_attentions=True)
                    patched_pt[(La, ha)] += float((op.attentions[LB][0, hB].float().cpu().numpy() * PT).sum())
                finally:
                    hk.remove()
            if sane is None:                                                   # patch with zero A_out -> attention unchanged
                kz = norm_mods[LB](resid)
                if a["is_gpt2"]:
                    d = a["d"]; ksl = a["cattn"][LB](kz)[..., d:2 * d]

                    def hz(m, inp, out, _k=ksl, _d=d):
                        out = out.clone(); out[..., _d:2 * _d] = _k; return out
                    hk = a["cattn"][LB].register_forward_hook(hz)
                else:
                    def prez(m, inp, _ki=kz):
                        return (_ki,) + inp[1:]
                    hk = a["kproj"][LB].register_forward_pre_hook(prez)
                with torch.no_grad():
                    az = model(input_ids=torch.tensor([c], device=dev), output_attentions=True).attentions[LB][0, hB]
                hk.remove()
                sane = float(np.abs(az.float().cpu().numpy() - o.attentions[LB][0, hB].float().cpu().numpy()).max())
    for h in hooks:
        h.remove()
    print(f"  [sanity] zero-patch attn vs clean max|Δ| = {sane:.2e}")

    clean = clean_pt / max(tot, 1)
    rows = []
    for (La, ha) in upstream:
        pat = patched_pt[(La, ha)] / max(tot, 1)
        rows.append({"head": f"{La}.{ha}", "rel_collapse": (clean - pat) / clean if clean > 1e-6 else 0.0,
                     "sink_mass": float(sk[La * H + ha])})
    rels = np.array([r["rel_collapse"] for r in rows])
    med = float(np.median(rels)); mad = float(np.median(np.abs(rels - med)))
    for r in rows:
        r["z"] = (r["rel_collapse"] - med) / (1.4826 * mad + 1e-6)
    rows.sort(key=lambda r: -r["rel_collapse"])
    top = rows[0]
    return {"model": model_id, "rope": not a["is_gpt2"], "pos": "RoPE" if not a["is_gpt2"] else "absolute",
            "prevtok_head": f"{LB}.{hB}", "prevtok_mass": float(pt[B]), "clean_prevtok": clean,
            "n_upstream": len(upstream), "sanity": sane, "top_head": top["head"], "top_collapse": top["rel_collapse"],
            "top_z": top["z"], "top_is_sink": top["sink_mass"] > 0.2, "median_collapse": med, "top_rows": rows[:6]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--max-upstream", type=int, default=48, help="upstream heads to patch (earliest layers first)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/gemma/key_patch_cross_model_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/key_patch_cross_model.png"))
    args = p.parse_args(argv)

    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = run_model(mid, args)
        except Exception as e:  # pragma: no cover - gated/missing model, OOM
            print(f"  [skip] {e}")
            r = {"model": mid, "error": str(e)}
        results.append(r)
        if "top_collapse" in r:
            print(f"  top KEY-patch collapser {r['top_head']} ({'SINK' if r['top_is_sink'] else 'non-sink'}): "
                  f"prev-token collapse {r['top_collapse']:+.0%} (robust z {r['top_z']:.1f} vs upstream bulk); "
                  f"median {r['median_collapse']:+.1%}")

    out = {"experiment": "key-only causal path-patch cross-model (does removing an upstream head's KEY content collapse prev-token?)",
           "ctx": args.ctx, "chunks": args.chunks, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    ok = [r for r in results if "top_collapse" in r]
    print("\n[cross-model] key-only causal patch — is prev-token attention carried by KEY CONTENT (absolute) or rotation (RoPE)?")
    for r in ok:
        live = "YES (key carries position)" if (r["top_z"] > 4 and r["top_collapse"] > 0.1) else "no (rotation carries position)"
        print(f"  {r['model']:>22} [{r['pos']:>8}]: prev-tok {r['prevtok_head']:>5}  top collapse {r['top_collapse']:+.0%} "
              f"z{r['top_z']:.1f} {'sink' if r['top_is_sink'] else 'non-sink'}  => {live}")
    gpt2 = next((r for r in ok if not r["rope"]), None); rope = [r for r in ok if r["rope"]]
    if gpt2 and rope:
        gpt2_live = gpt2["top_z"] > 4 and gpt2["top_collapse"] > 0.1
        rope_flat = all(not (r["top_z"] > 4 and r["top_collapse"] > 0.1) for r in rope)
        rmax = max(r["top_collapse"] for r in rope)
        if gpt2_live and rope_flat:
            verdict = (f"CONFIRMED CAUSALLY — GPT-2's prev-token attention is carried by KEY CONTENT: removing the "
                       f"{'SINK ' if gpt2['top_is_sink'] else ''}head {gpt2['top_head']} from its key collapses prev-token "
                       f"{gpt2['top_collapse']:+.0%} (z {gpt2['top_z']:.1f}), while NO upstream head collapses any RoPE model "
                       f"(max {rmax:+.0%}) — their rotation, untouched by the key-content patch, still aligns q−1. The causal "
                       f"complement to #26's representational result: absolute-position content (piped from sink heads) in "
                       f"GPT-2 vs intrinsic rotation in RoPE. Same GPT-2-family-is-special pattern as the sink + ceiling results.")
        else:
            verdict = f"MIXED: GPT-2 key-causal={gpt2_live}, all-RoPE-flat={rope_flat} — see table"
        print(f"\n[verdict] {verdict}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (axC, axZ) = plt.subplots(1, 2, figsize=(12.6, 5.0))
        names = [r["model"].split("/")[-1] for r in ok]
        cols = ["#d62728" if not r["rope"] else "#1f77b4" for r in ok]
        axC.bar(range(len(ok)), [r["top_collapse"] for r in ok], color=cols, edgecolor="k")
        axC.axhline(0.1, color="k", lw=0.6, ls=":"); axC.set_xticks(range(len(ok)))
        axC.set_xticklabels([f"{n}\n{r['pos']}" for n, r in zip(names, ok)], fontsize=8)
        axC.set_ylabel("top upstream-head prev-token collapse (KEY patch)")
        axC.set_title("removing an upstream head from the prev-token KEY", fontsize=10)
        axZ.bar(range(len(ok)), [r["top_z"] for r in ok], color=cols, edgecolor="k")
        axZ.axhline(4, color="k", lw=0.6, ls=":"); axZ.set_xticks(range(len(ok)))
        axZ.set_xticklabels([f"{n}\n{r['pos']}" for n, r in zip(names, ok)], fontsize=8)
        axZ.set_ylabel("robust z of top collapser vs upstream bulk")
        axZ.set_title("standout collapser (abs, red) vs flat (RoPE, blue)", fontsize=10)
        fig.suptitle("Key-only causal path-patch: GPT-2 prev-token reads absolute position from the KEY; RoPE from the rotation", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
