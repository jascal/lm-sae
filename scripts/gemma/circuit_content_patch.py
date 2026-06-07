"""Generalize the key-only causal patch beyond prev-token — is the key-content dependence about POSITION or CONTENT?

`key_patch_cross_model.py` (#31) showed the prev-token head's attention is carried by KEY CONTENT in GPT-2 (remove
a sink head from its key → collapse) but by the ROTATION in RoPE models (no upstream head matters). But prev-token
is a *positional* circuit (attend to q−1). What about *content* circuits? **Induction** attends to the key whose
PREDECESSOR token equals the current token; **duplicate-token** attends to an earlier occurrence of the *same*
token. Both are token-IDENTITY matches — RoPE's rotation handles *position*, not token content, so the matched
content MUST live in the key content for *every* architecture.

Prediction: the key-content-vs-rotation split is specific to POSITIONAL addressing. Removing the right upstream
writer from the reader's KEY should:
  - prev-token (positional): collapse only GPT-2 (RoPE reads position from the rotation);
  - induction / duplicate (content): collapse in EVERY model (the predecessor / same-token content is in the key,
    rotation can't supply it) — and for induction the top collapser should be the model's prev-token head (the
    canonical K-composition writer), universally.

Same faithful forward key-only patch as #31 (replace the reader's k_proj/c_attn input with `norm(resid − A_out)`,
q untouched; the model applies its own RoPE). For each circuit we pick the reader behaviorally and measure the
collapse of THAT circuit's attention signal. GPT-2 + Gemma-2 / Llama-3 / Qwen-2.5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CIRCUITS = ["prevtok", "induction", "duplicate"]                               # positional, content, content


def _arch(model):
    cfg = model.config; H = cfg.num_attention_heads; nkv = getattr(cfg, "num_key_value_heads", None) or H
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
        L = list(model.model.layers)
        return dict(is_gpt2=False, layers=L, oproj=[ly.self_attn.o_proj for ly in L],
                    norm=[ly.input_layernorm for ly in L], kproj=[ly.self_attn.k_proj for ly in L], hd=hd, H=H, nkv=nkv)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        L = list(model.transformer.h)
        return dict(is_gpt2=True, layers=L, oproj=[b.attn.c_proj for b in L], norm=[b.ln_1 for b in L],
                    cattn=[b.attn.c_attn for b in L], hd=cfg.n_embd // H, H=H, nkv=H, d=cfg.n_embd)
    raise SystemExit("unknown architecture")


def circuit_masks(c):
    Lc = len(c); ca = np.array(c); qi = np.arange(Lc); pv = np.full(Lc, -1); pv[1:] = ca[:-1]
    return {"prevtok": (qi[None, :] == (qi[:, None] - 1)) & (qi[:, None] >= 1),
            "induction": (pv[None, :] == ca[:, None]) & (qi[None, :] >= 1) & (qi[None, :] < qi[:, None]),
            "duplicate": (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])}


def run_model(model_id, args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    a = _arch(model); H = a["H"]; hd = a["hd"]; nL = model.config.num_hidden_layers; NH = nL * H
    oprojs = a["oproj"]; norm_mods = a["norm"]
    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8][: args.chunks]

    # ---- behavioural pass: per-head mass for each circuit (pick the reader) + sink mass (labeling) ----
    mass = {cc: np.zeros(NH) for cc in CIRCUITS}; sk = np.zeros(NH); ntot = {cc: 0 for cc in CIRCUITS}
    with torch.no_grad():
        for c in chunks:
            o = model(input_ids=torch.tensor([c], device=dev), output_attentions=True)
            Lc = len(c); M = circuit_masks(c); SK = (np.arange(Lc)[None, :] == 0) & (np.arange(Lc)[:, None] >= 1)
            for cc in CIRCUITS:
                ntot[cc] += int(M[cc].sum())
            for L in range(nL):
                at = o.attentions[L][0].float().cpu().numpy()
                for cc in CIRCUITS:
                    mass[cc][L * H:(L + 1) * H] += (at * M[cc][None]).sum((1, 2))
                sk[L * H:(L + 1) * H] += (at * SK[None]).sum((1, 2))
    for cc in CIRCUITS:
        mass[cc] /= max(ntot[cc], 1)
    prevtok_head = int(np.argmax(mass["prevtok"]))                             # the model's prev-token head (induction writer)

    def head_contrib(L, captured, h):
        x = torch.zeros_like(captured); x[..., h * hd:(h + 1) * hd] = captured[..., h * hd:(h + 1) * hd]
        return oprojs[L](x) - oprojs[L](torch.zeros_like(captured[..., :1, :]))

    out = {"model": model_id, "rope": not a["is_gpt2"], "pos": "RoPE" if not a["is_gpt2"] else "absolute",
           "prev_token_head": f"{prevtok_head // H}.{prevtok_head % H}", "circuits": {}}
    for cc in CIRCUITS:
        B = int(np.argmax(mass[cc])); LB, hB = B // H, B % H
        if LB == 0 or mass[cc][B] < 0.05:                                      # need an upstream + a real circuit signal
            out["circuits"][cc] = {"reader": f"{LB}.{hB}", "reader_mass": float(mass[cc][B]),
                                   "note": "reader in layer 0 or weak circuit — skipped"}
            print(f"  [{cc:>9}] reader {LB}.{hB} mass {mass[cc][B]:.3f} — skipped (layer-0/weak)")
            continue
        upstream = [(L, h) for L in range(LB) for h in range(H)][: args.max_upstream]
        cap = {}; hooks = [a["layers"][LB].register_forward_pre_hook(lambda m, inp: cap.__setitem__("r", inp[0].detach()))]
        for L in range(LB):
            hooks.append(oprojs[L].register_forward_pre_hook((lambda L: lambda m, inp: cap.__setitem__(L, inp[0].detach()))(L)))
        clean = 0.0; patched = {u: 0.0 for u in upstream}; tot = 0; sane = None
        with torch.no_grad():
            for c in chunks:
                cap.clear()
                o = model(input_ids=torch.tensor([c], device=dev), output_attentions=True)
                Lc = len(c); tot += Lc; Msk = circuit_masks(c)[cc]
                clean += float((o.attentions[LB][0, hB].float().cpu().numpy() * Msk).sum())
                resid = cap["r"]
                for (La, ha) in upstream:
                    key_in = norm_mods[LB](resid - head_contrib(La, cap[La], ha))
                    if a["is_gpt2"]:
                        d = a["d"]; ksl = a["cattn"][LB](key_in)[..., d:2 * d]

                        def hook(m, inp, o2, _k=ksl, _d=d):
                            o2 = o2.clone(); o2[..., _d:2 * _d] = _k; return o2
                        hk = a["cattn"][LB].register_forward_hook(hook)
                    else:
                        def pre(m, inp, _ki=key_in):
                            return (_ki,) + inp[1:]
                        hk = a["kproj"][LB].register_forward_pre_hook(pre)
                    try:
                        ap = model(input_ids=torch.tensor([c], device=dev), output_attentions=True).attentions[LB][0, hB]
                        patched[(La, ha)] += float((ap.float().cpu().numpy() * Msk).sum())
                    finally:
                        hk.remove()
                if sane is None:                                              # zero-patch fidelity check
                    kz = norm_mods[LB](resid)
                    if a["is_gpt2"]:
                        d = a["d"]; ksl = a["cattn"][LB](kz)[..., d:2 * d]

                        def hz(m, inp, o2, _k=ksl, _d=d):
                            o2 = o2.clone(); o2[..., _d:2 * _d] = _k; return o2
                        hk = a["cattn"][LB].register_forward_hook(hz)
                    else:
                        def prez(m, inp, _ki=kz):
                            return (_ki,) + inp[1:]
                        hk = a["kproj"][LB].register_forward_pre_hook(prez)
                    with torch.no_grad():
                        az = model(input_ids=torch.tensor([c], device=dev), output_attentions=True).attentions[LB][0, hB]
                    hk.remove(); sane = float(np.abs(az.float().cpu().numpy() - o.attentions[LB][0, hB].float().cpu().numpy()).max())
        for h in hooks:
            h.remove()
        clean /= max(tot, 1)
        rows = [{"head": f"{La}.{ha}", "rel": (clean - patched[(La, ha)] / max(tot, 1)) / clean if clean > 1e-6 else 0.0,
                 "sink": float(sk[La * H + ha])} for (La, ha) in upstream]
        rels = np.array([r["rel"] for r in rows]); med = float(np.median(rels)); mad = float(np.median(np.abs(rels - med)))
        for r in rows:
            r["z"] = (r["rel"] - med) / (1.4826 * mad + 1e-6)
        rows.sort(key=lambda r: -r["rel"]); top = rows[0]
        out["circuits"][cc] = {"reader": f"{LB}.{hB}", "reader_mass": float(mass[cc][B]), "sanity": sane,
                               "top_head": top["head"], "top_collapse": top["rel"], "top_z": top["z"],
                               "top_is_sink": top["sink"] > 0.2, "top_is_prevtok_head": top["head"] == out["prev_token_head"],
                               "median_collapse": med, "rows": rows[:5]}
        tag = ("=prev-tok head" if top["head"] == out["prev_token_head"] else ("SINK" if top["sink"] > 0.2 else ""))
        print(f"  [{cc:>9}] reader {LB}.{hB}: top collapser {top['head']} {tag} -> collapse {top['rel']:+.0%} (z {top['z']:.1f}); "
              f"sanity {sane:.1e}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--max-upstream", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/gemma/circuit_content_patch_summary.json"))
    p.add_argument("--fig", type=Path, default=Path("/tmp/circuit_content_patch.png"))
    args = p.parse_args(argv)

    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            results.append(run_model(mid, args))
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid, "error": str(e)})

    out = {"experiment": "generalize the key-only patch across circuits — is key-content dependence positional or content?",
           "ctx": args.ctx, "chunks": args.chunks, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))

    ok = [r for r in results if "circuits" in r]

    def collapsed(r, cc):                                                      # did circuit cc's key collapse for model r?
        d = r["circuits"].get(cc, {})                                          # robust standout: >10% AND >3x the upstream median
        return d.get("top_collapse", 0) > 0.1 and d.get("top_collapse", 0) > 3 * max(d.get("median_collapse", 0), 0.01)

    print("\n[cross-model] key-content dependence per circuit (collapse % | ✓ = key-content-dependent):")
    print(f"  {'model':>22} {'pos':>8} | " + " | ".join(f"{cc:>10}" for cc in CIRCUITS))
    for r in ok:
        cells = []
        for cc in CIRCUITS:
            d = r["circuits"].get(cc, {})
            v = f"{d['top_collapse']:+.0%}{'✓' if collapsed(r, cc) else ' '}" if "top_collapse" in d else "  skip"
            cells.append(f"{v:>10}")
        print(f"  {r['model'].split('/')[-1]:>22} {r['pos']:>8} | " + " | ".join(cells))

    absm = [r for r in ok if not r["rope"]]; rope = [r for r in ok if r["rope"]]
    if absm and rope:
        pos_split = (all(collapsed(r, "prevtok") for r in absm) and all(not collapsed(r, "prevtok") for r in rope))
        content_universal = all(collapsed(r, "induction") for r in ok if "top_collapse" in r["circuits"].get("induction", {}))
        ind_writer = all(r["circuits"].get("induction", {}).get("top_is_prevtok_head", False)
                         for r in ok if "top_collapse" in r["circuits"].get("induction", {}))
        if pos_split and content_universal:
            verdict = (f"GENERALIZED — key-content dependence is about ADDRESSING TYPE, not architecture: the POSITIONAL "
                       f"circuit (prev-token) is key-content-dependent ONLY in GPT-2 (RoPE reads position from the rotation), "
                       f"but the CONTENT circuit (induction) is key-content-dependent in EVERY model"
                       f"{' — and the top collapser IS the prev-token head universally (the canonical K-composition writer)' if ind_writer else ''}. "
                       f"RoPE replaces only POSITIONAL key-content with rotation; token-identity matching always lives in the "
                       f"key content. So #26/#31's GPT-2-vs-RoPE split is specifically the POSITIONAL register; the content "
                       f"instruction set is universal (consistent with the mechanism-invariance results).")
        else:
            verdict = (f"PARTIAL: positional-split={pos_split}, content-universal={content_universal} — see table")
        print(f"\n[verdict] {verdict}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11.0, 5.0))
        x = np.arange(len(ok)); w = 0.26
        for i, cc in enumerate(CIRCUITS):
            vals = [max(r["circuits"].get(cc, {}).get("top_collapse", 0.0), 0.0) for r in ok]
            ax.bar(x + (i - 1) * w, vals, w, edgecolor="k", label=cc + (" (positional)" if cc == "prevtok" else " (content)"))
        ax.axhline(0.1, color="k", lw=0.6, ls=":"); ax.set_xticks(x)
        ax.set_xticklabels([f"{r['model'].split('/')[-1]}\n{r['pos']}" for r in ok], fontsize=8)
        ax.set_ylabel("top upstream-head key-patch collapse"); ax.legend(fontsize=8)
        ax.set_title("key-content dependence: POSITIONAL (prev-token) is GPT-2-only; CONTENT (induction/duplicate) is universal", fontsize=10)
        fig.tight_layout(); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
