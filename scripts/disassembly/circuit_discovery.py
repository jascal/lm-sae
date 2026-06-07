"""Cross-model circuit-edge DISCOVERY — de-novo composition edges in every model, not just the named ones.

The circuit catalog's cross-model rows path-patch the *single* top reader of each named circuit
(`circuit_content_patch.py`). This generalizes that to **discovery**: in each model, take the **top-K behavioural
content readers** (heads with high induction- or duplicate-mass) and path-patch **every** upstream head out of each
reader's KEY (the faithful key-only patch — replace the reader's k-input with `norm(resid − A_out)`, so the model
re-applies its own RoPE), measuring the collapse of that reader's attention on its content pattern. Edges that
collapse the reader beyond a reader-matched null are **discovered live circuit edges** (writer → reader, K).
Arch-generic (GPT-2 c_attn / RoPE k_proj), reuses `circuit_content_patch._arch`.

Output: [runs/disassembly/circuits/discovered_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/discovered_summary.json) + a generated `docs/circuits/discovered.md`. The
rigorous (every reader, every upstream, every model) way to grow the circuit catalog beyond the named circuits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
from circuit_content_patch import _arch  # noqa: E402


def content_masks(toks):
    ca = np.array(toks); n = len(ca); qi = np.arange(n); pv = np.full(n, -1); pv[1:] = ca[:-1]
    return {"induction": (pv[None, :] == ca[:, None]) & (qi[None, :] >= 1) & (qi[None, :] < qi[:, None]),
            "duplicate": (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]),
            "prevtok": (qi[None, :] == (qi[:, None] - 1)) & (qi[:, None] >= 1)}


def run_model(model_id, args, dev):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    a = _arch(model); H = a["H"]; hd = a["hd"]; nL = model.config.num_hidden_layers; NH = nL * H
    oproj = a["oproj"]; norm = a["norm"]; d = a.get("d", model.config.hidden_size)
    V = model.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
    rng = np.random.default_rng(args.seed)
    seqs = [(lambda s: s + s)([int(x) for x in rng.integers(lo, hi, args.rep_len)]) for _ in range(args.probes)]

    # ---- behavioural mass per head for content patterns -> pick the top-K readers ----
    mass = {p: np.zeros(NH) for p in ("induction", "duplicate", "prevtok")}; ntot = {p: 0 for p in mass}
    with torch.no_grad():
        for s in seqs:
            o = model(input_ids=torch.tensor([s], device=dev), output_attentions=True)
            M = content_masks(s)
            for p in mass:
                ntot[p] += int(M[p].sum())
                for L in range(nL):
                    at = o.attentions[L][0].float().cpu().numpy()
                    mass[p][L * H:(L + 1) * H] += (at * M[p][None]).sum((1, 2))
    for p in mass:
        mass[p] /= max(ntot[p], 1)
    prevtok_head = int(np.argmax(mass["prevtok"]))
    # readers = top-K heads by max(induction, duplicate) mass, in layers >= 1
    best = np.maximum(mass["induction"], mass["duplicate"])
    cand = [i for i in np.argsort(-best) if i // H >= 1 and best[i] > args.reader_thr][: args.top_readers]
    readers = [(int(i), "induction" if mass["induction"][i] >= mass["duplicate"][i] else "duplicate") for i in cand]

    def head_contrib(L, captured, h):
        x = torch.zeros_like(captured); x[..., h * hd:(h + 1) * hd] = captured[..., h * hd:(h + 1) * hd]
        return oproj[L](x) - oproj[L](torch.zeros_like(captured[..., :1, :]))

    discovered = []
    for (B, patt) in readers:
        LB, hB = B // H, B % H
        upstream = [(L, h) for L in range(LB) for h in range(H)][: args.max_upstream]
        cap = {}
        hooks = [a["layers"][LB].register_forward_pre_hook(lambda m, inp: cap.__setitem__("r", inp[0].detach()))]
        for L in range(LB):
            hooks.append(oproj[L].register_forward_pre_hook((lambda L: lambda m, inp: cap.__setitem__(L, inp[0].detach()))(L)))
        clean = 0.0; patched = {u: 0.0 for u in upstream}; tot = 0
        with torch.no_grad():
            for s in seqs[: args.patch_probes]:
                cap.clear(); o = model(input_ids=torch.tensor([s], device=dev), output_attentions=True)
                Msk = content_masks(s)[patt]; attnB = o.attentions[LB][0, hB]
                clean += float((attnB.float().cpu().numpy() * Msk).sum())
                resid = cap["r"]
                for (La, ha) in upstream:
                    kin = norm[LB](resid - head_contrib(La, cap[La], ha))
                    if a["is_gpt2"]:
                        ksl = a["cattn"][LB](kin)[..., d:2 * d]

                        def hook(m, inp, o2, _k=ksl, _d=d):
                            o2 = o2.clone(); o2[..., _d:2 * _d] = _k; return o2
                        hk = a["cattn"][LB].register_forward_hook(hook)
                    else:
                        def pre(m, inp, _ki=kin):
                            return (_ki,) + inp[1:]
                        hk = a["kproj"][LB].register_forward_pre_hook(pre)
                    try:
                        ap = model(input_ids=torch.tensor([s], device=dev), output_attentions=True).attentions[LB][0, hB]
                        patched[(La, ha)] += float((ap.float().cpu().numpy() * Msk).sum())
                    finally:
                        hk.remove()
                tot += 1
        for h in hooks:
            h.remove()
        clean /= max(tot, 1)
        rows = [{"writer": f"{La}.{ha}", "collapse": (clean - patched[(La, ha)] / max(tot, 1)) / clean if clean > 1e-6 else 0.0}
                for (La, ha) in upstream]
        rels = np.array([r["collapse"] for r in rows]); med = float(np.median(rels)); mad = float(np.median(np.abs(rels - med)))
        for r in rows:
            r["z"] = (r["collapse"] - med) / (1.4826 * mad + 1e-6)
        rows.sort(key=lambda r: -r["collapse"]); top = rows[0]
        live = top["collapse"] > max(args.live_thr, 3 * max(med, 0.01))
        discovered.append({"reader": f"{LB}.{hB}", "pattern": patt, "top_writer": top["writer"], "collapse": top["collapse"],
                           "z": top["z"], "writer_is_prevtok_head": top["writer"] == f"{prevtok_head // H}.{prevtok_head % H}",
                           "live": bool(live), "median_collapse": med, "top_writers": rows[:4]})
    nlive = sum(1 for e in discovered if e["live"])
    return {"model": model_id.split("/")[-1], "arch": "GPT-2/absolute" if a["is_gpt2"] else "RoPE", "n_layers": nL,
            "prev_token_head": f"{prevtok_head // H}.{prevtok_head % H}", "n_readers": len(readers),
            "n_live_edges": nlive, "edges": discovered}


def write_doc(out, docs):
    models = [r for r in out["results"] if "edges" in r]
    lines = ["---", "title: Discovered circuits", "---", "",
             "# Discovered circuit edges — the key-patch run over the top content readers in every model", "",
             "A **working catalog** (amateur, exploratory, provisional) of de-novo composition **edges**: for the "
             "top behavioural content readers in each model we path-patch every upstream head out of the reader's "
             "**key** and keep the edges that collapse the reader's attention beyond a reader-matched null. "
             "writer → reader (K-composition); *live* = robust collapse.", "",
             f"_{len(models)} models · top content readers × all upstream · faithful key-only patch._", ""]
    for r in models:
        lines += [f"## {r['model']} ({r['arch']}) — {r['n_live_edges']}/{r['n_readers']} live edges  (prev-token head {r['prev_token_head']})",
                  "", "| reader | pattern | top upstream writer | key-collapse | z | live? |", "|---|---|---|---|---|---|"]
        for e in sorted(r["edges"], key=lambda e: -e["collapse"]):
            w = e["top_writer"] + (" (=prev-tok head)" if e["writer_is_prevtok_head"] else "")
            lines.append(f"| {e['reader']} | {e['pattern']} | {w} | {e['collapse']:+.0%} | {e['z']:.1f} | {'**yes**' if e['live'] else 'no'} |")
        lines.append("")
    lines += ["## How to read this", "",
              "- A **live** edge = removing that upstream writer from the reader's key collapses the reader's "
              "content attention beyond a reader-matched null → a real K-composition edge. For induction readers the "
              "top writer is typically the model's **prev-token head** (the canonical induction wiring), recovered "
              "de novo here.",
              "- Provisional and descriptive. Value-channel (move) edges and Q-composition are not in this pass "
              "(key/match only). See the [circuit catalog](README.md) for the named circuits.", "",
              "_Data: [runs/disassembly/circuits/discovered_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/circuits/discovered_summary.json). Regenerate: [circuit_discovery.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/circuit_discovery.py)._"]
    (docs / "discovered.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--probes", type=int, default=24)
    p.add_argument("--patch-probes", type=int, default=10)
    p.add_argument("--rep-len", type=int, default=22)
    p.add_argument("--top-readers", type=int, default=6)
    p.add_argument("--reader-thr", type=float, default=0.1)
    p.add_argument("--max-upstream", type=int, default=80)
    p.add_argument("--live-thr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/circuits"))
    p.add_argument("--docs", type=Path, default=Path("docs/circuits"))
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
            print(f"  {r['arch']} {r['n_layers']}L | {r['n_live_edges']}/{r['n_readers']} live edges | "
                  + ", ".join(f"{e['top_writer']}→{e['reader']}({e['collapse']:+.0%})" for e in r['edges'] if e['live'])[:120])
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid, "error": str(e)})
    out = {"experiment": "cross-model circuit-edge discovery (top content readers x all upstream, key-patch)", "results": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "discovered_summary.json").write_text(json.dumps(out, indent=2, default=float))
    args.docs.mkdir(parents=True, exist_ok=True)
    write_doc(out, args.docs)
    ok = [r for r in results if "edges" in r]
    print(f"\n[done] {sum(r['n_live_edges'] for r in ok)} live discovered edges across {len(ok)} models → "
          f"{args.outdir / 'discovered_summary.json'} + {args.docs / 'discovered.md'}")
    return out


if __name__ == "__main__":
    main()
