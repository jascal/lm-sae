"""Are the woven-in induction heads ≥2 SEPARABLE circuits, or ONE decomposition? — cluster by writer-dependency.

`circuit_ensemble.py` established the distributing induction circuit is **not** a weighted ensemble of duplicates
(OV-operation cosine ≈0 everywhere) but **structurally heterogeneous heads spread across depth** with overlapping
function. That leaves the open question this script settles: are those heterogeneous heads several **complete
parallel circuits** (each fed by its own upstream predecessor-writer) or **one circuit decomposed** behind a shared
front-end?

The discriminator is the **upstream writer-dependency** of each reader. For every induction head B in the population
we run the faithful **key-only causal patch** (the `circuit_content_patch` machinery, arch-generic): remove each
upstream head A's output from B's KEY (`norm(resid − A_out)`, q untouched so RoPE still applies) and measure the
collapse of B's induction attention. That yields a reader × writer dependency matrix.

  - ONE decomposition (shared front-end): every reader's top collapser is the SAME prev-token/predecessor writer;
    reader writer-profiles are highly similar (one circuit, labour split across the readers).
  - ≥2 SEPARABLE circuits: readers split into groups with DIFFERENT dominant writers; writer-profiles cluster.

Metrics per model: the number of distinct dominant writers across readers, the fraction of readers feeding from the
single most-common writer (shared-front-end fraction), and the mean pairwise cosine of reader writer-profiles
(high = shared front-end, low = distinct circuits). Tracked across the GPT-2 ladder + RoPE.

Output: runs/disassembly/circuits/writer_cluster_summary.json (merge-safe). Findings -> docs/FINDINGS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))


def induction_population(model, a, chunks, dev, top_k, mass_thresh):
    """Per-head induction attention mass (one pass) -> the population (heads with mass>thresh, layer>=1, top_k)."""
    import torch
    from circuit_content_patch import circuit_masks
    H = a["H"]; nL = len(a["layers"]); NH = nL * H
    mass = np.zeros(NH); ntot = 0
    with torch.no_grad():
        for c in chunks:
            o = model(input_ids=torch.tensor([c], device=dev), output_attentions=True)
            M = circuit_masks(c)["induction"]; ntot += int(M.sum())
            for L in range(nL):
                mass[L * H:(L + 1) * H] += (o.attentions[L][0].float().cpu().numpy() * M[None]).sum((1, 2))
    mass /= max(ntot, 1)
    order = np.argsort(-mass)
    pop = [int(i) for i in order if mass[int(i)] > mass_thresh and int(i) >= H][:top_k]   # layer>=1 (has upstream)
    return pop, mass


def reader_key_profile(model, a, reader_B, chunks, dev, max_upstream):
    """Faithful key-only patch for ONE reader: collapse of its induction attention when each upstream head is removed
    from its key. Returns {writer_name: rel_collapse}. Mirrors circuit_content_patch.run_model's per-reader patch."""
    import torch
    from circuit_content_patch import circuit_masks
    H = a["H"]; hd = a["hd"]; oprojs = a["oproj"]; norm_mods = a["norm"]
    LB, hB = reader_B // H, reader_B % H
    upstream = [(L, h) for L in range(LB) for h in range(H)][:max_upstream]

    def head_contrib(L, captured, h):
        x = torch.zeros_like(captured); x[..., h * hd:(h + 1) * hd] = captured[..., h * hd:(h + 1) * hd]
        return oprojs[L](x) - oprojs[L](torch.zeros_like(captured[..., :1, :]))

    cap = {}; hooks = [a["layers"][LB].register_forward_pre_hook(lambda m, inp: cap.__setitem__("r", inp[0].detach()))]
    for L in range(LB):
        hooks.append(oprojs[L].register_forward_pre_hook((lambda L: lambda m, inp: cap.__setitem__(L, inp[0].detach()))(L)))
    clean = 0.0; patched = {u: 0.0 for u in upstream}; tot = 0; sane = None
    with torch.no_grad():
        for c in chunks:
            cap.clear()
            o = model(input_ids=torch.tensor([c], device=dev), output_attentions=True)
            Msk = circuit_masks(c)["induction"]; attnB = o.attentions[LB][0, hB]
            clean += float((attnB.float().cpu().numpy() * Msk).sum()); tot += len(c)
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
            if sane is None:                                                  # zero-patch fidelity check
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
    clean = max(clean / max(tot, 1), 1e-9)
    return {f"{La}.{ha}": (clean - patched[(La, ha)] / max(tot, 1)) / clean for (La, ha) in upstream}, sane


def cluster_one_model(model_id, args, dev):
    import torch
    from circuit_content_patch import _arch
    is_gpt2 = "gpt2" in model_id.lower()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gkw = {"dtype": torch.bfloat16} if (not is_gpt2 or "xl" in model_id) else {}
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", **gkw).eval().to(dev)
    tok = AutoTokenizer.from_pretrained(model_id)
    a = _arch(model); H = a["H"]
    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:160000]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8][: args.chunks]
    pop, _ = induction_population(model, a, chunks, dev, args.top_k, args.mass_thresh)
    if len(pop) < 2:
        return {"model": model_id.split("/")[-1], "note": "induction population <2 above layer 0", "pop_size": len(pop)}

    readers = [f"{b // H}.{b % H}" for b in pop]
    profiles = {}; tops = []; sanities = []
    for b, rn in zip(pop, readers):
        prof, sane = reader_key_profile(model, a, b, chunks, dev, args.max_upstream)
        profiles[rn] = prof; sanities.append(sane)
        top_w = max(prof, key=prof.get); tops.append((rn, top_w, prof[top_w]))

    # union writer space, profile vectors aligned on it
    writers = sorted({w for p in profiles.values() for w in p})
    P = np.array([[profiles[rn].get(w, 0.0) for w in writers] for rn in readers])
    # shared-front-end metrics
    top_writers = [tw for _, tw, _ in tops]
    from collections import Counter
    wc = Counter(top_writers); most_w, most_n = wc.most_common(1)[0]
    shared_frac = most_n / len(readers)
    n_distinct = len(wc)
    # mean pairwise cosine of reader writer-profiles (high = shared front-end / one circuit; low = distinct circuits)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    iu = np.triu_indices(len(readers), 1)
    prof_cos = float((Pn @ Pn.T)[iu].mean()) if len(iu[0]) else 0.0
    # how separable: a connected-components count on the thresholded profile-similarity graph (cos > 0.5)
    G = (Pn @ Pn.T) > 0.5
    seen = set(); comps = 0
    for i in range(len(readers)):
        if i in seen:
            continue
        comps += 1; stack = [i]
        while stack:
            j = stack.pop(); seen.add(j)
            stack += [k for k in range(len(readers)) if G[j, k] and k not in seen]
    return {"model": model_id.split("/")[-1], "rope": not is_gpt2, "pop_size": len(readers),
            "readers": readers, "reader_top_writers": {rn: tw for rn, tw, _ in tops},
            "n_distinct_top_writers": n_distinct, "most_common_writer": most_w, "shared_front_end_frac": shared_frac,
            "profile_pairwise_cosine": prof_cos, "separable_components_cos0.5": comps,
            "sanity_max": float(max([s for s in sanities if s is not None], default=0.0))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,gpt2-xl,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=48)
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--top-k", type=int, default=8, help="induction-population size (readers to profile)")
    p.add_argument("--mass-thresh", type=float, default=0.03)
    p.add_argument("--max-upstream", type=int, default=48)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/circuits"))
    args = p.parse_args(argv)

    import torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = cluster_one_model(mid, args, dev)
            if dev == "cuda":
                torch.cuda.empty_cache()
            results.append(r)
            if "readers" in r:
                print(f"  pop {r['pop_size']} readers | distinct top-writers {r['n_distinct_top_writers']} | "
                      f"shared-front-end {r['shared_front_end_frac']:.0%} (writer {r['most_common_writer']}) | "
                      f"profile-cos {r['profile_pairwise_cosine']:+.2f} | components(cos>.5) {r['separable_components_cos0.5']} | "
                      f"sanity {r['sanity_max']:.1e}")
            else:
                print(f"  {r.get('note')}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "writer_cluster_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "induction population: separable parallel circuits vs one decomposition (writer-dependency clustering)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'readers' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
