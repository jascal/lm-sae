"""Characterize the REAL legible corner — a properly-trained 32×d SAE (jbloom GPT-2) — before training one in sae-forge.

#146 showed a *quick* 2×d SAE on the update is complete + big but never sparse/legible (L0 ≥ 0.30). The legible corner
done right needs 8–64×d + full SAE training — which is the sae-forge sub-project. Before committing to an OpenSpec plan
there, this prototypes the answer for free using the already-trained **jbloom GPT-2 SAEs (F=24,576 = 32×d, resid_pre,
all 12 layers)** via ResidualVM's loader + bottleneck. Measures the three triangle axes for a real monosemantic SAE:

  SIZE        — F / d (overcompleteness).
  SPARSITY    — mean L0 (fraction of features active per token) — the legibility proxy for SAEs (monosemantic = sparse).
  COMPLETE    — generic NLL with the SAE bottleneck (decode∘encode the residual at every layer) vs the base model.
  LEGIBLE     — logit-lens peak-z + closed-class fraction of the decoder directions (for comparison with #146's 2×d SAE).

If the real 32×d SAE is genuinely sparse (low L0) + ~complete, the legible corner is real and worth training properly in
sae-forge; the contrast with #146's dense 2×d confirms "legibility needs high overcompleteness." Output:
runs/disassembly/legible_corner_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_grammar import CLOSED, PUNCT  # noqa: E402


def run_model(mid, args):
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8][: args.eval]

    WU = vm.model.get_output_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
    cset = set()
    for w in CLOSED:
        for f in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            tt = tok(f, add_special_tokens=False)["input_ids"]
            if len(tt) == 1:
                cset.add(tt[0])
    for p_ in PUNCT:
        for f in (p_, " " + p_):
            tt = tok(f, add_special_tokens=False)["input_ids"]
            if len(tt) == 1:
                cset.add(tt[0])

    def legibility(W, n=400):
        zs = []; cf = []
        idx = np.linspace(0, W.shape[0] - 1, min(W.shape[0], n)).astype(int)
        for k in idx:
            v = W[k]; v = v / (np.linalg.norm(v) + 1e-9); lg = WU @ v
            zp = (lg.max() - lg.mean()) / (lg.std() + 1e-9); zn = (lg.mean() - lg.min()) / (lg.std() + 1e-9)
            top = (np.argsort(-lg) if zp >= zn else np.argsort(lg))[:10]
            zs.append(max(zp, zn)); cf.append(np.mean([int(i) in cset for i in top]))
        return float(np.mean(zs)), float(np.mean(cf))

    def gen_nll(ctxmgr=None):
        tot = 0.0; k = 0
        with t.no_grad():
            for c in chunks:
                if ctxmgr is None:
                    lp = t.log_softmax(vm.logits(c).float(), -1)
                else:
                    with ctxmgr():
                        lp = t.log_softmax(vm.logits(c).float(), -1)
                y = c[1:]
                for p in range(len(y)):
                    tot += float(-lp[p, y[p]]); k += 1
        return tot / max(k, 1)

    base_nll = gen_nll()
    layers = list(range(nL))
    bottleneck_nll = gen_nll(lambda: vm.sae_bottleneck(layers))        # decode∘encode the residual at every layer

    # per-layer L0 (sparsity) + decoder legibility + overcompleteness
    F = None; l0s = []; pz = []; cf = []
    for L in layers:
        sae = vm.load_sae(L); F = sae["wdec"].shape[0]
        resid = vm.trace(chunks[0])["resid"][L][0].float()            # (seq, d) resid_pre
        pre = (resid - sae["bdec"]) @ sae["wenc"] + sae["benc"]
        codes = t.relu(pre) if sae["thr"] is None else t.where(pre > sae["thr"], pre, t.zeros_like(pre))
        l0s.append(float((codes > 0).float().sum(-1).mean() / F))
        Wd = sae["wdec"].detach().cpu().numpy().astype(np.float64)     # (F, d)
        z, c = legibility(Wd); pz.append(z); cf.append(c)
    return {"model": mid.split("/")[-1], "n_layers": nL, "d_model": d, "sae_features": F,
            "overcompleteness": F / d, "base_nll": base_nll, "bottleneck_nll": bottleneck_nll,
            "bottleneck_nll_increase": bottleneck_nll - base_nll, "mean_L0_frac": float(np.mean(l0s)),
            "mean_L0_count": float(np.mean(l0s) * F), "decoder_peak_z": float(np.mean(pz)),
            "decoder_closed_frac": float(np.mean(cf))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--eval", type=int, default=20)
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
            print(f"  jbloom SAE F={r['sae_features']} ({r['overcompleteness']:.0f}×d)")
            print(f"  SPARSITY  mean L0 {r['mean_L0_frac']:.4f} (~{r['mean_L0_count']:.0f} of {r['sae_features']} features active/token)")
            print(f"  COMPLETE  base NLL {r['base_nll']:.3f} → SAE-bottleneck {r['bottleneck_nll']:.3f} "
                  f"(Δ {r['bottleneck_nll_increase']:+.3f})")
            print(f"  LEGIBLE   decoder peak-z {r['decoder_peak_z']:.2f} · closed-frac {r['decoder_closed_frac']:.2f} "
                  f"(vs #146 2×d SAE: peak-z ~4.8, L0 ~0.30)")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "legible_corner_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "characterize the real legible corner — a properly-trained 32×d SAE (jbloom GPT-2)", "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'sae_features' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
