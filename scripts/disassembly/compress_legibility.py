"""Does compression ALIGN with legibility, or trade off against it? — interpretability of trained vs PCA write directions.

#142 showed a *trained* rank-r bottleneck on each layer's update is ~lossless at ~1% of d (compression is tractable).
The user's feature-native endgame (train N× smaller → extract features/knowledge/circuits → the small Python program →
runtime explainability) hinges on a question that is NOT obviously true: when you compress by minimising NLL alone, do
the kept directions become MORE interpretable (compression disentangles) or LESS (it packs more per direction — the
forge tax)? If the latter, "extract clean features after compression" needs an explicit legibility objective (sae-forge),
not just distillation.

Test: train a rank-r update-bottleneck (init from PCA), then logit-lens-score the r directions it *writes into*
(rows of W_up) and compare to the untrained top-r PCA directions, on two interpretability axes:
  PEAK-Z      — how sharply each direction decodes to a few tokens (a monosemanticity proxy);
  CLOSED-FRAC — how often its top tokens are closed-class / punctuation (grammatical alignment, as in core_grammar).
trained ≫ PCA ⇒ compression cleans the features; trained ≪ PCA ⇒ NLL-only compression packs harder (needs a legibility term).

Output: runs/disassembly/compress_legibility_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_grammar import CLOSED, PUNCT  # noqa: E402


def run_model(mid, args):
    import urllib.request

    import torch
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d; r = args.rank
    for p in vm.model.parameters():
        p.requires_grad_(False)
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:300000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8]
    fit, train = chunks[: args.fit], chunks[args.fit: args.fit + args.train]

    # PCA bases of each layer's update
    cov = {L: np.zeros((d, d)) for L in range(nL)}; cap = {}
    hks = [vm.layers[L].register_forward_hook(
        (lambda L: lambda m, i, o: cap.__setitem__(L, ((o[0] if isinstance(o, tuple) else o) - i[0]).detach()))(L))
        for L in range(nL)]
    with t.no_grad():
        for c in fit:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                u = cap[L][0].float().cpu().numpy(); cov[L] += u.T @ u
    for h in hks:
        h.remove()
    pca = {L: np.linalg.eigh(cov[L])[1][:, ::-1][:, :r].astype(np.float32) for L in range(nL)}   # (d,r) top-r

    # train the rank-r bottleneck (init from PCA) — base frozen, only factors
    down = {L: t.tensor(pca[L].copy(), device=vm.dev, requires_grad=True) for L in range(nL)}
    up = {L: t.tensor(pca[L].T.copy(), device=vm.dev, requires_grad=True) for L in range(nL)}
    hs = []

    def mk(L):
        def hook(m, i, o):
            out = o[0] if isinstance(o, tuple) else o
            low = ((out - i[0]).float() @ down[L]) @ up[L]
            return (((i[0].float() + low).to(out.dtype)),) + tuple(o[1:]) if isinstance(o, tuple) else (i[0].float() + low).to(out.dtype)
        return hook
    for L in range(nL):
        hs.append(vm.layers[L].register_forward_hook(mk(L)))
    opt = torch.optim.Adam(list(down.values()) + list(up.values()), lr=args.lr)
    rng = np.random.default_rng(0)
    for s in range(args.steps):
        c = train[int(rng.integers(0, len(train)))]; ids = t.tensor([c], device=vm.dev)
        logits = vm.model(input_ids=ids).logits[0]
        loss = t.nn.functional.cross_entropy(logits[:-1].float(), ids[0, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
    for h in hs:
        h.remove()

    # ---- legibility: logit-lens the WRITE directions (rows of W_up, trained) vs PCA top-r ----
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

    def legibility(vecs):                                              # vecs: (d, k) columns = directions
        zs = []; cf = []
        for k in range(vecs.shape[1]):
            v = vecs[:, k]; v = v / (np.linalg.norm(v) + 1e-9); lg = WU @ v
            zp = (lg.max() - lg.mean()) / (lg.std() + 1e-9); zn = (lg.mean() - lg.min()) / (lg.std() + 1e-9)
            top = (np.argsort(-lg) if zp >= zn else np.argsort(lg))[:10]
            zs.append(max(zp, zn)); cf.append(np.mean([int(i) in cset for i in top]))
        return float(np.mean(zs)), float(np.mean(cf))

    pca_z = []; pca_c = []; tr_z = []; tr_c = []
    for L in range(nL):
        z, c = legibility(pca[L]); pca_z.append(z); pca_c.append(c)
        z, c = legibility(up[L].detach().cpu().numpy().T.astype(np.float64))   # W_up rows = write dirs
        tr_z.append(z); tr_c.append(c)
    return {"model": mid.split("/")[-1], "n_layers": nL, "d_model": d, "rank": r, "steps": args.steps,
            "pca_peak_z": float(np.mean(pca_z)), "trained_peak_z": float(np.mean(tr_z)),
            "pca_closed_frac": float(np.mean(pca_c)), "trained_closed_frac": float(np.mean(tr_c))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=30)
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
    args = p.parse_args(argv)

    import torch
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} (rank {args.rank}) ===")
        try:
            r = run_model(mid, args)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(r)
            print(f"  logit-lens peak-z:    PCA {r['pca_peak_z']:.2f} → trained {r['trained_peak_z']:.2f}")
            print(f"  closed-class frac:    PCA {r['pca_closed_frac']:.2f} → trained {r['trained_closed_frac']:.2f}")
            verdict = ("compression CLEANS features" if r['trained_peak_z'] > r['pca_peak_z']
                       else "NLL-only compression PACKS harder (needs a legibility term)")
            print(f"  → {verdict}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "compress_legibility_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "does NLL-only compression align with legibility — trained vs PCA write-direction interpretability",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'rank' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
