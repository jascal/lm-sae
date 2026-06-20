"""Extract the rank-r composition core as an explicit artifact — and test whether it IS the forge tax.

core_distill.py PROVED a per-layer rank-r update-bottleneck is near-lossless (the entangled core is
retrainable, not irreducible) — but it measured a number and threw the trained bonds away. This script:

  1. EXTRACTS the bonds. Trains the per-layer rank-r update bottleneck
         update' = (update · W_down[L]) · W_up[L]      W_down ∈ ℝ^{d×r}, W_up ∈ ℝ^{r×d}
     (init = top-r PCA of the update = the no-retrain floor; base frozen; distilled to match), then
     SERIALIZES {W_down[L], W_up[L]} as pylm/compose_core_<model>.npz — the first explicit, sized
     artifact of the rank-r composition channel (~0.15 M params for GPT-2 at r=8).

  2. TESTS THE NORTH-STAR CLAIM. The pylm thesis splits a model into a flat KNOWLEDGE store (~50% of
     tokens: induction + n-gram + grammar) and the COMPOSITION remainder (the forge tax — the computed,
     not retrieved, part). If the rank-r inter-layer channel is the *substrate of that composition*, then
     bottlenecking it should damage exactly the composition tokens and spare the retrieval ones. We tag
     every eval position retrieval-vs-composition with the actual pylm flat predictor, then measure
     KL(full ‖ rank-r) split by tag, swept over r.

       composition KL ≫ retrieval KL, gap widening as r↓   → the rank-r channel carries the forge tax
       both flat across r                                  → the channel is generic, not the composition

Honest scope: the bonds compress the inter-layer *channel* (the message each layer writes), proving the
composition is low-bandwidth and separable from retrieval — they do NOT replace the per-layer attention/MLP
*compute* (the open remainder of "minimum to run"). Pure cached path, CPU, no network.

Run:  .venv/bin/python scripts/disassembly/compose_core.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[2]


def load_eval_stream(model_short: str):
    """Held-out token ids (the pylm store did NOT memorise these) → fair retrieval/composition split."""
    for cand in (f"pylm/holdout_{model_short}.json", "pylm/holdout_ids.json"):
        p = REPO / cand
        if p.exists():
            d = json.loads(p.read_text())
            ids = d["holdout_ids"] if isinstance(d, dict) else d
            if ids:
                return [int(x) for x in ids]
    raise SystemExit("no holdout ids found")


def run(args):
    import torch
    from residual_vm import ResidualVM

    sys.path.insert(0, str(REPO / "pylm"))
    from lm import PyLM

    short = args.model.split("/")[-1]
    vm = ResidualVM(args.model, device=args.device)
    t = vm.torch
    nL, d = vm.nL, vm.d
    for p in vm.model.parameters():
        p.requires_grad_(False)

    store = REPO / "pylm" / f"store_{short}.json"
    if not store.exists():
        store = REPO / "pylm" / "store.json"
    pylm = PyLM(str(store))

    ids = load_eval_stream(short)
    ctx = args.ctx
    chunks = [ids[i:i + ctx] for i in range(0, len(ids), ctx) if len(ids[i:i + ctx]) >= 8]
    fit = chunks[:args.fit]
    train = chunks[args.fit:args.fit + args.train]
    ev = chunks[args.fit + args.train:args.fit + args.train + args.eval]
    print(f"[{short}] {nL}L d{d} | chunks: fit {len(fit)} train {len(train)} eval {len(ev)} (ctx {ctx})")

    # ---- per-layer update-covariance PCA bases (the no-retrain init) ----
    cov = {L: np.zeros((d, d), np.float64) for L in range(nL)}
    cap: dict = {}
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
    pca = {L: np.linalg.eigh(cov[L])[1][:, ::-1].copy().astype(np.float32) for L in range(nL)}  # cols desc

    # ---- full-model reference logits + per-position retrieval/composition tag ----
    full_lp, tags = [], []   # tags[c][p] in {"R","C"} for positions 0..len-2
    with t.no_grad():
        for c in ev:
            lg = vm.model(input_ids=t.tensor([c], device=vm.dev)).logits[0].float()
            full_lp.append(t.log_softmax(lg, -1))
            mt1 = lg.argmax(-1)                                   # model top-1 per position
            tg = []
            for p in range(len(c) - 1):
                pred = pylm.predict(c[:p + 1])                   # flat retrieval prediction of token p+1
                tg.append("R" if pred == int(mt1[p]) else "C")
            tags.append(tg)
    nR = sum(x == "R" for tg in tags for x in tg)
    nC = sum(x == "C" for tg in tags for x in tg)
    # confound diagnostic: the model's own top-1 confidence by token type (composition top-1s may be
    # marginal/fragile, so low retention could be fragility rather than the channel carrying composition).
    confR = confC = 0.0
    for flp, tg in zip(full_lp, tags):
        pr = flp.exp()
        for p, tag in enumerate(tg):
            mx = float(pr[p].max())
            if tag == "R":
                confR += mx
            else:
                confC += mx
    confR, confC = confR / max(nR, 1), confC / max(nC, 1)
    print(f"  eval positions: {nR} retrieval + {nC} composition  (decompilable frac {nR / (nR + nC):.1%})")
    print(f"  model top-1 confidence: retrieval {confR:.2f} | composition {confC:.2f}  "
          f"(confound check — is composition just lower-confidence?)")

    def make_hooks(down, up):
        hs = []
        for L in range(nL):
            def mk(L):
                def hook(m, i, o):
                    out = o[0] if isinstance(o, tuple) else o
                    upd = (out - i[0]).float()
                    low = (upd @ down[L]) @ up[L]
                    new = (i[0].float() + low).to(out.dtype)
                    return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
                return hook
            hs.append(vm.layers[L].register_forward_hook(mk(L)))
        return hs

    def kl_by_tag(down, up):
        """mean KL(full ‖ bottleneck) split by retrieval/composition tag, + top-1 retention by tag."""
        hs = make_hooks(down, up)
        klR = klC = 0.0; keepR = keepC = 0
        try:
            with t.no_grad():
                for c, flp, tg in zip(ev, full_lp, tags):
                    lg = vm.model(input_ids=t.tensor([c], device=vm.dev)).logits[0].float()
                    blp = t.log_softmax(lg, -1)
                    ph = flp.exp()
                    kl = (ph * (flp - blp)).sum(-1)              # per-position KL
                    ftop = flp.argmax(-1); btop = blp.argmax(-1)
                    for p, tag in enumerate(tg):
                        if tag == "R":
                            klR += float(kl[p]); keepR += int(btop[p] == ftop[p])
                        else:
                            klC += float(kl[p]); keepC += int(btop[p] == ftop[p])
        finally:
            for h in hs:
                h.remove()
        return (klR / max(nR, 1), klC / max(nC, 1), keepR / max(nR, 1), keepC / max(nC, 1))

    # random orthonormal rank-r basis per layer (the confound control: same rank, no structure)
    rng0 = np.random.default_rng(0)
    rand_full = {L: np.linalg.qr(rng0.standard_normal((d, d)))[0].astype(np.float32) for L in range(nL)}

    # ---- (1) rank-r channel swept over rank — PCA vs RANDOM control, top-1 retention by tag ----
    # top-1 retention is confounded by token-type peakedness (composition top-1s are marginal and flip
    # under ANY noise), so the clean signal is the STRUCTURED−RANDOM gap: how much a PCA channel of the
    # same rank preserves each type beyond a random channel. A bigger gap on composition ⇒ the structured
    # low-rank channel disproportionately carries the forge tax.
    ranks = sorted({min(int(x), d) for x in args.ranks.split(",")})
    print("\n  no-retrain rank-r channel — top-1 retention (PCA vs random control) by token type:")
    print(f"  {'rank':>5} | {'retr:PCA':>9}{'rand':>7}{'gap':>7} | {'comp:PCA':>9}{'rand':>7}{'gap':>7} | {'KL_r':>6}{'KL_c':>6}")
    no_retrain = []
    for r in ranks:
        dP = {L: t.tensor(pca[L][:, :r].copy(), device=vm.dev) for L in range(nL)}
        uP = {L: t.tensor(pca[L][:, :r].T.copy(), device=vm.dev) for L in range(nL)}
        kR, kC, tR, tC = kl_by_tag(dP, uP)
        dN = {L: t.tensor(rand_full[L][:, :r].copy(), device=vm.dev) for L in range(nL)}
        uN = {L: t.tensor(rand_full[L][:, :r].T.copy(), device=vm.dev) for L in range(nL)}
        _, _, rR, rC = kl_by_tag(dN, uN)
        no_retrain.append({"rank": r, "kl_retr": kR, "kl_comp": kC,
                           "top1_retr_pca": tR, "top1_retr_rand": rR, "gap_retr": tR - rR,
                           "top1_comp_pca": tC, "top1_comp_rand": rC, "gap_comp": tC - rC})
        print(f"  {r:>5} | {tR:>9.1%}{rR:>7.1%}{tR - rR:>+7.1%} | {tC:>9.1%}{rC:>7.1%}{tC - rC:>+7.1%} | "
              f"{kR:>6.2f}{kC:>6.2f}")

    # ---- (2) EXTRACT: train the rank-r bonds to REPRODUCE THE MODEL (KL distillation), serialize ----
    # The correct extraction objective is to match the model's distribution (KL to full logits), NOT
    # cross-entropy to true tokens (that recovers true-token NLL — #142's metric — while drifting from the
    # model's behaviour). An enable flag lets one forward give the teacher (off) and the next the student (on).
    r = args.extract_rank
    print(f"\n  extracting rank-{r} bonds — KL-distilling to the MODEL ({args.steps} steps) ...")
    down = {L: t.tensor(pca[L][:, :r].copy(), device=vm.dev, requires_grad=True) for L in range(nL)}
    up = {L: t.tensor(pca[L][:, :r].T.copy(), device=vm.dev, requires_grad=True) for L in range(nL)}
    pre_kR, pre_kC, pre_tR, pre_tC = kl_by_tag({L: down[L].detach() for L in down},
                                               {L: up[L].detach() for L in up})
    flag = {"on": True}
    hs = []
    for L in range(nL):
        def mk(L):
            def hook(m, i, o):
                if not flag["on"]:
                    return o
                out = o[0] if isinstance(o, tuple) else o
                upd = (out - i[0]).float()
                low = (upd @ down[L]) @ up[L]
                new = (i[0].float() + low).to(out.dtype)
                return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
            return hook
        hs.append(vm.layers[L].register_forward_hook(mk(L)))
    opt = t.optim.Adam(list(down.values()) + list(up.values()), lr=args.lr)
    rng = np.random.default_rng(0)
    tr = train if train else fit
    for s in range(args.steps):
        c = tr[int(rng.integers(0, len(tr)))]
        ids = t.tensor([c], device=vm.dev)
        flag["on"] = False
        with t.no_grad():
            teach = t.log_softmax(vm.model(input_ids=ids).logits[0].float(), -1)
        flag["on"] = True
        stud = t.log_softmax(vm.model(input_ids=ids).logits[0].float(), -1)
        loss = (teach.exp() * (teach - stud)).sum(-1).mean()      # KL(teacher ‖ student)
        opt.zero_grad(); loss.backward(); opt.step()
    for h in hs:
        h.remove()
    det_down = {L: down[L].detach() for L in down}; det_up = {L: up[L].detach() for L in up}
    post_kR, post_kC, post_tR, post_tC = kl_by_tag(det_down, det_up)
    print(f"  rank-{r} no-retrain  KL retr {pre_kR:.3f}/comp {pre_kC:.3f}  top-1 retr {pre_tR:.1%}/comp {pre_tC:.1%}")
    print(f"  rank-{r} KL-DISTILLED KL retr {post_kR:.3f}/comp {post_kC:.3f}  top-1 retr {post_tR:.1%}/comp {post_tC:.1%}")

    # serialize the explicit artifact
    art = {}
    for L in range(nL):
        art[f"down_{L}"] = det_down[L].cpu().numpy().astype(np.float32)
        art[f"up_{L}"] = det_up[L].cpu().numpy().astype(np.float32)
    artpath = REPO / "pylm" / f"compose_core_{short}.npz"
    np.savez_compressed(artpath, **art)
    n_params = nL * 2 * d * r
    full_params = sum(p.numel() for p in vm.model.parameters())
    store_mb = store.stat().st_size / 1e6
    art_mb = artpath.stat().st_size / 1e6
    print(f"\n  [artifact] {artpath.name}: {n_params:,} params ({art_mb:.2f} MB) = the rank-{r} composition channel")
    print(f"  decomposition sizes — flat knowledge store {store_mb:.2f} MB | composition program {art_mb:.2f} MB "
          f"| full model {full_params / 1e6:.0f} M params")

    out = {
        "model": short, "n_layers": nL, "d_model": d, "extract_rank": r,
        "eval_positions": {"retrieval": nR, "composition": nC, "decompilable_frac": nR / (nR + nC),
                           "conf_retr": confR, "conf_comp": confC},
        "no_retrain_by_rank": no_retrain,
        "trained_rank": {"rank": r, "objective": "KL-to-model",
                         "kl_retr_pre": pre_kR, "kl_comp_pre": pre_kC,
                         "top1_retr_pre": pre_tR, "top1_comp_pre": pre_tC,
                         "kl_retr_trained": post_kR, "kl_comp_trained": post_kC,
                         "top1_retr_trained": post_tR, "top1_comp_trained": post_tC},
        "artifact": {"file": artpath.name, "params": n_params, "mb": art_mb,
                     "flat_store_mb": store_mb, "full_model_params_m": full_params / 1e6},
    }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=20)
    p.add_argument("--train", type=int, default=120)
    p.add_argument("--eval", type=int, default=24)
    p.add_argument("--ranks", default="2,4,8,16,32,64")
    p.add_argument("--extract-rank", type=int, default=8)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", type=Path, default=REPO / "runs/disassembly")
    args = p.parse_args(argv)

    out = run(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    sp = args.outdir / "compose_core_summary.json"
    prior = json.loads(sp.read_text()).get("results", []) if sp.exists() else []
    merged = [out] + [r for r in prior if r.get("model") != out["model"]]
    sp.write_text(json.dumps({"experiment": "extract the rank-r composition core; is it the forge tax?",
                              "results": merged}, indent=2, default=float))
    print(f"\n[done] {sp}")
    return out


if __name__ == "__main__":
    main()
