"""Tease cross-layer COMMUNICATION CHANNELS (candidate circuits) out of the high-dimensional activations, and
represent each one separately.

The richer-form survey (#157–160) showed the COMPUTATION is sparse circuits even though the WEIGHTS are high-rank /
layer-distinct — a few sparse pathways threaded through distinct weight matrices. Cross-layer *weight* sharing only
hurt; but the cross-layer *activation* coupling is low-dimensional (the area-law bond χ≈16). This script decomposes that
coupling and isolates each pathway.

Construction. Capture every layer's residual WRITE Δ_L = block_out − block_in over a corpus. At a mid cut, form the
aggregate early write e_p = Σ_{L≤cut} Δ_{L,p} and late write ℓ_p = Σ_{L>cut} Δ_{L,p} per token position p, and the
d×d cross-covariance C = (1/N) Σ_p e_p ⊗ ℓ_p. Its SVD C = Σ σ_i u_i w_iᵀ gives CHANNELS: a direction u_i the early
layers write that the late layers' writes co-vary with along w_i — i.e. cross-layer communication. (`core_basis_decompile`
showed the static *write basis* ≠ the named operators, but it ignored the READER side; a channel is a writer↔reader pair.)

Each channel is then characterised as a circuit — separately represented:
  - WRITER / READER layer profile (which layers write u_i / w_i) — the circuit's span;
  - logit-lens of u_i and w_i through W_U (what tokens it promotes → grammar vs content);
  - FIRING pattern: the per-position channel activation a_i(p)=e_p·u_i, its top-activating contexts, and its correlation
    with the DUPLICATE-token signal (induction family) and with POSITION → labels it catalog (induction/positional) or NEW.

No retraining; pure analysis on the frozen model. Output: runs/disassembly/circuit_channels_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CORPUS = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _fetch(url, n):
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                      timeout=20).read().decode("utf-8", "ignore")[:n]
    except Exception:
        return ""


def run_model(mid, args):
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d
    cut = args.cut if args.cut > 0 else nL // 2
    WU = vm.model.get_output_embeddings().weight.detach().float().cpu().numpy()    # (vocab, d) unembedding, logit lens

    text = _fetch(CORPUS, args.chars)
    ids_all = tok(text)["input_ids"]
    chunks = [ids_all[i:i + args.ctx] for i in range(0, len(ids_all), args.ctx)
              if len(ids_all[i:i + args.ctx]) >= 16][: args.fit]

    def capture():
        cap = {}
        hks = [vm.layers[L].register_forward_hook(
            (lambda L: lambda m, i, o: cap.__setitem__(L, ((o[0] if isinstance(o, tuple) else o) - i[0]).detach()))(L))
            for L in range(nL)]
        return cap, hks

    # ---- pass 1: cross-covariance C(early,late) + store per-position early/late/ids/duplicate flag ----
    C = np.zeros((d, d)); E = []; Lt = []; IDS = []; DUP = []; POS = []
    cap, hks = capture()
    with t.no_grad():
        for c in chunks:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            early = sum(cap[L][0] for L in range(cut + 1)).float().cpu().numpy()       # (seq, d)
            late = sum(cap[L][0] for L in range(cut + 1, nL)).float().cpu().numpy()    # (seq, d)
            early = early / (np.linalg.norm(early, axis=1, keepdims=True) + 1e-9)      # DIRECTIONAL coupling — unit-norm
            late = late / (np.linalg.norm(late, axis=1, keepdims=True) + 1e-9)        # per position (else last-layer magnitude swamps it)
            C += early.T @ late
            seen = set()
            for p in range(len(c)):
                dup = 1.0 if c[p] in seen else 0.0; seen.add(c[p])
                E.append(early[p]); Lt.append(late[p]); IDS.append(c[p]); DUP.append(dup); POS.append(p)
    for h in hks:
        h.remove()
    C /= max(len(E), 1)
    E = np.asarray(E); Lt = np.asarray(Lt); IDS = np.asarray(IDS); DUP = np.asarray(DUP); POS = np.asarray(POS, float)

    U, S, Vh = np.linalg.svd(C)                                                        # channels: u_i=U[:,i], w_i=Vh[i]
    K = min(args.channels, len(S))

    # ---- pass 2: per-layer writer/reader profiles for the top-K channels ----
    Uk = U[:, :K].astype(np.float32); Wk = Vh[:K].T.astype(np.float32)                 # (d,K) early & late dirs
    wr = np.zeros((nL, K)); rd = np.zeros((nL, K)); lnorm = np.zeros(nL); ntok = 0
    cap, hks = capture()
    with t.no_grad():
        for c in chunks:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                dl = cap[L][0].float().cpu().numpy()                                   # (seq,d)
                wr[L] += np.abs(dl @ Uk).sum(0); rd[L] += np.abs(dl @ Wk).sum(0)
                lnorm[L] += np.linalg.norm(dl, axis=1).sum()
            ntok += len(c)
    for h in hks:
        h.remove()
    wr = wr / (lnorm[:, None] + 1e-9); rd = rd / (lnorm[:, None] + 1e-9)               # DIRECTIONAL: fraction of layer's write along the channel (not magnitude)

    def lens(vec, topn=8):
        s = WU @ vec; idx = np.argsort(-s)[:topn]
        return [tok.decode([int(i)]).replace("\n", "\\n") for i in idx]

    dup_base = DUP.mean()                                                              # base rate of duplicate tokens
    channels = []
    for i in range(K):
        ui = U[:, i]; wi = Vh[i]
        act = E @ ui                                                                   # (N,) channel activation (early write)
        aa = np.abs(act)
        order = np.argsort(-aa)[: args.top_ctx]
        # firing signal: does the channel fire on DUPLICATE tokens (induction family) or by POSITION?
        hi = aa >= np.quantile(aa, 0.9)                                                # top-decile firing mask
        dup_lift = (DUP[hi].mean() / max(dup_base, 1e-6))                              # duplicate enrichment when firing
        pos_corr = float(np.corrcoef(aa, POS)[0, 1])                                   # positional channel?
        wr_prof = wr[:, i] / max(wr[:, i].sum(), 1e-9); rd_prof = rd[:, i] / max(rd[:, i].sum(), 1e-9)
        writer_layers = [int(x) for x in np.argsort(-wr[:, i])[:3]]
        reader_layers = [int(x) for x in np.argsort(-rd[:, i])[:3]]
        top_ctx = []
        for p in order[:6]:
            j = int(p)                                                                 # show the top-firing token + whether it's a duplicate
            top_ctx.append({"token": tok.decode([int(IDS[j])]).replace("\n", "\\n"), "dup": float(DUP[j])})
        label = ("induction/duplicate" if dup_lift > 1.3 else
                 "positional" if abs(pos_corr) > 0.3 else "content/other")
        channels.append({
            "channel": i, "sigma": float(S[i]), "sigma_frac": float(S[i] / S.sum()),
            "writer_layers": writer_layers, "reader_layers": reader_layers,
            "writer_profile": [round(float(x), 3) for x in wr_prof], "reader_profile": [round(float(x), 3) for x in rd_prof],
            "lens_write": lens(ui), "lens_read": lens(wi),
            "dup_lift": round(float(dup_lift), 2), "pos_corr": round(pos_corr, 2), "label": label,
            "top_tokens_when_firing": top_ctx})

    # participation ratio of the coupling spectrum (the χ this decomposition is enumerating)
    p2 = (S / S.sum()); chi = float(1.0 / (p2 ** 2).sum())
    return {"model": mid.split("/")[-1], "n_layers": nL, "cut": cut, "d": d, "n_positions": len(E),
            "coupling_participation_ratio_chi": chi, "channels": channels}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-160m")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=120, help="corpus chunks")
    p.add_argument("--chars", type=int, default=300000)
    p.add_argument("--cut", type=int, default=0, help="layer cut (0 = nL//2)")
    p.add_argument("--channels", type=int, default=16, help="top coupling channels to characterise")
    p.add_argument("--top-ctx", type=int, default=200)
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
            print(f"  cut {r['cut']}/{r['n_layers']} · coupling χ≈{r['coupling_participation_ratio_chi']:.1f} · "
                  f"{r['n_positions']} positions · top-{len(r['channels'])} channels:")
            for ch in r["channels"]:
                print(f"   ch{ch['channel']:2d} σ {ch['sigma_frac']:.1%} | W{ch['writer_layers']}→R{ch['reader_layers']} "
                      f"| {ch['label']:20s} dup×{ch['dup_lift']:.2f} pos{ch['pos_corr']:+.2f} "
                      f"| read-lens: {' '.join(ch['lens_read'][:5])}")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "circuit_channels_summary.json"
    out = {"experiment": "cross-layer communication channels (candidate circuits) decomposed from the activation coupling",
           "results": results}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] → {sumpath}")
    return out


if __name__ == "__main__":
    main()
