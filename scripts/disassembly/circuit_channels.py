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


def build_corpus(args, tok):
    """single-source (Shakespeare) or DIVERSE (shakespeare+austen+code+wiki) chunks, each tagged with a source id."""
    def _chunk(txt):
        idx = tok(txt)["input_ids"]
        return [idx[i:i + args.ctx] for i in range(0, len(idx), args.ctx) if len(idx[i:i + args.ctx]) >= 16]
    if args.diverse:
        from min_to_run import _fetch_wiki, _gutenberg, _local_code
        per = max(args.chars, 120000); cap = max(1, args.fit // 4)
        sources = {"shakespeare": _fetch(CORPUS, per), "austen": _gutenberg(1342, per),
                   "code": _local_code(per), "wiki": _fetch_wiki(per)}
        chunks = []; csrc = []; names = []
        for si, (nm, txt) in enumerate(sources.items()):
            cc = _chunk(txt)[:cap]
            if cc:
                names.append(nm); chunks += cc; csrc += [si] * len(cc)
        return chunks, csrc, names
    chunks = _chunk(_fetch(CORPUS, args.chars))[: args.fit]
    return chunks, [0] * len(chunks), ["shakespeare"]


def _lens(WU, tok, vec, topn=8):
    s = WU @ vec; idx = np.argsort(-s)[:topn]
    return [tok.decode([int(i)]).replace("\n", "\\n") for i in idx]


def _firing(act, DUP, POS, CID, SRC, dup_base):
    """firing-pattern signatures of a channel's per-position activation: duplicate enrichment, positional correlation,
    PERSISTENCE (lag-1 autocorr within chunk = held/broadcast), REGISTER fraction (between-source variance = topic)."""
    aa = np.abs(act); hi = aa >= np.quantile(aa, 0.9)
    dup_lift = aa[hi].size and DUP[hi].mean() / max(dup_base, 1e-6) or 0.0
    pos_corr = float(np.corrcoef(aa, POS)[0, 1])
    acs = []
    for cid in np.unique(CID):
        a = act[CID == cid]
        if len(a) > 3:
            a0 = a - a.mean(); den = (a0[:-1] ** 2).sum()
            if den > 1e-9:
                acs.append(float((a0[:-1] * a0[1:]).sum() / den))
    persistence = float(np.mean(acs)) if acs else 0.0
    if len(np.unique(SRC)) > 1:
        means = np.array([act[SRC == s].mean() for s in np.unique(SRC)])
        register_frac = float(np.var(means) / (np.var(act) + 1e-9))
    else:
        register_frac = float("nan")
    label = ("induction/duplicate" if dup_lift > 1.3 else "positional" if abs(pos_corr) > 0.3 else "content/other")
    return {"dup_lift": round(float(dup_lift), 2), "pos_corr": round(pos_corr, 2), "persistence": round(persistence, 2),
            "register_frac": round(register_frac, 2), "label": label,
            "broadcast": bool(persistence > 0.5 and (register_frac > 0.3 or np.isnan(register_frac)))}


def run_resolved(mid, args):
    """PER-LAYER-RESOLVED channels (faithful core_mps coupling, χ≈16). Per-layer top-r PCA + standardise each layer's
    write; at a cut, SVD the (early-layers·r)×(late-layers·r) standardised cross-covariance into channels that MIX
    layers. Each channel: writer/reader LAYER profile (which layers' coords carry it), logit-lens of the early/late
    d-space directions (reconstructed from the per-layer PCA bases), and firing pattern."""
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d
    r = args.rank; cut = args.cut if args.cut > 0 else nL // 2
    WU = vm.model.get_output_embeddings().weight.detach().float().cpu().numpy()
    chunks, csrc, src_names = build_corpus(args, tok)

    def capture():
        cap = {}
        hks = [vm.layers[L].register_forward_hook(
            (lambda L: lambda m, i, o: cap.__setitem__(L, ((o[0] if isinstance(o, tuple) else o) - i[0]).detach()))(L))
            for L in range(nL)]
        return cap, hks

    # pass 1: per-layer write covariance → top-r PCA basis + std (for standardising the coords)
    cov = {L: np.zeros((d, d)) for L in range(nL)}
    cap, hks = capture()
    with t.no_grad():
        for c in chunks:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                u = cap[L][0].float().cpu().numpy(); cov[L] += u.T @ u
    for h in hks:
        h.remove()
    bases = {}; std = {}
    for L in range(nL):
        w, V = np.linalg.eigh(cov[L]); order = np.argsort(-w)[:r]
        bases[L] = V[:, order].astype(np.float32); std[L] = np.sqrt(np.clip(w[order], 1e-9, None)).astype(np.float32)

    # pass 2: standardised per-layer coords → (early·r)×(late·r) cross-covariance + stored coords for firing
    el = list(range(cut + 1)); ll = list(range(cut + 1, nL)); nE = len(el) * r; nLt = len(ll) * r
    Cb = np.zeros((nE, nLt)); EC = []; IDS = []; DUP = []; POS = []; CID = []; SRC = []
    cap, hks = capture()
    with t.no_grad():
        for ci, c in enumerate(chunks):
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            ec = np.concatenate([(cap[L][0].float().cpu().numpy() @ bases[L]) / std[L] for L in el], 1)   # (seq,nE)
            lc = np.concatenate([(cap[L][0].float().cpu().numpy() @ bases[L]) / std[L] for L in ll], 1)   # (seq,nLt)
            Cb += ec.T @ lc
            seen = set()
            for p in range(len(c)):
                dup = 1.0 if c[p] in seen else 0.0; seen.add(c[p])
                EC.append(ec[p]); IDS.append(c[p]); DUP.append(dup); POS.append(p); CID.append(ci); SRC.append(csrc[ci])
    for h in hks:
        h.remove()
    Cb /= max(len(EC), 1)
    EC = np.asarray(EC); DUP = np.asarray(DUP); POS = np.asarray(POS, float); CID = np.asarray(CID); SRC = np.asarray(SRC)

    U, S, Vh = np.linalg.svd(Cb, full_matrices=False)                                   # channels mixing layers
    p2 = S / S.sum(); chi = float(1.0 / (p2 ** 2).sum())
    K = min(args.channels, len(S)); dup_base = DUP.mean()
    channels = []
    for i in range(K):
        pi = U[:, i]; qi = Vh[i]
        wr = np.array([np.abs(pi[k * r:(k + 1) * r]).sum() for k in range(len(el))])    # early-layer profile
        rd = np.array([np.abs(qi[k * r:(k + 1) * r]).sum() for k in range(len(ll))])    # late-layer profile
        ud = sum((bases[el[k]] @ pi[k * r:(k + 1) * r]) for k in range(len(el)))        # early d-direction
        wd = sum((bases[ll[k]] @ qi[k * r:(k + 1) * r]) for k in range(len(ll)))        # late d-direction
        fs = _firing(EC @ pi, DUP, POS, CID, SRC, dup_base)
        channels.append({"channel": i, "sigma": float(S[i]), "sigma_frac": float(S[i] / S.sum()),
                         "writer_layers": [int(el[k]) for k in np.argsort(-wr)[:3]],
                         "reader_layers": [int(ll[k]) for k in np.argsort(-rd)[:3]],
                         "lens_write": _lens(WU, tok, ud), "lens_read": _lens(WU, tok, wd), **fs})
    return {"model": mid.split("/")[-1], "n_layers": nL, "cut": cut, "d": d, "rank": r, "n_positions": len(EC),
            "coupling_participation_ratio_chi": chi, "resolved": True, "channels": channels}


def run_model(mid, args):
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device, dtype="fp32"); t = vm.torch; tok = vm.tok; nL = vm.nL; d = vm.d
    cut = args.cut if args.cut > 0 else nL // 2
    WU = vm.model.get_output_embeddings().weight.detach().float().cpu().numpy()    # (vocab, d) unembedding, logit lens

    chunks, csrc, src_names = build_corpus(args, tok)             # single-source or diverse (sources tagged)

    def capture():
        cap = {}
        hks = [vm.layers[L].register_forward_hook(
            (lambda L: lambda m, i, o: cap.__setitem__(L, ((o[0] if isinstance(o, tuple) else o) - i[0]).detach()))(L))
            for L in range(nL)]
        return cap, hks

    # ---- pass 1: cross-covariance C(early,late) + store per-position early/late/ids/duplicate flag ----
    C = np.zeros((d, d)); E = []; Lt = []; IDS = []; DUP = []; POS = []; CID = []; SRC = []
    cap, hks = capture()
    with t.no_grad():
        for ci, c in enumerate(chunks):
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            early = sum(cap[L][0] for L in range(cut + 1)).float().cpu().numpy()       # (seq, d)
            late = sum(cap[L][0] for L in range(cut + 1, nL)).float().cpu().numpy()    # (seq, d)
            early = early / (np.linalg.norm(early, axis=1, keepdims=True) + 1e-9)      # DIRECTIONAL coupling — unit-norm
            late = late / (np.linalg.norm(late, axis=1, keepdims=True) + 1e-9)        # per position (else last-layer magnitude swamps it)
            C += early.T @ late
            seen = set()
            for p in range(len(c)):
                dup = 1.0 if c[p] in seen else 0.0; seen.add(c[p])
                E.append(early[p]); Lt.append(late[p]); IDS.append(c[p]); DUP.append(dup)
                POS.append(p); CID.append(ci); SRC.append(csrc[ci])
    for h in hks:
        h.remove()
    C /= max(len(E), 1)
    E = np.asarray(E); Lt = np.asarray(Lt); IDS = np.asarray(IDS); DUP = np.asarray(DUP); POS = np.asarray(POS, float)
    CID = np.asarray(CID); SRC = np.asarray(SRC)

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
        # BROADCAST / TOPIC signatures: persistence (held across positions) + register fraction (encodes the source/topic)
        acs = []
        for cid in np.unique(CID):
            a = act[CID == cid]
            if len(a) > 3:
                a0 = a - a.mean(); den = (a0[:-1] ** 2).sum()
                if den > 1e-9:
                    acs.append(float((a0[:-1] * a0[1:]).sum() / den))             # lag-1 autocorr within chunk
        persistence = float(np.mean(acs)) if acs else 0.0
        if len(np.unique(SRC)) > 1:
            means = np.array([act[SRC == s].mean() for s in np.unique(SRC)])
            register_frac = float(np.var(means) / (np.var(act) + 1e-9))           # between-source variance fraction
        else:
            register_frac = float("nan")
        broadcast = bool(persistence > 0.5 and (register_frac > 0.3 or np.isnan(register_frac)))
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
            "persistence": round(persistence, 2), "register_frac": round(register_frac, 2), "broadcast": broadcast,
            "top_tokens_when_firing": top_ctx})

    # ---- CAUSAL: project a channel OUT at the cut (late layers can't read it) → targeted behaviour drop? ----
    if args.causal:
        PUNCT = set('.,;:!?"\'()-\n—’“”…`')
        vocab = WU.shape[0]
        punct_id = np.zeros(vocab, bool)
        for v in range(vocab):
            s = tok.decode([v]).strip()
            if s != "" and all(ch in PUNCT for ch in s):
                punct_id[v] = True

        def nll_split(udir):                                          # mean NLL on punct / duplicate / other next-tokens
            hk = []
            if udir is not None:
                uu = t.tensor(udir, device=vm.dev, dtype=t.float32); uu = uu / uu.norm()

                def hook(m, i, o):
                    out = o[0] if isinstance(o, tuple) else o
                    new = (out.float() - (out.float() @ uu)[..., None] * uu).to(out.dtype)
                    return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
                hk.append(vm.layers[cut].register_forward_hook(hook))
            tot = {"punct": 0.0, "dup": 0.0, "other": 0.0}; cnt = {"punct": 0, "dup": 0, "other": 0}
            with t.no_grad():
                for c in chunks:
                    lp = t.log_softmax(vm.model(input_ids=t.tensor([c], device=vm.dev)).logits[0].float(), -1)
                    seen = set()
                    for pp in range(len(c) - 1):
                        seen.add(c[pp]); nxt = c[pp + 1]
                        cat = "punct" if punct_id[nxt] else ("dup" if nxt in seen else "other")
                        tot[cat] += -float(lp[pp, nxt]); cnt[cat] += 1
            for h in hk:
                h.remove()
            return {k: tot[k] / max(cnt[k], 1) for k in tot}
        base = nll_split(None)
        for ch in channels[: args.causal_k]:                         # ablate each top channel's early-dir at the cut
            ab = nll_split(U[:, ch["channel"]])
            ch["causal_dNLL"] = {k: round(ab[k] - base[k], 3) for k in base}
        print(f"  [causal] baseline NLL punct {base['punct']:.2f} · dup {base['dup']:.2f} · other {base['other']:.2f}")

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
    p.add_argument("--resolved", action="store_true", help="PER-LAYER-RESOLVED coupling (core_mps-style, exposes χ≈16) vs aggregate cut")
    p.add_argument("--rank", type=int, default=24, help="per-layer PCA rank for the resolved coupling")
    p.add_argument("--diverse", action="store_true", help="diverse corpus (shakespeare+austen+code+wiki) → register/topic-broadcast test")
    p.add_argument("--causal", action="store_true", help="ablate each top channel at the cut, measure targeted ΔNLL (punct/dup/other)")
    p.add_argument("--causal-k", type=int, default=8, help="how many top channels to causally ablate")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly"))
    args = p.parse_args(argv)

    import torch
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            r = run_resolved(mid, args) if args.resolved else run_model(mid, args)
            if args.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            results.append(r)
            print(f"  cut {r['cut']}/{r['n_layers']} · coupling χ≈{r['coupling_participation_ratio_chi']:.1f} · "
                  f"{r['n_positions']} positions · top-{len(r['channels'])} channels:")
            for ch in r["channels"]:
                cz = ch.get("causal_dNLL")
                ctxt = (f" | ablate ΔNLL punct {cz['punct']:+.2f} dup {cz['dup']:+.2f} other {cz['other']:+.2f}" if cz else "")
                bc = " «BROADCAST»" if ch.get("broadcast") else ""
                print(f"   ch{ch['channel']:2d} σ {ch['sigma_frac']:.1%} | W{ch['writer_layers']}→R{ch['reader_layers']} "
                      f"| {ch['label']:20s} dup×{ch['dup_lift']:.2f} persist {ch['persistence']:+.2f} reg {ch['register_frac']}{bc} "
                      f"| read-lens: {' '.join(ch['lens_read'][:4])}{ctxt}")
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
