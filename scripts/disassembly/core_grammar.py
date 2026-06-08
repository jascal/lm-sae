"""Is the entangled core a GENERIC GRAMMAR (à la Chomsky)? — test the shared write-basis for the two properties a
universal-grammar core would have: GENERIC (corpus-invariant) and GRAMMATICAL (closed-class / function-word, not content).

`core_basis_decompile.py` found the shared cross-layer write-basis Ug is logit-lens-readable and its top directions
decode to grammatical *classes* (determiners, punctuation, pronouns, a verb axis) — not to content / facts. That
invites the hypothesis the user named: **the entangled core is a generic grammar** — the content-free syntactic
scaffolding every layer reuses, with lexical/factual content carried elsewhere. A grammar core makes two predictions:

  GENERIC  — it should be ~the SAME basis no matter what corpus you fit it on (grammar is content-independent). Fit the
             shared core on structurally different corpora (Shakespeare drama / a modern novel / Python source) and
             measure pairwise subspace overlap. Contrast against the UNSHARED TAIL (the least-shared singular
             directions of the same stacked basis = the content candidate): a grammar core should be FAR more
             corpus-invariant than the content tail.
  GRAMMATICAL — its directions should promote CLOSED-CLASS tokens (function words + punctuation), not open-class
             content words. Logit-lens each direction; score the closed-class fraction of its top tokens. Contrast
             core vs tail vs random: grammar core ≫ tail (content) in closed-class mass.

The dissociation (core = generic & closed-class; tail = corpus-specific & open-class) is the decisive evidence for /
against the grammar reading. Pure analysis on the frozen model, no retraining. Output: runs/disassembly/core_grammar_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_rank import participation_ratio  # noqa: E402

# Closed-class (function-word) lexicon — determiners, prepositions, conjunctions, pronouns, auxiliaries, particles.
CLOSED = (
    "the a an this that these those my your his her its our their some any no each every either neither "
    "of to in for on with at by from up about into over after under above below between out off down through "
    "and or but nor so yet because although while if then than as when where while since unless whereas "
    "i you he she it we they me him us them mine yours hers ours theirs who whom whose which what "
    "is are was were be been being am do does did have has had having will would shall should can could may might must "
    "not no nor too very just only also there here now then thus hence however therefore "
    "he's it's that's there's i'm you're we're they're don't can't won't"
).split()
PUNCT = list(".,;:!?'\"()[]{}-—…/")


def closed_ids(tok):
    ids = set()
    for w in CLOSED:
        for form in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            t = tok(form, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.add(t[0])
    for p in PUNCT:
        for form in (p, " " + p):
            t = tok(form, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.add(t[0])
    # newline / paragraph tokens are structural too
    for form in ("\n", "\n\n"):
        t = tok(form, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ids.add(t[0])
    return ids


def captured_fraction(A, B):
    """Fraction of subspace B (d,kb) captured by A (d,ka), both orthonormal: ‖Aᵀ B‖²_F / kb."""
    return float(np.square(A.T @ B).sum() / B.shape[1])


def fetch(url, n=200000):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read().decode("utf-8", "ignore")[:n]


def local_code(n=200000):
    """Python source from this repo — a structurally distinct 'corpus' (no network)."""
    root = Path(__file__).resolve().parents[2]
    txt = []
    for p in sorted(root.glob("scripts/disassembly/*.py")) + sorted(root.glob("pylm/*.py")):
        txt.append(p.read_text(errors="ignore"))
        if sum(len(x) for x in txt) > n:
            break
    return "".join(txt)[:n]


def stacked_basis(vm, chunks, share_rank):
    """Per-layer residual-update PCA → stacked SVD → (Uall columns desc by sharedness, singular values, K=union rank)."""
    t = vm.torch; nL = vm.nL; d = vm.d
    cov = {L: np.zeros((d, d)) for L in range(nL)}; cap = {}
    hks = [vm.layers[L].register_forward_hook(
        (lambda L: lambda m, i, o: cap.__setitem__(L, ((o[0] if isinstance(o, tuple) else o) - i[0]).detach()))(L))
        for L in range(nL)]
    with t.no_grad():
        for c in chunks:
            cap.clear(); vm.model(input_ids=t.tensor([c], device=vm.dev))
            for L in range(nL):
                u = cap[L][0].float().cpu().numpy(); cov[L] += u.T @ u
    for h in hks:
        h.remove()
    rs = min(share_rank, d // 2); cols = []
    for L in range(nL):
        w, V = np.linalg.eigh(cov[L]); order = np.argsort(-w)
        cols.append(V[:, order[:rs]].astype(np.float64))
    Uall, sv, _ = np.linalg.svd(np.concatenate(cols, axis=1), full_matrices=False)
    K = max(1, min(int(round(participation_ratio(sv ** 2))), d))
    return Uall, sv, K


def run_model(mid, args):
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); tok = vm.tok; d = vm.d
    cset = closed_ids(tok)
    WU = vm.model.get_output_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)

    corpora = {
        "shakespeare": fetch("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"),
        "novel": fetch("https://www.gutenberg.org/files/1342/1342-0.txt"),       # Pride & Prejudice (modern prose)
        "code": local_code(),                                                    # Python source (distinct grammar)
    }
    bases = {}
    Kref = None
    for name, txt in corpora.items():
        ids = tok(txt)["input_ids"]
        chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8][: args.fit]
        Uall, sv, K = stacked_basis(vm, chunks, args.share_rank)
        if Kref is None:
            Kref = K                                                             # one K for fair cross-corpus overlap
        bases[name] = {"core": Uall[:, :Kref], "tail": Uall[:, -Kref:], "K_native": K, "n_chunks": len(chunks)}
    K = Kref

    # ---- GENERIC: pairwise cross-corpus subspace overlap, core vs content tail (chance = K/d) ----
    names = list(corpora)
    pairs = [(a, b) for ia, a in enumerate(names) for b in names[ia + 1:]]
    core_ov = {f"{a}~{b}": captured_fraction(bases[a]["core"], bases[b]["core"]) for a, b in pairs}
    tail_ov = {f"{a}~{b}": captured_fraction(bases[a]["tail"], bases[b]["tail"]) for a, b in pairs}

    # ---- GRAMMATICAL: closed-class fraction of logit-lens top tokens, core vs tail vs random ----
    rng = np.random.default_rng(0)
    Rrand = np.linalg.svd(rng.standard_normal((d, K)), full_matrices=False)[0]

    def closed_frac(vec, topn=10):
        lg = WU @ vec
        pos = [int(i) for i in np.argsort(-lg)[:topn]]; neg = [int(i) for i in np.argsort(lg)[:topn]]
        # pick the more peaked pole (consistent with core_basis lens), score its closed-class hit-rate
        zp = (lg.max() - lg.mean()) / (lg.std() + 1e-9); side = pos if zp >= (lg.mean() - lg.min()) / (lg.std() + 1e-9) else neg
        return float(np.mean([i in cset for i in side]))

    def basis_closed(B):
        return float(np.mean([closed_frac(B[:, k]) for k in range(B.shape[1])])) if B.shape[1] else 0.0

    # the columns of "core" are sorted by SHAREDNESS (top of the stacked-SVD spectrum). Grammar (if it exists) should
    # be the MOST-shared head; content fills the deeper, less-shared bulk. So bin closed-class fraction by rank.
    ref = bases["shakespeare"]
    bins = [("top16", 0, 16), ("mid16_64", 16, 64), ("deep64_K", 64, K)]
    gram = {f"closed_{nm}": basis_closed(ref["core"][:, lo:hi]) for nm, lo, hi in bins}
    gram.update({"closed_tail": basis_closed(ref["tail"]), "closed_random": basis_closed(Rrand),
                 "closed_core_all": basis_closed(ref["core"]), "closed_vocab_size": len(cset)})

    # GENERIC, binned by sharedness: is the most-shared head MORE corpus-invariant than the deeper/content bulk?
    def binned_overlap(lo, hi):
        vals = [captured_fraction(bases[a]["core"][:, lo:hi], bases[b]["core"][:, lo:hi]) for a, b in pairs]
        return float(np.mean(vals)), (hi - lo) / d                       # (mean overlap, chance for this bin)
    gen_bins = {}
    for nm, lo, hi in bins:
        ov, ch = binned_overlap(lo, hi); gen_bins[nm] = {"overlap": ov, "chance": ch}
    tail_ov_mean = float(np.mean(list(tail_ov.values())))

    def examples(B, n=6, topn=8):
        out = []
        for k in range(min(n, B.shape[1])):
            lg = WU @ B[:, k]
            zp = (lg.max() - lg.mean()) / (lg.std() + 1e-9); zn = (lg.mean() - lg.min()) / (lg.std() + 1e-9)
            idx = np.argsort(-lg)[:topn] if zp >= zn else np.argsort(lg)[:topn]
            out.append([tok.convert_ids_to_tokens(int(i)).replace("Ġ", "_").replace("Ċ", "\\n") for i in idx])
        return out

    return {"model": mid.split("/")[-1], "d_model": d, "core_dim_K": K,
            "corpora": {n: bases[n]["K_native"] for n in names}, "chance_overlap": K / d,
            "generic": {"core_overlap": core_ov, "tail_overlap": tail_ov,
                        "mean_core_overlap": float(np.mean(list(core_ov.values()))),
                        "mean_tail_overlap": tail_ov_mean, "binned": gen_bins},
            "grammatical": gram,
            "core_examples": examples(ref["core"]), "tail_examples": examples(ref["tail"])}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--fit", type=int, default=40)
    p.add_argument("--share-rank", type=int, default=64)
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
            g = r["generic"]; gr = r["grammatical"]; gb = g["binned"]
            print(f"  d{r['d_model']} | core dim K={r['core_dim_K']} (chance overlap {r['chance_overlap']:.2f})")
            print("  GENERIC  cross-corpus overlap by sharedness rank (overlap vs chance):")
            for nm in ("top16", "mid16_64", "deep64_K"):
                print(f"             {nm:9s} {gb[nm]['overlap']:.2f} vs {gb[nm]['chance']:.2f}")
            print(f"             content-tail {g['mean_tail_overlap']:.2f} vs {r['chance_overlap']:.2f}")
            print(f"  GRAMMAR  closed-class fraction by rank: top16 {gr['closed_top16']:.2f} · "
                  f"mid {gr['closed_mid16_64']:.2f} · deep {gr['closed_deep64_K']:.2f} · "
                  f"tail {gr['closed_tail']:.2f} · random {gr['closed_random']:.2f} (vocab {gr['closed_vocab_size']})")
            print("  core dirs:  " + " | ".join(" ".join(e) for e in r["core_examples"][:4]))
            print("  tail dirs:  " + " | ".join(" ".join(e) for e in r["tail_examples"][:4]))
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "core_grammar_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "is the entangled core a generic grammar — corpus-invariance + closed-class dominance of the shared write-basis",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'core_dim_K' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
