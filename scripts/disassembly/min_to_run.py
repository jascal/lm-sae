"""Minimum-to-run frontier — fidelity vs stored size as the transformer weights are low-rank factorized.

The flat pylm program reproduces ~half the model at ~0 matmul; the rest needs compute. This measures the other end of
that frontier: how small can the model's STORED weights get (low-rank factorization of every attention + MLP weight
matrix) while still reproducing the full model, on a fidelity-vs-size curve. No-retrain baseline first (SVD-truncate the
weights to rank r); distillation can push it later. Embeddings/unembedding (the vocab/flat-knowledge) are left intact —
this measures the COMPOSITION weights' compressibility, the part that isn't flat lookup.

Per rank r: SVD-truncate c_attn / attn.c_proj / mlp.c_fc / mlp.c_proj of every layer to rank r, then measure
  FIDELITY  generic next-token NLL, and top-1 AGREEMENT with the unmodified model (the same metric as pylm's
            decompilable fraction — what fraction of the model's tokens the compressed model still predicts);
  SIZE      stored params of the factored transformer weights (r·(m+n) per matrix) vs the full m·n, as a ratio.
GPT-2 only (the anchor; nn.Conv1D weights). Output: runs/disassembly/min_to_run_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _fetch(url, n):
    try:
        return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                      timeout=20).read().decode("utf-8", "ignore")[:n]
    except Exception:
        return ""


def _gutenberg(book_id, n):
    """Project Gutenberg plaintext — try the two common URL layouts."""
    for url in (f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
                f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"):
        txt = _fetch(url, n)
        if len(txt) > 1000:
            return txt
    return ""


def _fetch_wiki(n):
    """encyclopedic prose from the Wikipedia API (plaintext extracts of diverse topics) — a big slice of GPT-2's data."""
    titles = ("Physics|World_War_II|Photosynthesis|Roman_Empire|Jupiter|Computer|Evolution|Democracy|Coffee|"
              "Mathematics|Climate|Internet|Medicine|Music|Volcano|Economics|Bacteria|Renaissance|Galaxy|Election|"
              "Quantum_mechanics|French_Revolution|DNA|Ancient_Egypt|Black_hole|Linguistics|Immune_system|Philosophy|"
              "Ocean|Architecture|Chemistry|Buddhism|Glacier|Algorithm|Constitution|Neuron|Painting|Earthquake|"
              "Probability|Ecosystem")
    out = []
    for batch in [titles.split("|")[i:i + 20] for i in range(0, len(titles.split("|")), 20)]:
        url = ("https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&exlimit=20"
               "&titles=" + "|".join(batch))
        try:
            import json as _j
            raw = _j.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                                  timeout=25).read().decode("utf-8", "ignore"))
            out += [p.get("extract", "") for p in raw["query"]["pages"].values()]
        except Exception:
            pass
    return "\n\n".join(out)[:n]


_LANG_TITLES = {  # major (long) articles per language — multilingual diversity (GPT-2's OOD tail)
    "fr": "France|Physique|Histoire|Mathématiques|Musique|Philosophie|Science|Allemagne|Paris|Univers|Biologie|Guerre",
    "de": "Deutschland|Physik|Geschichte|Mathematik|Musik|Philosophie|Wissenschaft|Frankreich|Berlin|Universum|Biologie|Krieg",
    "es": "España|Física|Historia|Matemáticas|Música|Filosofía|Ciencia|Francia|Madrid|Universo|Biología|Guerra",
    "ru": "Россия|Физика|История|Математика|Музыка|Философия|Наука|Франция|Москва|Вселенная|Биология|Война",
    "it": "Italia|Fisica|Storia|Matematica|Musica|Filosofia|Scienza|Francia|Roma|Universo|Biologia|Guerra",
    "zh": "中国|物理学|历史|数学|音乐|哲学|科学|法国|北京|宇宙|生物学|战争",
    "ja": "日本|物理学|歴史|数学|音楽|哲学|科学|フランス|東京|宇宙|生物学|戦争"}


def _fetch_wiki_lang(lang, n):
    """plaintext of major articles in a given language — multilingual diversity (GPT-2's OOD tail)."""
    url = (f"https://{lang}.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&exlimit=20"
           "&titles=" + urllib.parse.quote(_LANG_TITLES.get(lang, ""), safe="|"))
    try:
        import json as _j
        raw = _j.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                              timeout=25).read().decode("utf-8", "ignore"))
        return "\n\n".join(p.get("extract", "") for p in raw["query"]["pages"].values())[:n]
    except Exception:
        return ""


def _local_code(n):
    root = Path(__file__).resolve().parents[2]; txt = []
    for p in sorted(root.glob("scripts/disassembly/*.py")) + sorted(root.glob("pylm/*.py")):
        txt.append(p.read_text(errors="ignore"))
        if sum(len(x) for x in txt) > n:
            break
    return "".join(txt)[:n]


def build_chunks(args, tok):
    """single-domain (Shakespeare) or DIVERSE (drama + novels + code) chunk lists; diverse eval samples across domains."""
    SH = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if not args.diverse:
        ids = tok(_fetch(SH, args.corpus_chars))["input_ids"]
        ch = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
        return ch[args.eval: args.eval + args.train], ch[: args.eval], ["shakespeare"] * len(ch[: args.eval])
    per = max(args.corpus_chars, 300000)
    books = {"austen": 1342, "shelley": 84, "melville": 2701, "doyle": 1661, "dickens": 98, "darwin": 1228,
             "stoker": 345, "carroll": 11, "grimm": 2591, "machiavelli": 1232, "twain": 76, "kjv": 10}
    sources = {"shakespeare": _fetch(SH, per), "code": _local_code(per), "wiki": _fetch_wiki(per)}
    for name, bid in books.items():                                   # drama + 12 books across genres/eras + code + wiki
        sources[name] = _gutenberg(bid, per)
    if args.multilingual:                                             # + random Wikipedia in 7 other languages
        for lang in ("fr", "de", "es", "ru", "it", "zh", "ja"):
            sources[f"wiki_{lang}"] = _fetch_wiki_lang(lang, per)
    train = []; ev = []; evdom = []; ne = max(1, args.eval // max(sum(1 for v in sources.values() if v), 1))
    for name, txt in sources.items():
        if not txt:
            continue
        ids = tok(txt)["input_ids"]
        ch = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]
        ev += ch[:ne]; evdom += [name] * len(ch[:ne]); train += ch[ne:]   # held-out eval from EACH domain (labelled)
    rng = np.random.default_rng(0); rng.shuffle(train)
    return train[: args.train], ev, evdom


def run_model(mid, args):
    import torch
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL
    train_chunks, chunks, ev_dom = build_chunks(args, tok)

    # the composition weight matrices (GPT-2 Conv1D: weight shape (in, out))
    mats = []
    for L in range(nL):
        blk = vm.model.transformer.h[L]
        for mod in (blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj):
            mats.append(mod.weight)
    full_params = int(sum(W.shape[0] * W.shape[1] for W in mats))
    orig = [W.detach().clone() for W in mats]
    # precompute SVD of each matrix once
    svds = []
    for W in mats:
        U, S, Vh = torch.linalg.svd(W.detach().float(), full_matrices=False)
        svds.append((U, S, Vh))

    def gen(metric_top1=None):
        tot = 0.0; k = 0; agree = 0; preds = []; dom_hit = {}; dom_tot = {}
        with t.no_grad():
            for ci, c in enumerate(chunks):
                lg = vm.logits(c).float(); lp = t.log_softmax(lg, -1); y = c[1:]
                top1 = lg[:-1].argmax(-1)
                preds.append(top1.cpu())
                for p in range(len(y)):
                    tot += float(-lp[p, y[p]]); k += 1
                if metric_top1 is not None:
                    a = int((top1 == metric_top1[ci].to(top1.device)).sum()); agree += a
                    d = ev_dom[ci]; dom_hit[d] = dom_hit.get(d, 0) + a; dom_tot[d] = dom_tot.get(d, 0) + len(y)
        per_dom = {d: dom_hit[d] / max(dom_tot[d], 1) for d in dom_tot}
        return tot / max(k, 1), preds, (agree / max(k, 1) if metric_top1 is not None else None), per_dom

    full_nll, full_top1, _, _ = gen()

    def set_rank(r):
        for (W, (U, S, Vh)) in zip(mats, svds):
            Wr = (U[:, :r] * S[:r]) @ Vh[:r]
            W.data.copy_(Wr.to(W.dtype))

    def restore():
        for W, o in zip(mats, orig):
            W.data.copy_(o)

    train = train_chunks                                              # diverse or single-domain (from build_chunks)

    def fit_distill(r, steps):
        """factor every weight W≈A·B at rank r (SVD init), TRAIN the factors to match the full model (others frozen)."""
        for p in vm.model.parameters():
            p.requires_grad_(False)
        A = []; B = []
        for (W, (U, S, Vh)) in zip(mats, svds):
            s = S[:r].sqrt()
            A.append((U[:, :r] * s).detach().clone().requires_grad_(True))
            B.append((s[:, None] * Vh[:r]).detach().clone().requires_grad_(True))
        mods = []
        for L in range(nL):
            blk = vm.model.transformer.h[L]
            mods += [blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj]
        factor_on = [True]; hs = []

        def mk(j):
            def hook(m, i, o):
                return i[0] @ (A[j] @ B[j]).to(i[0].dtype) + m.bias if factor_on[0] else None   # off → teacher
            return hook
        for j, mod in enumerate(mods):
            hs.append(mod.register_forward_hook(mk(j)))
        opt = torch.optim.Adam(A + B, lr=args.lr); rng = np.random.default_rng(0); T = 2.0
        for s in range(steps):
            j = int(rng.integers(0, len(train))); tid = t.tensor([train[j]], device=vm.dev)
            if args.match_teacher:                                    # soft-KL distillation to the full model
                factor_on[0] = False
                with t.no_grad():
                    teach = t.log_softmax(vm.model(input_ids=tid).logits[0][:-1].float() / T, -1)
                factor_on[0] = True
                student = t.log_softmax(vm.model(input_ids=tid).logits[0][:-1].float() / T, -1)
                loss = t.nn.functional.kl_div(student, teach, log_target=True, reduction="batchmean") * (T * T)
            else:                                                     # corpus NLL (capable, not faithful)
                logits = vm.model(input_ids=tid).logits[0]
                loss = t.nn.functional.cross_entropy(logits[:-1].float(), tid[0, 1:])
            opt.zero_grad(); loss.backward(); opt.step()
        factor_on[0] = True
        nll, _, agree, per_dom = gen(metric_top1=full_top1)
        for h in hs:
            h.remove()
        return nll, agree, per_dom

    ranks = sorted({int(x) for x in args.ranks.split(",")})
    curve = []
    for r in ranks:
        set_rank(r); nll0, _, agree0, _ = gen(metric_top1=full_top1); restore()   # no-retrain SVD baseline
        nll_d, agree_d, per_dom = (fit_distill(r, args.distill_steps) if args.distill_steps > 0 else (None, None, None))
        stored = int(sum(r * (W.shape[0] + W.shape[1]) for W in mats))
        curve.append({"rank": r, "svd_nll_increase": nll0 - full_nll, "svd_agreement": agree0,
                      "distilled_nll_increase": (nll_d - full_nll) if nll_d is not None else None,
                      "distilled_agreement": agree_d, "distilled_per_domain": per_dom, "stored_params": stored,
                      "compression_ratio": stored / full_params, "params_saved_frac": 1 - stored / full_params})
    return {"model": mid.split("/")[-1], "n_layers": nL, "full_nll": full_nll,
            "transformer_weight_params_M": full_params / 1e6, "distill_steps": args.distill_steps, "curve": curve}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--eval", type=int, default=24)
    p.add_argument("--ranks", default="8,16,32,64,128,256,512", help="SVD ranks to truncate the weight matrices to")
    p.add_argument("--distill-steps", type=int, default=0, help="if >0, also distill the low-rank factors (train them)")
    p.add_argument("--train", type=int, default=200, help="train chunks for distillation")
    p.add_argument("--corpus-chars", type=int, default=120000, help="chars of corpus to fetch (more = more distill data)")
    p.add_argument("--diverse", action="store_true", help="distill on a diverse multi-domain corpus (drama+novels+code)")
    p.add_argument("--multilingual", action="store_true", help="add random Wikipedia in 7 non-English languages")
    p.add_argument("--lr", type=float, default=1e-3, help="distillation learning rate")
    p.add_argument("--match-teacher", action="store_true", help="distill to the full model's top-1 (faithful) vs corpus NLL (capable)")
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
            print(f"  transformer-weight params {r['transformer_weight_params_M']:.0f}M | full NLL {r['full_nll']:.3f} "
                  f"| distill steps {r['distill_steps']}")
            print("  rank → stored% · no-retrain SVD (ΔNLL/agree) · DISTILLED (ΔNLL/agree):")
            for c in r["curve"]:
                dd = (f"distilled ΔNLL {c['distilled_nll_increase']:+.3f} agree {c['distilled_agreement']:.0%}"
                      if c['distilled_nll_increase'] is not None else "distilled --")
                print(f"    rank {c['rank']:4d}  stored {c['compression_ratio']:.0%} ({c['params_saved_frac']:.0%} saved)  "
                      f"| SVD ΔNLL {c['svd_nll_increase']:+.2f} agree {c['svd_agreement']:.0%}  | {dd}")
                if c.get("distilled_per_domain"):
                    print("        per-domain faithful agreement: " +
                          " ".join(f"{d} {v:.0%}" for d, v in sorted(c["distilled_per_domain"].items(), key=lambda x: -x[1])))
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "min_to_run_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "minimum-to-run frontier — fidelity vs stored size under no-retrain low-rank weight factorization", "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'curve' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
