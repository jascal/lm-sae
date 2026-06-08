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
Arch-generic: GPT-2 (Conv1D), Pythia/GPT-NeoX and Llama/Qwen (nn.Linear). Output: runs/disassembly/min_to_run_summary.json.
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


def composition_mats(model):
    """(module, is_linear) for every attention+MLP weight matrix, arch-generic.

    The forward-hook below replaces each module's output with a low-rank product, so we need the *effective* in→out map
    M such that `out = x @ M (+ bias)`:  Conv1D (GPT-2) stores weight as (in, out) → M = W;  nn.Linear (GPT-NeoX/Pythia,
    Llama/Qwen) stores it as (out, in) → M = Wᵀ. We carry the is_linear flag so SVD and writeback use the right side.
    """
    recs = []
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):           # GPT-2 (Conv1D)
        for blk in model.transformer.h:
            recs += [(blk.attn.c_attn, False), (blk.attn.c_proj, False),
                     (blk.mlp.c_fc, False), (blk.mlp.c_proj, False)]
    elif hasattr(model, "gpt_neox"):                                                 # Pythia / GPT-NeoX (Linear, fused qkv)
        for ly in model.gpt_neox.layers:
            recs += [(ly.attention.query_key_value, True), (ly.attention.dense, True),
                     (ly.mlp.dense_h_to_4h, True), (ly.mlp.dense_4h_to_h, True)]
    elif hasattr(model, "model") and hasattr(model.model, "layers"):                 # Llama/Qwen RoPE (Linear, split qkv)
        for ly in model.model.layers:
            a, mlp = ly.self_attn, ly.mlp
            recs += [(a.q_proj, True), (a.k_proj, True), (a.v_proj, True), (a.o_proj, True),
                     (mlp.gate_proj, True), (mlp.up_proj, True), (mlp.down_proj, True)]
    else:
        raise SystemExit("min_to_run: unknown architecture for weight factorization")
    return recs


def _student_dtype(mid):
    """fp32 for small models (stable factor training); bf16 only for the genuinely large ones (memory)."""
    big = any(s in mid.lower() for s in ("-xl", "1.4b", "1.5b", "2.8b", "6.9b", "7b", "8b"))
    return "auto" if big else "fp32"


def run_model(mid, args):
    import torch
    from residual_vm import ResidualVM
    from transformers import AutoTokenizer
    t = torch
    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    train_chunks, chunks, ev_dom = build_chunks(args, tok)

    # SEPARATE teacher (e.g. gpt2-xl → gpt2-large): precompute its targets with ONLY the teacher loaded, then FREE it
    # BEFORE the student loads — so teacher + student never coexist on the GPU (teacher for precompute, student for training).
    teacher_top1 = None; teacher_soft = None; TT = 2.0; TK = 40
    if args.teacher:
        teacher_model = ResidualVM(args.teacher, device=args.device).model
        teacher_top1 = []; teacher_soft = []
        with t.no_grad():
            for c in chunks:
                teacher_top1.append(teacher_model(input_ids=t.tensor([c], device=dev)).logits[0, :-1].argmax(-1).cpu())
            for c in train_chunks:
                tlp = t.log_softmax(teacher_model(input_ids=t.tensor([c], device=dev)).logits[0][:-1].float() / TT, -1)
                vals, idx = tlp.topk(TK, -1)
                teacher_soft.append((idx.cpu(), t.log_softmax(vals, -1).cpu()))   # renorm over top-k; CPU to save GPU
        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    vm = ResidualVM(mid, device=args.device, dtype=_student_dtype(mid)); nL = vm.nL   # student alone on GPU (fp32 → stable)

    # the composition weight matrices, arch-generic (Conv1D or Linear)
    recs = composition_mats(vm.model)
    mods = [m for (m, _) in recs]; mats = [m.weight for (m, _) in recs]; is_lin = [lin for (_, lin) in recs]

    def eff(W, lin):                       # effective in→out matrix (out = x @ eff): Conv1D weight is (in,out), Linear (out,in)
        return W.t() if lin else W

    full_params = int(sum(W.shape[0] * W.shape[1] for W in mats))
    orig = [W.detach().clone() for W in mats]

    def input_covariances():
        """per-matrix input covariance Cⱼ = E[xxᵀ] over the corpus — for FUNCTIONAL (data-aware) low-rank, which
        minimizes ‖X(W−AB)‖ (the directions the activations actually visit) instead of ‖W−AB‖_F (every direction)."""
        cov = [None] * len(mods); cnt = [0] * len(mods); hs = []
        for j, mod in enumerate(mods):
            def mk(j):
                def hook(m, i):
                    x = i[0].detach().reshape(-1, i[0].shape[-1]).float(); g = x.t() @ x
                    cov[j] = g if cov[j] is None else cov[j] + g; cnt[j] += x.shape[0]
                return hook
            hs.append(mod.register_forward_pre_hook(mk(j)))
        with t.no_grad():
            for c in (chunks + train_chunks):
                vm.model(input_ids=t.tensor([c], device=vm.dev))
        for h in hs:
            h.remove()
        return [(cov[j] / max(cnt[j], 1)).cpu() for j in range(len(mods))]

    def daware_factor(M, C, ridge=1e-3):
        """functional rank-r factors: SVD in the data norm. Returns (P, S, Vh) where M_r = (P[:,:r]*S[:r])@Vh[:r],
        P = C^{-1/2}U taking U's role — so set_rank / fit_distill below are unchanged (plain SVD is P=U, C=I)."""
        C = C + ridge * C.diag().mean() * t.eye(C.shape[0])
        w, Q = t.linalg.eigh(C); w = w.clamp_min(1e-8)
        Chalf = (Q * w.sqrt()) @ Q.t(); Cinv = (Q * w.rsqrt()) @ Q.t()
        U, S, Vh = t.linalg.svd(Chalf @ M, full_matrices=False)
        return Cinv @ U, S, Vh

    # precompute (data-aware or plain) low-rank factors of each EFFECTIVE matrix once — kept on CPU (GPU fits student+teacher)
    if args.data_aware:
        covs = input_covariances()
        svds = [daware_factor(eff(W, lin).detach().float().cpu(), covs[j]) for j, (W, lin) in enumerate(zip(mats, is_lin))]
    else:
        svds = [torch.linalg.svd(eff(W, lin).detach().float().cpu(), full_matrices=False) for W, lin in zip(mats, is_lin)]

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
        for (W, lin, (U, S, Vh)) in zip(mats, is_lin, svds):
            Mr = (U[:, :r] * S[:r]) @ Vh[:r]                          # (in, out) effective; transpose back if Linear
            Wr = Mr.t() if lin else Mr
            W.data.copy_(Wr.to(device=W.device, dtype=W.dtype))

    def restore():
        for W, o in zip(mats, orig):
            W.data.copy_(o)

    train = train_chunks                                              # diverse or single-domain (from build_chunks)

    def fit_distill(r, steps):
        """factor every weight W≈A·B at rank r (SVD init), TRAIN the factors to match the full model (others frozen)."""
        for p in vm.model.parameters():
            p.requires_grad_(False)
        A = []; B = []; alpha = []
        for (W, (U, S, Vh)) in zip(mats, svds):
            s = S[:r].sqrt()
            A.append((U[:, :r] * s).detach().clone().to(vm.dev).requires_grad_(True))   # svds live on CPU
            B.append((s[:, None] * Vh[:r]).detach().clone().to(vm.dev).requires_grad_(True))
            # nonlinear bottleneck: code → code + α·GELU(code); α a learnable per-code-channel gate init 0 (strict
            # superset of the linear factor — at α=0 it IS the data-aware linear map, so the init is the linear baseline)
            alpha.append(t.zeros(r, device=vm.dev, requires_grad=True) if args.nonlinear else None)
        factor_on = [True]; hs = []                                   # reuse the arch-generic `mods` from run_model

        def mk(j):
            def hook(m, i, o):
                if not factor_on[0]:                                  # off → teacher (self-distill case)
                    return None
                code = i[0] @ A[j].to(i[0].dtype)                     # (.., r) bottleneck code
                if args.nonlinear:
                    code = code + alpha[j].to(code.dtype) * t.nn.functional.gelu(code)
                y = code @ B[j].to(i[0].dtype)                        # low-rank product, no d×4d materialization
                return y if m.bias is None else y + m.bias            # Llama/Qwen Linear carries no bias
            return hook
        for j, mod in enumerate(mods):
            hs.append(mod.register_forward_hook(mk(j)))
        params = A + B + ([a for a in alpha if a is not None])        # alpha trained alongside the factors
        if args.teacher:                                              # gradient checkpointing — fits fp32-large training on a small GPU
            vm.model.config.use_cache = False
            vm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        opt = torch.optim.Adam(params, lr=args.lr); rng = np.random.default_rng(0); T = 2.0
        for s in range(steps):
            j = int(rng.integers(0, len(train))); tid = t.tensor([train[j]], device=vm.dev)
            if args.match_teacher and teacher_soft is not None:       # top-k KL to the precomputed SEPARATE-teacher targets
                slp = t.log_softmax(vm.model(input_ids=tid).logits[0][:-1].float() / TT, -1)
                idx, t_lp = teacher_soft[j]; idx = idx.to(vm.dev); t_lp = t_lp.to(vm.dev)   # precomputed on CPU
                s_lp = t.log_softmax(slp.gather(-1, idx), -1)         # student renormalised over the teacher's top-k
                loss = (t_lp.exp() * (t_lp - s_lp)).sum(-1).mean() * (TT * TT)
            elif args.match_teacher:                                  # self-teacher: student WITHOUT its factors
                with t.no_grad():
                    factor_on[0] = False
                    teach = t.log_softmax(vm.model(input_ids=tid).logits[0][:-1].float() / T, -1)
                    factor_on[0] = True
                student = t.log_softmax(vm.model(input_ids=tid).logits[0][:-1].float() / T, -1)
                loss = t.nn.functional.kl_div(student, teach, log_target=True, reduction="batchmean") * (T * T)
            else:                                                     # corpus NLL (capable, not faithful)
                logits = vm.model(input_ids=tid).logits[0]
                loss = t.nn.functional.cross_entropy(logits[:-1].float(), tid[0, 1:])
            opt.zero_grad(); loss.backward(); t.nn.utils.clip_grad_norm_(params, 1.0); opt.step()   # clip → stable
        factor_on[0] = True
        if args.teacher:
            vm.model.gradient_checkpointing_disable(); vm.model.config.use_cache = True
        nll, preds, agree, per_dom = gen(metric_top1=full_top1)
        agree_teacher = (float(np.mean([float((p == teacher_top1[i].to(p.device)).float().mean())
                                        for i, p in enumerate(preds)])) if teacher_top1 is not None else None)
        for h in hs:
            h.remove()
        return nll, agree, per_dom, agree_teacher

    ranks = sorted({int(x) for x in args.ranks.split(",")})
    curve = []
    for r in ranks:
        set_rank(r); nll0, _, agree0, _ = gen(metric_top1=full_top1); restore()   # no-retrain SVD baseline
        try:
            nll_d, agree_d, per_dom, agree_tch = (fit_distill(r, args.distill_steps) if args.distill_steps > 0 else (None, None, None, None))
        except torch.cuda.OutOfMemoryError:                            # one rank OOM (big factors) → drop that rank, keep the curve
            nll_d, agree_d, per_dom, agree_tch = (None, None, None, None)
            print(f"    rank {r:4d}: distill OOM — skipped (SVD baseline kept)"); torch.cuda.empty_cache()
        stored = int(sum(r * (W.shape[0] + W.shape[1]) for W in mats))
        curve.append({"rank": r, "svd_nll_increase": nll0 - full_nll, "svd_agreement": agree0,
                      "distilled_nll_increase": (nll_d - full_nll) if nll_d is not None else None,
                      "distilled_agreement": agree_d, "distilled_agreement_teacher": agree_tch,
                      "distilled_per_domain": per_dom, "stored_params": stored,
                      "compression_ratio": stored / full_params, "params_saved_frac": 1 - stored / full_params})
    tag = args.tag or (f"teacher={args.teacher.split('/')[-1]}" if args.teacher
                       else ("self" if args.match_teacher else "nll"))   # distinguish self/cross-teacher runs in the json
    if args.data_aware:
        tag += "+daware"
    if args.nonlinear:
        tag += "+nl"
    return {"model": mid.split("/")[-1] + "@" + tag, "n_layers": nL, "full_nll": full_nll, "teacher": args.teacher,
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
    p.add_argument("--match-teacher", action="store_true", help="distill to a teacher (faithful) vs corpus NLL (capable)")
    p.add_argument("--teacher", default="", help="a SEPARATE teacher model (same tokenizer), e.g. gpt2-xl → gpt2-large core")
    p.add_argument("--tag", default="", help="suffix for the summary-json model key (auto: self/teacher=<id>/nll)")
    p.add_argument("--data-aware", action="store_true", help="functional (activation-weighted) low-rank vs Frobenius SVD")
    p.add_argument("--nonlinear", action="store_true", help="α-gated GELU bottleneck (code+α·gelu(code)); superset of linear, α init 0")
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
                tch = (f"  | agree-w-teacher {c['distilled_agreement_teacher']:.0%}"
                       if c.get("distilled_agreement_teacher") is not None else "")
                print(f"    rank {c['rank']:4d}  stored {c['compression_ratio']:.0%} ({c['params_saved_frac']:.0%} saved)  "
                      f"| SVD ΔNLL {c['svd_nll_increase']:+.2f} agree {c['svd_agreement']:.0%}  | {dd}{tch}")
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
