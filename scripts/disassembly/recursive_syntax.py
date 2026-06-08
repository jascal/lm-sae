"""Where does RECURSIVE/HIERARCHICAL syntax live? — subject–verb agreement across attractor depth: model vs the
flat decompilation vs the composition.

The core-structure work (#133) found the static write-basis holds only a *categorial* grammar — POS-class directions
(determiner-slot, verb-slot), a distributional scaffold. The open hypothesis: the genuinely Chomskyan part — hierarchy,
long-range dependency, recursion — is NOT in that static basis but in the **composition** (how categories combine
across positions/layers), i.e. the entangled bulk that pays the forge tax. This probes it with the cleanest
hierarchical dependency there is: **subject–verb number agreement across intervening attractor nouns** (Linzen et al.).

  "The key  near the cabinets  is/are"   — the verb agrees with the HEAD ("key", singular), not the nearest noun
                                            ("cabinets", plural). A flat local program follows the nearest noun (wrong);
                                            true hierarchical syntax tracks the head across depth.

Three measurements, all read-only:
  MODEL — logit-diff lp(correct-number verb) − lp(wrong-number verb) vs attractor DEPTH (0,1,2,3). Does it agree
          hierarchically (resist the attractor), and how does it degrade with depth (bounded forward-pass depth)?
  FLAT (pylm) — does the decompiled program (n-gram + induction + categorial grammar) pick the head's number or the
          nearest attractor's? If it follows the attractor, the hierarchical dependency is NOT in the flat decompilation.
  COMPOSITION — ablate all attention (mean) vs all MLP (mean): if agreement collapses toward the attractor under
          attention-ablation but "predict a verb" survives, the dependency lives in the attention composition — the
          entangled core, not the static grammar head.

Output: runs/disassembly/recursive_syntax_summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

NOUNS = [("key", "keys"), ("dog", "dogs"), ("book", "books"), ("car", "cars"), ("door", "doors"),
         ("girl", "girls"), ("boy", "boys"), ("author", "authors"), ("painting", "paintings"),
         ("computer", "computers"), ("student", "students"), ("teacher", "teachers"),
         ("window", "windows"), ("table", "tables"), ("chair", "chairs"), ("friend", "friends")]
PREPS = ["near", "behind", "beside", "by"]
VERBS = [(" is", " are"), (" was", " were"), (" has", " have")]   # (singular, plural)


def single(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"]) == 1


def build_stimuli(tok, depths, rng):
    """For each depth, sentences 'The {head} [{prep} the {attractor}]*d' with attractors of OPPOSITE number to head."""
    nouns = [(s, p) for (s, p) in NOUNS if single(tok, " " + s) and single(tok, " " + p)]
    verbs = [(a, b) for (a, b) in VERBS if single(tok, a) and single(tok, b)]
    stim = {d: [] for d in depths}
    for d in depths:
        for (sg, pl) in nouns:
            for hnum in (0, 1):                                          # 0 = singular head, 1 = plural head
                head = sg if hnum == 0 else pl
                opp = [(p if hnum == 0 else s) for (s, p) in nouns if (s, p) != (sg, pl)]
                atts = [opp[int(i)] for i in rng.integers(0, len(opp), d)]
                phrase = "The " + head + "".join(f" {PREPS[int(rng.integers(0, len(PREPS)))]} the {a}" for a in atts)
                for (vs, vp) in verbs:
                    correct = vs if hnum == 0 else vp                   # agrees with the HEAD
                    wrong = vp if hnum == 0 else vs                     # agrees with the nearest ATTRACTOR (if d>0)
                    stim[d].append({"text": phrase, "correct": correct, "wrong": wrong, "hnum": hnum})
    return stim


def run_model(mid, args):
    from residual_vm import ResidualVM
    vm = ResidualVM(mid, device=args.device); t = vm.torch; tok = vm.tok
    rng = np.random.default_rng(0)
    depths = list(range(args.max_depth + 1))
    stim = build_stimuli(tok, depths, rng)

    # corpus means for mean-ablation (the composition test)
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + 64] for i in range(0, len(ids), 64) if len(ids[i:i + 64]) >= 8][:24]
    vm.fit_means(chunks)
    all_heads = [(L, h) for L in range(vm.nL) for h in range(vm.H)]

    def score(sents, ctxmgr=None):
        """mean signed logit-diff (correct − wrong) + accuracy (sign correct) under an optional intervention."""
        diffs = []
        for s in sents:
            ctx = tok(s["text"], add_special_tokens=False)["input_ids"]
            ci = tok(s["correct"], add_special_tokens=False)["input_ids"][0]
            wi = tok(s["wrong"], add_special_tokens=False)["input_ids"][0]
            if ctxmgr is None:
                lp = t.log_softmax(vm.logits(ctx).float(), -1)[-1]
            else:
                with ctxmgr():
                    lp = t.log_softmax(vm.logits(ctx).float(), -1)[-1]
            diffs.append(float(lp[ci] - lp[wi]))
        diffs = np.array(diffs)
        return {"mean_logit_diff": float(diffs.mean()), "accuracy": float((diffs > 0).mean()), "n": len(diffs)}

    conditions = {
        "full": None,
        "attn_ablated": (lambda: vm.ablate_heads(all_heads, mode="mean")),
        "mlp_ablated": (lambda: vm.ablate_mlps(list(range(vm.nL)), mode="mean")),
    }
    by_depth = {}
    for d in depths:
        by_depth[d] = {cond: score(stim[d], cm) for cond, cm in conditions.items()}

    # ---- the flat decompilation (pylm): does it follow the head or the nearest attractor? ----
    pylm_rows = {}
    if args.pylm_store and Path(args.pylm_store).exists():
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".." / "pylm"))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pylm"))
        from lm import PyLM
        lm = PyLM(args.pylm_store)
        vset = {}
        for (vs, vp) in VERBS:
            for form, num in ((vs, 0), (vp, 1)):
                ids_ = tok(form, add_special_tokens=False)["input_ids"]
                if len(ids_) == 1:
                    vset[ids_[0]] = num                                  # verb token → number
        for d in depths:
            head_match = att_match = found = 0
            for s in stim[d]:
                ctx = tok(s["text"], add_special_tokens=False)["input_ids"]
                ranked = lm.ranked(ctx, k=60)
                vnum = next((vset[c] for c in ranked if c in vset), None)  # pylm's first verb-token's number
                if vnum is None:
                    continue
                found += 1
                head_match += int(vnum == s["hnum"])                     # agrees with the HEAD (correct)
                att_match += int(vnum != s["hnum"])                      # agrees with the nearest ATTRACTOR (d>0 ⇒ wrong)
            pylm_rows[d] = {"head_agreement": head_match / max(found, 1),
                            "attractor_agreement": att_match / max(found, 1), "found": found}

    return {"model": mid.split("/")[-1], "n_layers": vm.nL, "depths": depths,
            "by_depth": {str(d): by_depth[d] for d in depths},
            "pylm": {str(d): pylm_rows[d] for d in pylm_rows} if pylm_rows else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2")
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--pylm-store", default="pylm/store_grammar.json")
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
            print("  MODEL agreement accuracy (logit-diff) by attractor depth:")
            for d in r["depths"]:
                bd = r["by_depth"][str(d)]
                print(f"    depth {d}: full {bd['full']['accuracy']:.0%} (Δ{bd['full']['mean_logit_diff']:+.2f}) · "
                      f"attn-ablated {bd['attn_ablated']['accuracy']:.0%} (Δ{bd['attn_ablated']['mean_logit_diff']:+.2f}) · "
                      f"mlp-ablated {bd['mlp_ablated']['accuracy']:.0%} (Δ{bd['mlp_ablated']['mean_logit_diff']:+.2f})")
            if r["pylm"]:
                print("  FLAT pylm — follows the HEAD (correct) vs the nearest ATTRACTOR:")
                for d in r["depths"]:
                    pr = r["pylm"].get(str(d))
                    if pr:
                        print(f"    depth {d}: head {pr['head_agreement']:.0%} · attractor {pr['attractor_agreement']:.0%} "
                              f"(n={pr['found']})")
        except Exception as e:  # pragma: no cover
            import traceback
            traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "recursive_syntax_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "recursive syntax — subject-verb agreement across attractor depth: model vs flat pylm vs composition",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'by_depth' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
