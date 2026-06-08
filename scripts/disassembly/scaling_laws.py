"""Scaling laws on a CONTROLLED ladder (Pythia) — turn "tracks scale" into laws with architecture held fixed.

Across the program, many findings "track scale" — but they were read off the GPT-2 ladder (3-4 points) plus a
heterogeneous mix of RoPE models where architecture and scale are confounded. The Pythia ladder (EleutherAI/pythia,
14m→1.4b: ONE GPT-NeoX architecture, SAME training data, 6 sizes) is the clean control. This runs the arch-generic
themes (no head resolution needed — block-level + logit-lens, so it works on any decoder-layer LM) across the ladder:

  INDUCTION   — induction-NLL on repeated-random (strength), + mean-ablate each LAYER BLOCK -> Δinduction-NLL
                (distribution: effective-N blocks, top-block share, all-block necessity);
  KNOWLEDGE EMERGENCE — the relation table's completeness (capital/language accuracy) as a function of size: at what
                scale does the database fill? (the tiny models won't know it);
  READ-OUT DEPTH — logit-lens depth at which the relation resolves, among the models that know it.

So the recurring scale observations become curves on a controlled axis. Arch-generic harness (duck-types layers /
final-norm / unembedding) — GPT-NeoX here; re-runnable on the other families.

Output: runs/disassembly/scaling_laws_summary.json (merge-safe). Findings -> docs/scaling.md (the central-thesis page).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fact_edit_xmodel import FACTS, LANG, single_tok  # noqa: E402


def find_stack(model):
    """(decoder layers, final norm) for any HF causal LM, by duck-typing the known families."""
    for rn in ("gpt_neox", "model", "transformer", "backbone"):
        root = getattr(model, rn, None)
        if root is None:
            continue
        for la in ("layers", "h"):
            layers = getattr(root, la, None)
            if layers is None:
                continue
            for nm in ("final_layer_norm", "norm", "ln_f", "norm_f"):
                if hasattr(root, nm):
                    return layers, getattr(root, nm)
    raise SystemExit("unknown architecture")


class GenericLM:
    def __init__(self, model_id, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        big = any(s in model_id for s in ("1b", "1.4b", "2.8b"))
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **({"dtype": torch.bfloat16} if big else {})).eval()
        self.dev = device if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.dev)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.layers, self.norm_f = find_stack(self.model)
        self.WU = self.model.get_output_embeddings().weight.detach()
        self.nL = len(self.layers)
        self.bos = self.tok.bos_token_id is not None
        self.mean_contrib = None

    def logits(self, ids):
        t = self.torch
        with t.no_grad():
            return self.model(input_ids=t.tensor([ids], device=self.dev)).logits[0]

    def trace_resid(self, ids):
        t = self.torch; cap = {}
        hs = [self.layers[L].register_forward_pre_hook((lambda L: lambda m, i: cap.__setitem__(L, i[0].detach()))(L)) for L in range(self.nL)]
        with t.no_grad():
            self.model(input_ids=t.tensor([ids], device=self.dev))
        for h in hs:
            h.remove()
        return cap

    def fit_block_means(self, chunks):
        t = self.torch; acc = {L: [] for L in range(self.nL)}

        def mk(L):
            def hook(m, i, o):
                out = o[0] if isinstance(o, tuple) else o
                acc[L].append((out - i[0]).detach().reshape(-1, out.shape[-1]))
            return hook
        hs = [self.layers[L].register_forward_hook(mk(L)) for L in range(self.nL)]
        with t.no_grad():
            for c in chunks:
                self.model(input_ids=t.tensor([c], device=self.dev))
        for h in hs:
            h.remove()
        self.mean_contrib = {L: t.cat(acc[L], 0).mean(0) for L in range(self.nL)}

    @contextlib.contextmanager
    def ablate_blocks(self, layers):
        def mk(L):
            def hook(m, i, o):
                out = (o[0] if isinstance(o, tuple) else o).clone()
                out[:] = i[0] + self.mean_contrib[L].to(out.dtype)
                return (out,) + tuple(o[1:]) if isinstance(o, tuple) else out
            return hook
        hs = [self.layers[L].register_forward_hook(mk(L)) for L in layers]
        try:
            yield self
        finally:
            for h in hs:
                h.remove()


def hill(x):
    x = np.clip(x, 0, None); s = x.sum()
    return float((s * s) / (np.square(x).sum() + 1e-12)) if s > 0 else 0.0


def run_model(mid, args):
    import urllib.request
    vm = GenericLM(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL
    rng = np.random.default_rng(args.seed)
    add_special = vm.bos
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + 64] for i in range(0, len(pids), 64) if len(pids[i:i + 64]) >= 8][: args.chunks]
    vm.fit_block_means(chunks)
    cnt = {}
    for x in pids:
        cnt[x] = cnt.get(x, 0) + 1
    vocab = [x for x, _ in sorted(cnt.items(), key=lambda r: -r[1])[:400]]
    rep_seqs = [(lambda s: s + s)([int(vocab[i]) for i in rng.integers(0, len(vocab), 22)]) for _ in range(args.probes)]

    def ind_nll(ablate=()):
        with (vm.ablate_blocks(ablate) if ablate else contextlib.nullcontext()):
            tot = 0.0; k = 0
            with t.no_grad():
                for s in rep_seqs:
                    lp = t.log_softmax(vm.logits(s).float(), -1); Lh = len(s) // 2
                    for p in range(Lh, 2 * Lh - 1):
                        tot += float(-lp[p, s[p + 1]]); k += 1
            return tot / max(k, 1)

    base = ind_nll(); allabl = ind_nll(list(range(nL)))
    per = np.array([ind_nll([L]) - base for L in range(nL)]); pos = np.clip(per, 0, None)
    induction = {"base_nll": base, "all_ablated_delta": allabl - base, "effective_n_blocks": hill(pos),
                 "top_block_share": float(pos.max() / (pos.sum() + 1e-12)) if pos.sum() > 0 else 0.0,
                 "top_block": int(np.argmax(per)), "top_block_frac": float(np.argmax(per) / max(nL - 1, 1))}

    def read_relation(template, objmap):
        facts = [(c, single_tok(tok, c, not add_special), single_tok(tok, o, not add_special)) for c, o in objmap.items()]
        facts = [(c, s, o) for c, s, o in facts if s is not None and o is not None]
        if len(facts) < 6:
            return None
        obj_ids = sorted({o for _, _, o in facts}); correct = 0; depths = []
        for (c, sid, oid) in facts:
            ids = tok(template.format(S=c), add_special_tokens=add_special)["input_ids"]
            res = vm.trace_resid(ids); lg = vm.logits(ids)[-1]
            if obj_ids[int(t.tensor([float(lg[o]) for o in obj_ids]).argmax())] == oid:
                correct += 1
                d = 1.0
                with t.no_grad():
                    for L in range(1, nL):
                        h = vm.norm_f(res[L][0, -1]); lo = vm.WU.float() @ h.float()
                        if obj_ids[int(t.tensor([float(lo[o]) for o in obj_ids]).argmax())] == oid:
                            d = L / (nL - 1); break
                depths.append(d)
        return {"n": len(facts), "accuracy": correct / len(facts), "chance": 1.0 / len(obj_ids),
                "mean_readout_depth": float(np.mean(depths)) if depths else None}

    read = {"capital": read_relation("The capital of {S} is", {c: cap for c, cap in FACTS}),
            "language": read_relation("The language of {S} is", dict(LANG))}
    return {"model": mid.split("/")[-1], "n_layers": nL, "d_model": vm.model.config.hidden_size,
            "induction": induction, "read": read}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-14m,EleutherAI/pythia-70m,EleutherAI/pythia-160m,EleutherAI/pythia-410m,EleutherAI/pythia-1b,EleutherAI/pythia-1.4b")
    p.add_argument("--chunks", type=int, default=12)
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
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
            ind = r["induction"]; rd = r["read"]
            print(f"  d{r['d_model']} {r['n_layers']}L | induction base-NLL {ind['base_nll']:.2f}, all-abl Δ "
                  f"{ind['all_ablated_delta']:+.2f}, eff-N blocks {ind['effective_n_blocks']:.1f} (top {ind['top_block_share']:.0%} @ {ind['top_block_frac']:.0%})")
            for rel in ("capital", "language"):
                if rd.get(rel):
                    dd = f"{rd[rel]['mean_readout_depth']:.0%}" if rd[rel]["mean_readout_depth"] is not None else "n/a"
                    print(f"  {rel}: table acc {rd[rel]['accuracy']:.0%} (chance {rd[rel]['chance']:.0%}), read-out depth {dd}")
        except Exception as e:  # pragma: no cover
            import traceback; traceback.print_exc(); print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "scaling_laws_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "controlled scaling laws on the Pythia ladder (induction strength/distribution, knowledge emergence, read-out depth)",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'induction' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
