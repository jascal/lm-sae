"""Mamba (SSM) across the catalog's themes — the non-attention CONTRAST (induction / knowledge READ + WRITE).

Mamba has no attention heads and no per-layer MLP — just a residual stream of SSM `mixer` blocks — so the
attention-based catalog (heads, K/Q/V composition, name-movers) doesn't apply, and Mamba has only ever been a
behavioural footnote ("induction present"). But the arch-generic themes DO apply to a state-space model, via a
Mamba-specific harness (the ResidualVM is transformer-only). For the Mamba ladder (130m/370m/790m) we run the same
themes as the transformers, as a contrast:

  INDUCTION (distribution)  — induction-NLL on repeated-random; mean-ablate each SSM LAYER -> Δinduction-NLL (the
                              layer is the unit, there are no heads): is in-context copy concentrated in a few layers
                              or spread? (the SSM analog of head-distribution); + all-layer-ablated necessity.
  KNOWLEDGE READ            — dump the capital/language table (accuracy) + logit-lens read-out depth (where the
                              relation resolves), directly comparable to relation_decompile.
  KNOWLEDGE WRITE           — graft a donor subject's early-layer residual into S's run (band patch): does the fact
                              transplant? does the LANGUAGE leak too (entity-vs-fact), as in fact_edit?

So the SSM is placed on the same axes as the six transformers. Mamba-specific module paths
(backbone.layers[L] = norm+mixer, backbone.norm_f, lm_head); residual ablation = replace a layer's contribution
(output−input) with its corpus mean.

Output: runs/disassembly/mamba_themes_summary.json (merge-safe). Findings -> docs/FINDINGS.md.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fact_edit_xmodel import FACTS, LANG, single_tok, subj_pos  # noqa: E402


class MambaProbe:
    def __init__(self, model_id, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()
        self.dev = device if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.dev)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.layers = self.model.backbone.layers
        self.norm_f = self.model.backbone.norm_f
        self.WU = self.model.lm_head.weight.detach()
        self.nL = len(self.layers)
        self.mean_contrib = None

    def logits(self, ids):
        t = self.torch
        with t.no_grad():
            return self.model(input_ids=t.tensor([ids], device=self.dev)).logits[0]

    def trace_resid(self, ids):
        """Residual entering each layer (pre-hook input) for one sequence: dict L -> (1,seq,d)."""
        t = self.torch; cap = {}
        hs = [self.layers[L].register_forward_pre_hook((lambda L: lambda m, i: cap.__setitem__(L, i[0].detach()))(L)) for L in range(self.nL)]
        with t.no_grad():
            self.model(input_ids=t.tensor([ids], device=self.dev))
        for h in hs:
            h.remove()
        return cap

    def fit_means(self, chunks):
        """Per-layer mean residual CONTRIBUTION (output−input), the mean-ablation reference."""
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
    def ablate_layers(self, layers):
        """Replace each layer's residual contribution (output−input) with its corpus mean."""
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

    @contextlib.contextmanager
    def patch_layers(self, layers, donor_out, pos):
        """Graft donor's layer output at `pos` (the residual store edit)."""
        def mk(L):
            def hook(m, i, o):
                out = (o[0] if isinstance(o, tuple) else o).clone()
                out[0, pos] = donor_out[L][0, pos].to(out.dtype)
                return (out,) + tuple(o[1:]) if isinstance(o, tuple) else out
            return hook
        hs = [self.layers[L].register_forward_hook(mk(L)) for L in layers]
        try:
            yield self
        finally:
            for h in hs:
                h.remove()

    def trace_out(self, ids):
        """Each layer's OUTPUT residual for one sequence (for donor capture)."""
        t = self.torch; cap = {}
        hs = [self.layers[L].register_forward_hook((lambda L: lambda m, i, o: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))(L)) for L in range(self.nL)]
        with t.no_grad():
            self.model(input_ids=t.tensor([ids], device=self.dev))
        for h in hs:
            h.remove()
        return cap


def hill(x):
    x = np.clip(x, 0, None); s = x.sum()
    return float((s * s) / (np.square(x).sum() + 1e-12)) if s > 0 else 0.0


def run_model(mid, args):
    import urllib.request
    vm = MambaProbe(mid, device=args.device); t = vm.torch; tok = vm.tok; nL = vm.nL
    rng = np.random.default_rng(args.seed)
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    pids = tok(prose)["input_ids"]
    chunks = [pids[i:i + 64] for i in range(0, len(pids), 64) if len(pids[i:i + 64]) >= 8][: args.chunks]
    vm.fit_means(chunks)
    cnt = {}
    for x in pids:
        cnt[x] = cnt.get(x, 0) + 1
    vocab = [x for x, _ in sorted(cnt.items(), key=lambda r: -r[1])[:400]]
    rep_seqs = [(lambda s: s + s)([int(vocab[i]) for i in rng.integers(0, len(vocab), 22)]) for _ in range(args.probes)]

    def ind_nll(ablate=()):
        with (vm.ablate_layers(ablate) if ablate else contextlib.nullcontext()):
            tot = 0.0; k = 0
            with t.no_grad():
                for s in rep_seqs:
                    lp = t.log_softmax(vm.logits(s).float(), -1); Lh = len(s) // 2
                    for p in range(Lh, 2 * Lh - 1):
                        tot += float(-lp[p, s[p + 1]]); k += 1
            return tot / max(k, 1)

    base_ind = ind_nll()
    all_abl = ind_nll(list(range(nL)))
    per_layer = np.array([ind_nll([L]) - base_ind for L in range(nL)])               # Δind per ablated layer
    pos = np.clip(per_layer, 0, None)
    induction = {"base_nll": base_ind, "all_ablated_delta": all_abl - base_ind,
                 "top_layers": [{"layer": int(L), "delta": float(per_layer[L])} for L in np.argsort(-per_layer)[:5]],
                 "effective_n_layers": hill(pos), "top_layer_share": float(pos.max() / (pos.sum() + 1e-12)) if pos.sum() > 0 else 0.0}

    # ---- KNOWLEDGE READ: relation table + logit-lens read-out depth ----
    def read_relation(template, objmap):
        facts = [(c, single_tok(tok, c, False), single_tok(tok, o, False)) for c, o in objmap.items()]
        facts = [(c, s, o) for c, s, o in facts if s is not None and o is not None]
        if len(facts) < 6:
            return None
        obj_ids = sorted({o for _, _, o in facts}); correct = 0; depths = []
        for (c, sid, oid) in facts:
            ids = tok(template.format(S=c))["input_ids"]
            res = vm.trace_resid(ids); lg = vm.logits(ids)[-1]
            pick = obj_ids[int(t.tensor([float(lg[o]) for o in obj_ids]).argmax())]; correct += int(pick == oid)
            d = 1.0
            with t.no_grad():
                for L in range(1, nL):
                    h = vm.norm_f(res[L][0, -1]); lo = vm.WU.float() @ h.float()
                    if obj_ids[int(t.tensor([float(lo[o]) for o in obj_ids]).argmax())] == oid:
                        d = L / (nL - 1); break
            depths.append(d)
        return {"n": len(facts), "accuracy": correct / len(facts), "chance": 1.0 / len(obj_ids),
                "mean_readout_depth": float(np.mean(depths))}
    read = {"capital": read_relation("The capital of {S} is", {c: cap for c, cap in FACTS}),
            "language": read_relation("The language of {S} is", dict(LANG))}

    # ---- KNOWLEDGE WRITE: residual-store band patch (fact transplant + entity-leakage) ----
    band = list(range(max(2, nL // 5)))
    facts = [(c, single_tok(tok, c, False), single_tok(tok, cap, False), single_tok(tok, LANG.get(c, ""), False))
             for c, cap in FACTS]
    facts = [(c, s, o, lg) for c, s, o, lg in facts if s is not None and o is not None]
    eff = 0; leak = 0; npw = 0; leak_n = 0
    for i, (Sc, Sid, Scap, Slang) in enumerate(facts):
        D = facts[(i + 1) % len(facts)]; Dc, Did, Dcap, Dlang = D
        if Dcap == Scap:
            continue
        ids_S = tok(f"The capital of {Sc} is")["input_ids"]; ids_D = tok(f"The capital of {Dc} is")["input_ids"]
        pS = subj_pos(ids_S, Sid); pD = subj_pos(ids_D, Did)
        if pS is None or pD is None or pS != pD:
            continue
        npw += 1; D_out = vm.trace_out(ids_D)
        with vm.patch_layers(band, D_out, pS):
            lc = t.log_softmax(vm.logits(ids_S).float(), -1)[-1]
        if float(lc[Dcap] - lc[Scap]) > 0:
            eff += 1
        if Slang is not None and Dlang is not None and Slang != Dlang:
            idsL = tok(f"The language of {Sc} is")["input_ids"]; pLS = subj_pos(idsL, Sid)
            if pLS is not None:
                D_outL = vm.trace_out(tok(f"The language of {Dc} is")["input_ids"])  # donor language-context store
                with vm.patch_layers(band, D_outL, pLS):
                    ll = t.log_softmax(vm.logits(idsL).float(), -1)[-1]
                leak_n += 1
                if float(ll[Dlang] - ll[Slang]) > 0:
                    leak += 1
    write = {"n_pairs": npw, "transplant_rate": eff / npw if npw else None,
             "entity_leakage_rate": (leak / leak_n) if leak_n else None, "band_layers": len(band)}
    return {"model": mid.split("/")[-1], "n_layers": nL, "induction": induction, "read": read, "write": write}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="state-spaces/mamba-130m-hf,state-spaces/mamba-370m-hf,state-spaces/mamba-790m-hf")
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
            ind = r["induction"]; rd = r["read"]; w = r["write"]
            print(f"  induction: base-NLL {ind['base_nll']:.2f}, all-ablated Δ {ind['all_ablated_delta']:+.2f}, "
                  f"eff-N layers {ind['effective_n_layers']:.1f}, top L{ind['top_layers'][0]['layer']} "
                  f"({ind['top_layer_share']:.0%})")
            for rel in ("capital", "language"):
                if rd.get(rel):
                    print(f"  read {rel}: acc {rd[rel]['accuracy']:.0%} (chance {rd[rel]['chance']:.0%}), "
                          f"read-out depth {rd[rel]['mean_readout_depth']:.0%}")
            lk = f"{w['entity_leakage_rate']:.0%}" if w["entity_leakage_rate"] is not None else "n/a"
            tr = f"{w['transplant_rate']:.0%}" if w["transplant_rate"] is not None else "n/a"
            print(f"  write: transplant {tr}, entity-leakage {lk} ({w['n_pairs']} pairs)")
        except Exception as e:  # pragma: no cover
            import traceback; traceback.print_exc()
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "mamba_themes_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "Mamba (SSM) across the themes — induction-distribution / knowledge READ + WRITE, the non-attention contrast",
           "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'induction' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
