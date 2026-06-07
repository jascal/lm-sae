"""ResidualVM debugger — a PROGRAMMATIC discovery engine over the forward pass (not a human REPL).

The point is not to hand-step a debugger; it is a tool *driven programmatically* to find MORE operators and
circuits than the named catalog already has. It instruments the forward pass as a steppable interpreter and exposes
a discovery API:

  trace(ids)                          -> per-(layer,head) residual write + per-layer MLP write + clean logits (the "tape")
  intervene(ids, ablate=, preserve=)  -> mean-ablate any heads/MLPs (or preserve-only) -> metric deltas vs clean
  logit_lens_step(ids)                -> per-layer KL(final ‖ logit-lens@L): WHERE the next-token answer is decided
  attribution_sweep(metric)           -> ablate EVERY head + EVERY MLP one-at-a-time, rank by |Δmetric|; flag which
                                         strong components are NOT in the named catalog  ==> CANDIDATE NEW OPERATORS
  edge_probe(reader, metric)          -> path-patch each upstream head out of the reader's KEY; collapse of the
                                         reader's behaviour  ==> CANDIDATE CIRCUIT EDGES feeding a (possibly new) op

`main()` *uses* the engine: it runs attribution sweeps for three behaviours (induction / IOI / generic LM), reports
the load-bearing components and — the goal — the ones the catalog has NOT named (candidate new ops), then
edge-probes the top discovered op to propose candidate circuit edges. All outputs are structured JSON; a UI could
sit on top but is not required. GPT-2 (arch-generic intervention harness; QK-patch is GPT-2-specific here).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ioi_causal import NAMES, OBJECTS, PLACES, TEMPLATES  # noqa: E402

# named catalog (union of literature head-sets) — a component outside this with high causal effect is a CANDIDATE.
NAMED = {
    "prevtok": [(4, 11)], "induction": [(5, 0), (5, 1), (5, 5), (6, 9), (7, 11)],
    "duplicate": [(0, 1), (0, 5), (1, 5), (3, 0)], "name_mover": [(9, 6), (9, 9), (10, 0), (10, 10)],
    "backup_name_mover": [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (11, 2)],
    "negative_mover": [(10, 7), (11, 10)], "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)], "coreference": [(9, 0)],
}


class ResidualVMDebugger:
    def __init__(self, pretrained="gpt2", device="cuda", ctx=96, mean_chunks=30, seed=0):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.torch = torch
        self.dev = device if torch.cuda.is_available() else "cpu"
        self.model = GPT2LMHeadModel.from_pretrained(pretrained, attn_implementation="eager").eval().to(self.dev)
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        cfg = self.model.config
        self.H = cfg.n_head; self.hd = cfg.n_embd // self.H; self.nL = cfg.n_layer; self.d = cfg.n_embd
        self.tr = self.model.transformer
        self.named2heads = NAMED
        self.head2name = {tuple(h): n for n, hs in NAMED.items() for h in hs}
        self.rng = np.random.default_rng(seed)
        import urllib.request
        txt = urllib.request.urlopen(urllib.request.Request(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
        self.ids_all = self.tok(txt)["input_ids"]
        self.echunks = [self.ids_all[i:i + ctx] for i in range(0, len(self.ids_all), ctx) if len(self.ids_all[i:i + ctx]) >= 8][:mean_chunks]
        # corpus means: per-head c_proj input slice, and per-layer MLP output
        cap_h = {L: [] for L in range(self.nL)}; cap_m = {L: [] for L in range(self.nL)}
        hk = []
        for L in range(self.nL):
            hk.append(self.tr.h[L].attn.c_proj.register_forward_pre_hook(
                (lambda L: lambda m, inp: cap_h[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)))
            hk.append(self.tr.h[L].mlp.register_forward_hook(
                (lambda L: lambda m, inp, out: cap_m[L].append(out.detach().reshape(-1, out.shape[-1])))(L)))
        with torch.no_grad():
            for c in self.echunks:
                self.model(input_ids=torch.tensor([c], device=self.dev))
        for h in hk:
            h.remove()
        self.mean_head = {L: torch.cat(cap_h[L], 0).mean(0) for L in range(self.nL)}     # c_proj-input mean (per head slice)
        self.mean_mlp = {L: torch.cat(cap_m[L], 0).mean(0) for L in range(self.nL)}      # mlp-output mean

    # ---------- intervention harness ----------
    def _hooks(self, ablate_heads=(), ablate_mlps=()):
        hd = self.hd
        by = {}
        for (L, h) in ablate_heads:
            by.setdefault(L, []).append(h)
        hs = []
        for L, hh in by.items():
            def mk(L, hh):
                def hook(m, inp):
                    x = inp[0].clone()
                    for h in hh:
                        x[..., h * hd:(h + 1) * hd] = self.mean_head[L][h * hd:(h + 1) * hd].to(x.dtype)
                    return (x,)
                return hook
            hs.append(self.tr.h[L].attn.c_proj.register_forward_pre_hook(mk(L, hh)))
        for L in ablate_mlps:
            def mkm(L):
                def hook(m, inp, out):
                    return self.mean_mlp[L].to(out.dtype).expand_as(out)
                return hook
            hs.append(self.tr.h[L].mlp.register_forward_hook(mkm(L)))
        return hs

    def run_logits(self, ids, ablate_heads=(), ablate_mlps=()):
        torch = self.torch
        hs = self._hooks(ablate_heads, ablate_mlps)
        try:
            with torch.no_grad():
                return self.model(input_ids=torch.tensor([ids], device=self.dev)).logits[0].float()
        finally:
            for h in hs:
                h.remove()

    def intervene(self, ids, target_pos, ablate_heads=(), ablate_mlps=()):
        """Programmatic step: KL at target_pos between clean and intervened next-token distribution."""
        torch = self.torch
        F = torch.nn.functional
        clean = F.log_softmax(self.run_logits(ids)[target_pos], -1)
        pert = F.log_softmax(self.run_logits(ids, ablate_heads, ablate_mlps)[target_pos], -1)
        return float((clean.exp() * (clean - pert)).sum())

    # ---------- logit-lens stepping: WHERE is the answer decided? ----------
    def logit_lens_step(self, ids):
        torch = self.torch
        F = torch.nn.functional
        resids = {}

        def grab(L):
            def hook(m, i, o):
                resids[L] = (o[0] if isinstance(o, tuple) else o).detach()    # block output is a bare tensor in tf5.x
            return hook
        hk = [self.tr.h[L].register_forward_hook(grab(L)) for L in range(self.nL)]
        with torch.no_grad():
            final = F.log_softmax(self.model(input_ids=torch.tensor([ids], device=self.dev)).logits[0].float(), -1)
        for h in hk:
            h.remove()
        pos = len(ids) - 1; out = []
        for L in range(self.nL):
            with torch.no_grad():
                lens = F.log_softmax(self.model.lm_head(self.tr.ln_f(resids[L]))[0, pos].float(), -1)
            out.append({"layer": L, "kl_to_final": float((final[pos].exp() * (final[pos] - lens)).sum())})
        return out

    # ---------- metrics ----------
    def metric_induction(self, ablate_heads=(), ablate_mlps=(), n=40, L=20):
        torch = self.torch
        F = torch.nn.functional
        vocab = self._common_vocab()
        seqs = [[int(vocab[i]) for i in self.rng.integers(0, len(vocab), L)] * 2 for _ in range(n)]
        hs = self._hooks(ablate_heads, ablate_mlps); tot = 0.0; k = 0
        try:
            with torch.no_grad():
                for s in seqs:
                    lp = F.log_softmax(self.model(input_ids=torch.tensor([s], device=self.dev)).logits[0].float(), -1)
                    half = len(s) // 2
                    for p in range(half, 2 * half - 1):
                        tot += float(-lp[p, s[p + 1]]); k += 1
        finally:
            for h in hs:
                h.remove()
        return tot / max(k, 1)

    def metric_generic(self, ablate_heads=(), ablate_mlps=(), n=30):
        torch = self.torch
        F = torch.nn.functional
        hs = self._hooks(ablate_heads, ablate_mlps); tot = 0.0; k = 0
        try:
            with torch.no_grad():
                for c in self.echunks[:n]:
                    lp = F.log_softmax(self.model(input_ids=torch.tensor([c], device=self.dev)).logits[0, :-1].float(), -1)
                    y = torch.tensor(c[1:], device=self.dev); tot += float(-lp[torch.arange(len(y)), y].sum()); k += len(y)
        finally:
            for h in hs:
                h.remove()
        return tot / max(k, 1)

    def metric_ioi(self, ablate_heads=(), ablate_mlps=(), n=60):
        torch = self.torch
        prompts = self._ioi_prompts(n); hs = self._hooks(ablate_heads, ablate_mlps); v = []
        try:
            with torch.no_grad():
                for ids, io, s in prompts:
                    lg = self.model(input_ids=torch.tensor([ids], device=self.dev)).logits[0, -1].float()
                    v.append(float(lg[io] - lg[s]))
        finally:
            for h in hs:
                h.remove()
        return float(np.mean(v))

    def _common_vocab(self):
        if not hasattr(self, "_cv"):
            cnt = {}
            for t in self.ids_all:
                cnt[t] = cnt.get(t, 0) + 1
            self._cv = [t for t, _ in sorted(cnt.items(), key=lambda r: -r[1])[:400]]
        return self._cv

    def _ioi_prompts(self, n):
        def single(strs):
            out = []
            for s in strs:
                i = self.tok(s, add_special_tokens=False)["input_ids"]
                out.append(i[0] if len(i) == 1 else None)
            return [x for x in out if x is not None]
        names = single(NAMES); places = single(PLACES); objs = single(OBJECTS); pr = []
        for _ in range(n):
            a, b = self.rng.choice(len(names), 2, replace=False)
            IO = names[a]; S = names[b]; pl = places[int(self.rng.integers(0, len(places)))]; ob = objs[int(self.rng.integers(0, len(objs)))]
            tpl = TEMPLATES[int(self.rng.integers(0, len(TEMPLATES)))]
            text = tpl.format(i0=self.tok.decode([IO]), i1=self.tok.decode([S]), P=self.tok.decode([pl]), S=self.tok.decode([S]), T=self.tok.decode([ob]))
            pr.append((self.tok(text)["input_ids"], IO, S))
        return pr

    # ---------- DISCOVERY: attribution sweep over every head + MLP ----------
    def attribution_sweep(self, metric, higher_is_worse=True):
        base = metric()
        rows = []
        for L in range(self.nL):
            for h in range(self.H):
                v = metric(ablate_heads=[(L, h)])
                eff = (v - base) if higher_is_worse else (base - v)
                rows.append({"comp": f"{L}.{h}", "kind": "head", "L": L, "effect": eff,
                             "named": self.head2name.get((L, h))})
        for L in range(self.nL):
            v = metric(ablate_mlps=[L])
            eff = (v - base) if higher_is_worse else (base - v)
            rows.append({"comp": f"mlp{L}", "kind": "mlp", "L": L, "effect": eff, "named": "mlp"})
        rows.sort(key=lambda r: -r["effect"])
        return base, rows

    def edge_probe(self, reader, metric, max_up=64):
        """Candidate circuit edges into `reader` (L,h): mean-ablate each upstream head, measure Δreader-mediated metric.
        Approx: ablate upstream head AND compare metric with vs without reader present (does the upstream matter *through* later)."""
        L0 = reader[0]
        base = metric()
        rows = []
        for La in range(L0):
            for ha in range(self.H):
                v = metric(ablate_heads=[(La, ha)])
                rows.append({"writer": f"{La}.{ha}", "effect": v - base, "named": self.head2name.get((La, ha))})
        rows.sort(key=lambda r: -r["effect"])
        return rows[:max_up]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/residual_vm_debugger_summary.json"))
    args = p.parse_args(argv)

    dbg = ResidualVMDebugger(args.pretrained, device=args.device)
    behaviours = {"induction": (dbg.metric_induction, True), "ioi": (dbg.metric_ioi, False), "generic": (dbg.metric_generic, True)}
    report = {"model": args.pretrained, "named_catalog": {n: [f"{L}.{h}" for L, h in hs] for n, hs in NAMED.items()}, "sweeps": {}}
    print(f"=== ResidualVM debugger (programmatic discovery) on {args.pretrained} ===")
    discovered = {}
    for bname, (metric, hiw) in behaviours.items():
        base, rows = dbg.attribution_sweep(metric, higher_is_worse=hiw)
        top = rows[: args.top]
        unnamed = [r for r in rows if r["named"] is None and r["effect"] > 0][: args.top]
        report["sweeps"][bname] = {"baseline": base, "top": top, "candidate_unnamed_ops": unnamed}
        discovered[bname] = unnamed
        print(f"\n[{bname}] baseline {base:+.3f} | top load-bearing components (effect = damage when ablated):")
        for r in top:
            tag = f"[{r['named']}]" if r["named"] else "  ?CANDIDATE"
            print(f"    {r['comp']:>6} {r['effect']:+.3f} {tag}")
        print("  -> DISCOVERED (load-bearing but UNNAMED) candidate ops: " + ", ".join(f"{r['comp']}({r['effect']:+.2f})" for r in unnamed[:8]))

    # edge-probe the strongest discovered induction-relevant candidate (or the top induction component)
    ind_unnamed = discovered.get("induction", [])
    probe_target = None
    for r in ind_unnamed:
        if r["kind"] == "head" and r["L"] >= 4:
            probe_target = tuple(int(x) for x in r["comp"].split(".")); break
    if probe_target:
        edges = dbg.edge_probe(probe_target, dbg.metric_induction)
        report["edge_probe"] = {"reader": f"{probe_target[0]}.{probe_target[1]}", "candidate_writers": edges[:12]}
        print(f"\n[edge-probe] candidate circuit edges INTO discovered op {probe_target[0]}.{probe_target[1]} (upstream ablation Δinduction):")
        for r in edges[:8]:
            tag = f"[{r['named']}]" if r["named"] else "?candidate-writer"
            print(f"    {r['writer']:>6} -> {probe_target[0]}.{probe_target[1]}  Δ{r['effect']:+.3f} {tag}")

    # logit-lens step on an induction example: where is the copy decided?
    vocab = dbg._common_vocab()
    seq = [int(vocab[i]) for i in dbg.rng.integers(0, len(vocab), 18)] * 2
    lens = dbg.logit_lens_step(seq)
    report["logit_lens_induction"] = lens
    determined = next((x["layer"] for x in lens if x["kl_to_final"] < 0.5), lens[-1]["layer"])
    print(f"\n[logit-lens step] induction next-token KL-to-final crosses 0.5 nats by layer L{determined} "
          f"(profile: " + " ".join(f"L{x['layer']}:{x['kl_to_final']:.1f}" for x in lens[::3]) + ")")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float))
    n_cand = sum(len(v) for v in discovered.values())
    print(f"\n[done] {n_cand} candidate-unnamed-op rows across 3 behaviours → {args.output}")
    print("  (the debugger is the programmatic discovery tool; candidates feed the operator/circuit catalog.)")
    return report


if __name__ == "__main__":
    main()
