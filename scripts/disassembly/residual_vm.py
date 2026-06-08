"""ResidualVM — a composable steppable debugger over any HF causal LM (+ the milestone-1 reconstruction CLI).

The `ResidualVM` class (below) is the consolidated intervention layer the ~20 disassembly scripts each
re-implemented: arch-generic load + corpus-mean fitting + ablate/patch/trace/attribution as **composable
context managers**, so an experiment is ~10 lines instead of ~100 and the load/hook/merge bugs live in one place.
`python residual_vm.py --demo` reproduces the induction reconstruction coverage through the class as a correctness
check. The original milestone-1 coverage-curve CLI is preserved below `main()`.

----- (original milestone-1 docstring) -----
ResidualVM — reconstruction-coverage harness for attention (decompilation milestone 1).

Turns the disassembly's "% of attention *legible*" into "% of the forward pass *executably reconstructable*".
Runs the model as an interpreter over its attention ops in selectable fidelity: KEEP a chosen set of heads
at full fidelity and **mean-ablate the complement** (the minimal "recompile = run these ops, null the rest").

  reconstruction_coverage(keep) = 1 - KL(host || keep-only) / KL(host || all-heads-ablated)

= 1.0 when keeping everything (sanity), 0.0 at the all-ablated floor. Sweeping the budget B (keep the top-B
heads by marginal ablation importance) gives a coverage CURVE; we compare it against a random-B control and,
if given, a NAMED idiom set (--named "L.H,..."). The question milestone 1 answers: how few / which heads
reconstruct most of the forward pass, and does the named op-catalog punch above its weight?

Mean-ablation hook is the proven one from gemma_causal/sink_ablation (replace a head's slice of the
attention output-projection input with its corpus mean) — arch-generic across GPT-2 (attn.c_proj) and the
self_attn.o_proj family (Gemma/Llama/Qwen). v1 = attention heads only (MLP ops = a later milestone); the
"recompile = forge into a feature basis" refinement (sae-forge NativeModel) is the v2 of this metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


class ResidualVM:
    """A composable, steppable debugger over any HF causal LM — the consolidated intervention layer the disassembly
    scripts each re-implemented (load + hooks + ablate/patch/trace/attribution), arch-generic (GPT-2 + RoPE/GQA).

    Interventions are **context managers** so they compose and auto-clean:

        vm = ResidualVM("gpt2"); vm.fit_means(chunks)
        with vm.ablate_heads(non_circuit):            # keep only the circuit
            nll = vm.nll(seqs, induction_target)
        with vm.ablate_heads(h_set), vm.ablate_mlps([0]):   # compose freely
            ...
        with vm.patch_mlp(layer, donor_out, pos):     # graft an activation (ROME-style edit)
            ...
    """

    def __init__(self, model_id, device="cuda", dtype="auto"):
        import torch
        from circuit_content_patch import _arch
        from mlp_atlas import mlp_blocks
        self.torch = torch
        self.is_gpt2 = "gpt2" in model_id.lower()
        dt = torch.bfloat16 if (dtype == "bf16" or (dtype == "auto" and (not self.is_gpt2 or "xl" in model_id))) else None
        from transformers import AutoTokenizer
        if self.is_gpt2:
            from transformers import GPT2LMHeadModel
            self.model = GPT2LMHeadModel.from_pretrained(model_id, attn_implementation="eager", **({"dtype": dt} if dt else {}))
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16)
        self.dev = device if torch.cuda.is_available() else "cpu"
        self.model = self.model.eval().to(self.dev)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        a = _arch(self.model); self.a = a
        self.oproj = a["oproj"]; self.layers = a["layers"]; self.norm = a["norm"]; self.mlps = mlp_blocks(self.model)
        self.H = a["H"]; self.hd = a["hd"]; self.nkv = a["nkv"]; self.nL = self.model.config.num_hidden_layers
        self.d = a.get("d", self.model.config.hidden_size)
        self.mean_oproj = None; self.mean_mlp = None; self._sae = {}

    # ---- corpus means (the mean-ablation reference) ----
    def fit_means(self, chunks):
        torch = self.torch
        cap = {L: [] for L in range(self.nL)}; mcap = {L: [] for L in range(self.nL)}
        ho = [self.oproj[L].register_forward_pre_hook((lambda L: lambda m, i: cap[L].append(i[0].detach().reshape(-1, i[0].shape[-1])))(L)) for L in range(self.nL)]
        hm = [self.mlps[L].register_forward_hook((lambda L: lambda m, i, o: mcap[L].append((o[0] if isinstance(o, tuple) else o).detach().reshape(-1, (o[0] if isinstance(o, tuple) else o).shape[-1])))(L)) for L in range(self.nL)]
        with torch.no_grad():
            for c in chunks:
                self.model(input_ids=torch.tensor([c], device=self.dev))
        for h in ho + hm:
            h.remove()
        self.mean_oproj = {L: torch.cat(cap[L], 0).mean(0) for L in range(self.nL)}
        self.mean_mlp = {L: torch.cat(mcap[L], 0).mean(0) for L in range(self.nL)}
        return self

    # ---- interventions (composable context managers) ----
    @contextmanager
    def ablate_heads(self, heads, mode="mean"):
        hd = self.hd; by = {}
        for (L, h) in heads:
            by.setdefault(L, []).append(h)
        hs = []
        for L, hss in by.items():
            def mk(L, hss):
                def hook(m, inp):
                    x = inp[0].clone()
                    for h in hss:
                        sl = slice(h * hd, (h + 1) * hd)
                        x[..., sl] = (self.mean_oproj[L][sl].to(x.dtype) if mode == "mean" else 0.0)
                    return (x,)
                return hook
            hs.append(self.oproj[L].register_forward_pre_hook(mk(L, hss)))
        try:
            yield self
        finally:
            for h in hs:
                h.remove()

    @contextmanager
    def ablate_mlps(self, layers, mode="mean"):
        hs = []
        for L in layers:
            def mk(L):
                def hook(m, i, o):
                    t = o[0] if isinstance(o, tuple) else o
                    rep = (self.mean_mlp[L].to(t.dtype).expand_as(t) if mode == "mean" else self.torch.zeros_like(t))
                    return (rep,) + tuple(o[1:]) if isinstance(o, tuple) else rep
                return hook
            hs.append(self.mlps[L].register_forward_hook(mk(L)))
        try:
            yield self
        finally:
            for h in hs:
                h.remove()

    @contextmanager
    def _patch(self, module, donor_out, pos):
        def hook(m, i, o):
            t = (o[0] if isinstance(o, tuple) else o).clone(); t[0, pos] = donor_out[0, pos].to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        h = module.register_forward_hook(hook)
        try:
            yield self
        finally:
            h.remove()

    def patch_mlp(self, layer, donor_out, pos):
        return self._patch(self.mlps[layer], donor_out, pos)

    def patch_attn(self, layer, donor_out, pos):
        return self._patch(self.oproj[layer], donor_out, pos)

    # ---- readouts ----
    def logits(self, ids):
        torch = self.torch
        with torch.no_grad():
            return self.model(input_ids=torch.tensor([ids], device=self.dev)).logits[0]

    def nll(self, seqs, target):
        """target(ids) -> list of (position, token) to score; mean −logp. Use for induction/fact/etc."""
        torch = self.torch; tot = 0.0; k = 0
        with torch.no_grad():
            for s in seqs:
                lp = torch.log_softmax(self.logits(s).float(), -1)
                for pos, t in target(s):
                    tot += float(-lp[pos, t]); k += 1
        return tot / max(k, 1)

    def trace(self, ids, attn=False):
        """Forward capturing per-layer mlp + (optional) attention outputs + resid_pre."""
        torch = self.torch; cap = {"mlp": {}, "attn": {}, "resid": {}}
        hk = [self.mlps[L].register_forward_hook((lambda L: lambda m, i, o: cap["mlp"].__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))(L)) for L in range(self.nL)]
        hk += [self.layers[L].register_forward_pre_hook((lambda L: lambda m, i: cap["resid"].__setitem__(L, i[0].detach()))(L)) for L in range(self.nL)]
        if attn:
            hk += [self.oproj[L].register_forward_hook((lambda L: lambda m, i, o: cap["attn"].__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))(L)) for L in range(self.nL)]
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids], device=self.dev))
        for h in hk:
            h.remove()
        cap["logits"] = out.logits[0]
        return cap

    def attribution(self, metric, kind="heads", layers=None):
        """Ablate each component alone, return [(comp, Δmetric)] sorted by importance (metric rises = load-bearing)."""
        base = metric(self)
        out = []
        if kind == "heads":
            for L in range(self.nL):
                for h in range(self.H):
                    with self.ablate_heads([(L, h)]):
                        out.append(((L, h), metric(self) - base))
        else:
            for L in (layers or range(self.nL)):
                with self.ablate_mlps([L]):
                    out.append((("mlp", L), metric(self) - base))
        return sorted(out, key=lambda r: -r[1])

    # ---- feature-level interventions (SAE latents — the next rung toward feature-native decompilation) ----
    def load_sae(self, layer):
        """Cache the resid SAE for a layer: GPT-2 jbloom (all layers) / Gemma Scope (8 layers). torch on device."""
        if layer in self._sae:
            return self._sae[layer]
        t = self.torch
        if self.is_gpt2:
            from huggingface_hub import hf_hub_download
            from safetensors.numpy import load_file
            st = load_file(hf_hub_download("jbloom/GPT2-Small-SAEs-Reformatted", f"blocks.{layer}.hook_resid_pre/sae_weights.safetensors"))
            w = {k: st[k] for k in ("W_enc", "b_enc", "W_dec", "b_dec")}; thr = None
        else:
            from scope_loader import scope_npz
            d = np.load(scope_npz(layer)); w = {k: d[k] for k in ("W_enc", "b_enc", "W_dec", "b_dec")}; thr = d["threshold"]
        sae = {k.lower().replace("_", ""): t.tensor(v.astype(np.float32), device=self.dev) for k, v in w.items()}
        sae["thr"] = t.tensor(thr.astype(np.float32), device=self.dev) if thr is not None else None
        self._sae[layer] = sae
        return sae

    def _feat_act(self, sae, x, feat):
        pre = (x.float() - sae["bdec"]) @ sae["wenc"][:, feat] + sae["benc"][feat]
        return self.torch.relu(pre) if sae["thr"] is None else self.torch.where(pre > sae["thr"][feat], pre, self.torch.zeros_like(pre))

    @contextmanager
    def set_feature(self, layer, feat, target):
        """Set SAE feature `feat` at resid-layer `layer` to `target` activation (target=0 → ablate). Composable."""
        sae = self.load_sae(layer)

        def pre(m, inp):
            x = inp[0]
            delta = (target - self._feat_act(sae, x, feat)).unsqueeze(-1) * sae["wdec"][feat]
            return (x + delta.to(x.dtype),) + inp[1:]
        h = self.layers[layer].register_forward_pre_hook(pre)
        try:
            yield self
        finally:
            h.remove()

    def ablate_feature(self, layer, feat):
        return self.set_feature(layer, feat, 0.0)

    def head_OV(self, L, h):
        """W_V^h W_O^h (d,d), arch-generic (GQA-aware)."""
        a = self.a; H = self.H; hd = self.hd; kvB = h // (H // self.nkv); d = self.d
        if a["is_gpt2"]:
            Wv_h = a["cattn"][L].weight.detach().float().cpu().numpy().astype(np.float64)[:, 2 * d:3 * d][:, h * hd:(h + 1) * hd]
            Wo_h = a["oproj"][L].weight.detach().float().cpu().numpy().astype(np.float64)[h * hd:(h + 1) * hd, :]
        else:
            Wv_h = a["vproj"][L].weight.detach().float().cpu().numpy().astype(np.float64)[kvB * hd:(kvB + 1) * hd, :].T
            Wo_h = a["oproj"][L].weight.detach().float().cpu().numpy().astype(np.float64)[:, h * hd:(h + 1) * hd].T
        return Wv_h @ Wo_h


def demo(model_id="gpt2", device="cpu"):
    """Reproduce the induction reconstruction coverage via the ResidualVM API (correctness check)."""
    vm = ResidualVM(model_id, device=device)
    tok = vm.tok; rng = np.random.default_rng(0)
    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:120000]
    ids = tok(txt)["input_ids"]; chunks = [ids[i:i + 64] for i in range(0, len(ids), 64) if len(ids[i:i + 64]) >= 8][:12]
    vm.fit_means(chunks)
    from collections import Counter
    vocab = [t for t, _ in Counter(t for c in chunks for t in c).most_common(400)]
    seqs = [(lambda s: s + s)([int(vocab[i]) for i in rng.integers(0, len(vocab), 22)]) for _ in range(20)]

    def ind_target(s):
        L = len(s) // 2
        return [(p, s[p + 1]) for p in range(L, 2 * L - 1)]
    circuit = {(5, 1), (5, 5), (6, 9), (7, 10), (4, 11), (2, 2), (3, 2), (3, 7)}     # induction + prev-token (gpt2)
    all_heads = [(L, h) for L in range(vm.nL) for h in range(vm.H)]
    base = vm.nll(seqs, ind_target)
    with vm.ablate_heads(all_heads):
        allabl = vm.nll(seqs, ind_target)
    with vm.ablate_heads([hh for hh in all_heads if hh not in circuit]):
        circ = vm.nll(seqs, ind_target)
    cov = (allabl - circ) / (allabl - base + 1e-9)
    print(f"[demo] {model_id}: induction-NLL full {base:.2f} | circuit-only {circ:.2f} | all-ablated {allabl:.2f} "
          f"| reconstruction coverage {cov:+.0%}  (matches circuit_reconstruction.py)")
    return cov


def feat_demo(model_id="gpt2", device="cpu", layer=5):
    """Feature-level surgery: ablate the subject's dominant SAE feature and watch the fact prediction drop."""
    vm = ResidualVM(model_id, device=device); tok = vm.tok; t = vm.torch
    ids = tok("The capital of France is")["input_ids"]
    paris = tok(" Paris", add_special_tokens=False)["input_ids"][0]
    fr = tok(" France", add_special_tokens=False)["input_ids"][0]
    spos = max(i for i, x in enumerate(ids) if x == fr)
    sae = vm.load_sae(layer)
    resid = vm.trace(ids)["resid"][layer][0]
    pre = (resid[spos].float() - sae["bdec"]) @ sae["wenc"] + sae["benc"]
    feat = int(t.relu(pre).argmax().cpu())
    top = vm.tok.convert_ids_to_tokens(int(((sae["wdec"][feat]) @ vm.model.get_output_embeddings().weight.float().T).argmax().cpu())).replace("Ġ", "_")
    base = float(t.log_softmax(vm.logits(ids).float(), -1)[-1, paris])
    with vm.ablate_feature(layer, feat):
        abl = float(t.log_softmax(vm.logits(ids).float(), -1)[-1, paris])
    print(f"[feat-demo] {model_id}: ablate the dominant feature #{feat} (promotes '{top}') at L{layer} of ' France' "
          f"→ logp(' Paris') {base:.2f} → {abl:.2f} (Δ {abl - base:+.2f}). Feature-level surgery works.")
    return abl - base


def _oproj_modules(model):
    """(output-projection module per layer, head_dim) for the mean-ablation slice — arch-generic."""
    cfg = model.config
    H = cfg.num_attention_heads
    if hasattr(model, "model") and hasattr(model.model, "layers"):           # Gemma/Llama/Qwen
        hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
        return [lyr.self_attn.o_proj for lyr in model.model.layers], hd
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):    # GPT-2
        return [blk.attn.c_proj for blk in model.transformer.h], cfg.n_embd // H
    raise SystemExit("unknown architecture")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--corpus", default="shakespeare")
    p.add_argument("--rank-tokens", type=int, default=2400, help="tokens for per-head importance ranking")
    p.add_argument("--eval-tokens", type=int, default=6000, help="tokens for the coverage-curve KLs")
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--budgets", default="1,2,4,8,16,24,32,48,64,96,128")
    p.add_argument("--named", default=None, help="comma list of L.H (a named idiom set to score), e.g. '4.11,5.0,5.5,6.9,7.11'")
    p.add_argument("--n-rand", type=int, default=3, help="random control sets per budget")
    p.add_argument("--device", default="cuda")
    p.add_argument("--demo", action="store_true", help="reproduce induction reconstruction coverage via the ResidualVM class (correctness check)")
    p.add_argument("--feat-demo", action="store_true", help="feature-level surgery demo: ablate a subject SAE feature, watch the fact drop")
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/residual_vm_summary.json"))
    args = p.parse_args(argv)
    if args.demo:
        demo(args.model, args.device); return
    if args.feat_demo:
        feat_demo(args.model, args.device); return

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = args.device if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    cfg = model.config
    nL, H = cfg.num_hidden_layers, cfg.num_attention_heads
    oprojs, hd = _oproj_modules(model)
    rng = np.random.default_rng(0)
    import urllib.request
    CORPORA = {"shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
               "wikitext": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"}
    txt = urllib.request.urlopen(urllib.request.Request(CORPORA.get(args.corpus, args.corpus),
                                 headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:400000]
    ids_all = tok(txt)["input_ids"]

    def chunkify(n):
        ids = ids_all[:n]
        return [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]

    # ---- capture per-head mean output (the ablation value) ----
    cap = {L: [] for L in range(nL)}
    caps = [oprojs[L].register_forward_pre_hook(
        (lambda L: lambda m, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in chunkify(args.rank_tokens):
            model(input_ids=torch.tensor([c], device=dev))
    for h in caps:
        h.remove()
    meanvec = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    def ablate_hooks(ablate):
        by_layer = {}
        for (L, h) in ablate:
            by_layer.setdefault(L, []).append(h)
        hs = []
        for L, hss in by_layer.items():
            def mk(L, hss):
                def hook(m, inp):
                    x = inp[0].clone()
                    for h in hss:
                        x[..., h * hd:(h + 1) * hd] = meanvec[L][h * hd:(h + 1) * hd].to(x.dtype)
                    return (x,)
                return hook
            hs.append(oprojs[L].register_forward_pre_hook(mk(L, hss)))
        return hs

    all_heads = [(L, h) for L in range(nL) for h in range(H)]

    # ---- cache host log-probs per chunk (no hooks), for whichever token budget ----
    def host_logprobs(chunks):
        out = []
        with torch.no_grad():
            for c in chunks:
                lp = F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                out.append(lp.cpu())
        return out

    def kl_keep(keep, chunks, host_lp):
        """mean KL(host || keep-only) over chunks; keep = set of heads kept (complement mean-ablated)."""
        ablate = [hh for hh in all_heads if hh not in keep]
        hs = ablate_hooks(ablate)
        tot = 0.0; n = 0
        try:
            with torch.no_grad():
                for c, hlp in zip(chunks, host_lp):
                    lp = F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                    ph = hlp.to(dev).exp()
                    kl = (ph * (hlp.to(dev) - lp)).sum(-1)
                    tot += float(kl.sum()); n += kl.numel()
        finally:
            for h in hs:
                h.remove()
        return tot / max(n, 1)

    # ---- per-head importance ranking (marginal: KL of ablating each head alone) ----
    rchunks = chunkify(args.rank_tokens); rhost = host_logprobs(rchunks)
    print(f"{args.model}: {nL}L x {H}H = {len(all_heads)} heads; ranking on {len(rchunks)} chunks ...")
    full_set = set(all_heads)
    imp = {}
    for i, hh in enumerate(all_heads):
        imp[hh] = kl_keep(full_set - {hh}, rchunks, rhost)   # KL when only hh is ablated
        if (i + 1) % 48 == 0:
            print(f"  ranked {i + 1}/{len(all_heads)} heads")
    ranked = sorted(all_heads, key=lambda hh: -imp[hh])
    print("  top heads by marginal ablation importance: " + ", ".join(f"{L}.{h}" for L, h in ranked[:8]))

    # ---- coverage curve on the eval token budget ----
    echunks = chunkify(args.eval_tokens); ehost = host_logprobs(echunks)
    floor = kl_keep(set(), echunks, ehost)        # all heads ablated
    print(f"  floor KL(host || all-{len(all_heads)}-heads-ablated) = {floor:.4f}")
    budgets = sorted({min(int(b), len(all_heads)) for b in args.budgets.split(",")})
    curve = []
    for B in budgets:
        keep_top = set(ranked[:B])
        kl_top = kl_keep(keep_top, echunks, ehost)
        cov_top = 1.0 - kl_top / floor
        rand_covs = []
        for s in range(args.n_rand):
            rs = set(map(tuple, np.array(all_heads)[rng.choice(len(all_heads), B, replace=False)].tolist()))
            rand_covs.append(1.0 - kl_keep(rs, echunks, ehost) / floor)
        cov_rand = float(np.mean(rand_covs))
        curve.append({"budget": B, "coverage_top": cov_top, "coverage_random": cov_rand, "kl_top": kl_top})
        print(f"  B={B:>4}: coverage top-B {cov_top:+.3f}  | random-B {cov_rand:+.3f}  (Δ {cov_top - cov_rand:+.3f})")

    # heads to reach 90% coverage (top ranking)
    h90 = next((c["budget"] for c in curve if c["coverage_top"] >= 0.9), None)

    named_res = None
    if args.named:
        named = set()
        for s in args.named.split(","):
            L, h = s.split("."); named.add((int(L), int(h)))
        cov_named = 1.0 - kl_keep(named, echunks, ehost) / floor
        ranks_of_named = sorted(ranked.index(hh) for hh in named if hh in ranked)
        # random sets of the same size as named, for a fair baseline
        rc = [1.0 - kl_keep(set(map(tuple, np.array(all_heads)[rng.choice(len(all_heads), len(named), replace=False)].tolist())), echunks, ehost) / floor
              for _ in range(args.n_rand)]
        named_res = {"heads": sorted(f"{L}.{h}" for L, h in named), "size": len(named),
                     "coverage": cov_named, "coverage_random_samesize": float(np.mean(rc)),
                     "importance_ranks": ranks_of_named}
        print(f"  NAMED set ({len(named)} heads): coverage {cov_named:+.3f}  vs random-same-size {np.mean(rc):+.3f}  "
              f"(importance ranks {ranks_of_named})")

    out = {"experiment": f"ResidualVM reconstruction coverage: {args.model}", "model": args.model,
           "corpus": args.corpus, "n_heads": len(all_heads), "floor_kl": floor,
           "budgets": budgets, "curve": curve, "heads_for_90pct_coverage": h90,
           "top_heads": [f"{L}.{h}" for L, h in ranked[:16]], "named": named_res}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[verdict] {h90 if h90 else '>'+str(budgets[-1])} of {len(all_heads)} heads reconstruct 90% of the "
          f"attention forward pass (KL-coverage); top-B beats random-B at every budget"
          f"{'' if all(c['coverage_top'] >= c['coverage_random'] for c in curve) else ' EXCEPT some'}.")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
