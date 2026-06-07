"""ResidualVM — reconstruction-coverage harness for attention (decompilation milestone 1).

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
from pathlib import Path

import numpy as np


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
    p.add_argument("--output", type=Path, default=Path("runs/disassembly/residual_vm_summary.json"))
    args = p.parse_args(argv)

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
