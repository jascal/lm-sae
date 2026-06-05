"""DEGENERACY DIAGNOSTIC — forging a RAW slice of an over-complete SAELens SAE fails.

Result (do not read as a faithfulness curve): forging GPT-2 over the top-N SAELens
decoder atoms (no basis conditioning) gives numerically DEGENERATE logits —
meanKL(host||forged) = 58 (N=256), 17245 (N=768), 15 (N=1536): huge and
non-monotonic, i.e. garbage. You cannot naively forge a 32x-over-complete basis;
the basis must be CONDITIONED first. The repo's path does this with polygram
compression (examples/forge_gpt2_real_sae.py -> sane KL ~21 on an 11-feature smoke).

So this script stands as the *negative control*: it demonstrates why GPT-2's
production SAE needs the polygram whole-loop, not a raw slice. A faithful forge +
the forged-cov95 tax (hook the forged layer-8 residual, re-score the oracle) is the
GPU-scale next build.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/allans/code/bio-sae/scripts")
import forge_capability_eval as fce  # noqa: E402

EVAL_PROMPTS = [
    "The mitochondrion is the powerhouse of the",
    "To be or not to be, that is the",
    "All happy families are alike; each unhappy family is",
    "In the beginning God created the heavens and the",
    "The capital of France is Paris, a city renowned for its",
    "Newton's third law states that every action has an equal and opposite",
]


def _mean_kl(host, forged, tok, device="cpu"):
    """mean_t KL(softmax(host_logits_t) || softmax(forged_logits_t)) over prompts/positions."""
    import torch

    kls = []
    with torch.no_grad():
        for p in EVAL_PROMPTS:
            ids = torch.tensor([tok(p)["input_ids"]])
            hl = host(ids).logits[0]                     # (L, V)
            fo = forged(ids)
            fl = (fo[0] if isinstance(fo, (tuple, list)) else fo)[0]
            lp_h = torch.log_softmax(hl.float(), -1)
            lp_f = torch.log_softmax(fl.float(), -1)
            kl = (lp_h.exp() * (lp_h - lp_f)).sum(-1)    # (L,)
            kls.append(kl.mean().item())
    return float(np.mean(kls))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sae-repo", default="jbloom/GPT2-Small-SAEs-Reformatted")
    p.add_argument("--sae-file", default="blocks.8.hook_resid_pre/sae_weights.safetensors")
    p.add_argument("--ns", default="64,128,256,512,768,1024,1536")
    p.add_argument("--output", type=Path, default=Path("runs/forge_gpt2_summary.json"))
    args = p.parse_args(argv)

    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from saeforge.basis import FeatureBasis
    from transformers import GPT2TokenizerFast

    Wdec = load_file(hf_hub_download(repo_id=args.sae_repo, filename=args.sae_file))["W_dec"].astype(np.float64)
    d_model = Wdec.shape[1]
    order = np.argsort(-np.linalg.norm(Wdec, axis=1))
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ns = [int(x) for x in args.ns.split(",")]
    print(f"[forge-gpt2] SAE {args.sae_file}  d_model={d_model}  sweeping N={ns}")

    rows = []
    host_ref = None
    for n in ns:
        Wd = Wdec[order[:n]]
        norms = np.linalg.norm(Wd, axis=1)
        basis = FeatureBasis(kept_ids=np.arange(n, dtype=np.int64), W_dec=Wd,
                             merged_norms=norms, original_norms=norms,
                             metadata={"src": f"gpt2 saelens top-{n}"})
        forged, host = fce._forge(basis, "gpt2", "cpu", "auto")
        host_ref = host
        kl = _mean_kl(host, forged, tok)
        rows.append({"n_features": n, "over_complete": round(n / d_model, 2), "kl": kl})
        print(f"    N={n:>5} ({n/d_model:.2f}x)  meanKL(host||forged)={kl:.3f}")
    # host self-KL is 0 by definition; report a random-basis floor for scale
    out = {"sae_repo": args.sae_repo, "sae_file": args.sae_file, "d_model": d_model,
           "metric": "mean KL(host||forged) on held-out prompts (lower=faithful)",
           "sweep": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
