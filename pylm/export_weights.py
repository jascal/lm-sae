"""One-time weight export — dump a HF GPT-2 to flat .npz arrays (the Tier-B composition kernel's data).

This is the *build* step (needs torch + transformers, run once on a real machine). The runtime kernel `numpy_lm.py`
that consumes these arrays needs only numpy — no torch, no GPU. The composition (attention + MLP) is the part pylm's flat
retrieval cannot do (it is genuine dense computation, the forge tax); this exports the weights it runs on.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LAYER_KEYS = ["ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight",
              "attn.c_proj.bias", "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias",
              "mlp.c_proj.weight", "mlp.c_proj.bias"]


def main(argv=None):
    import torch
    from transformers import GPT2LMHeadModel
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--out", type=Path, default=Path("pylm/weights_gpt2.npz"))
    p.add_argument("--dtype", default="float32", help="float32 (exact) or float16 (half the bytes)")
    args = p.parse_args(argv)

    m = GPT2LMHeadModel.from_pretrained(args.model).eval()
    sd = m.state_dict(); cfg = m.config

    def npy(name):
        return sd[name].detach().to(torch.float32).numpy().astype(args.dtype)

    W = {"wte": npy("transformer.wte.weight"), "wpe": npy("transformer.wpe.weight"),
         "ln_f.weight": npy("transformer.ln_f.weight"), "ln_f.bias": npy("transformer.ln_f.bias"),
         "config": np.array([cfg.n_layer, cfg.n_head, cfg.n_embd, cfg.n_positions, cfg.vocab_size], dtype=np.int64)}
    for L in range(cfg.n_layer):
        for k in LAYER_KEYS:
            W[f"h{L}.{k}"] = npy(f"transformer.h.{L}.{k}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **W)
    print(f"[export] {args.model}: {len(W)} flat arrays · {args.dtype} → {args.out} "
          f"({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
