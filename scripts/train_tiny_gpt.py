"""Train a tiny GPT-2-config LM from scratch (the CPU-feasible 'nanochat' stand-in).

A real nanochat needs a GPU + an unimplemented sae-forge adapter. A tiny
GPT2LMHeadModel (n_embd=128, 4 layers) is architecturally a GPT, reuses sae-forge's
existing GPT2Adapter, and trains + forges on CPU. This is the TRAINABLE-host arm
(econ's 'concentrate' regime) — the complement to frozen GPT-2's preserve regime —
and it makes the lm-sae whole loop (train → SAE → forge → forged-cov95) tractable.

Saves runs/tiny_gpt.pt (config + state_dict).
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FALLBACK = (
    "In 1492 Columbus sailed the ocean blue. The Dow rose 3.2% on Tuesday. "
    "Dr. Smith paid $45 for 12 apples in New York. Is it 5 o'clock yet? "
) * 400


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-chars", type=int, default=400000)
    p.add_argument("--out", type=Path, default=Path("runs/tiny_gpt.pt"))
    args = p.parse_args(argv)

    import numpy as np
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    try:
        text = urllib.request.urlopen(CORPUS_URL, timeout=8).read().decode("utf-8", "ignore")
    except Exception as e:
        print(f"[warn] corpus fetch failed ({type(e).__name__}); fallback"); text = FALLBACK
    text = text[: args.max_chars]
    ids = np.array(tok(text)["input_ids"], dtype=np.int64)
    print(f"[1] corpus {len(ids)} tokens")

    cfg = GPT2Config(vocab_size=tok.vocab_size, n_positions=args.ctx, n_ctx=args.ctx,
                     n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head)
    model = GPT2LMHeadModel(cfg).train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2] tiny GPT: n_embd={args.n_embd} n_layer={args.n_layer} params={n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(0)
    for step in range(args.steps):
        starts = torch.randint(0, len(ids) - args.ctx - 1, (args.batch,), generator=g)
        x = torch.stack([torch.from_numpy(ids[s: s + args.ctx]) for s in starts])
        y = torch.stack([torch.from_numpy(ids[s + 1: s + 1 + args.ctx]) for s in starts])
        out = model(input_ids=x, labels=y)
        opt.zero_grad(); out.loss.backward(); opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"    step {step:>4}  loss {out.loss.item():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict()}, args.out)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
