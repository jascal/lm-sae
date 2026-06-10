"""Faithfulness gate for the pure-numpy Qwen3-MoE kernel (`numpy_moe.py`).

The methodology ported from fieldrun (`scripts/gemma3_ref.py`): build a *tiny* random-init `Qwen3MoeForCausalLM`,
randomise its norms (so the RMSNorm / QK-norm weights aren't a no-op identity that would hide a bug), export it with
`export_weights_moe.py`, run the numpy kernel, and check **top-1 agreement against torch** on the same fixed-window
contexts. No gated download — a tiny instance exercises every code path (dense layers, MoE layers, the router top-k,
GQA, QK-norm, and the sliding window) identically to the full model. This is the same f32 bar the other pylm kernels
hold (GPT-2 / RoPE / Gemma-2 all hit 100% vs torch).

Run standalone:  .venv/bin/python pylm/test_numpy_moe.py
Or under pytest:  .venv/bin/pytest pylm/test_numpy_moe.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_weights_moe import main as export_main  # noqa: E402
from numpy_moe import NumpyMoE  # noqa: E402

CTX, N_EVAL = 16, 60


def _tiny_cfg(swa: bool):
    import torch
    from transformers import Qwen3MoeConfig
    return Qwen3MoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,        # dense (non-sparse) layers
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, tie_word_embeddings=False, attention_bias=False,
        use_sliding_window=swa, sliding_window=4 if swa else None,   # small so seq>window actually masks
        decoder_sparse_step=2, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        norm_topk_prob=True, max_position_embeddings=256,
        attn_implementation="eager", torch_dtype=torch.float32,
    )


def _agreement(swa: bool, seed: int = 0):
    import torch
    from transformers import Qwen3MoeForCausalLM
    torch.manual_seed(seed)
    cfg = _tiny_cfg(swa)
    model = Qwen3MoeForCausalLM(cfg).eval()
    # randomise the norm weights (incl. QK-norm) to mean-1 — at the ones() default a QK-norm / RMSNorm bug is invisible.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(torch.randn_like(p) * 0.1 + 1.0)

    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td, safe_serialization=True)
        npz = Path(td) / "w.npz"
        export_main(["--model", td, "--out", str(npz), "--dtype", "float32"])

        g = torch.Generator().manual_seed(1000 + seed)
        ids = torch.randint(0, cfg.vocab_size, (CTX + N_EVAL + 4,), generator=g).tolist()
        end = min(CTX + N_EVAL, len(ids))
        torch_pred, lm = [], NumpyMoE(str(npz))
        numpy_pred = []
        with torch.no_grad():
            for i in range(CTX, end):
                ctx = ids[max(0, i - CTX):i]
                torch_pred.append(int(model(input_ids=torch.tensor([ctx])).logits[0, -1].argmax()))
                numpy_pred.append(lm.predict(ctx))
    agree = sum(a == b for a, b in zip(torch_pred, numpy_pred))
    return agree, len(torch_pred)


def test_numpy_moe_faithful():
    for swa in (False, True):
        agree, n = _agreement(swa)
        assert agree == n, f"qwen3moe{'-swa' if swa else ''}: only {agree}/{n} top-1 agree with torch"


if __name__ == "__main__":
    ok = True
    for swa in (False, True):
        agree, n = _agreement(swa)
        tag = "qwen3moe-swa" if swa else "qwen3moe   "
        print(f"[numpy-moe gate] {tag}: {agree}/{n} top-1 agree with torch  {'OK' if agree == n else 'FAIL'}")
        ok = ok and agree == n
    sys.exit(0 if ok else 1)
