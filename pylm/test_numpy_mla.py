"""Faithfulness gate for the pure-numpy MLA kernel (`numpy_mla.py`) — DeepSeek-V3, plain + YaRN.

Same tiny-random-instance methodology as `test_numpy_moe.py` (ported from fieldrun's `gemma3_ref.py`): build a tiny
`DeepseekV3ForCausalLM`, randomise its norms (and the router's `e_score_correction_bias`, so the choice-bias path isn't a
no-op), export with `export_weights_mla.py`, and check top-1 agreement against torch on the same fixed-window contexts.
Two cases: plain MLA and **YaRN** (with the deliberately large init that makes the rotary ramp / mscale / interleave a
discriminating gate — a wrong interleave agrees only ~11/60). No gated download.

Run standalone:  .venv/bin/python pylm/test_numpy_mla.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_weights_mla import main as export_main  # noqa: E402
from numpy_mla import NumpyMLA  # noqa: E402

CTX, N_EVAL = 16, 60


def _tiny_cfg(yarn: bool):
    import torch
    from transformers import DeepseekV3Config
    extra = {}
    if yarn:
        extra = dict(initializer_range=0.2, max_position_embeddings=128,
                     rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                                      "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 0.8, "mscale_all_dim": 0.5,
                                      "original_max_position_embeddings": 32})
    return DeepseekV3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=16, kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2, n_group=4, topk_group=2,
        norm_topk_prob=True, routed_scaling_factor=2.5, first_k_dense_replace=1, rms_norm_eps=1e-6,
        tie_word_embeddings=False, attn_implementation="eager", torch_dtype=torch.float32,
        **({"max_position_embeddings": 256} if not yarn else {}), **extra,
    )


def _agreement(yarn: bool, seed: int = 0):
    import torch
    from transformers import DeepseekV3ForCausalLM
    torch.manual_seed(seed)
    cfg = _tiny_cfg(yarn)
    model = DeepseekV3ForCausalLM(cfg).eval()
    # randomise norms (mean-1 under YaRN keeps attention sharp so the rotary details discriminate; mean-0 otherwise) and
    # the router choice-bias (a buffer/param that inits to 0 — at 0 the bias-for-choice path is invisible).
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(torch.randn_like(p) * 0.1 + (1.0 if yarn else 0.0))
            elif name.endswith("e_score_correction_bias"):
                p.copy_(torch.randn_like(p) * 0.2)
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.2)

    with tempfile.TemporaryDirectory() as td:
        model.save_pretrained(td, safe_serialization=True)
        npz = Path(td) / "w.npz"
        export_main(["--model", td, "--out", str(npz), "--dtype", "float32"])

        g = torch.Generator().manual_seed(1000 + seed)
        ids = torch.randint(0, cfg.vocab_size, (CTX + N_EVAL + 4,), generator=g).tolist()
        end = min(CTX + N_EVAL, len(ids))
        lm = NumpyMLA(str(npz))
        torch_pred, numpy_pred = [], []
        with torch.no_grad():
            for i in range(CTX, end):
                ctx = ids[max(0, i - CTX):i]
                torch_pred.append(int(model(input_ids=torch.tensor([ctx])).logits[0, -1].argmax()))
                numpy_pred.append(lm.predict(ctx))
    agree = sum(a == b for a, b in zip(torch_pred, numpy_pred))
    return agree, len(torch_pred)


def test_numpy_mla_faithful():
    for yarn in (False, True):
        agree, n = _agreement(yarn)
        assert agree == n, f"mla{'-yarn' if yarn else ''}: only {agree}/{n} top-1 agree with torch"


if __name__ == "__main__":
    ok = True
    for yarn in (False, True):
        agree, n = _agreement(yarn)
        tag = "mla-yarn" if yarn else "mla     "
        print(f"[numpy-mla gate] {tag}: {agree}/{n} top-1 agree with torch  {'OK' if agree == n else 'FAIL'}")
        ok = ok and agree == n
    sys.exit(0 if ok else 1)
