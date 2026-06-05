"""lm-sae WHOLE LOOP on the tiny GPT — host cov95 vs FORGED cov95 (the actual tax).

train (done) -> extract final-layer acts + exact-lexical oracle -> train small TopK
SAE -> FORGE the tiny transformer over the SAE basis -> extract forged final hidden
(basis coords) -> decode -> re-score -> forged cov95. The forged-vs-host cov95 drop
is the forge tax on a (trainable) language model, end to end on CPU — the thing the
GPT-2 + 24k-SAELens path couldn't reach (over-completeness wall, GPU-scale).

SAE on the FINAL layer (like bio) so the forged residual is directly decodable — no
mid-model hook. Reuses the TopK SAE + scorer from forge_cov_mechanism.py and the
oracle from build_lm_bundle.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lm_bundle import COMMON, _lexical_features  # noqa: E402
from forge_cov_mechanism import _best_auc_per_label, _encode, _per_tier, _train_topk_sae  # noqa: E402


def _build_oracle(tok, all_ids, tok_strs, min_pos, n):
    feat, cols, tiers = [], [], []
    for c in COMMON:
        cid = tok(c, add_special_tokens=False)["input_ids"]
        if len(cid) == 1:
            feat.append(f"token:{c!r}"); tiers.append("token")
            cols.append(np.array([1 if j == cid[0] else 0 for j in all_ids], dtype=np.uint8))
    lex = [_lexical_features(s) for s in tok_strs]
    for name in lex[0]:
        feat.append(name); tiers.append("struct" if name.startswith("struct") else "lexical")
        cols.append(np.array([r[name] for r in lex], dtype=np.uint8))
    Y = np.stack(cols, 1)
    npos = Y.sum(0)
    keep = (npos >= min_pos) & (npos <= n - min_pos)
    return Y[:, keep], [f for f, k in zip(feat, keep) if k], [t for t, k in zip(tiers, keep) if k]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/tiny_gpt.pt"))
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--output", type=Path, default=Path("runs/whole_loop_tiny_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    from saeforge.basis import FeatureBasis

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT2LMHeadModel(GPT2Config(**ck["config"]))
    model.load_state_dict(ck["state_dict"]); model.eval()
    transformer = model.transformer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]

    print("[1] extract FINAL-layer host activations")
    acts, all_ids = [], []
    with torch.no_grad():
        for i in range(0, len(ids), args.ctx):
            ch = ids[i:i + args.ctx]
            h = transformer(input_ids=torch.tensor([ch])).last_hidden_state[0]
            acts.append(h.float().numpy()); all_ids.extend(ch)
    Xraw = np.concatenate(acts, 0).astype(np.float32)
    tok_strs = tok.convert_ids_to_tokens(all_ids)
    N, d = Xraw.shape
    Y, vocab, tiers = _build_oracle(tok, all_ids, tok_strs, args.min_pos, N)
    mu, sd = Xraw.mean(0, keepdims=True), Xraw.std(0, keepdims=True) + 1e-6
    X = ((Xraw - mu) / sd).astype(np.float32)
    print(f"    X={X.shape}  Y={Y.shape}  tiers={dict((t, tiers.count(t)) for t in set(tiers))}")

    print(f"[2] train TopK SAE (width {args.width}, {args.width/d:.1f}x)")
    params = _train_topk_sae(X, args.width, args.k, args.steps, 1e-3, 0)
    host = _per_tier(_best_auc_per_label(_encode(X, params, args.k), Y), tiers)
    print(f"[host]   cov95={host['all']['cov95']:.3f} mAUC={host['all']['mauc']:.3f}  "
          + "  ".join(f"{t}={host[t]['cov95']:.2f}" for t in host if t != "all"))

    print("[3] FORGE the tiny transformer over the SAE basis")
    Wdec = params[1].numpy().astype(np.float64)          # (d, width)
    Wdec_rows = Wdec.T                                   # (width, d) atoms-as-rows
    norms = np.linalg.norm(Wdec_rows, axis=1)
    basis = FeatureBasis(kept_ids=np.arange(args.width, dtype=np.int64), W_dec=Wdec_rows,
                         merged_norms=norms, original_norms=norms, metadata={"src": "tiny-gpt sae"})
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge import SubspaceProjector
    projector = SubspaceProjector(basis, scale_boost="auto")
    adapter = adapter_for(transformer)
    weights = projector.project_module(transformer, attention_width="host")
    cfg = adapter.build_native_config(transformer, basis.n_features)
    cfg.forward_mode = "native_in_basis"
    forged = NativeModel.from_projected_weights(cfg, weights).torch_module
    forged.eval()

    print("[4] extract FORGED activations (pre-lm_head residual) -> decode -> re-score")
    Wdec_t = torch.from_numpy(Wdec.astype(np.float32))   # (d, width): host = basis @ Wdec^T
    cap = {}

    def _pre(mod, inp):
        cap["h"] = (inp[0] if isinstance(inp, (tuple, list)) else inp).detach()
    handle = forged.lm_head.register_forward_pre_hook(_pre)
    facts = []
    with torch.no_grad():
        for i in range(0, len(ids), args.ctx):
            ch = ids[i:i + args.ctx]
            forged(torch.tensor([ch]))
            fh = cap["h"][0].float()                      # (seq, width) basis-coord final residual
            facts.append((fh @ Wdec_t.t()).numpy())      # (seq, d) host coords
    handle.remove()
    Xf = np.concatenate(facts, 0).astype(np.float32)
    Xf = ((Xf - mu) / sd).astype(np.float32)             # same z-score as host
    forged_t = _per_tier(_best_auc_per_label(_encode(Xf, params, args.k), Y), tiers)
    print(f"[forged] cov95={forged_t['all']['cov95']:.3f} mAUC={forged_t['all']['mauc']:.3f}  "
          + "  ".join(f"{t}={forged_t[t]['cov95']:.2f}" for t in forged_t if t != "all"))

    out = {"model": "tiny-gpt (n_embd %d, %d layers)" % (d, GPT2Config(**ck["config"]).n_layer),
           "d_model": d, "sae_width": args.width, "over_complete": round(args.width / d, 2),
           "n_samples": int(N), "host": host, "forged": forged_t,
           "tax_cov95": {t: round(host[t]["cov95"] - forged_t[t]["cov95"], 3)
                         for t in host}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[TAX] cov95 host->forged: "
          + "  ".join(f"{t} {host[t]['cov95']:.2f}->{forged_t[t]['cov95']:.2f}" for t in host))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
