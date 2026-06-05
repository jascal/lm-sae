"""Prototype: does FORGE-DURING-TRAINING avoid the cov95 tax?

Free-train-then-forge pays the tax (host cov95 0.65 -> forged 0.12). Idea: impose the
basis geometry DURING training so the model only learns what the basis can hold. Test:
fine-tune the tiny GPT through an SAE-reconstruction bottleneck (final residual ->
decode(TopK(encode(.))) before lm_head) — a differentiable stand-in for the forge
constraint — then compare host cov95, FORGED cov95, and LM loss to the free baseline.

If geo-forcing keeps forged cov95 high (no collapse) at acceptable LM loss, the idea
works; if LM loss craters, that's the capability cost of forgeable geometry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forge_cov_mechanism import _best_auc_per_label, _encode, _per_tier, _train_topk_sae  # noqa: E402
from preserve_hybrid_tiny import _build_oracle  # noqa: E402

CTX, MAXTOK, WIDTH, K = 96, 8000, 512, 24


def main():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    from saeforge import SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.basis import FeatureBasis
    from saeforge.model import NativeModel

    ck = torch.load("runs/tiny_gpt.pt", map_location="cpu", weights_only=False)

    def fresh_model():
        m = GPT2LMHeadModel(GPT2Config(**ck["config"])); m.load_state_dict(ck["state_dict"]); m.eval()
        return m

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][:MAXTOK]
    chunks = [ids[i:i + CTX] for i in range(0, len(ids), CTX) if len(ids[i:i + CTX]) > 1]

    def host_acts(model):
        with torch.no_grad():
            return np.concatenate([model.transformer(input_ids=torch.tensor([c])).last_hidden_state[0]
                                   .float().numpy() for c in chunks], 0).astype(np.float32)

    base = fresh_model()
    Xraw = host_acts(base)
    all_ids = [j for c in chunks for j in c]
    Y, tiers = _build_oracle(tok, all_ids, tok.convert_ids_to_tokens(all_ids), 30, Xraw.shape[0])
    mu, sd = Xraw.mean(0, keepdims=True), Xraw.std(0, keepdims=True) + 1e-6
    muT, sdT = torch.from_numpy(mu), torch.from_numpy(sd)
    params = _train_topk_sae(((Xraw - mu) / sd).astype(np.float32), WIDTH, K, 600, 1e-3, 0)
    Wenc, Wdec, benc, bdec = params

    def cov95(acts_raw):
        z = _encode(((acts_raw - mu) / sd).astype(np.float32), params, K)
        return _per_tier(_best_auc_per_label(z, Y), tiers)

    def lm_loss(model, bottleneck=False):
        h = None
        if bottleneck:
            h = model.transformer.register_forward_hook(_mk_bottleneck())
        tot = 0.0
        with torch.no_grad():
            for c in chunks[:40]:
                t = torch.tensor([c])
                tot += model(input_ids=t, labels=t).loss.item()
        if h:
            h.remove()
        return tot / 40

    def _mk_bottleneck():
        def hook(mod, inp, out):
            x = out.last_hidden_state
            xz = (x - muT) / sdT
            pre = torch.relu((xz - bdec) @ Wenc.t() + benc)
            tv, ti = pre.topk(K, dim=-1)
            z = torch.zeros_like(pre).scatter(-1, ti, tv)
            out.last_hidden_state = (z @ Wdec.t() + bdec) * sdT + muT
            return out
        return hook

    def forge_cov95(model):
        tr = model.transformer
        basis = FeatureBasis(kept_ids=np.arange(WIDTH, dtype=np.int64), W_dec=Wdec.numpy().T.astype(np.float64),
                             merged_norms=np.linalg.norm(Wdec.numpy().T, axis=1),
                             original_norms=np.linalg.norm(Wdec.numpy().T, axis=1), metadata={})
        proj = SubspaceProjector(basis, scale_boost="auto")
        w = proj.project_module(tr, attention_width="host")
        cfg = adapter_for(tr).build_native_config(tr, WIDTH); cfg.forward_mode = "native_in_basis"
        forged = NativeModel.from_projected_weights(cfg, w).torch_module; forged.eval()
        cap = {}
        forged.lm_head.register_forward_pre_hook(lambda m, i: cap.__setitem__("h", i[0]))
        outs = []
        with torch.no_grad():
            for c in chunks:
                forged(torch.tensor([c]))
                outs.append((cap["h"][0] @ Wdec.t()).numpy())
        return cov95(np.concatenate(outs, 0).astype(np.float32))

    print("[baseline] free model")
    b_host = cov95(Xraw)["all"]
    b_forge = forge_cov95(base)["all"]
    b_loss = lm_loss(base)
    print(f"  host cov95={b_host['cov95']:.3f}  forged cov95={b_forge['cov95']:.3f}  LM loss={b_loss:.3f}")

    print("[geo-forced] fine-tune THROUGH the SAE-recon bottleneck (impose basis geometry)")
    geo = fresh_model()
    hh = geo.transformer.register_forward_hook(_mk_bottleneck())
    opt = torch.optim.AdamW(geo.parameters(), lr=5e-4)
    g = torch.Generator().manual_seed(0)
    geo.train()
    for step in range(300):
        c = chunks[int(torch.randint(0, len(chunks), (1,), generator=g))]
        t = torch.tensor([c])
        loss = geo(input_ids=t, labels=t).loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"    step {step}  bottlenecked LM loss {loss.item():.3f}")
    hh.remove(); geo.eval()

    g_host = cov95(host_acts(geo))["all"]
    g_forge = forge_cov95(geo)["all"]
    g_loss = lm_loss(geo)
    g_loss_bn = lm_loss(geo, bottleneck=True)
    print(f"  host cov95={g_host['cov95']:.3f}  forged cov95={g_forge['cov95']:.3f}  "
          f"LM loss={g_loss:.3f} (through-bottleneck {g_loss_bn:.3f})")

    out = {
        "free":       {"host_cov95": b_host["cov95"], "forged_cov95": b_forge["cov95"], "lm_loss": b_loss},
        "geo_forced": {"host_cov95": g_host["cov95"], "forged_cov95": g_forge["cov95"],
                       "lm_loss": g_loss, "lm_loss_through_bottleneck": g_loss_bn},
        "tax_free": round(b_host["cov95"] - b_forge["cov95"], 3),
        "tax_geo":  round(g_host["cov95"] - g_forge["cov95"], 3),
    }
    Path("runs/forge_aware_train_tiny_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n=== VERDICT ===")
    print(f"  free:       host {b_host['cov95']:.2f} -> forged {b_forge['cov95']:.2f}  (tax {out['tax_free']}), LM {b_loss:.2f}")
    print(f"  geo-forced: host {g_host['cov95']:.2f} -> forged {g_forge['cov95']:.2f}  (tax {out['tax_geo']}), LM {g_loss:.2f}")
    print("[done] runs/forge_aware_train_tiny_summary.json")


if __name__ == "__main__":
    main()
