"""Phase 1b, corrected: the M0..Mn tower with GEOMETRY-FORCING retrain (basis refreshed).

The complement-routing retrain backfired (freed capacity -> the model learns to
entangle). The fix: the retrain step needs forgeability PRESSURE. So each round:
  1. fit an SAE on the current residual (the refreshed basis),
  2. RETRAIN the model with LM loss while its final residual is passed through that
     SAE's differentiable bottleneck decode(TopK(encode(.))) before lm_head -- the model
     is pressured to live in the SAE-representable (low-chi / forgeable) subspace,
  3. harvest the cleanest atoms as the level Mk,
  4. repeat (basis refreshes as the model changes).
Then a fixed v0 decomposition of the adapted model: does the entangled core drop BELOW
the original 0.24? (Geometry-forcing halved the tax in forge_aware_train_tiny; iterating
with basis refresh should push further.) Caveat: fine-tunes the existing tiny GPT.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lm_bundle import COMMON, _lexical_features  # noqa: E402
from forge_cov_mechanism import _column_ranks, _encode, _train_topk_sae  # noqa: E402
from preserve_hybrid_tiny import _auc_matrix  # noqa: E402


def _oracle(tok, all_ids, tok_strs, min_pos, n):
    cols = []
    for c in COMMON:
        cid = tok(c, add_special_tokens=False)["input_ids"]
        if len(cid) == 1:
            cols.append(np.array([1 if j == cid[0] else 0 for j in all_ids], np.uint8))
    lex = [_lexical_features(s) for s in tok_strs]
    for name in lex[0]:
        cols.append(np.array([r[name] for r in lex], np.uint8))
    Y = np.stack(cols, 1); npos = Y.sum(0)
    keep = (npos >= min_pos) & (npos <= n - min_pos)
    return Y[:, keep]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("runs/tiny_gpt.pt"))
    p.add_argument("--max-tokens", type=int, default=10000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--harvest", type=int, default=24)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--sae-steps", type=int, default=350)
    p.add_argument("--ft-steps", type=int, default=120)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--output", type=Path, default=Path("runs/mps_tower_geoforce_tiny_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT2LMHeadModel(GPT2Config(**ck["config"])); model.load_state_dict(ck["state_dict"])
    tr = model.transformer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    import urllib.request
    txt = urllib.request.urlopen(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        timeout=8).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx)]
    all_ids = [j for c in chunks for j in c]
    Y = _oracle(tok, all_ids, tok.convert_ids_to_tokens(all_ids), args.min_pos, len(all_ids))

    def residual():
        model.eval()
        with torch.no_grad():
            return np.concatenate([tr(input_ids=torch.tensor([c])).last_hidden_state[0].float().numpy()
                                   for c in chunks], 0).astype(np.float32)

    # differentiable SAE bottleneck on lm_head input, toggled during geometry-forcing retrain
    bn = {"active": False, "We": None, "Wd": None, "be": None, "bd": None, "k": args.k}

    def pre(mod, inp):
        if not bn["active"]:
            return None
        h = inp[0]
        pre_a = torch.relu((h - bn["bd"]) @ bn["We"].t() + bn["be"])
        topv, topi = pre_a.topk(bn["k"], dim=-1)
        z = torch.zeros_like(pre_a).scatter(-1, topi, topv)
        return (z @ bn["Wd"].t() + bn["bd"],)
    model.lm_head.register_forward_pre_hook(pre)

    def free_lm_loss():
        bn["active"] = False; model.eval()
        with torch.no_grad():
            ls = [float(model(input_ids=torch.tensor([c]), labels=torch.tensor([c])).loss)
                  for c in chunks[:20]]
        return float(np.mean(ls))

    orig_loss = free_lm_loss()
    rounds = []
    for rd in range(args.rounds):
        X = residual()
        We, Wd, be, bd = _train_topk_sae(X, args.width, args.k, args.sae_steps, 1e-3, rd)
        bn.update(We=We, Wd=Wd, be=be, bd=bd)
        # geometry-forcing retrain: LM loss through the SAE bottleneck
        bn["active"] = True; model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        g = torch.Generator().manual_seed(rd)
        ft = None
        for _ in range(args.ft_steps):
            c = chunks[int(torch.randint(0, len(chunks), (1,), generator=g))]
            t = torch.tensor([c])
            out = model(input_ids=t, labels=t)
            opt.zero_grad(); out.loss.backward(); opt.step()
            ft = float(out.loss.item())
        bn["active"] = False
        # harvest cleanest atoms from the (now more forgeable) residual
        X2 = residual()
        z = _encode(X2, (We, Wd, be, bd), args.k)
        valid = z.std(0) > 0
        A, _ok = _auc_matrix(_column_ranks(z), Y)
        atom_mono = np.where(valid, A.max(0), -np.inf)
        sel = [s for s in np.argsort(-atom_mono)[:args.harvest] if valid[s]]
        mono = float(np.mean([atom_mono[s] for s in sel])) if sel else float("nan")
        rounds.append({"round": rd, "harvest_mono": mono, "ft_loss_bottlenecked": ft})
        print(f"  M{rd}: harvest_mono={mono:.3f}  ft_loss(bottlenecked)={ft:.3f}", flush=True)

    geo_loss = free_lm_loss()

    # ---- fixed v0 decomposition of the adapted model: the core verdict ----
    print(f"\n[Phase B] fixed decomposition of the GEO-FORCED model (vs original core 0.239)", flush=True)
    bn["active"] = False
    R = residual(); totB = float((R ** 2).sum()); fixed = []
    for rd in range(8):
        We, Wd, be, bd = _train_topk_sae(R, args.width, args.k, args.sae_steps, 1e-3, 200 + rd)
        z = _encode(R, (We, Wd, be, bd), args.k); Wdn = Wd.numpy().astype(np.float32)
        valid = z.std(0) > 0
        A, _ok = _auc_matrix(_column_ranks(z), Y)
        atom_mono = np.where(valid, A.max(0), -np.inf)
        sel = [s for s in np.argsort(-atom_mono)[:args.harvest] if valid[s]]
        R = (R - z[:, sel] @ Wdn[:, sel].T).astype(np.float32)
        fixed.append({"round": rd, "mono": float(np.mean([atom_mono[s] for s in sel])),
                      "core": float((R ** 2).sum()) / totB})
        print(f"  fixed M{rd}: mono={fixed[-1]['mono']:.3f}  core={fixed[-1]['core']:.3f}", flush=True)
    geo_core = fixed[-1]["core"]

    out = {"experiment": "MPS tower + geometry-forcing retrain (corrected Phase 1b)",
           "fixed_v0_core": 0.239, "orig_lm_loss": orig_loss, "geo_lm_loss": geo_loss,
           "adapt_rounds": rounds, "fixed_decomp_of_geoforced": fixed, "geoforced_core": geo_core}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[verdict] core: original 0.239 -> geo-forced {geo_core:.3f}  "
          f"({'LOWER — geometry-forcing made it MORE forgeable' if geo_core < 0.239 else 'NOT lower'})")
    print(f"          LM loss: {orig_loss:.3f} -> {geo_loss:.3f} (capability cost {geo_loss-orig_loss:+.3f} nats)")
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
