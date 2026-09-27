"""Is the SHARED category subspace causally used? (universal-grammar track)

For each target language t, the UPOS concept is erased from the residual stream entering block
--layer (every position), and we measure the increase in next-token loss on the FIRST subword of
each held-out word, split by that word's gold UPOS into function words (ADP DET AUX SCONJ CCONJ PRON
PART) and content words (NOUN VERB ADJ ADV PROPN).

Main intervention: closed-form LEACE (Belrose et al. 2023) erasure of ALL linearly decodable UPOS
information, fitted on the other languages — a principled "the concept is gone" edit. (An earlier
variant projected out a rank-k probe's readout covectors; it left the information in place and gave
inconsistent, control-sized effects, so it was dropped.) Control:
LEACE-style erasure of a RANDOM subspace of the same rank in the same whitened space. The
prediction of a universal, causally used category code: erasure learned WITHOUT t hurts t's
prediction far more than the random control, in every language.

  .venv/bin/python scripts/grammar/ug_causal.py --model Qwen/Qwen2.5-0.5B --model-dir data/ug/Qwen2.5-0.5B \
      --layer 12 --out runs/grammar/qwen05_causal_L12_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ug_lib as U
from ug_extract import PUD, prefix_ids, read_conllu, spans

FUNC = {"ADP", "DET", "AUX", "SCONJ", "CCONJ", "PRON", "PART"}
CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


def aligned(tok, lang: str, pre: list[int]):
    """Held-out sentences: (ids, [(position predicting word w's first token, target id, upos)])."""
    out = []
    for si, s in enumerate(read_conllu(PUD / f"{lang}_pud.conllu")):
        if si % 5 != 0:
            continue
        sp = spans(s)
        if sp is None:
            continue
        enc = tok(s["text"], add_special_tokens=True, return_offsets_mapping=True)
        ids, offs = pre + enc["input_ids"], [(0, 0)] * len(pre) + enc["offset_mapping"]
        targets = []
        for a, b, widx in sp:
            if len(widx) != 1:
                continue
            hits = [t for t, (x, y) in enumerate(offs) if y > x and x < b and y > a]
            if hits and hits[0] >= 1:
                targets.append((hits[0] - 1, ids[hits[0]], s["words"][widx[0]]["upos"]))
        out.append((ids, targets))
    return out


def leace(parts, n_cls: int,
          rng: torch.Generator | None = None, chunk: int = 16384):
    """Closed-form LEACE (Belrose et al. 2023) eraser for a categorical concept. `parts` are
    (per-language z-scored states, that language's σ, labels): x_raw − μ_l = z·σ_l, so states are
    centred per language. Covariances are accumulated in float64 chunks (no full-precision copy
    of the pooled states). Returns (A, W, rank) with erase(x) = x − A (W (x − μ)). With `rng`, the
    same-rank control: a RANDOM subspace of the whitened space instead of the concept's span."""
    S = XZ = zsum = xsum = None
    n = 0
    for Xz, sigma, y in parts:
        if S is None:
            d = Xz.shape[1]
            S = torch.zeros(d, d, dtype=torch.float64, device=U.DEV)
            XZ = torch.zeros(d, n_cls, dtype=torch.float64, device=U.DEV)
            zsum = torch.zeros(n_cls, dtype=torch.float64, device=U.DEV)
            xsum = torch.zeros(d, dtype=torch.float64, device=U.DEV)
        yt = torch.from_numpy(y).to(U.DEV)
        for s in range(0, Xz.shape[0], chunk):
            X = Xz[s:s + chunk].double() * sigma.double()
            Z = torch.nn.functional.one_hot(yt[s:s + chunk], n_cls).double()
            S += X.T @ X
            XZ += X.T @ Z
            zsum += Z.sum(0)
            xsum += X.sum(0)
            n += X.shape[0]
    S /= n
    XZ = XZ / n - torch.outer(xsum / n, zsum / n)     # cross-covariance with centred labels
    evals, evecs = torch.linalg.eigh(S)
    keep = evals > evals.max() * 1e-6
    W = evecs[:, keep] @ torch.diag(evals[keep].rsqrt()) @ evecs[:, keep].T       # Σ^{-1/2}
    Winv = evecs[:, keep] @ torch.diag(evals[keep].sqrt()) @ evecs[:, keep].T     # Σ^{1/2}
    U_, sv, _ = torch.linalg.svd(W @ XZ, full_matrices=False)
    r = int((sv > sv.max() * 1e-6).sum())
    if rng is None:
        Q = U_[:, :r]
    else:
        Q, _ = torch.linalg.qr(torch.randn(d, r, generator=rng, dtype=torch.float64).to(U.DEV))
    return (Winv @ Q @ Q.T).float(), W.float(), r


def subspace(W: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis (d×k) of the raw-space readout covectors of a standardised-space probe."""
    Q, _ = torch.linalg.qr((W / sigma[None, :]).T.double())
    return Q.float()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--langs", nargs="+")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    inf = U.info(a.model_dir)
    langs = a.langs or list(inf["langs"])
    # word states are streamed from disk one language at a time (a 2.6B model's weights plus 13
    # languages of states do not fit an 8 GB GPU together)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    block = model.base_model.layers[a.layer - 1]      # hidden_states[layer] is this block's output
    pre = prefix_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    state = {"Q": None, "mu": None}

    def hook(_mod, _inp, out):
        if state["Q"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        Q, mu = state["Q"], state["mu"]
        hf = h.float()
        if isinstance(Q, tuple):                          # LEACE: x − A B (x − μ)
            A, Bw = Q
            hf = hf - ((hf - mu) @ Bw.T) @ A.T
        else:                                             # orthogonal projection-out
            hf = hf - ((hf - mu) @ Q) @ Q.T
        h2 = hf.to(h.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    block.register_forward_hook(hook)
    g = torch.Generator(device="cpu").manual_seed(0)
    res = {"model": a.model, "layer": a.layer, "k": a.k, "langs": langs, "per_lang": {}}
    for t in langs:
        mu, sigma = U.load(a.model_dir, t, a.layer).stats
        torch.cuda.empty_cache()

        def parts():
            """LEACE input: the OTHER languages' states, each centred on its own (unlabelled) mean."""
            for l in langs:
                if l != t:
                    Dl = U.load(a.model_dir, l, a.layer)
                    m = ~Dl.test & ~Dl.mwt
                    yield Dl.X[torch.from_numpy(m).to(U.DEV)], Dl.stats[1], Dl.upos[m]
                    del Dl

        A, Bw, r = leace(parts(), len(U.UPOS))
        Ar, Brw, _ = leace(parts(), len(U.UPOS), rng=g)
        torch.cuda.empty_cache()
        conds = {"none": None, "leace_shared": (A, Bw), "leace_random": (Ar, Brw)}
        sents = aligned(tok, t, pre)
        nll = {c: {"func": [], "content": []} for c in conds}
        for c, Q in conds.items():
            state["Q"], state["mu"] = Q, mu
            for s in range(0, len(sents), a.batch):
                chunk = sents[s:s + a.batch]
                Lmax = max(len(x[0]) for x in chunk)
                ids = torch.full((len(chunk), Lmax), pad, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for r, (x, _) in enumerate(chunk):
                    ids[r, : len(x)] = torch.tensor(x)
                    mask[r, : len(x)] = 1
                with torch.no_grad():
                    logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
                for r, (_, targets) in enumerate(chunk):
                    for pos, tid, upos in targets:
                        cls = "func" if upos in FUNC else "content" if upos in CONTENT else None
                        if cls:
                            nll[c][cls].append(float(-torch.log_softmax(logits[r, pos].float(), -1)[tid]))
        state["Q"] = None
        base = {k: float(np.mean(v)) for k, v in nll["none"].items()}
        row = {"base_nll": base, "n": {k: len(v) for k, v in nll["none"].items()}, "leace_rank": r}
        for c in [c for c in conds if c != "none"]:
            row[c] = {k: float(np.mean(nll[c][k]) - base[k]) for k in base}
        res["per_lang"][t] = row
        print(f"{t}: ΔNLL func/content " + "  ".join(
            f"{c} {row[c]['func']:+.3f}/{row[c]['content']:+.3f}" for c in conds if c != "none"), flush=True)
    for c in ("leace_shared", "leace_random"):
        res[f"mean_{c}"] = {k: float(np.mean([res["per_lang"][l][c][k] for l in langs])) for k in ("func", "content")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
