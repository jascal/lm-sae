"""POST-HOC control for P4 of docs/TPR_VS_SAE.md (not pre-registered): a permutation noise floor for the latent
'interaction share'. Cell means over a sparse (filler, role) grid are noisy, and noise alone leaves variance that an
additive row+column fit cannot explain. Re-trains the SAE with tpr_vs_sae.py's exact splits and settings; then, per
latent, compares the observed interaction share with the same statistic after shuffling the latent's activations
across contexts (structure destroyed, marginal kept).

    .venv/bin/python scripts/tpr/latent_null.py   # -> runs/tpr/latent_null_summary.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tpr_vs_sae import (NOUNS, OCCUPATIONS, VERBS, TopKSAE, fit_sae, latent_structure, make_list, make_svo,  # noqa: E402
                        pad_pairs, period_activations, single_token)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # noqa: E402


def cell_means(a, f, r, m, n_fill, n_role):
    sums, counts = np.zeros((n_fill, n_role)), np.zeros((n_fill, n_role))
    for p in range(f.shape[1]):
        keep = m[:, p] > 0
        np.add.at(sums, (f[keep, p], r[keep, p]), a[keep])
        np.add.at(counts, (f[keep, p], r[keep, p]), 1)
    return sums / np.maximum(counts, 1), counts


def additive_fit(mean, w, iters=50):
    grand = (mean * w).sum() / max(w.sum(), 1e-12)
    row, col = np.zeros(mean.shape[0]), np.zeros(mean.shape[1])
    for _ in range(iters):
        row = np.where(w.sum(1) > 0, (w * (mean - grand - col[None, :])).sum(1) / np.maximum(w.sum(1), 1e-12), 0)
        col = np.where(w.sum(0) > 0, (w * (mean - grand - row[:, None])).sum(0) / np.maximum(w.sum(0), 1e-12), 0)
    return grand + row[:, None] + col[None, :]


def split_half(codes, f, r, m, n_fill, n_role, seed, min_active=50, min_count=3):
    """Held-out test of conjunction: fit on half A, score cell means of half B. 'Conjunctive' iff raw A cell means
    predict B better than the additive (row+col) fit of A, on cells with >= min_count in both halves."""
    codes, f, r, m = codes.numpy(), f.numpy(), r.numpy(), m.numpy()
    half = np.random.default_rng(seed).random(len(codes)) < 0.5
    wins, gains = [], []
    for j in np.flatnonzero((codes > 0).sum(0) >= min_active):
        a = codes[:, j]
        ma, ca = cell_means(a[half], f[half], r[half], m[half], n_fill, n_role)
        mb, cb = cell_means(a[~half], f[~half], r[~half], m[~half], n_fill, n_role)
        cells = (ca >= min_count) & (cb >= min_count)
        if cells.sum() < 10:
            continue
        w = np.where(cells, cb, 0.0)
        add = additive_fit(np.where(cells, ma, 0.0), np.where(cells, ca, 0.0))
        err_add = (w * (mb - add) ** 2).sum()
        err_raw = (w * (mb - ma) ** 2).sum()
        wins.append(err_raw < err_add)
        gains.append(1 - err_raw / max(err_add, 1e-12))
    return dict(n=len(wins), conjunctive_heldout_frac=float(np.mean(wins)) if wins else None,
                median_gain=float(np.median(gains)) if gains else None)


def splits(rows, withheld, seed):
    rng = random.Random(seed + 1)
    order = list(range(len(rows)))
    rng.shuffle(order)
    n_read, n_dict = int(0.33 * len(rows)), int(0.42 * len(rows))
    dict_rows = [rows[i] for i in order[n_read:n_read + n_dict] if not any(tuple(p) in withheld for p in rows[i]["pairs"])]
    return dict_rows, [rows[i] for i in order[n_read + n_dict:]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--widths", type=int, nargs="+", default=[1024, 8192])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sae-steps", type=int, default=4000)
    ap.add_argument("--out", default="runs/tpr/latent_null_summary.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token, tok.padding_side = tok.eos_token, "right"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    nouns, occ, verbs = single_token(tok, NOUNS, 100), single_token(tok, OCCUPATIONS, 40), single_token(tok, VERBS, 16)
    rng = random.Random(args.seed)
    list_rows, list_roles = make_list(rng, nouns, 24000)  # same generator order and sizes as tpr_vs_sae.py defaults
    svo_rows, _ = make_svo(rng, occ, verbs, 16000)
    wrng = random.Random(args.seed + 100)
    list_w = set(wrng.sample([(a, s) for a in range(len(nouns)) for s in range(len(list_roles))], len(nouns) * 12 // 10))
    svo_w = set(wrng.sample([(a, s) for a in range(len(occ)) for s in (0, 2)], len(occ) * 2 // 10))
    out = dict(tag="empirical", post_hoc=True, layer=args.layer, families={})
    for name, rows, withheld, n_fill, n_role in (("LIST", list_rows, list_w, len(nouns), 12),
                                                  ("SVO", svo_rows, svo_w, len(occ) + len(verbs), 3)):
        dict_rows, test_rows = splits(rows, withheld, args.seed)
        both = dict_rows + test_rows
        x = period_activations(model, tok, both, [args.layer], device)[args.layer].to(device)
        mu, sd = x[:len(dict_rows)].mean(0), x[:len(dict_rows)].std(0) + 1e-6
        x = (x - mu) / sd
        f, r, m = pad_pairs(both)
        fam = {}
        for width in args.widths:
            torch.manual_seed(args.seed)
            sae = TopKSAE(x.shape[1], width, 32).to(device)
            fit_sae(sae, x[:len(dict_rows)], args.sae_steps, seed=args.seed)
            with torch.no_grad():
                codes = sae.codes(x).cpu()
            observed = latent_structure(codes, f, r, m, n_fill, n_role)
            gen = torch.Generator().manual_seed(args.seed)
            shuffled = codes[torch.randperm(len(codes), generator=gen)]
            null = latent_structure(shuffled, f, r, m, n_fill, n_role)
            obs = np.array([o["interaction"] for o in observed])
            nul = np.array([o["interaction"] for o in null])
            floor = float(np.quantile(nul, 0.95)) if len(nul) else None
            fam[str(width)] = dict(n_latents=len(obs), conjunctive_frac=float((obs > 0.5).mean()),
                                   null_n=len(nul), null_median=float(np.median(nul)) if len(nul) else None,
                                   null_q95=floor, null_conjunctive_frac=float((nul > 0.5).mean()) if len(nul) else None,
                                   conjunctive_above_null_frac=float(((obs > 0.5) & (obs > floor)).mean())
                                   if floor is not None else None,
                                   observed_median=float(np.median(obs)),
                                   split_half=split_half(codes, f, r, m, n_fill, n_role, args.seed),
                                   split_half_shuffled=split_half(shuffled, f, r, m, n_fill, n_role, args.seed))
            print(name, width, fam[str(width)], flush=True)
        out["families"][name] = fam
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
