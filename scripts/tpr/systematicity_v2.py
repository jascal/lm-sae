"""TPR systematicity v2: pre-registered in docs/TPR_SYSTEMATICITY_PREREG.md (do not change choices here without
marking them post hoc there).

(a) P2 under the paper's conditions: L2,1-regularised TPR, a 6-layer unpacking decoder trained on REAL period
    encodings, mid-layer sites, withheld (filler, role) pairs, the 1/n_w! strong baseline.
(b) The split-half conjunction test for TopK SAE latents, with a permutation null and a planted-additive
    specificity control.

    .venv/bin/python scripts/tpr/systematicity_v2.py --model gpt2
    .venv/bin/python scripts/tpr/systematicity_v2.py --model Qwen/Qwen2.5-0.5B
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent_null import additive_fit, cell_means  # noqa: E402
from tpr_vs_sae import (NOUNS, OCCUPATIONS, VERBS, TopKSAE, fit_sae, make_list, make_svo,  # noqa: E402
                        pad_pairs, period_activations)

# Fresh LIST vocabulary: candidates disjoint from v1's NOUNS (filtered at runtime to single-token in BOTH tokenizers).
NEW_NOUNS = """anchor arrow axe badge banana barrel bean bell belt bench berry blade blanket bolt bone boot brush bucket
bulb button cabin cake canal candy cannon carrot cart chain chalk cherry chest chimney coat? collar comb cookie copper
cord corn cotton crown crystal cushion dagger diamond dish drawer dress dust feather fence fiddle flute fountain frame
fruit gate gem goat grape hammer harp helmet hook horn hose jar jewel kettle lemon lens lid lock locket marble mask mat
medal melon mill mint mud nail needle nest net nut onion orange oven paint palace pearl pen pie pillow pipe pistol pump
quilt rabbit? rack raft rail razor ribbon rice rifle rocket rope sack saddle sail sauce saw scarf scroll seed shell
shield shovel silk skirt sled slipper soap sock sofa spear sponge spring stamp statue stick stool straw string sword
tank tent thread thumb tile tin tooth torch towel tower toy? trap tray tub tube tulip umbrella vase vest violin wagon
wallet wand wheel whistle wig wire wool yarn""".replace("?", "").split()
LAYER_FRACTIONS = (0.25, 0.5, 0.75)
SEEDS = dict(stimulus=11, withheld=111, split=12, fit=(0, 1, 2), sae=0, halves=7)
LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2)


def both_single(toks, words, cap, exclude=()):
    out = [w for w in dict.fromkeys(words) if w not in exclude
           and all(len(t.encode(" " + w, add_special_tokens=False)) == 1 for t in toks)]
    return out[:cap]


# ---------------------------------------------------------------- (a) unpacking decoder and TPR

class Unpacker(nn.Module):
    """Decoder-only Transformer: [prefix(encoding), y_0..y_{T-1}] -> y_0..y_T (EOS). 6 layers, 1024, 16 heads."""

    def __init__(self, d_in, n_vocab, max_len, d=1024, layers=6, heads=16, ff=4096, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.tok = nn.Embedding(n_vocab, d)
        self.pos = nn.Embedding(max_len + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, n_vocab)
        self.max_len, self.eos = max_len, n_vocab - 1

    def forward(self, x, y_in):
        h = torch.cat([self.inp(x)[:, None], self.tok(y_in)], 1)
        h = h + self.pos(torch.arange(h.shape[1], device=x.device))[None]
        mask = nn.Transformer.generate_square_subsequent_mask(h.shape[1], device=x.device)
        return self.out(self.body(h, mask=mask, is_causal=True))

    @torch.no_grad()
    def greedy(self, x):
        y = torch.zeros(len(x), 0, dtype=torch.long, device=x.device)
        for _ in range(self.max_len + 1):
            nxt = self(x, y)[:, -1].argmax(-1)
            y = torch.cat([y, nxt[:, None]], 1)
        return y


def targets(rows, eos, max_len):
    """Filler sequence in surface order + EOS, padded with EOS. Surface order = order of pairs as generated."""
    y = torch.full((len(rows), max_len + 1), eos, dtype=torch.long)
    for i, row in enumerate(rows):
        for j, (a, _) in enumerate(row["pairs"]):
            y[i, j] = a
    return y


def exact(pred, y, eos):
    """Exact match up to and including the first EOS of the target."""
    ok = []
    for p, t in zip(pred.tolist(), y.tolist(), strict=True):
        n = t.index(eos) + 1
        ok.append(p[:n] == t[:n])
    return torch.tensor(ok)


SMOKE = False


def train_unpacker(x, y, n_vocab, max_len, seed, epochs=40, batch=256, lr=3e-4):
    epochs = 1 if SMOKE else epochs
    torch.manual_seed(seed)
    model = Unpacker(x.shape[1], n_vocab, max_len).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(x), generator=g).split(batch):
            idx = idx.to(x.device)
            yi = y[idx]
            logits = model(x[idx], yi[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), yi.flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


class TPR(nn.Module):
    def __init__(self, n_fill, n_role, d, d_f=32):
        super().__init__()
        self.f = nn.Embedding(n_fill, d_f)
        self.r = nn.Embedding(n_role, n_role)
        self.W = nn.Linear(d_f * n_role, d)
        nn.init.normal_(self.f.weight, std=0.3)
        nn.init.normal_(self.r.weight, std=0.3)

    def forward(self, f, r, m):
        bound = torch.einsum("bpi,bpj->bpij", self.f(f), self.r(r)) * m[..., None, None]
        return self.W(bound.sum(1).flatten(1))

    def l21(self):
        return self.f.weight.norm(dim=1).sum() + self.r.weight.norm(dim=1).sum()


class Atomic(nn.Module):
    def __init__(self, n_fill, n_role, d):
        super().__init__()
        self.n_role = n_role
        self.e = nn.Embedding(n_fill * n_role, d)
        nn.init.zeros_(self.e.weight)  # unseen pairs keep a zero atom
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, f, r, m):
        return (self.e(f * self.n_role + r) * m[..., None]).sum(1) + self.b


def fit(module, f, r, m, x, lam=0.0, steps=3000, lr=3e-3, seed=0):
    steps = 50 if SMOKE else steps
    torch.manual_seed(seed)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(module(f, r, m), x)
        if lam:
            loss = loss + lam * module.l21()
        loss.backward()
        opt.step()
    return module


def binom_tail(k, n, p):
    """P(X >= k), X ~ Bin(n, p)."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def strata_of(rows, withheld):
    return torch.tensor([min(3, sum(tuple(p) in withheld for p in r["pairs"])) for r in rows])


def by_stratum(ok, strata):
    return {str(k): (float(ok[strata == k].float().mean()), int((strata == k).sum())) if (strata == k).any() else None
            for k in range(4)}


# ---------------------------------------------------------------- (b) split-half conjunction test

def split_half_latents(codes, f, r, m, n_fill, n_role, half, min_active=50, min_count=3):
    """Per eligible latent: True iff A's raw cell means predict B's better than A's additive fit."""
    out = {}
    for j in np.flatnonzero((codes > 0).sum(0) >= min_active):
        a = codes[:, j]
        ma, ca = cell_means(a[half], f[half], r[half], m[half], n_fill, n_role)
        mb, cb = cell_means(a[~half], f[~half], r[~half], m[~half], n_fill, n_role)
        cells = (ca >= min_count) & (cb >= min_count)
        if cells.sum() < 10:
            continue
        w = np.where(cells, cb, 0.0)
        add = additive_fit(np.where(cells, ma, 0.0), np.where(cells, ca, 0.0))
        out[int(j)] = bool((w * (mb - ma) ** 2).sum() < (w * (mb - add) ** 2).sum())
    return out


def additive_design(f, r, m, n_fill, n_role):
    X = np.zeros((len(f), 1 + n_fill + n_role))
    X[:, 0] = 1
    for p in range(f.shape[1]):
        keep = m[:, p] > 0
        np.add.at(X, (np.flatnonzero(keep), 1 + f[keep, p]), 1)
        np.add.at(X, (np.flatnonzero(keep), 1 + n_fill + r[keep, p]), 1)
    return X


def conjunction_test(codes, f, r, m, n_fill, n_role, seed):
    rng = np.random.default_rng(seed)
    half = rng.random(len(codes)) < 0.5
    observed = split_half_latents(codes, f, r, m, n_fill, n_role, half)
    perm = codes[rng.permutation(len(codes))]
    null = split_half_latents(perm, f, r, m, n_fill, n_role, half)
    elig = sorted(observed)
    planted = relu = {}
    if elig:
        X = additive_design(f, r, m, n_fill, n_role)
        A = codes[:, elig]
        coef, *_ = np.linalg.lstsq(X, A, rcond=None)
        fitted = X @ coef
        sigma = (A - fitted).std(0)
        synth = fitted + rng.normal(size=fitted.shape) * sigma
        planted = split_half_latents(synth, f, r, m, n_fill, n_role, half, min_active=0)
        freq = (A > 0).mean(0)
        thr = np.array([np.quantile(synth[:, i], 1 - freq[i]) for i in range(len(elig))])
        relu = split_half_latents(np.maximum(synth - thr, 0), f, r, m, n_fill, n_role, half)

    def frac(d):
        return float(np.mean(list(d.values()))) if d else None

    return dict(n_eligible=len(observed), conjunctive=frac(observed), null_n=len(null), null=frac(null),
                planted_n=len(planted), planted_additive=frac(planted), relu_n=len(relu),
                relu_additive_diagnostic=frac(relu))


def p4_prime(res):
    ok = res["conjunctive"] is not None and res["planted_additive"] is not None
    specific = ok and res["planted_additive"] <= 0.10
    passes = specific and res["conjunctive"] >= 0.50 and res["conjunctive"] - (res["null"] or 0.0) >= 0.30
    refuting = specific and res["conjunctive"] <= (res["null"] or 0.0) + 0.10
    return dict(specific=specific, passes=passes, refuting=refuting)


# ---------------------------------------------------------------- driver

def run_family(name, rows, n_fill, n_role, withheld, model, tok, layers, primary, device, args, log):
    rng = random.Random(SEEDS["split"])
    order = list(range(len(rows)))
    rng.shuffle(order)
    n_dec, n_dict = int(0.33 * len(rows)), int(0.42 * len(rows))
    dec_rows = [rows[i] for i in order[:n_dec]]
    dict_all = [rows[i] for i in order[n_dec:n_dec + n_dict]]
    test_rows = [rows[i] for i in order[n_dec + n_dict:]]
    dict_rows = [r for r in dict_all if not any(tuple(p) in withheld for p in r["pairs"])]
    n_val = len(dict_rows) // 10
    val_rows, fit_rows = dict_rows[:n_val], dict_rows[n_val:]
    strata = strata_of(test_rows, withheld)
    max_len = max(len(r["pairs"]) for r in rows)
    eos = n_fill
    log(f"[{name}] dec {len(dec_rows)} fit {len(fit_rows)} val {len(val_rows)} test {len(test_rows)} "
        f"strata {[int((strata == k).sum()) for k in range(4)]}")
    groups = dict(dec=dec_rows, fit=fit_rows, val=val_rows, test=test_rows)
    all_rows = [r for g in groups.values() for r in g]
    acts = period_activations(model, tok, all_rows, layers, device, batch=args.batch)
    sl, start = {}, 0
    for k, g in groups.items():
        sl[k] = slice(start, start + len(g))
        start += len(g)
    pf, pr, pm = (t.to(device) for t in pad_pairs(all_rows))
    y = targets(all_rows, eos, max_len).to(device)
    out = {}
    for layer in layers:
        x_raw = acts[layer].to(device)
        mu, sd = x_raw[sl["fit"]].mean(0), x_raw[sl["fit"]].std(0) + 1e-6
        x = (x_raw - mu) / sd
        dec = train_unpacker(x[sl["dec"]], y[sl["dec"]], n_fill + 1, max_len, seed=SEEDS["fit"][0])

        def accuracy(vecs, part):
            return exact(dec.greedy(vecs), y[sl[part]], eos)

        res = dict(real=by_stratum(accuracy(x[sl["test"]], "test"), strata))
        fargs = (pf[sl["fit"]], pr[sl["fit"]], pm[sl["fit"]], x[sl["fit"]])
        targs = (pf[sl["test"]], pr[sl["test"]], pm[sl["test"]])
        vargs = (pf[sl["val"]], pr[sl["val"]], pm[sl["val"]])
        val_acc = {}
        for lam in LAMBDAS:
            tpr = fit(TPR(n_fill, n_role, x.shape[1]).to(device), *fargs, lam=lam, seed=SEEDS["fit"][0])
            with torch.no_grad():
                val_acc[str(lam)] = float(accuracy(tpr(*vargs), "val").float().mean())
        lam = float(max(val_acc, key=val_acc.get))
        per_seed = []
        for seed in SEEDS["fit"]:
            tpr = fit(TPR(n_fill, n_role, x.shape[1]).to(device), *fargs, lam=lam, seed=seed)
            with torch.no_grad():
                per_seed.append(accuracy(tpr(*targs), "test"))
        stack = torch.stack([s.float() for s in per_seed])
        seed_mean = stack.mean(0)
        tpr_res = {"lambda": lam, "val_acc_by_lambda": val_acc,
                   "by_stratum": {k: (float(seed_mean[strata == int(k)].mean()), int((strata == int(k)).sum()))
                                  if (strata == int(k)).any() else None for k in map(str, range(4))},
                   "by_stratum_seed_range": {str(k): [float(stack[i][strata == k].mean()) for i in range(len(per_seed))]
                                             for k in range(4) if (strata == k).any()}}
        atomic = fit(Atomic(n_fill, n_role, x.shape[1]).to(device), *fargs, seed=SEEDS["fit"][0])
        with torch.no_grad():
            res["atomic"] = by_stratum(accuracy(atomic(*targs), "test"), strata)
        res["tpr"] = tpr_res
        acc0 = tpr_res["by_stratum"]["0"][0]
        pos = strata >= 1
        acc_pos = float(seed_mean[pos].mean()) if pos.any() else None
        strict = acc_pos is not None and acc_pos >= acc0 - 0.15
        paper = {}
        for mm in (2, 3):
            n = int((strata == mm).sum())
            if n >= 30:
                k = round(float(seed_mean[strata == mm].mean()) * n)
                base = 1 / math.factorial(mm)  # stratum 3 pools n_w >= 3; 1/3! is the most lenient baseline there
                paper[str(mm)] = dict(n=n, hits=k, baseline=base, p=binom_tail(k, n, base))
        res["P2_strict"] = dict(acc_nw0=acc0, acc_nw_ge1=acc_pos, passes=bool(strict))
        res["P2_paper"] = dict(strata=paper, passes=any(v["p"] < 0.01 and v["hits"] / v["n"] > v["baseline"]
                                                         for v in paper.values()))
        if layer == primary:
            conj = {}
            xall = torch.cat([x[sl["fit"]], x[sl["val"]], x[sl["test"]]])
            fa = torch.cat([pf[sl["fit"]], pf[sl["val"]], pf[sl["test"]]]).cpu().numpy()
            ra = torch.cat([pr[sl["fit"]], pr[sl["val"]], pr[sl["test"]]]).cpu().numpy()
            ma = torch.cat([pm[sl["fit"]], pm[sl["val"]], pm[sl["test"]]]).cpu().numpy()
            for width in (1024, 8192):
                torch.manual_seed(SEEDS["sae"])
                sae = TopKSAE(x.shape[1], width, 32).to(device)
                fit_sae(sae, torch.cat([x[sl["fit"]], x[sl["val"]]]), 4000, seed=SEEDS["sae"])
                with torch.no_grad():
                    codes = sae.codes(xall).cpu().numpy()
                c = conjunction_test(codes, fa, ra, ma, n_fill, n_role, SEEDS["halves"])
                c["P4_prime"] = p4_prime(c)
                conj[str(width)] = c
            res["conjunction"] = conj
            res["P4_prime"] = dict(passes=all(v["P4_prime"]["passes"] for v in conj.values()),
                                   specific=all(v["P4_prime"]["specific"] for v in conj.values()),
                                   refuting=all(v["P4_prime"]["refuting"] for v in conj.values()))
        out[str(layer)] = res
        log(f"[{name} L{layer}{'*' if layer == primary else ''}] real {res['real']} | tpr(λ={lam}) "
            f"{tpr_res['by_stratum']} | atomic {res['atomic']} | P2s {res['P2_strict']['passes']} "
            f"P2p {res['P2_paper']['passes']}" + (f" | P4' {res['P4_prime']} {res['conjunction']}"
                                                   if layer == primary else ""))
    return dict(n_fill=n_fill, n_role=n_role, n_withheld=len(withheld), primary_layer=primary, layers=out,
                split={k: len(v) for k, v in groups.items()}, test_strata=[int((strata == k).sum()) for k in range(4)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=("gpt2", "Qwen/Qwen2.5-0.5B"))
    ap.add_argument("--list-n", type=int, default=24000)
    ap.add_argument("--svo-n", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true",
                    help="bug check only: stimulus seed 999, tiny sizes, 1 epoch; numbers are NOT results")
    args = ap.parse_args()
    if args.smoke:
        SEEDS["stimulus"], args.list_n, args.svo_n = 999, 1500, 1500
        globals()["SMOKE"] = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    toks = [AutoTokenizer.from_pretrained(n) for n in ("gpt2", "Qwen/Qwen2.5-0.5B")]
    tok = toks[0] if args.model == "gpt2" else toks[1]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device).eval()
    n_layers = model.config.num_hidden_layers
    layers = [round(fr * n_layers) for fr in LAYER_FRACTIONS]
    primary = layers[1]
    nouns = both_single(toks, NEW_NOUNS, 80, exclude=set(NOUNS))
    occupations, verbs = both_single(toks, OCCUPATIONS, 40), both_single(toks, VERBS, 16)
    log(f"{args.model}: layers {layers} (primary {primary}); {len(nouns)} nouns, {len(occupations)} occupations, "
        f"{len(verbs)} verbs; device {device}")
    rng = random.Random(SEEDS["stimulus"])
    list_rows, list_roles = make_list(rng, nouns, args.list_n)
    svo_rows, _ = make_svo(rng, occupations, verbs, args.svo_n)
    wrng = random.Random(SEEDS["withheld"])
    list_pairs = [(a, s) for a in range(len(nouns)) for s in range(len(list_roles))]
    svo_pairs = [(a, s) for a in range(len(occupations)) for s in (0, 2)]
    list_w = set(wrng.sample(list_pairs, len(list_pairs) // 10))
    svo_w = set(wrng.sample(svo_pairs, len(svo_pairs) // 4))
    summary = dict(tag="empirical", prereg="docs/TPR_SYSTEMATICITY_PREREG.md", model=args.model, seeds=SEEDS,
                   layers=layers, primary_layer=primary,
                   vocab=dict(nouns=nouns, occupations=occupations, verbs=verbs))
    summary["LIST"] = run_family("LIST", list_rows, len(nouns), len(list_roles), list_w, model, tok, layers, primary,
                                 device, args, log)
    summary["SVO"] = run_family("SVO", svo_rows, len(occupations) + len(verbs), 3, svo_w, model, tok, layers,
                                primary, device, args, log)
    summary["seconds"] = round(time.time() - t0, 1)
    default = f"runs/tpr/systematicity_v2_{args.model.split('/')[-1]}_summary.json"
    out = Path(args.out or ("/tmp/claude-wt/smoke.json" if args.smoke else default))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
