"""Binding vs atoms on GPT-2 small: supervised TPR dictionary vs atomic / additive / bag nulls vs TopK SAEs.

Pre-registered in docs/TPR_VS_SAE.md (McCoy, Soulos, Linzen & Smolensky 2026, arXiv:2608.29530). Residual at the
final period of templated LIST / SVO stimuli; every dictionary is fit in per-dimension z-space; reconstructions are
scored by (1) exact-match structure readout with linear readouts trained on REAL encodings, (2) FVU, and (3) splice
KL recovered at the period. Contexts containing withheld (filler, role) pairs never reach dictionary training.

    .venv/bin/python scripts/tpr/tpr_vs_sae.py            # -> runs/tpr/tpr_vs_sae_summary.json
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

NOUNS = """apple table river horse window garden pencil mirror candle bottle rabbit forest island castle bridge jacket
carpet basket kitchen engine ladder pepper rubber lion trouble kindness coffee chair house money water music paper stone
bread cloud train plane ship tree flower school church phone glass metal wood fire snow rain wind salt sugar milk cheese
butter egg fish bird dog cat cow pig sheep mouse snake bear wolf fox deer duck eagle shark whale frog bee rose grass sand
rock gold silver iron steel coal oil glove shirt shoe hat coat ring clock lamp bed door wall roof floor road street park
farm field hill lake sea ocean beach desert valley cave star moon sun planet book letter map flag box bag cup plate knife
spoon bowl pot pan key coin ticket card camera radio piano guitar drum ball kite toy doll game bike truck boat car bus""".split()
OCCUPATIONS = """doctor lawyer teacher nurse pilot farmer baker chef judge poet painter singer dancer actor writer author
editor banker soldier sailor driver student scientist engineer artist captain priest waiter guard coach player officer
agent tourist manager clerk tailor butcher miner hunter spy king queen prince""".split()
VERBS = """helped saw called met thanked liked hated pushed followed visited praised blamed hired paid warned chased loved
watched kissed attacked admired avoided""".split()


def single_token(tok, words, cap):
    out = [w for w in words if len(tok.encode(" " + w)) == 1]
    return out[:cap]


def make_list(rng, nouns, n_contexts):
    roles = [(i, n - 1 - i) for n in (3, 4, 5) for i in range(n)]
    rid = {r: k for k, r in enumerate(roles)}
    seen, rows = set(), []
    while len(rows) < n_contexts:
        n = rng.choice((3, 4, 5))
        words = tuple(rng.sample(range(len(nouns)), n))
        if words in seen:
            continue
        seen.add(words)
        text = "Here is a list of words: " + ", ".join(nouns[w] for w in words) + "."
        rows.append(dict(text=text, length=n, pairs=[(w, rid[(i, n - 1 - i)]) for i, w in enumerate(words)]))
    return rows, roles


def make_svo(rng, occupations, verbs, n_contexts):
    combos = [(s, v, o) for s in range(len(occupations)) for o in range(len(occupations)) if s != o
              for v in range(len(verbs))]
    rng.shuffle(combos)
    rows = []
    for s, v, o in combos[:n_contexts]:
        text = f"The {occupations[s]} {verbs[v]} the {occupations[o]}."
        # fillers: occupations then verbs share one filler vocabulary
        rows.append(dict(text=text, length=3, pairs=[(s, 0), (len(occupations) + v, 1), (o, 2)]))
    return rows, ["subject", "verb", "object"]


@torch.no_grad()
def period_activations(model, tok, rows, layers, device, batch=256):
    acts = {layer: [] for layer in layers}
    for start in range(0, len(rows), batch):
        enc = tok([r["text"] for r in rows[start:start + batch]], return_tensors="pt", padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(len(last), device=device)
        for layer in layers:
            acts[layer].append(out.hidden_states[layer + 1][idx, last].float().cpu())
    return {layer: torch.cat(v) for layer, v in acts.items()}


def pad_pairs(rows, width=5):
    f = torch.zeros(len(rows), width, dtype=torch.long)
    r = torch.zeros(len(rows), width, dtype=torch.long)
    m = torch.zeros(len(rows), width)
    for i, row in enumerate(rows):
        for j, (a, s) in enumerate(row["pairs"]):
            f[i, j], r[i, j], m[i, j] = a, s, 1.0
    return f, r, m


class Supervised(nn.Module):
    """kind in {tpr, atomic, additive, bag}; input = padded (filler, role, mask)."""

    def __init__(self, kind, n_fill, n_role, d, d_f=32):
        super().__init__()
        self.kind, self.n_role = kind, n_role
        self.b = nn.Parameter(torch.zeros(d))
        if kind == "tpr":
            self.f = nn.Embedding(n_fill, d_f)
            self.r = nn.Embedding(n_role, n_role)
            self.W = nn.Linear(d_f * n_role, d, bias=False)
            nn.init.normal_(self.f.weight, std=0.3)
            nn.init.normal_(self.r.weight, std=0.3)
        elif kind == "atomic":
            self.e = nn.Embedding(n_fill * n_role, d)
            nn.init.zeros_(self.e.weight)  # unseen pairs keep a zero atom: no systematic binding by construction
        else:
            self.ef = nn.Embedding(n_fill, d)
            nn.init.zeros_(self.ef.weight)
            if kind == "additive":
                self.er = nn.Embedding(n_role, d)
                nn.init.zeros_(self.er.weight)

    def forward(self, f, r, m):
        if self.kind == "tpr":
            bound = torch.einsum("bpi,bpj->bpij", self.f(f), self.r(r)) * m[..., None, None]
            return self.W(bound.sum(1).flatten(1)) + self.b
        if self.kind == "atomic":
            return (self.e(f * self.n_role + r) * m[..., None]).sum(1) + self.b
        out = (self.ef(f) * m[..., None]).sum(1)
        if self.kind == "additive":
            out = out + (self.er(r) * m[..., None]).sum(1)
        return out + self.b

    def n_params(self, used_pairs=None):
        if self.kind == "atomic" and used_pairs is not None:
            return used_pairs * self.b.numel() + self.b.numel()
        return sum(p.numel() for p in self.parameters())


class TopKSAE(nn.Module):
    def __init__(self, d, m, k):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d, m)
        self.dec = nn.Linear(m, d)
        with torch.no_grad():
            self.dec.weight.copy_(F.normalize(torch.randn(d, m), dim=0))
            self.enc.weight.copy_(self.dec.weight.T)
            self.enc.bias.zero_()

    def codes(self, x):
        pre = F.relu(self.enc(x - self.dec.bias))
        top = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)

    def forward(self, x):
        return self.dec(self.codes(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def fit_supervised(model, f, r, m, x, steps, lr=3e-3, wd=1e-5):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for step in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(model(f, r, m), x)
        loss.backward()
        opt.step()
    return float(loss.detach())


def fit_sae(sae, x, steps, lr=1e-3, batch=512, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for step in range(steps):
        idx = torch.randint(0, len(x), (batch,), device=x.device, generator=g)
        opt.zero_grad()
        loss = F.mse_loss(sae(x[idx]), x[idx])
        loss.backward()
        opt.step()
        with torch.no_grad():
            sae.dec.weight.copy_(F.normalize(sae.dec.weight, dim=0))
    return float(loss)


class Readout(nn.Module):
    """Linear 'period-unpacking': one softmax over fillers per role (+ a length head for variable-length lists)."""

    def __init__(self, d, n_fill, n_role, lengths):
        super().__init__()
        self.heads = nn.Linear(d, n_fill * n_role)
        self.n_fill, self.n_role, self.lengths = n_fill, n_role, lengths
        self.len_head = nn.Linear(d, len(lengths)) if len(lengths) > 1 else None

    def logits(self, x):
        return self.heads(x).view(len(x), self.n_role, self.n_fill)

    def fit(self, x, f, r, m, length_idx, steps=600, lr=3e-3, wd=1e-4):
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        for _ in range(steps):
            opt.zero_grad()
            lg = self.logits(x)
            sel = lg[torch.arange(len(x), device=x.device)[:, None], r]  # [N, P, n_fill]
            loss = (F.cross_entropy(sel.flatten(0, 1), f.flatten(), reduction="none") * m.flatten()).sum() / m.sum()
            if self.len_head is not None:
                loss = loss + F.cross_entropy(self.len_head(x), length_idx)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def exact(self, x, f, r, m, length_idx):
        ok = torch.ones(len(x), dtype=torch.bool, device=x.device)
        if self.len_head is not None:
            ok &= self.len_head(x).argmax(-1) == length_idx
        pred = self.logits(x)[torch.arange(len(x), device=x.device)[:, None], r].argmax(-1)
        ok &= ((pred == f) | (m == 0)).all(-1)
        return ok

    @torch.no_grad()
    def role_accuracy(self, x, f, r, m):
        """Secondary metric (amendment after the smoke test): mean per-role filler accuracy over present roles."""
        pred = self.logits(x)[torch.arange(len(x), device=x.device)[:, None], r].argmax(-1)
        return ((pred == f).float() * m).sum(-1) / m.sum(-1)


@torch.no_grad()
def splice_kl(model, tok, rows, layer, vectors, device, batch=128):
    """KL(orig || spliced) of the next-token distribution at the period, writing `vectors` into block `layer`'s output."""
    kls = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        enc = tok([r["text"] for r in chunk], return_tensors="pt", padding=True).to(device)
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(len(last), device=device)
        base = model(**enc).logits[idx, last].float().log_softmax(-1)
        vec = vectors[start:start + batch].to(device)

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            hs[idx, last] = vec.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        handle = model.transformer.h[layer].register_forward_hook(hook)
        try:
            new = model(**enc).logits[idx, last].float().log_softmax(-1)
        finally:
            handle.remove()
        kls.append((base.exp() * (base - new)).sum(-1).cpu())
    return torch.cat(kls)


def latent_structure(codes, f, r, m, n_fill, n_role, min_active=50, min_count=3):
    """Per active latent: share of between-cell variance of mean activation over the (filler, role) grid that a
    row+column (additive) model leaves unexplained (interaction share), plus filler-only / role-only R²."""
    codes, f, r, m = codes.cpu().numpy(), f.cpu().numpy(), r.cpu().numpy(), m.cpu().numpy()
    out = []
    for j in np.flatnonzero((codes > 0).sum(0) >= min_active):
        a = codes[:, j]
        sums = np.zeros((n_fill, n_role))
        counts = np.zeros((n_fill, n_role))
        for p in range(f.shape[1]):
            keep = m[:, p] > 0
            np.add.at(sums, (f[keep, p], r[keep, p]), a[keep])
            np.add.at(counts, (f[keep, p], r[keep, p]), 1)
        cells = counts >= min_count
        if cells.sum() < 10:
            continue
        mean = np.where(cells, sums / np.maximum(counts, 1), 0.0)
        w = np.where(cells, counts, 0.0)
        grand = (mean * w).sum() / w.sum()
        ss_tot = (w * (mean - grand) ** 2).sum()
        if ss_tot <= 1e-12:
            continue
        row, col = np.zeros(n_fill), np.zeros(n_role)
        for _ in range(50):  # weighted two-way additive fit by backfitting
            row = np.where(w.sum(1) > 0, (w * (mean - grand - col[None, :])).sum(1) / np.maximum(w.sum(1), 1e-12), 0)
            col = np.where(w.sum(0) > 0, (w * (mean - grand - row[:, None])).sum(0) / np.maximum(w.sum(0), 1e-12), 0)
        fit = grand + row[:, None] + col[None, :]
        r_add = 1 - (w * (mean - fit) ** 2).sum() / ss_tot
        r_row = (w * (row[:, None]) ** 2).sum() / ss_tot
        r_col = (w * (col[None, :]) ** 2).sum() / ss_tot
        out.append(dict(interaction=float(1 - r_add), filler_r2=float(r_row), role_r2=float(r_col)))
    return out


def summarize_latents(lat):
    if not lat:
        return dict(n=0)
    inter = np.array([x["interaction"] for x in lat])
    fr = np.array([x["filler_r2"] for x in lat])
    rr = np.array([x["role_r2"] for x in lat])
    return dict(n=len(lat), conjunctive_frac=float((inter > 0.5).mean()), median_interaction=float(np.median(inter)),
                filler_dominant_frac=float(((fr > 0.5) & (inter <= 0.5)).mean()),
                role_dominant_frac=float(((rr > 0.5) & (inter <= 0.5)).mean()))


def run_family(name, rows, n_fill, n_role, withheld, model, tok, layers, device, args, log):
    rng = random.Random(args.seed + 1)
    order = list(range(len(rows)))
    rng.shuffle(order)
    n_read, n_dict = int(0.33 * len(rows)), int(0.42 * len(rows))
    read_rows = [rows[i] for i in order[:n_read]]
    dict_rows_all = [rows[i] for i in order[n_read:n_read + n_dict]]
    test_rows = [rows[i] for i in order[n_read + n_dict:]]
    dict_rows = [r for r in dict_rows_all if not any(tuple(p) in withheld for p in r["pairs"])]
    strata = torch.tensor([min(2, sum(tuple(p) in withheld for p in r["pairs"])) for r in test_rows])
    lengths = sorted({r["length"] for r in rows})
    lidx = lambda rs: torch.tensor([lengths.index(r["length"]) for r in rs], device=device)  # noqa: E731
    log(f"[{name}] readout {len(read_rows)}  dict {len(dict_rows)} (of {len(dict_rows_all)})  test {len(test_rows)} "
        f"strata {[int((strata == k).sum()) for k in range(3)]}")
    all_rows = read_rows + dict_rows + test_rows
    acts = period_activations(model, tok, all_rows, layers, device)
    sl = dict(read=slice(0, len(read_rows)), dict=slice(len(read_rows), len(read_rows) + len(dict_rows)),
              test=slice(len(read_rows) + len(dict_rows), len(all_rows)))
    pf, pr, pm = (t.to(device) for t in pad_pairs(all_rows))
    used_pairs = len({tuple(p) for r in dict_rows for p in r["pairs"]})
    results = {}
    for layer in layers:
        x_raw = acts[layer].to(device)
        mu, sd = x_raw[sl["dict"]].mean(0), x_raw[sl["dict"]].std(0) + 1e-6
        x = (x_raw - mu) / sd
        readout = Readout(x.shape[1], n_fill, n_role, lengths).to(device)
        readout.fit(x[sl["read"]], pf[sl["read"]], pr[sl["read"]], pm[sl["read"]], lidx(read_rows))
        tf, tr, tm, tl = pf[sl["test"]], pr[sl["test"]], pm[sl["test"]], lidx(test_rows)
        xt = x[sl["test"]]
        var = ((xt - xt.mean(0)) ** 2).sum()
        kl_rows = test_rows[:args.kl_n]
        kl_mean = splice_kl(model, tok, kl_rows, layer, mu.expand(len(kl_rows), -1).cpu(), device)

        def evaluate(recon, n_params):
            ok = readout.exact(recon, tf, tr, tm, tl).cpu()
            racc = readout.role_accuracy(recon, tf, tr, tm).cpu()
            kl = splice_kl(model, tok, kl_rows, layer, (recon[:len(kl_rows)] * sd + mu).cpu(), device)
            return dict(params=int(n_params), fvu=float(((xt - recon) ** 2).sum() / var),
                        readout=float(ok.float().mean()), role_acc=float(racc.mean()),
                        role_acc_by_withheld={str(k): float(racc[strata == k].mean()) if (strata == k).any() else None
                                              for k in range(3)},
                        readout_by_withheld={str(k): float(ok[strata == k].float().mean()) if (strata == k).any() else None
                                             for k in range(3)},
                        kl_recovered=float(1 - kl.mean() / kl_mean.mean()))

        real_ok = readout.exact(xt, tf, tr, tm, tl).cpu()
        real_racc = readout.role_accuracy(xt, tf, tr, tm).cpu()
        res = dict(real=dict(readout=float(real_ok.float().mean()), role_acc=float(real_racc.mean()),
                             role_acc_by_withheld={str(k): float(real_racc[strata == k].mean())
                                                   if (strata == k).any() else None for k in range(3)},
                             readout_by_withheld={str(k): float(real_ok[strata == k].float().mean())
                                                  if (strata == k).any() else None for k in range(3)}),
                   kl_mean_splice=float(kl_mean.mean()), dictionaries={})
        df, dr, dm, xd = pf[sl["dict"]], pr[sl["dict"]], pm[sl["dict"]], x[sl["dict"]]
        tpr_params = None
        for kind in ("tpr", "atomic", "additive", "bag"):
            torch.manual_seed(args.seed)
            sup = Supervised(kind, n_fill, n_role, x.shape[1]).to(device)
            fit_supervised(sup, df, dr, dm, xd, args.steps)
            with torch.no_grad():
                recon = sup(tf, tr, tm)
            params = sup.n_params(used_pairs)
            tpr_params = params if kind == "tpr" else tpr_params
            res["dictionaries"][kind] = evaluate(recon, params)
        matched = max(8, round(tpr_params / (2 * x.shape[1] + 1)))
        for width in (matched, 1024, 8192):
            torch.manual_seed(args.seed)
            k = min(32, max(1, width // 4))
            sae = TopKSAE(x.shape[1], width, k).to(device)
            fit_sae(sae, xd, args.sae_steps, seed=args.seed)
            with torch.no_grad():
                recon = sae(xt)
                codes = sae.codes(torch.cat([xd, xt]))
            entry = evaluate(recon, sae.n_params())
            entry.update(width=width, k=k, matched=width == matched,
                         dead_frac=float(((codes > 0).sum(0) == 0).float().mean()),
                         latents=summarize_latents(latent_structure(
                             codes, torch.cat([df, tf]), torch.cat([dr, tr]), torch.cat([dm, tm]), n_fill, n_role)))
            res["dictionaries"][f"sae-{width}"] = entry
        results[str(layer)] = res
        d = res["dictionaries"]
        log(f"[{name} L{layer}] real ro={res['real']['readout']:.3f} ra={res['real']['role_acc']:.3f} | " + " | ".join(
            f"{k} ro={v['readout']:.3f} ra={v['role_acc']:.3f} fvu={v['fvu']:.3f} kl={v['kl_recovered']:.3f}" for k, v in d.items()))
    return dict(n_fill=n_fill, n_role=n_role, n_withheld=len(withheld), used_pairs=used_pairs,
                split=dict(readout=len(read_rows), dict=len(dict_rows), dict_before_filter=len(dict_rows_all),
                           test=len(test_rows), test_strata=[int((strata == k).sum()) for k in range(3)]),
                layers=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9, 11])
    ap.add_argument("--list-n", type=int, default=24000)
    ap.add_argument("--svo-n", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--sae-steps", type=int, default=4000)
    ap.add_argument("--kl-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/tpr/tpr_vs_sae_summary.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    nouns, occupations, verbs = single_token(tok, NOUNS, 100), single_token(tok, OCCUPATIONS, 40), single_token(tok, VERBS, 16)
    log(f"vocab: {len(nouns)} nouns, {len(occupations)} occupations, {len(verbs)} verbs; device {device}")
    rng = random.Random(args.seed)
    list_rows, list_roles = make_list(rng, nouns, args.list_n)
    svo_rows, svo_roles = make_svo(rng, occupations, verbs, args.svo_n)
    # Withheld pairs are drawn BEFORE any split or fit.
    wrng = random.Random(args.seed + 100)
    list_pairs = [(a, s) for a in range(len(nouns)) for s in range(len(list_roles))]
    svo_pairs = [(a, s) for a in range(len(occupations)) for s in (0, 2)]
    list_withheld = set(wrng.sample(list_pairs, len(list_pairs) // 10))
    svo_withheld = set(wrng.sample(svo_pairs, len(svo_pairs) // 10))
    summary = dict(tag="empirical", prereg="docs/TPR_VS_SAE.md", model="gpt2", args=vars(args), device=device,
                   vocab=dict(nouns=nouns, occupations=occupations, verbs=verbs))
    summary["LIST"] = run_family("LIST", list_rows, len(nouns), len(list_roles), list_withheld, model, tok,
                                 args.layers, device, args, log)
    summary["SVO"] = run_family("SVO", svo_rows, len(occupations) + len(verbs), 3, svo_withheld, model, tok,
                                args.layers, device, args, log)
    summary["seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
