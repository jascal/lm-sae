"""Explore shared write directions from fieldrun source dumps across languages.

This is a new measurement on final-norm-folded, per-position writes. It is not
numerically interchangeable with core_grammar.py's raw, all-position HF hooks.
See docs/GRAMMAR_FIELD_RUN.md for the collection protocol.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


LAYER = re.compile(r"^L(\d+)\.(attn|mlp)$")
PUNCT = list(".,;:!?()[]{}\"'—-…")


def read_dump(path: Path, max_positions: int) -> tuple[np.ndarray, list[str], list[str]]:
    """Return [positions, layers, dim] writes, sentence ids, and block names."""
    rows, sids = [], []
    labels = None
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if labels is None:
                labels = row["blocks"]
            if row["blocks"] != labels:
                raise ValueError(f"{path}:{line_no}: block layout changed")
            rows.append(np.asarray(row["d"], dtype=np.float32))
            sids.append(str(row.get("sid", "single")))
            if max_positions and len(rows) >= max_positions:
                break
    if not rows:
        raise ValueError(f"{path}: empty dump")
    D = np.stack(rows)
    if not np.isfinite(D).all():
        raise ValueError(f"{path}: nonfinite write")
    layers = sorted({int(m.group(1)) for label in labels if (m := LAYER.match(label))})
    if layers != list(range(len(layers))):
        raise ValueError(f"{path}: expected contiguous layers, found {layers}")
    writes = []
    for layer in layers:
        idx = [i for i, label in enumerate(labels) if (m := LAYER.match(label)) and int(m.group(1)) == layer]
        if len(idx) != 2:
            raise ValueError(f"{path}: layer {layer} needs attn and mlp writes")
        writes.append(D[:, idx].sum(axis=1))
    return np.stack(writes, axis=1), sids, labels


def fit_basis(writes: np.ndarray, share_rank: int, seed: int = 0) -> tuple[np.ndarray, int]:
    """Per-layer PCA followed by the original stacked-basis SVD.

    Use exact small-sample Gram PCA, then randomized range finding when a
    large dump would make a dense position covariance expensive.
    """
    n, n_layers, d = writes.shape
    cols = []
    rng = np.random.default_rng(seed)
    for layer in range(n_layers):
        X = writes[:, layer, :].astype(np.float64)
        if d <= 256 and n > d:
            _, _, vt = np.linalg.svd(X, full_matrices=False)
            basis = vt[:share_rank].T
        elif n <= 256:
            gram = X @ X.T
            vals, vecs = np.linalg.eigh(gram)
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:, order]
            keep = vals > max(vals[0], 0) * 1e-10
            vals, vecs = vals[keep][:share_rank], vecs[:, keep][:, :share_rank]
            basis = X.T @ (vecs / np.sqrt(vals)[None, :])
        else:
            sketch = min(min(n, d), share_rank + 16)
            Y = X @ rng.standard_normal((d, sketch))
            for _ in range(2):
                Y = X @ (X.T @ Y)
                Y, _ = np.linalg.qr(Y, mode="reduced")
            Q, _ = np.linalg.qr(Y, mode="reduced")
            _, _, vt = np.linalg.svd(Q.T @ X, full_matrices=False)
            basis = vt[:share_rank].T
        cols.append(basis)
    stack = np.concatenate(cols, axis=1)
    U, sv, _ = np.linalg.svd(stack, full_matrices=False)
    sq = sv * sv
    effective_rank = int(round(float(sq.sum() ** 2 / np.square(sq).sum())))
    return U, effective_rank


def overlap(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.square(A.T @ B).sum() / B.shape[1])


def token_ids(tok: Tokenizer, words: list[str]) -> tuple[set[int], set[int]]:
    out, punctuation = set(), set()
    for word in words:
        for form in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            ids = tok.encode(form, add_special_tokens=False).ids
            if len(ids) == 1:
                out.add(ids[0])
    for mark in PUNCT + ["\n", "\n\n"]:
        for form in (mark, " " + mark):
            ids = tok.encode(form, add_special_tokens=False).ids
            if len(ids) == 1:
                punctuation.add(ids[0])
    return out - punctuation, punctuation


def lens_fraction(U: np.ndarray, W: np.ndarray, ids: set[int], topn: int = 10) -> float:
    if not ids:
        return float("nan")
    logits = W @ U
    mean = logits.mean(axis=0)
    positive = (logits.max(axis=0) - mean) >= (mean - logits.min(axis=0))
    top = np.argpartition(logits, -topn, axis=0)[-topn:]
    bottom = np.argpartition(logits, topn - 1, axis=0)[:topn]
    selected = np.where(positive[None, :], top, bottom)
    # The peak side uses the same sign convention as core_grammar.py.
    return float(np.isin(selected, list(ids)).mean())


def lens_examples(U: np.ndarray, W: np.ndarray, tok: Tokenizer, n: int = 8) -> list[dict]:
    examples = []
    for k in range(min(n, U.shape[1])):
        logits = W @ U[:, k]
        mean = logits.mean()
        side = 1 if logits.max() - mean >= mean - logits.min() else -1
        ids = np.argsort(-(side * logits))[:10]
        examples.append({"direction": k, "tokens": [
            {"id": int(i), "text": tok.decode([int(i)], skip_special_tokens=False)[:80]}
            for i in ids]})
    return examples


def split_indices(sids: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    unique = list(dict.fromkeys(sids))
    if len(unique) < 4:
        return None
    first = set(unique[::2])
    left = np.array([i for i, sid in enumerate(sids) if sid in first])
    right = np.array([i for i, sid in enumerate(sids) if sid not in first])
    return left, right


def run(args: argparse.Namespace) -> dict:
    corpora = {}
    for spec in args.corpus:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path or name in corpora:
            raise ValueError(f"invalid or duplicate --corpus NAME=PATH: {spec}")
        corpora[name] = Path(path)
    if len(corpora) < 2:
        raise ValueError("at least two corpora are required")
    W = np.load(args.tensors)["U"].astype(np.float64)
    tok = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    if W.shape[0] < vocab_size:
        raise ValueError("unembedding has fewer rows than the tokenizer vocabulary")
    W = W[:vocab_size]
    lexicons = json.loads(args.lexicons.read_text()) if args.lexicons else {}
    bases, splits, counts, grammar, examples, lexicon_ids = {}, {}, {}, {}, {}, {}
    d = W.shape[1]
    if max(args.bins) > d:
        raise ValueError("largest bin exceeds model width")
    rng = np.random.default_rng(args.seed)
    random_basis, _ = np.linalg.qr(rng.standard_normal((d, max(args.bins))))
    for name, path in corpora.items():
        writes, sids, _ = read_dump(path, args.max_positions)
        if writes.shape[2] != d:
            raise ValueError(f"{name}: dump dimension {writes.shape[2]} != unembedding {d}")
        if writes.shape[0] < args.share_rank:
            raise ValueError(f"{name}: need at least {args.share_rank} positions")
        U, K = fit_basis(writes, args.share_rank, args.seed)
        bases[name] = U
        examples[name] = lens_examples(U, W, tok)
        counts[name] = {"positions": len(sids), "sentences": len(set(sids)), "effective_rank": K}
        pair = split_indices(sids)
        if pair is not None and min(map(len, pair)) >= args.share_rank:
            splits[name] = (fit_basis(writes[pair[0]], args.share_rank, args.seed)[0],
                            fit_basis(writes[pair[1]], args.share_rank, args.seed)[0])
        if name in lexicons:
            words, punct = token_ids(tok, lexicons[name])
            lexicon_ids[name] = words
            grammar[name] = {"word_ids": len(words), "punctuation_ids": len(punct),
                             "word_vocab_fraction": len(words) / vocab_size,
                             "bins": {}}
            for lo, hi in zip((0,) + tuple(args.bins[:-1]), args.bins):
                if hi > U.shape[1]:
                    continue
                grammar[name]["bins"][f"{lo}:{hi}"] = {
                    "closed_word": lens_fraction(U[:, lo:hi], W, words),
                    "punctuation": lens_fraction(U[:, lo:hi], W, punct),
                    "random_word_direction": lens_fraction(random_basis[:, lo:hi], W, words),
                }
    transfer = {}
    for source, U in bases.items():
        transfer[source] = {}
        for language, ids in lexicon_ids.items():
            transfer[source][language] = lens_fraction(U[:, :min(16, U.shape[1])], W, ids)
    overlap_rows = []
    names = list(corpora)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for lo, hi in zip((0,) + tuple(args.bins[:-1]), args.bins):
                if hi > min(bases[a].shape[1], bases[b].shape[1]):
                    continue
                within = {}
                for name in (a, b):
                    if name in splits:
                        x, y = splits[name]
                        within[name] = overlap(x[:, lo:hi], y[:, lo:hi])
                overlap_rows.append({"a": a, "b": b, "bin": f"{lo}:{hi}",
                                     "cross": overlap(bases[a][:, lo:hi], bases[b][:, lo:hi]),
                                     "random_subspace": (hi - lo) / d,
                                     "within_split": within})
    return {"method": "fieldrun final-norm-folded per-position layer writes; uncentered stacked PCA",
            "tensors": str(args.tensors), "tokenizer": str(args.tokenizer),
            "corpora": {name: str(path) for name, path in corpora.items()},
            "counts": counts, "overlap": overlap_rows, "closed_class": grammar,
            "direction_examples": examples, "word_transfer_top16": transfer}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", action="append", required=True, help="NAME=fieldrun-source-dump.jsonl")
    p.add_argument("--tensors", type=Path, required=True, help="fieldrun --tensors-export NPZ")
    p.add_argument("--tokenizer", type=Path, required=True, help="bundle.tokenizer.json")
    p.add_argument("--lexicons", type=Path, help="JSON mapping corpus name to closed-class word list")
    p.add_argument("--share-rank", type=int, default=32)
    p.add_argument("--bins", type=int, nargs="+", default=[16, 64, 128])
    p.add_argument("--max-positions", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.bins != sorted(set(args.bins)) or args.bins[0] < 1:
        p.error("--bins must be strictly increasing positive integers")
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(result['overlap'])} comparisons to {args.out}")


if __name__ == "__main__":
    main()
