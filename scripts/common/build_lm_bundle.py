"""Build an lm-sae ground-truth bundle: GPT-2 residual-stream activations + an
EXACT-lexical feature oracle (Recipe A, the frozen-LLM / preserve regime).

The whole point of the *-sae substrates is a *known* feature factorization to grade
the SAE/forge against. Real LLMs have no oracle (manifesto: "no oracle"). This
manufactures a partial-but-EXACT one: per GPT-2 token we compute deterministic
lexical labels (no tagger, no noise) — specific-token detectors, capitalization,
punctuation, digits, length, word-boundary. cov95 then asks: does a single SAE
latent on GPT-2's activations detect "is the token ' the'", "is capitalized",
"is punctuation" at AUC >= 0.95?

Tiers (the sharp->diffuse axis, like bio's Pfam/GO):
  token   - "this token == <specific common token>"  (sharpest; one-token detectors)
  lexical - capitalization / punctuation / digit / length buckets (sharp)
  struct  - word-boundary (leading space) / newline                (medium)

Outputs data/lm_bundle_gpt2.npz (X float32 (N,d), Y uint8 (N,M)) + lm_labels.json
(feature_vocab, tiers, token strings). Self-contained: only torch + transformers,
GPT-2 from the HF cache. (Self-trained SAE downstream stands in for a SAELens
production dictionary until sae_lens is installed.)
"""
from __future__ import annotations

import argparse
import json
import string
import urllib.request
from pathlib import Path

import numpy as np

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FALLBACK = (
    "In 1492 Columbus sailed the ocean blue. The Dow rose 3.2% on Tuesday, "
    "to 34,012 points. Dr. Smith paid $45 for 12 apples in New York. "
    "WARNING: do not press the red button! Is it 5 o'clock yet? "
) * 200

# specific common GPT-2 tokens to build one-token detectors for (the sharp tier)
COMMON = [" the", " of", " and", " to", " a", " in", " that", " is", " was",
          " for", " it", " he", " his", " with", " as", ".", ",", " I", "\n", ";"]


def _fetch_corpus(max_chars: int) -> str:
    try:
        with urllib.request.urlopen(CORPUS_URL, timeout=8) as r:
            return r.read().decode("utf-8", "ignore")[:max_chars]
    except Exception as e:
        print(f"    [warn] corpus fetch failed ({type(e).__name__}); using fallback")
        return FALLBACK[:max_chars]


def _lexical_features(tok_str: str) -> dict:
    """Deterministic per-token labels from the decoded token string."""
    s = tok_str.replace("Ġ", " ")  # GPT-2 BPE word-boundary marker
    core = s.strip()
    alpha = [c for c in core if c.isalpha()]
    return {
        "lex:has_leading_space": int(tok_str.startswith("Ġ")),
        "lex:is_capitalized": int(bool(alpha) and alpha[0].isupper()),
        "lex:is_all_caps": int(bool(alpha) and all(c.isupper() for c in alpha) and len(alpha) > 1),
        "lex:is_alpha": int(bool(core) and core.isalpha()),
        "lex:has_digit": int(any(c.isdigit() for c in core)),
        "lex:is_punct": int(bool(core) and all(c in string.punctuation for c in core)),
        "struct:has_newline": int("\n" in tok_str),
        "lex:len1": int(len(core) == 1),
        "lex:len2": int(len(core) == 2),
        "lex:len3_4": int(3 <= len(core) <= 4),
        "lex:len5p": int(len(core) >= 5),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--layer", type=int, default=6, help="residual-stream layer (hidden_states index)")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--max-chars", type=int, default=120000)
    p.add_argument("--min-pos", type=int, default=40)
    p.add_argument("--out-npz", type=Path, default=Path("data/lm_bundle_gpt2.npz"))
    p.add_argument("--out-labels", type=Path, default=Path("data/lm_labels.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import GPT2TokenizerFast, GPT2Model

    print(f"[1] load {args.model} (from HF cache)")
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2Model.from_pretrained(args.model, output_hidden_states=True).eval()
    d_model = model.config.n_embd

    print("[2] corpus + tokenize")
    text = _fetch_corpus(args.max_chars)
    ids = tok(text)["input_ids"][: args.max_tokens]
    print(f"    {len(ids)} tokens, ctx={args.ctx}, layer={args.layer}/{model.config.n_layer}")

    print("[3] extract residual-stream activations")
    acts, all_ids = [], []
    with torch.no_grad():
        for i in range(0, len(ids), args.ctx):
            chunk = ids[i: i + args.ctx]
            t = torch.tensor([chunk])
            h = model(input_ids=t).hidden_states[args.layer][0]  # (seq, d)
            acts.append(h.float().numpy())
            all_ids.extend(chunk)
    X = np.concatenate(acts, axis=0).astype(np.float32)          # (N, d)
    tok_strs = tok.convert_ids_to_tokens(all_ids)
    decoded = [tok.convert_tokens_to_string([t]) for t in tok_strs]
    N = X.shape[0]
    print(f"    X={X.shape}")

    print("[4] build EXACT-lexical oracle Y")
    feat_names: list[str] = []
    cols: list[np.ndarray] = []
    tiers: list[str] = []
    # one-token detectors (sharp 'token' tier)
    id_by_tok = {t: i for i, t in enumerate(tok_strs)}  # noqa: F841 (kept for debug)
    common_ids = {c: tok(c, add_special_tokens=False)["input_ids"] for c in COMMON}
    for c, cid in common_ids.items():
        if len(cid) != 1:
            continue
        col = np.array([1 if j == cid[0] else 0 for j in all_ids], dtype=np.uint8)
        feat_names.append(f"token:{c!r}"); cols.append(col); tiers.append("token")
    # lexical/struct features
    lex_rows = [_lexical_features(s) for s in tok_strs]
    for name in lex_rows[0]:
        col = np.array([r[name] for r in lex_rows], dtype=np.uint8)
        feat_names.append(name); cols.append(col)
        tiers.append("struct" if name.startswith("struct") else "lexical")

    Y = np.stack(cols, axis=1)                                   # (N, M)
    # prevalence filter (need enough +/- like bio's min-n-pos)
    npos = Y.sum(0)
    keep = (npos >= args.min_pos) & (npos <= N - args.min_pos)
    Y = Y[:, keep]
    feat_names = [n for n, k in zip(feat_names, keep) if k]
    tiers = [t for t, k in zip(tiers, keep) if k]
    from collections import Counter
    print(f"    Y={Y.shape}  tiers={dict(Counter(tiers))}")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, X=X, Y=Y)
    args.out_labels.write_text(json.dumps({
        "model": args.model, "layer": args.layer, "d_model": int(d_model),
        "n_samples": int(N), "feature_vocab": feat_names, "tiers": tiers,
        "tokens": decoded,
    }))
    print(f"[done] {args.out_npz}  +  {args.out_labels}")


if __name__ == "__main__":
    main()
