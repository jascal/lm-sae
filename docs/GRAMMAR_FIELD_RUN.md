# Testing the proposed shared grammar with fieldrun

The result in [FINDINGS.md](FINDINGS.md) compares two English texts and Python
source. It is a cross-corpus result, not a cross-language result. This workflow
tests whether the same **model-internal write subspace** persists across
languages, and whether its directions favor each language's function words.

`scripts/disassembly/fieldrun_grammar.py` consumes `fieldrun --source-dump`
JSONL and `--tensors-export` NPZ. It never loads the model in Python. The dump
contains final-norm-folded, **per-position** attn and MLP writes. The script
sums the two writes for each layer, fits uncentered layer PCAs, stacks the
leading layer bases, and measures subspace overlap in rank bins. This is a
related experiment, not a numeric reproduction of the raw-write,
all-position Hugging Face hook in `core_grammar.py`.

## Collect matched data

Use one multilingual model and its own tokenizer for every language. Prepare a
JSONL file with `{"sid":"en-001","text":"..."}` rows for each language.
Collect at least four independent documents and several hundred scored tokens
per language. Use the same genre, approximate token count, and sampling method
across languages; keep documents disjoint between fitting and holdout.
`--source-dump` runs a forward pass for every scored position, so start with
short passages and increase the sample size after a pilot. Keep `--n` equal
across languages. Do not use the opening of one book as the entire corpus.

Example using the existing Llama bundle (substitute a larger multilingual
bundle for the main result):

```bash
FR=/home/allans/code/fieldrun/target/release/fieldrun
B=/home/allans/code/fieldrun/bundles/llama-3.2-1b
OUT=/tmp/grammar-fieldrun
mkdir -p "$OUT"

"$FR" --bundle "$B" --recursion-explain --tensors-export "$OUT/tensors.npz"
"$FR" --bundle "$B" --recursion-explain --texts corpora/en.jsonl --source-dump "$OUT/en.jsonl" --n 64
"$FR" --bundle "$B" --recursion-explain --texts corpora/fr.jsonl --source-dump "$OUT/fr.jsonl" --n 64
"$FR" --bundle "$B" --recursion-explain --texts corpora/de.jsonl --source-dump "$OUT/de.jsonl" --n 64

.venv/bin/python scripts/disassembly/fieldrun_grammar.py \
  --corpus en="$OUT/en.jsonl" --corpus fr="$OUT/fr.jsonl" --corpus de="$OUT/de.jsonl" \
  --tensors "$OUT/tensors.npz" --tokenizer "$B.tokenizer.json" \
  --lexicons docs/grammar_closed_class.json --out "$OUT/report.json"
```

The lexicon is an **explicit probe**, not a grammar definition. `closed_class`
reports separate function-word and punctuation fractions among each
direction's ten most extreme unembedding tokens. Compare function words with
the random-direction control and the lexicon's raw vocabulary fraction.
`word_transfer_top16` scores each fitted basis against *every* language's
function words; shared geometry alone is insufficient if the same space
does not favor words from both languages. Lexicon coverage and tokenization
differ by language, so raw fractions across languages are not directly
comparable. Inspect `direction_examples` before naming axes.

## Read the report

- `overlap.cross` is the cross-language subspace overlap. `random_subspace` is
  the geometric expectation for independent random subspaces of that bin size.
- `within_split` fits separate document halves of each language. A convincing
  shared subspace should retain a substantial fraction of within-language
  overlap across languages; a large ratio over random alone is insufficient.
- The first 16 directions are a predeclared hypothesis from FINDINGS. Read the
  later bins too: the existing Llama result shows considerable closed-class
  enrichment beyond direction 16.

The next causal step is to project the candidate subspace out of **layer
writes** during inference and measure the change in held-out grammatical
choices versus matched content choices, with equal-rank random and adjacent
rank-bin controls. The current `fieldrun --source-dump` is observational and
cannot perform that intervention. A rosetta grammar/skeleton rule should be
admitted only after it predicts held-out choices and survives the existing
causal-confirmation and certificate path.

## Plumbing check

I ran the export and dump path on the installed Pythia-70m bundle using four
short authored English sentences and four French sentences, eight scored
positions each. `fieldrun` reconstructed its own top candidate at all 32
positions in each language, and the analyzer produced cross-language overlap,
document-split overlap, and token examples. In this tiny check the first
direction's extreme tokens were mostly whitespace runs, and the closed-class
score was zero. This confirms that the pipeline runs; it is **not** evidence
for a shared grammar. A larger multilingual model and balanced natural text
are needed to test the claim.

## Result on balanced multilingual text (UD PUD)

Run on Qwen2.5-0.5B-Instruct through fieldrun, 40 held-out PUD sentences each in en/de/es/ja/zh (parallel
translations, so content is matched across languages; `scripts/grammar/ug_fieldrun.py` prepares the exact-id rows).
fieldrun's layer-12 residual, rebuilt from the folded writes and divided by γ, matches HF at mean cosine 0.9996
(`runs/grammar/fieldrun_parity_summary.json`).

- **Shared subspace: supported.** Top-16 cross-language overlap 0.68, against 0.74 within a language (document
  halves) and 0.02 for random subspaces. That is 93% of within-language overlap, including ja and zh.
- **Closed-class lexicon through the unembedding: not supported.** Function-word fraction 0.00–0.04 per bin,
  indistinguishable from the random-direction control, and the extreme tokens are junk.

So the shared write subspace is universal, but it is not readable as "directions that vote for function words".
The grammar is relational geometry. [UNIVERSAL_GRAMMAR.md](UNIVERSAL_GRAMMAR.md) reads it with probes and LEACE
erasure across four models and 13 languages (`runs/grammar/fieldrun_grammar_pud_summary.json`).
