"""Bridge the universal-grammar track to the fieldrun runtime (see docs/GRAMMAR_FIELD_RUN.md).

  prepare : write fieldrun `--texts` JSONL rows (exact `ids`, the same leading special token as
            ug_extract, plus one trailing token so the last word is a scored position) for PUD
            held-out sentences, and a side file with each word's token position + gold UPOS.
  parity  : rebuild the layer-L residual from fieldrun's final-norm-folded block writes,
            r_L = (embed + Σ_{blocks < L} d̃_b) / γ  (d̃_b = γ ⊙ w_b / rms(h_final), so r_L ∝ h_L per position),
            and compare with HF hidden_states[L] of the same checkpoint at the same positions (cosine).

  .venv/bin/python scripts/grammar/ug_fieldrun.py prepare --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
      --langs en de es ja zh --n 40 --out /tmp/ug-fieldrun
  fieldrun --bundle <stem> --recursion-explain --texts /tmp/ug-fieldrun/en.jsonl --source-dump /tmp/ug-fieldrun/en.dump.jsonl --n 400
  .venv/bin/python scripts/grammar/ug_fieldrun.py parity --model Qwen/Qwen2.5-0.5B-Instruct --dir /tmp/ug-fieldrun \
      --tensors /tmp/ug-fieldrun/tensors.npz --layer 12 --langs en de es ja zh --out runs/grammar/fieldrun_parity_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ug_extract import PUD, prefix_ids, read_conllu, spans


def prepare(a) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    pre = prefix_ids(tok)
    tail = tok("\n", add_special_tokens=False)["input_ids"]
    a.out.mkdir(parents=True, exist_ok=True)
    for lang in a.langs:
        rows, words = [], {}
        for si, s in enumerate(read_conllu(PUD / f"{lang}_pud.conllu")):
            if si % 5 != 0 or len(rows) >= a.n:
                continue
            sp = spans(s)
            if sp is None:
                continue
            enc = tok(s["text"], add_special_tokens=True, return_offsets_mapping=True)
            ids, offs = pre + enc["input_ids"], [(0, 0)] * len(pre) + enc["offset_mapping"]
            sid = f"{lang}-{si}"
            words[sid] = []
            for a_, b_, widx in sp:
                hits = [t for t, (x, y) in enumerate(offs) if y > x and x < a_ + (b_ - a_) and y > a_]
                if hits and len(widx) == 1:
                    words[sid].append({"pos": hits[-1], "upos": s["words"][widx[0]]["upos"]})
            rows.append({"sid": sid, "ids": ids + tail})
        (a.out / f"{lang}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (a.out / f"{lang}.words.json").write_text(json.dumps(words))
        print(f"{lang}: {len(rows)} sentences, {sum(len(r['ids']) for r in rows)} tokens")


def parity(a) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    gamma = np.load(a.tensors)["gamma"].astype(np.float64)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, device_map="cuda").eval()
    res = {"model": a.model, "layer": a.layer, "per_lang": {}}
    for lang in a.langs:
        rows = {json.loads(l)["sid"]: json.loads(l)["ids"] for l in open(a.dir / f"{lang}.jsonl")}
        cos, upos_cos = [], []
        words = json.loads((a.dir / f"{lang}.words.json").read_text())
        dump: dict[str, dict[int, np.ndarray]] = {}
        keep_blocks = None
        with open(a.dir / f"{lang}.dump.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if keep_blocks is None:
                    keep_blocks = [i for i, b in enumerate(r["blocks"])
                                   if b == "embed" or int(b[1:].split(".")[0]) < a.layer]
                d = np.asarray(r["d"], dtype=np.float64)[keep_blocks].sum(0) / gamma
                dump.setdefault(r["sid"], {})[r["pos"]] = d
        for sid, ids in rows.items():
            with torch.no_grad():
                h = model(input_ids=torch.tensor([ids]).cuda(), output_hidden_states=True).hidden_states[a.layer][0]
            h = h.double().cpu().numpy()
            wpos = {w["pos"] for w in words.get(sid, [])}
            for pos, r in dump.get(sid, {}).items():
                c = float(r @ h[pos] / (np.linalg.norm(r) * np.linalg.norm(h[pos]) + 1e-12))
                cos.append(c)
                if pos in wpos:
                    upos_cos.append(c)
        res["per_lang"][lang] = {"positions": len(cos), "cos_mean": float(np.mean(cos)), "cos_min": float(np.min(cos)),
                                 "word_positions": len(upos_cos)}
        print(lang, res["per_lang"][lang])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--tokenizer", required=True)
    q.add_argument("--langs", nargs="+", required=True)
    q.add_argument("--n", type=int, default=40)
    q.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("parity")
    q.add_argument("--model", required=True)
    q.add_argument("--dir", type=Path, required=True)
    q.add_argument("--tensors", type=Path, required=True)
    q.add_argument("--layer", type=int, default=12)
    q.add_argument("--langs", nargs="+", required=True)
    q.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prepare(a) if a.mode == "prepare" else parity(a)


if __name__ == "__main__":
    main()
