"""Word-aligned hidden states for UD PUD treebanks (the universal-grammar track).

PUD is the same 1000 sentences translated into each language, with gold UPOS /
head / deprel. For every syntactic word we take the hidden state of the LAST
subword token of its surface span (a causal model has then read the whole
word). Multiword tokens (de "zur" = zu der) give every component word the
surface token's state and are flagged `mwt` so category probes can exclude them.

Output (under --out/<model-tag>/): `<lang>.meta.npz` (word metadata) and
`<lang>.L<k>.npy` (fp16 [n_words, d]) for each requested layer; hidden_states
index 0 is the embedding output.

Usage:
  .venv/bin/python scripts/grammar/ug_extract.py --model Qwen/Qwen2.5-0.5B \
      --langs en de ja --layers all --max-sents 400 --out data/ug
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PUD = Path(__file__).resolve().parents[2] / "data" / "ud_pud"


def read_conllu(path: Path) -> list[dict]:
    """Sentences as {sid, text, words:[...], surface:[(form, [word idx])]}."""
    sents, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# sent_id"):
            cur = {"sid": line.split("=", 1)[1].strip(), "words": [], "surface": [], "_mwt": {}}
        elif line.startswith("# text ="):
            cur["text"] = line.split("=", 1)[1].strip()
        elif line and not line.startswith("#"):
            c = line.split("\t")
            if "." in c[0]:
                continue  # empty nodes
            if "-" in c[0]:
                a, b = map(int, c[0].split("-"))
                cur["surface"].append((c[1], list(range(a - 1, b))))
                cur["_mwt"].update({i: True for i in range(a - 1, b)})
                continue
            i = int(c[0]) - 1
            cur["words"].append({"form": c[1], "lemma": c[2], "upos": c[3], "head": int(c[6]) - 1,
                                 "deprel": c[7].split(":")[0], "mwt": i in cur["_mwt"]})
            if i not in cur["_mwt"]:
                cur["surface"].append((c[1], [i]))
        elif not line and cur is not None:
            if cur["words"]:
                sents.append(cur)
            cur = None
    if cur and cur["words"]:
        sents.append(cur)
    return sents


def spans(sent: dict) -> list[tuple[int, int, list[int]]] | None:
    """Char spans of each surface token in the sentence text, or None if unalignable."""
    out, cursor, text = [], 0, sent["text"]
    for form, widx in sent["surface"]:
        at = text.find(form, cursor)
        if at < 0 or text[cursor:at].strip():
            return None
        out.append((at, at + len(form), widx))
        cursor = at + len(form)
    return out


def prefix_ids(tok) -> list[int]:
    """One leading special token so no word sits on position 0 (the attention sink)."""
    probe = tok("a", add_special_tokens=True)["input_ids"]
    if tok.bos_token_id is not None and probe and probe[0] == tok.bos_token_id:
        return []  # the tokenizer already prepends BOS
    special = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    return [special]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", help="output subdir (default: model basename)")
    p.add_argument("--langs", nargs="+", required=True)
    p.add_argument("--layers", default="all", help="'all' or comma list of hidden_states indices")
    p.add_argument("--max-sents", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--random-init", action="store_true", help="control: same architecture, untrained weights")
    p.add_argument("--out", type=Path, default=Path("data/ug"))
    args = p.parse_args()

    tag = args.tag or args.model.rstrip("/").split("/")[-1] + ("-randinit" if args.random_init else "")
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = getattr(torch, args.dtype)
    if args.random_init:
        from transformers import AutoConfig
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.model), torch_dtype=dtype).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda")
    model.eval()
    n_hidden = model.config.get_text_config().num_hidden_layers + 1
    layers = list(range(n_hidden)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    pre = prefix_ids(tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    info = {"model": args.model, "n_hidden": n_hidden, "layers": layers, "prefix_ids": pre,
            "random_init": args.random_init, "langs": {}}

    for lang in args.langs:
        sents = read_conllu(PUD / f"{lang}_pud.conllu")
        if args.max_sents:
            sents = sents[: args.max_sents]
        feats = {l: [] for l in layers}
        meta = {k: [] for k in ("sent", "widx", "upos", "head", "deprel", "mwt", "form")}
        sids, skipped = [], 0
        jobs = []
        for si, s in enumerate(sents):
            sp = spans(s)
            if sp is None:
                skipped += 1
                continue
            enc = tok(s["text"], add_special_tokens=True, return_offsets_mapping=True)
            ids, offs = pre + enc["input_ids"], [(0, 0)] * len(pre) + enc["offset_mapping"]
            wtok = [0] * len(s["words"])
            ok = True
            for a, b, widx in sp:
                hits = [t for t, (x, y) in enumerate(offs) if y > x and x < b and y > a]
                if not hits:
                    ok = False
                    break
                for w in widx:
                    wtok[w] = hits[-1]
            if not ok:
                skipped += 1
                continue
            jobs.append((si, s, ids, wtok))
        for start in range(0, len(jobs), args.batch):
            chunk = jobs[start:start + args.batch]
            L = max(len(j[2]) for j in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), L), dtype=torch.long)
            for r, (_, _, jids, _) in enumerate(chunk):
                ids[r, : len(jids)] = torch.tensor(jids)
                mask[r, : len(jids)] = 1
            with torch.no_grad():
                # base model only: hidden states without the LM head (gemma's 256k-vocab logits are the memory peak)
                hs = model.base_model(input_ids=ids.cuda(), attention_mask=mask.cuda(),
                                      output_hidden_states=True).hidden_states
            for r, (si, s, _, wtok) in enumerate(chunk):
                idx = torch.tensor(wtok, device="cuda")
                for l in layers:
                    feats[l].append(hs[l][r, idx].float().cpu().numpy().astype(np.float16))
                sids.append(s["sid"])
                for w, word in enumerate(s["words"]):
                    meta["sent"].append(si)
                    meta["widx"].append(w)
                    meta["upos"].append(word["upos"])
                    meta["head"].append(word["head"])
                    meta["deprel"].append(word["deprel"])
                    meta["mwt"].append(word["mwt"])
                    meta["form"].append(word["form"])
        for l in layers:
            np.save(out / f"{lang}.L{l}.npy", np.concatenate(feats[l]))
        np.savez(out / f"{lang}.meta.npz", sid=np.array(sids), **{k: np.array(v) for k, v in meta.items()})
        info["langs"][lang] = {"sentences": len(sids), "skipped": skipped, "words": len(meta["upos"])}
        print(f"[{tag}] {lang}: {len(sids)} sents ({skipped} skipped), {len(meta['upos'])} words", flush=True)
    (out / "info.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
