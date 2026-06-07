"""SAE-feature operands per operator (the dossier's section-G gap): what each operator READS and WRITES in
*feature* space, not just token space.

The token-operand catalog says which TOKENS an operator binds; this says which monosemantic **SAE features** it
reads (the concept at the key positions it attends to) and writes (the concept its OV moves it toward), using the
published per-layer GPT-2 SAEs (jbloom/GPT2-Small-SAEs-Reformatted, resid_pre, 24576 feats/layer). For each
operator's top head (from the catalog):

  READ  = the attention-weighted dominant key-feature (the SAE feature most present at the positions the head
          attends to), content-filtered (structural newline/punct features dropped) and glossed by its top tokens.
  WRITE = the feature its OV maps that read-feature toward: argmax_f ( (d_read · OV_h) · W_enc[:,f] ), glossed.

GPT-2 has all-layer SAEs → every operator gets feature operands (Gemma Scope / Qwen are single-layer, a follow-up).
CPU-runnable; downloads SAE weights on demand. Emits runs/.../operator_sae_operands_summary.json + a docs page.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def is_struct(g):
    s = g.replace("_", " ").strip()
    return s == "" or all(c in "\\/|-_=+*.,;:!?'\"()[]{}<>`~@#$%^&" for c in s) or g in ("Ċ", "ĉ")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--sae-repo", default="jbloom/GPT2-Small-SAEs-Reformatted")
    p.add_argument("--atlas", type=Path, default=Path("runs/disassembly/operators/atlas_summary.json"))
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--chunks", type=int, default=40)
    p.add_argument("--top-read", type=int, default=3, help="top read-features to report per operator")
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/operators"))
    p.add_argument("--docs", type=Path, default=Path("docs/operators"))
    args = p.parse_args(argv)

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval().to(dev)
    tr = model.transformer; H = model.config.n_head; hd = model.config.n_embd // H; d = model.config.n_embd
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    WU = model.get_output_embeddings().weight.detach().cpu().numpy().astype(np.float64)   # (vocab, d) unembed (tied)

    atlas = json.loads(args.atlas.read_text())
    g = [r for r in atlas["results"] if r["model"] == args.pretrained][0]
    ops = {op: g["cells"][op]["top_head"] for op in atlas["operators"]}
    for op, heads in atlas.get("gpt2_circuit_ops", {}).items():
        ops[op] = heads[0]
    kind = {**atlas["kinds"], **{op: "circuit" for op in atlas.get("gpt2_circuit_ops", {})}}

    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8][: args.chunks]

    # OV per head (W_V^h W_O^h), folding ln_1 gain into the read-feature direction later
    Wv = [tr.h[L].attn.c_attn.weight.detach().cpu().numpy().astype(np.float64)[:, 2 * d:3 * d] for L in range(model.config.n_layer)]
    Wo = [tr.h[L].attn.c_proj.weight.detach().cpu().numpy().astype(np.float64) for L in range(model.config.n_layer)]

    # group operators by layer so each SAE loads once
    by_layer = {}
    for op, head in ops.items():
        by_layer.setdefault(int(head.split(".")[0]), []).append((op, head))

    results = {}
    for L in sorted(by_layer):
        st = load_file(hf_hub_download(args.sae_repo, f"blocks.{L}.hook_resid_pre/sae_weights.safetensors"))
        Wenc = st["W_enc"].astype(np.float64); benc = st["b_enc"].astype(np.float64)
        Wdec = st["W_dec"].astype(np.float64); bdec = st["b_dec"].astype(np.float64)
        F = Wenc.shape[1]; ln_w = tr.h[L].ln_1.weight.detach().cpu().numpy().astype(np.float64)

        # encode resid_pre[L] over the corpus -> dominant feature per position; collect per-head attention-weighted key-feature
        heads_here = sorted({h for _, hh in by_layer[L] for h in [int(hh.split(".")[1])]})
        read_w = {h: np.zeros(F) for h in heads_here}                                # attention-weighted key-feature mass
        domcount = np.zeros(F); tokfreq = {}                                         # for glossing
        with torch.no_grad():
            for c in chunks:
                o = model(input_ids=torch.tensor([c], device=dev), output_hidden_states=True, output_attentions=True)
                hs = o.hidden_states[L][0].float().cpu().numpy()                     # (Lc, d) resid_pre[L]
                a = np.maximum((hs - bdec) @ Wenc + benc, 0.0)
                dom = np.where(a.max(1) > 0, a.argmax(1), -1)                        # (Lc,) dominant feature per pos
                for f in dom[dom >= 0]:
                    domcount[f] += 1
                for pos, f in enumerate(dom):
                    if f >= 0:
                        tokfreq.setdefault(int(f), Counter())[c[pos]] += 1
                Lc = len(c)
                for h in heads_here:
                    at = o.attentions[L][0, h].float().cpu().numpy()                 # (Lc, Lc)
                    for q in range(1, Lc):
                        kf = dom[:q]
                        m = kf >= 0
                        if m.any():
                            np.add.at(read_w[h], kf[m], at[q, :q][m])

        def gloss(f):
            top = tokfreq.get(int(f), Counter()).most_common(3)
            return "/".join(tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t, _ in top) or "?"

        for op, head in by_layer[L]:
            h = int(head.split(".")[1])
            OV = Wv[L][:, h * hd:(h + 1) * hd] @ Wo[L][h * hd:(h + 1) * hd, :]       # (d, d)
            order = np.argsort(-read_w[h])
            reads = []
            for f in order:
                if read_w[h][f] <= 0 or len(reads) >= args.top_read:
                    break
                gl = gloss(f)
                if is_struct(gl.split("/")[0]):                                      # content-filter the read basis
                    continue
                # copy-score on the feature's own top tokens: attending to token t, does the OV raise (+, copy)
                # or lower (−, copy-suppression) t's own logit? (OV→unembed diagonal, ln-folded on the read side)
                tids = [t for t, _ in tokfreq.get(int(f), Counter()).most_common(3)]
                cs = []
                for t in tids:
                    e = ln_w * WU[t]; out = e @ OV
                    cs.append(float(out @ WU[t] / (np.linalg.norm(out) * np.linalg.norm(WU[t]) + 1e-9)))
                copy = float(np.mean(cs)) if cs else 0.0
                reads.append({"read_feature": int(f), "read_gloss": gl, "attn_mass": float(read_w[h][f]), "copy_score": copy})
            results[op] = {"head": head, "kind": kind.get(op, "?"), "layer": L, "operands": reads}
            top = reads[0] if reads else None
            print(f"  {op:>16} {head:>6}: " + (f"reads [{top['read_gloss']}] copy-score {top['copy_score']:+.2f}" if top else "(no content read-feature)"))
        del st, Wenc, Wdec

    out = {"experiment": "SAE-feature operands per operator (section G)", "model": args.pretrained,
           "sae_repo": args.sae_repo, "operators": results}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "operator_sae_operands_summary.json").write_text(json.dumps(out, indent=2, default=float))
    write_doc(out, args.docs)
    print(f"\n[done] {len(results)} operators → {args.outdir / 'operator_sae_operands_summary.json'} + {args.docs / 'sae_operands.md'}")
    return out


def write_doc(out, docs):
    L = ["---", "title: SAE-feature operands", "---", "",
         f"# SAE-feature operands per operator ({out['model']})", "",
         "The token-operand catalog says which **tokens** an operator binds; this says which monosemantic **SAE "
         "features** it reads and writes — the dossier's *section-G* layer. Using the published per-layer GPT-2 SAEs "
         f"([{out['sae_repo']}](https://huggingface.co/{out['sae_repo']}), resid_pre, 24576 features/layer): for each "
         "operator's top head, **READ** = the attention-weighted dominant key-feature (the SAE feature most present "
         "where the head attends; content-filtered, glossed by its top tokens). **copy-score** = the OV→unembed "
         "diagonal on that feature's own tokens — attending to those tokens, does the head **raise** their own logit "
         "(**+ = copies**) or **lower** it (**− = copy-suppression / negative head**)? Provisional, single corpus "
         "(Shakespeare prose); `_` = leading space.", "",
         "> **Read this with care.** Only **content / circuit** operators bind on *content*; **positional / "
         "addressing** operators (prev-token, local, sink, self) attend by *position* or to key-0, so their "
         "\"read-feature\" is whatever token happened to sit there — incidental, not a content bind. The copy-score "
         "is the load-bearing column.", "",
         "| operator | head | kind | reads (SAE feature) | copy-score (OV) |",
         "|---|---|---|---|---|"]
    for op, r in out["operators"].items():
        if not r["operands"]:
            L.append(f"| `{op}` | {r['head']} | {r['kind']} | _(no content read-feature)_ | — |"); continue
        o = r["operands"][0]
        rd = "; ".join(f"**{x['read_gloss']}**" for x in r["operands"])
        cs = f"{o['copy_score']:+.2f} ({'copies' if o['copy_score'] > 0.03 else 'suppresses' if o['copy_score'] < -0.03 else '≈neutral'})"
        L.append(f"| `{op}` | {r['head']} | {r['kind']} | {rd} | {cs} |")
    L += ["", "_Read-features by attention mass; copy-score is for the top read-feature. "
          "GPT-2 only here (all-layer SAEs); Gemma Scope / Qwen are single-layer (a follow-up). "
          "Data: [operator_sae_operands_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/operator_sae_operands_summary.json). "
          "Regenerate: [operator_sae_operands.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_sae_operands.py). "
          "See the [operator catalog](README.md) and the [token-operand opcode tables](../DISASSEMBLY.md)._"]
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "sae_operands.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
