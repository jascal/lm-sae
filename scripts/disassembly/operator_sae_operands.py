"""SAE-feature operands per operator (the dossier's section-G gap): what each operator READS and writes in
*feature* space, not just token space. Arch-aware (GPT-2 + Gemma-2), multi-model (merges into one summary).

The token-operand catalog says which TOKENS an operator binds; this says which monosemantic **SAE features** it
reads, plus whether its OV **copies** that content or **suppresses** it. For each operator's top head:

  READ  = the attention-weighted dominant key-feature (the SAE feature most present at the positions the head
          attends to), content-filtered (structural newline/punct dropped) and glossed by its top tokens.
  copy-score = the OV→unembed diagonal on that feature's own tokens (ln-folded read side): attending to those
          tokens, does the head raise (+ copies) or lower (− copy-suppression) their own logit?

SAEs are loaded straight from cached safetensors / npz — no `sae_lens` dependency:
  GPT-2 : jbloom/GPT2-Small-SAEs-Reformatted, resid_pre, all layers (24576 feats) → exact per-head layer.
  Gemma : Gemma Scope (gemma-scope-2b-pt-res, JumpReLU, 16384 feats) at layers {0,3,6,9,12,18,21,24} → the head's
          NEAREST available layer (offsets are ≤1 for every Gemma operator; READ is token-dominated so this is a
          fair proxy, and the copy-score uses the head's exact OV regardless). The cached Qwen SAE is qwen2-0.5b
          (a different model) so Qwen is not covered.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemma"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_content_patch import _arch  # noqa: E402

GEMMA_SAE_LAYERS = [0, 3, 6, 9, 12, 18, 21, 24]


def is_struct(g):
    s = g.replace("_", " ").strip()
    return s == "" or all(c in "\\/|-_=+*.,;:!?'\"()[]{}<>`~@#$%^&" for c in s) or g in ("Ċ", "ĉ")


def san(g):
    """Sanitize a gloss for markdown-table display (newlines/pipes break tables)."""
    return g.replace("\n", "⏎").replace("\r", "").replace("Ċ", "⏎").replace("|", "¦")


def load_model(model_id, dev):
    import torch
    is_gpt2 = "gpt2" in model_id.lower()
    from transformers import AutoTokenizer
    if is_gpt2:
        from transformers import GPT2LMHeadModel
        m = GPT2LMHeadModel.from_pretrained(model_id, attn_implementation="eager").eval().to(dev)
    else:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
    return m, AutoTokenizer.from_pretrained(model_id), _arch(m), is_gpt2


def head_OV(a, L, h, d):
    """W_V^h W_O^h (d,d), arch-generic (GQA-aware)."""
    H = a["H"]; hd = a["hd"]; kvB = h // (H // a["nkv"])
    if a["is_gpt2"]:
        Wv_h = a["cattn"][L].weight.detach().float().cpu().numpy().astype(np.float64)[:, 2 * d:3 * d][:, h * hd:(h + 1) * hd]
        Wo_h = a["oproj"][L].weight.detach().float().cpu().numpy().astype(np.float64)[h * hd:(h + 1) * hd, :]
    else:
        Wv_h = a["vproj"][L].weight.detach().float().cpu().numpy().astype(np.float64)[kvB * hd:(kvB + 1) * hd, :].T
        Wo_h = a["oproj"][L].weight.detach().float().cpu().numpy().astype(np.float64)[:, h * hd:(h + 1) * hd].T
    return Wv_h @ Wo_h


def ln_gain(a, L, is_gpt2, gemma):
    w = a["norm"][L].weight.detach().float().cpu().numpy().astype(np.float64)
    return (1.0 + w) if gemma else w


def load_sae(model_id, layer, is_gpt2):
    """Return (W_enc (d,F), b_enc, b_dec, threshold|None) for the SAE at `layer` (jbloom or Gemma Scope)."""
    if is_gpt2:
        from huggingface_hub import hf_hub_download
        from safetensors.numpy import load_file
        st = load_file(hf_hub_download(model_id_to_sae_repo(model_id), f"blocks.{layer}.hook_resid_pre/sae_weights.safetensors"))
        return st["W_enc"].astype(np.float64), st["b_enc"].astype(np.float64), st["b_dec"].astype(np.float64), None
    from scope_loader import scope_npz
    d = np.load(scope_npz(layer))
    return d["W_enc"].astype(np.float64), d["b_enc"].astype(np.float64), d["b_dec"].astype(np.float64), d["threshold"].astype(np.float64)


def model_id_to_sae_repo(model_id):
    return "jbloom/GPT2-Small-SAEs-Reformatted"


def encode_dom(hs, Wenc, benc, bdec, thr):
    """Dominant SAE feature per position. jbloom: relu((x-b_dec)@W_enc+b_enc). Gemma JumpReLU: gate pre>threshold."""
    pre = (hs - bdec) @ Wenc + benc if thr is None else hs @ Wenc + benc
    act = np.maximum(pre, 0.0) if thr is None else np.where(pre > thr, pre, 0.0)
    return np.where(act.max(1) > 0, act.argmax(1), -1)


def run_model(model_id, args, dev):
    import torch
    m, tok, a, is_gpt2 = load_model(model_id, dev)
    gemma = "gemma" in model_id.lower()
    d = a.get("d", m.config.hidden_size)
    WU = m.get_output_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)

    atlas = json.loads(args.atlas.read_text())
    short = model_id.split("/")[-1]
    g = next((r for r in atlas["results"] if r["model"] in (short, model_id)), None)
    ops = {op: g["cells"][op]["top_head"] for op in atlas["operators"]}
    kind = dict(atlas["kinds"])
    if is_gpt2:                                                                       # circuit ops only have GPT-2 head-sets
        for op, heads in atlas.get("gpt2_circuit_ops", {}).items():
            ops[op] = heads[0]; kind[op] = "circuit"

    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
    ids = tok(txt)["input_ids"]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8][: args.chunks]

    # op -> (head_layer, head, sae_layer)
    plan = {}
    for op, head in ops.items():
        hl, h = int(head.split(".")[0]), int(head.split(".")[1])
        sl = hl if is_gpt2 else min(GEMMA_SAE_LAYERS, key=lambda s: abs(s - hl))
        plan[op] = (hl, h, sl, head)

    results = {}
    for sl in sorted({p[2] for p in plan.values()}):
        Wenc, benc, bdec, thr = load_sae(model_id, sl, is_gpt2)
        here = {op: p for op, p in plan.items() if p[2] == sl}
        read_w = {op: np.zeros(Wenc.shape[1]) for op in here}; tokfreq = {}
        with torch.no_grad():
            for c in chunks:
                o = m(input_ids=torch.tensor([c], device=dev), output_hidden_states=True, output_attentions=True)
                hs = o.hidden_states[sl][0].float().cpu().numpy()
                dom = encode_dom(hs, Wenc, benc, bdec, thr)
                for pos, f in enumerate(dom):
                    if f >= 0:
                        tokfreq.setdefault(int(f), Counter())[c[pos]] += 1
                Lc = len(c)
                for op, (hl, h, _sl, _hd) in here.items():
                    at = o.attentions[hl][0, h].float().cpu().numpy()
                    for q in range(2, Lc):                                            # skip q=1 (only key is pos-0)
                        kf = dom[1:q]; msk = kf >= 0                                   # exclude key position 0 (BOS/sink — not a content read)
                        if msk.any():
                            np.add.at(read_w[op], kf[msk], at[q, 1:q][msk])

        def gloss(f):
            return "/".join(tok.convert_ids_to_tokens(t).replace("Ġ", "_") for t, _ in tokfreq.get(int(f), Counter()).most_common(3)) or "?"

        for op, (hl, h, _sl, head) in here.items():
            OV = head_OV(a, hl, h, d); lg = ln_gain(a, hl, is_gpt2, gemma)
            reads = []
            for f in np.argsort(-read_w[op]):
                if read_w[op][f] <= 0 or len(reads) >= args.top_read:
                    break
                gl = gloss(f)
                if is_struct(gl.split("/")[0]):
                    continue
                tids = [t for t, _ in tokfreq.get(int(f), Counter()).most_common(3)]
                cs = [float((e := lg * WU[t]) @ OV @ WU[t] / (np.linalg.norm(e @ OV) * np.linalg.norm(WU[t]) + 1e-9)) for t in tids]
                reads.append({"read_feature": int(f), "read_gloss": gl, "attn_mass": float(read_w[op][f]),
                              "copy_score": float(np.mean(cs)) if cs else 0.0})
            results[op] = {"head": head, "kind": kind.get(op, "?"), "head_layer": hl, "sae_layer": sl,
                           "sae_offset": sl - hl, "operands": reads}
            top = reads[0] if reads else None
            print(f"  {op:>16} {head:>6} (SAE L{sl}{'' if sl == hl else f', head L{hl}'}): " +
                  (f"reads [{top['read_gloss']}] copy-score {top['copy_score']:+.2f}" if top else "(no content read-feature)"))
        del Wenc
        if dev == "cuda":
            torch.cuda.empty_cache()
    return {"model": short, "sae": "Gemma Scope (gemma-scope-2b-pt-res, JumpReLU)" if gemma else model_id_to_sae_repo(model_id),
            "operators": results}


def write_doc(summary, docs):
    models = summary["models"]
    L = ["---", "title: SAE-feature operands", "---", "", "# SAE-feature operands per operator", "",
         "The token-operand catalog says which **tokens** an operator binds; this says which monosemantic **SAE "
         "features** it reads, and whether its OV **copies** that content (+) or **suppresses** it (−) — the dossier's "
         "*section-G* layer. **READ** = the attention-weighted dominant key-feature (the SAE feature most present where "
         "the head attends; content-filtered, glossed by top tokens). **copy-score** = the OV→unembed diagonal on that "
         "feature's own tokens. Provisional, single corpus (Shakespeare prose); `_` = leading space.", "",
         "> Only **content / circuit** operators bind on content; **positional / addressing** ops (prev-token, local, "
         "sink, self) attend by position, so their read-feature is incidental — the copy-score column is load-bearing.", ""]
    for mid, mr in models.items():
        L += [f"## {mid} — {mr['sae']}", "",
              "| operator | head | kind | reads (SAE feature) | copy-score (OV) |",
              "|---|---|---|---|---|"]
        for op, r in mr["operators"].items():
            if not r["operands"]:
                L.append(f"| `{op}` | {r['head']} | {r['kind']} | _(no content read-feature)_ | — |"); continue
            o = r["operands"][0]
            off = "" if r.get("sae_offset", 0) == 0 else f" _(SAE L{r['sae_layer']}, head L{r['head_layer']})_"
            rd = "; ".join(f"**{san(x['read_gloss'])}**" for x in r["operands"])
            cs = f"{o['copy_score']:+.2f} ({'copies' if o['copy_score'] > 0.03 else 'suppresses' if o['copy_score'] < -0.03 else '≈neutral'})"
            L.append(f"| `{op}` | {r['head']}{off} | {r['kind']} | {rd} | {cs} |")
        L.append("")
    L += ["_GPT-2 has all-layer SAEs (exact per-head layer); Gemma Scope is 8 layers so each Gemma op uses its "
          "nearest available SAE layer (offset ≤1, annotated). **Gemma's read-features come out noisier** than "
          "GPT-2's — its heads put heavy attention on `<bos>`/structural tokens on this non-repetitive prose, so the "
          "dominant *content* key-feature is weaker (a corpus + attention-budget effect, not a tooling one); the "
          "copy-score still uses the head's exact OV. The cached Qwen SAE is for qwen2-0.5b (a different model). "
          "Data: [operator_sae_operands_summary.json](https://github.com/jascal/lm-sae/blob/main/runs/disassembly/operators/operator_sae_operands_summary.json). "
          "Regenerate: [operator_sae_operands.py](https://github.com/jascal/lm-sae/blob/main/scripts/disassembly/operator_sae_operands.py)._"]
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "sae_operands.md").write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--atlas", type=Path, default=Path("runs/disassembly/operators/atlas_summary.json"))
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--chunks", type=int, default=40)
    p.add_argument("--top-read", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/operators"))
    p.add_argument("--docs", type=Path, default=Path("docs/operators"))
    p.add_argument("--docs-only", action="store_true")
    args = p.parse_args(argv)
    sumpath = args.outdir / "operator_sae_operands_summary.json"
    summary = json.loads(sumpath.read_text()) if sumpath.exists() else {"experiment": "SAE-feature operands per operator (section G)", "models": {}}
    if "models" not in summary:                                                       # migrate old single-model schema
        summary = {"experiment": summary.get("experiment", ""), "models": {}}
    if args.docs_only:
        write_doc(summary, args.docs); print(f"[docs-only] re-rendered {args.docs / 'sae_operands.md'}"); return summary

    import torch
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"\n=== {args.model} (dev {dev}) ===")
    mr = run_model(args.model, args, dev)
    summary["models"][mr["model"]] = mr
    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath.write_text(json.dumps(summary, indent=2, default=float))
    write_doc(summary, args.docs)
    print(f"\n[done] {args.model}: {len(mr['operators'])} operators → {sumpath} + {args.docs / 'sae_operands.md'}")
    return summary


if __name__ == "__main__":
    main()
