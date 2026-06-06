"""Portable disassembly (architecture-agnostic) — behavioral idioms + coverage scorecard on ANY HF causal LM.

The weight-space opcode tables are GPT-2-specific (Conv1D, ln_1, no RoPE/GQA). But the BEHAVIORAL layer —
idiom signatures (prev-token / duplicate / induction) and the attention-bucket coverage split — only needs
the attention maps, which HF standardizes (`output_attentions=True` -> `.attentions`, a length-n_layer tuple
of (batch, n_head, seq, seq); GQA is already expanded to n_attention_heads). So this ports to recent models
(Gemma-2, Llama-3, Qwen) unchanged. Structural-token detection is generalized across tokenizers (GPT-2 BPE
'Ġ/Ċ', SentencePiece '▁', raw newlines). Run on GPT-2 first to confirm it reproduces the known numbers
(prev-token 4.11, sink ~45%), then point --model at a recent model.

Reports, per model: top heads for prev/duplicate/induction (+literature check for GPT-2) and the attention
budget (self/sink/prev/structural/local/long_range = the coverage scorecard's plumbing-vs-content split).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CORPORA = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "wikitext": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
}
BUCKETS = ["self", "sink", "prev", "structural", "local", "long_range"]
# GPT-2-small literature heads (only used for the validation line when --model is gpt2)
GPT2_KNOWN = {"prev_token": {"4.11"}, "induction": {"5.0", "5.1", "5.5", "6.9", "7.11"},
              "duplicate_token": {"0.1", "0.5", "1.5", "3.0"}}


def _struct_token(s: str) -> bool:
    """Structural (delimiter/whitespace) token, across tokenizer surface forms."""
    s = s.replace("Ġ", "").replace("▁", "").replace("Ċ", "\n")
    s = s.strip()
    return s == "" or s in {"<0x0A>"} or all(not ch.isalnum() for ch in s)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2", help="HF causal-LM id (e.g. gpt2, google/gemma-2-2b)")
    p.add_argument("--corpus", default="shakespeare", help="preset (shakespeare/wikitext) or raw-text URL")
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--local-max", type=int, default=8)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-chars", type=int, default=400000)
    p.add_argument("--output", type=Path, default=Path("runs/disasm_portable_summary.json"))
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    print(f"[load] {args.model}  dtype={args.dtype} device={args.device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager", torch_dtype=dt).eval().to(args.device)
    import urllib.request
    url = CORPORA.get(args.corpus, args.corpus)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    txt = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")[: args.n_chars]
    ids = tok(txt)["input_ids"][: args.max_tokens]
    chunks = [ids[i:i + args.ctx] for i in range(0, len(ids), args.ctx) if len(ids[i:i + args.ctx]) >= 8]

    nL = nH = None
    pt = dup = ind = None
    ptn = 0; dupn = 0; dupb = 0.0; indn = 0; indb = 0.0
    bacc = None; btot = None
    with torch.no_grad():
        for c in chunks:
            o = model(input_ids=torch.tensor([c], device=args.device), output_attentions=True)
            atts = o.attentions
            if nL is None:
                nL = len(atts); nH = atts[0].shape[1]
                pt = np.zeros((nL, nH)); dup = np.zeros((nL, nH)); ind = np.zeros((nL, nH))
                bacc = np.zeros((nL, nH, len(BUCKETS))); btot = np.zeros((nL, nH))
            Lc = len(c); ca = np.array(c); qi = np.arange(Lc)
            toks = tok.convert_ids_to_tokens(c)
            struct = np.array([_struct_token(t) for t in toks])
            prevtok = np.full(Lc, -1); prevtok[1:] = ca[:-1]
            DM = (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None])
            IM = (prevtok[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]) & (qi[None, :] >= 1)
            ptn += Lc - 1; dupn += int(DM.any(1).sum()); indn += int(IM.any(1).sum())
            dq = DM.any(1); iq = IM.any(1)
            if dq.any():
                dupb += float((DM.sum(1)[dq] / np.maximum(qi[dq], 1)).sum())
            if iq.any():
                indb += float((IM.sum(1)[iq] / np.maximum(qi[iq], 1)).sum())
            S, T = np.meshgrid(qi, qi); delta = T - S
            bid = np.full((Lc, Lc), 5, dtype=int)
            bid[(delta >= 2) & (delta <= args.local_max)] = 4
            bid[struct[None, :].repeat(Lc, 0)] = 3
            bid[delta == 1] = 2
            bid[S == 0] = 1
            bid[S == T] = 0
            bid[S > T] = -1
            for L in range(nL):
                a = atts[L][0].float().cpu().numpy()
                pt[L] += np.diagonal(a, offset=-1, axis1=1, axis2=2).sum(1)
                dup[L] += (a * DM[None]).sum((1, 2)); ind[L] += (a * IM[None]).sum((1, 2))
                btot[L] += a.sum((1, 2))
                for b in range(len(BUCKETS)):
                    bacc[L, :, b] += (a * (bid == b)[None]).sum((1, 2))

    prevv = (pt / max(ptn, 1)).reshape(-1)
    dupv = (dup / max(dupn, 1) - dupb / max(dupn, 1)).reshape(-1)
    indv = (ind / max(indn, 1) - indb / max(indn, 1)).reshape(-1)
    frac = bacc / np.maximum(btot, 1e-9)[:, :, None]
    budget = {b: float(np.mean(frac[:, :, i])) for i, b in enumerate(BUCKETS)}
    heads = [f"{L}.{h}" for L in range(nL) for h in range(nH)]

    def topk(v):
        return [(heads[i], float(v[i])) for i in np.argsort(-v)[:args.top_k]]
    sigs = {"prev_token": topk(prevv), "duplicate_token": topk(dupv), "induction": topk(indv)}

    print(f"\n{args.model}: layers={nL} heads/layer={nH}  ({len(chunks)} chunks)")
    print("\n[behavioral idioms] top heads:")
    for k, v in sigs.items():
        print(f"  {k:16} " + ", ".join(f"{n}({s:+.2f})" for n, s in v))
    print("\n[attention budget] mean per-head mass:")
    for b in BUCKETS:
        mark = "  <- SINK" if b == "sink" else ("  <- content" if b == "long_range" else "")
        print(f"  {b:11} {budget[b]:6.1%}{mark}")
    plumbing = sum(budget[b] for b in ("self", "sink", "prev", "structural", "local"))
    print(f"  plumbing total {plumbing:.1%}  |  content (long_range) {budget['long_range']:.1%}")

    out = {"experiment": "portable disassembly (behavioral + coverage)", "model": args.model,
           "corpus": args.corpus, "n_layers": nL, "n_heads": nH, "idioms": sigs, "attention_budget": budget,
           "plumbing": plumbing}
    if args.model == "gpt2":
        val = {}
        for idiom, kset in GPT2_KNOWN.items():
            found = {n for n, _ in sigs[idiom]}
            val[idiom] = sorted(kset & found)
        out["gpt2_literature_check"] = val
        print("\n[gpt2 literature check] " + "; ".join(f"{k}->{v}" for k, v in val.items()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=float))
    print(f"[done] {args.output}")
    return out


if __name__ == "__main__":
    main()
