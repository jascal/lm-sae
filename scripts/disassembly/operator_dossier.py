"""A DEEP per-operator dossier — the full battery on ONE named instruction, not one measurement scattered per script.

The disassembly studied each named op piecemeal: induction in `composition_dag`/`ssm_induction`, prev-token in
`cross_model_positional`/`key_patch_cross_model`, name-movers in `self_repair`, the key/value channel in
`circuit_content_patch`, reuse-across-tasks in `instruction_reuse`. This consolidates them: pick ONE operator and
run EVERY measurement on it, deeply, in a single report. Sections:

  A. IDENTITY        — find the op's heads BEHAVIOURALLY (not hardcoded) + their attention-signal strength + depth;
                       which match the literature set.
  B. CAUSAL × TASKS  — mean-ablate the op's heads, damage to generic LM / induction / copy-names / successor / IOI
                       (+ a random-head control) → which programs the op serves.
  C. CHANNELS (K/V)  — for the op's reader head, decompose WHICH channel carries it: KEY/match (remove the upstream
                       writer from the reader's key → attention collapse) vs VALUE/move (ΔV-out).
  D. COMPOSITION     — the op's local call-graph: IN-edges (which earlier heads' OV compose into its key, weight +
                       behavioural path-patch) and OUT-edges (whose value its OV feeds — composed "virtual heads").
  E. REDUNDANCY      — the ablation CURVE on the op's primary task: cumulative head-ablation (sharp bottleneck vs
                       graceful population) + per-head leave-one-out marginals + the single-head-vs-population gap.
  F. CROSS-MODEL     — does the op's behavioural signal survive architecture? (gain across a GPT-2 size + a RoPE model).

Default `--op induction` (the keystone op). The OPS registry makes it extensible: `--op prevtok|duplicate|
name_mover|s_inhibition`. GPT-2 for the deep sections A–E; F loads a couple extra models. SAE-feature operands
(what the op reads/writes in feature space) are the next layer and need a SAE — flagged, not run here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ioi_causal import NAMES, OBJECTS, PLACES, TEMPLATES  # noqa: E402

# literature head sets, used ONLY to annotate the behaviourally-found heads (not to define them)
LIT = {
    "prevtok": [(4, 11)],
    "induction": [(5, 0), (5, 1), (5, 5), (6, 9), (7, 11)],
    "duplicate": [(0, 1), (0, 5), (3, 0), (1, 5)],
    "name_mover": [(9, 6), (9, 9), (10, 0), (10, 10)],
    "backup_name_mover": [(9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (11, 2)],     # self-repair backups (Wang et al.)
    "negative_mover": [(10, 7), (11, 10)],                                         # copy-suppression / negative name-movers
    "s_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "coreference": [(9, 0)],                                                       # pronoun -> antecedent (exploratory)
}


def op_masks(toks):
    """For a token list, the (query,key) boolean mask each op attends along (on repeated-random / prose seqs)."""
    ca = np.array(toks); n = len(ca); qi = np.arange(n); pv = np.full(n, -1); pv[1:] = ca[:-1]
    return {
        "prevtok": (qi[None, :] == (qi[:, None] - 1)) & (qi[:, None] >= 1),          # attend to position q-1
        "induction": (pv[None, :] == ca[:, None]) & (qi[None, :] >= 1) & (qi[None, :] < qi[:, None]),  # key's predecessor==query tok
        "duplicate": (ca[None, :] == ca[:, None]) & (qi[None, :] < qi[:, None]),     # earlier same token
        "sink": (qi[None, :] == 0) & (qi[:, None] >= 1),                             # attend to key-0 (the no-op / idle register)
    }


LIT_OPS = {"name_mover", "backup_name_mover", "negative_mover", "s_inhibition", "coreference"}  # head-sets from literature (DLA/IOI)
BEHAV_OPS = {"prevtok", "induction", "duplicate", "sink"}                                        # heads found by attention-mask mass

OPS = {
    "induction": dict(kind="content", primary="induction", repeat=True,
                      desc="in-context copy: attend to the key whose predecessor token == current token, copy it"),
    "prevtok": dict(kind="positional", primary="copy_names", repeat=False,
                    desc="previous-token head: attend to position q-1 (the induction writer / local addressing)"),
    "duplicate": dict(kind="content", primary="successor", repeat=True,
                      desc="duplicate-token head: attend to an earlier occurrence of the same token"),
    "sink": dict(kind="addressing", primary="generic", repeat=False,
                 desc="attention sink: park attention on key-0 (the no-op / idle register)"),
    "name_mover": dict(kind="output", primary="ioi", repeat=False,
                       desc="IOI name-mover: copy the indirect-object name to the logits (output head)"),
    "backup_name_mover": dict(kind="output", primary="ioi", repeat=False,
                              desc="IOI backup name-mover: the self-repair spares that wake when primaries are ablated"),
    "negative_mover": dict(kind="output", primary="ioi", repeat=False,
                           desc="copy-suppression / negative name-mover: writes against the copied token"),
    "s_inhibition": dict(kind="output", primary="ioi", repeat=False,
                         desc="IOI S-inhibition: suppress the subject so the name-mover writes IO"),
    "coreference": dict(kind="content", primary="generic", repeat=False,
                        desc="coreference (exploratory): pronoun -> earlier antecedent (no clean task probe here)"),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", default="induction", choices=list(OPS))
    p.add_argument("--pretrained", default="gpt2")
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--eval-chunks", type=int, default=40)
    p.add_argument("--probe-seqs", type=int, default=40, help="repeated-random/prose probes for behavioural head-ID")
    p.add_argument("--probe-len", type=int, default=24)
    p.add_argument("--n-task", type=int, default=70, help="examples per causal task")
    p.add_argument("--top-heads", type=int, default=5, help="how many behavioural heads define the op")
    p.add_argument("--max-upstream", type=int, default=80)
    p.add_argument("--cross-models", default="gpt2-medium,Qwen/Qwen2.5-1.5B")
    p.add_argument("--no-cross", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outroot", type=Path, default=Path("runs/disassembly/operators"), help="operator-catalog tree root")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--fig", type=Path, default=None)
    args = p.parse_args(argv)
    op = args.op; spec = OPS[op]
    mtag = args.pretrained.split("/")[-1]                                           # dossiers/<op>/<model>_summary.{json,png}
    args.output = args.output or (args.outroot / "dossiers" / op / f"{mtag}_summary.json")
    args.fig = args.fig or (args.outroot / "dossiers" / op / f"{mtag}.png")

    import torch
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    dev = args.device if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(args.pretrained, attn_implementation="eager").eval().to(dev)
    tr = model.transformer; cfg = model.config; H = cfg.n_head; hd = cfg.n_embd // H; nL = cfg.n_layer; d = cfg.n_embd; NH = nL * H
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    rng = np.random.default_rng(args.seed)
    import urllib.request
    txt = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:200000]
    ids_all = tok(txt)["input_ids"]
    echunks = [ids_all[i:i + args.ctx] for i in range(0, len(ids_all), args.ctx) if len(ids_all[i:i + args.ctx]) >= 8][: args.eval_chunks]
    cnt = {}
    for t in ids_all:
        cnt[t] = cnt.get(t, 0) + 1
    vocab = [t for t, _ in sorted(cnt.items(), key=lambda r: -r[1])[:400]]

    def name_of(i):
        return f"{i // H}.{i % H}"

    # ============================ A. IDENTITY (behavioural head-ID) ============================
    # probes: repeated-random sequences for content ops (induction/duplicate), prose for positional/output ops.
    if spec["repeat"]:
        probes = [(lambda s: s + s)([int(vocab[i]) for i in rng.integers(0, len(vocab), args.probe_len)]) for _ in range(args.probe_seqs)]
    else:
        probes = echunks[: args.probe_seqs]
    msk_op = op if op in BEHAV_OPS else "induction"                                # lit/output ops: identified via LIT below
    mass = np.zeros(NH); pmass = np.zeros(NH); ntot = 0
    with torch.no_grad():
        for s in probes:
            o = model(input_ids=torch.tensor([s], device=dev), output_attentions=True)
            M = op_masks(s); want = M[msk_op]; prev = M["prevtok"]; ntot += int(want.sum())
            for L in range(nL):
                at = o.attentions[L][0].float().cpu().numpy()
                mass[L * H:(L + 1) * H] += (at * want[None]).sum((1, 2))
                pmass[L * H:(L + 1) * H] += (at * prev[None]).sum((1, 2))
    mass /= max(ntot, 1)
    prevtok_head = int(np.argmax(pmass / max(ntot, 1)))                            # the model's prev-token head (induction writer)

    if op in LIT_OPS:                                                              # output/circuit ops: rank by IOI direct effect, not attention
        op_heads_idx = [L * H + h for (L, h) in LIT[op]]                            # literature set (DLA-defined; not weight/attn-readable)
        id_note = "circuit op — heads from literature (DLA-defined; not attention-mask-readable)"
        ident = [{"head": name_of(i), "signal": None, "depth": (i // H) / (nL - 1)} for i in op_heads_idx]
    else:
        order = np.argsort(-mass)
        op_heads_idx = [int(i) for i in order[: args.top_heads] if mass[int(i)] > 0.02]
        id_note = f"behavioural: top heads by attention mass on the {op} pattern (>0.02)"
        ident = [{"head": name_of(int(i)), "signal": float(mass[int(i)]), "depth": (int(i) // H) / (nL - 1),
                  "in_lit": [int(i) // H, int(i) % H] in [list(x) for x in LIT.get(op, [])]} for i in order[: args.top_heads + 3]]
    op_heads = [(i // H, i % H) for i in op_heads_idx]
    reader = op_heads_idx[0] if op_heads_idx else int(np.argmax(mass)); LB, hB = reader // H, reader % H

    # ============================ mean-ablation harness ============================
    cap = {L: [] for L in range(nL)}
    hk = [tr.h[L].attn.c_proj.register_forward_pre_hook(
        (lambda L: lambda m, inp: cap[L].append(inp[0].detach().reshape(-1, inp[0].shape[-1])))(L)) for L in range(nL)]
    with torch.no_grad():
        for c in echunks:
            model(input_ids=torch.tensor([c], device=dev))
    for h in hk:
        h.remove()
    meanv = {L: torch.cat(cap[L], 0).mean(0) for L in range(nL)}

    def ablate_hooks(ablate):
        by = {}
        for (L, h) in ablate:
            by.setdefault(L, []).append(h)
        hs = []
        for L, hss in by.items():
            def mk(L, hss):
                def hook(m, inp):
                    x = inp[0].clone()
                    for h in hss:
                        x[..., h * hd:(h + 1) * hd] = meanv[L][h * hd:(h + 1) * hd].to(x.dtype)
                    return (x,)
                return hook
            hs.append(tr.h[L].attn.c_proj.register_forward_pre_hook(mk(L, hss)))
        return hs

    # ============================ tasks (from instruction_reuse) ============================
    def single(strs):
        out = []
        for s in strs:
            i = tok(s, add_special_tokens=False)["input_ids"]
            out.append(i[0] if len(i) == 1 else None)
        return [x for x in out if x is not None]
    names = single(NAMES); places = single(PLACES); objs = single(OBJECTS)
    ind_seqs = [(lambda s: s + s)([int(vocab[i]) for i in rng.integers(0, len(vocab), 20)]) for _ in range(args.n_task)]
    kname = min(20, len(names)); name_seqs = [(lambda s: s + s)([int(names[i]) for i in rng.choice(len(names), kname, replace=False)]) for _ in range(args.n_task)]
    nums = {}
    for v in range(1, 220):
        i = tok(f" {v}", add_special_tokens=False)["input_ids"]
        if len(i) == 1:
            nums[v] = i[0]
    runlen = 6; starts = [a for a in nums if all((a + i) in nums for i in range(runlen + 1))]
    succ_seqs = [([nums[(a := starts[int(rng.integers(0, len(starts)))]) + i] for i in range(runlen)], nums[a + runlen]) for _ in range(args.n_task)]
    ioi = []
    for _ in range(args.n_task):
        a, b = rng.choice(len(names), 2, replace=False)
        P = names[a]; S = names[b]; pl = places[int(rng.integers(0, len(places)))]; ob = objs[int(rng.integers(0, len(objs)))]
        tpl = TEMPLATES[int(rng.integers(0, len(TEMPLATES)))]
        text = tpl.format(i0=tok.decode([P]), i1=tok.decode([S]), P=tok.decode([pl]), S=tok.decode([S]), T=tok.decode([ob]))
        ioi.append((tok(text)["input_ids"], P, S))

    def copy_nll(ablate, seqs):
        hs = ablate_hooks(ablate); tot = 0.0; n = 0
        try:
            with torch.no_grad():
                for s in seqs:
                    lp = F.log_softmax(model(input_ids=torch.tensor([s], device=dev)).logits[0].float(), -1)
                    L = len(s) // 2
                    for pos in range(L, 2 * L - 1):
                        tot += float(-lp[pos, s[pos + 1]]); n += 1
        finally:
            for h in hs:
                h.remove()
        return tot / max(n, 1)

    def lm_nll(ablate):
        hs = ablate_hooks(ablate); tot = 0.0; n = 0
        try:
            with torch.no_grad():
                for c in echunks:
                    lp = F.log_softmax(model(input_ids=torch.tensor([c], device=dev)).logits[0, :-1].float(), -1)
                    y = torch.tensor(c[1:], device=dev); tot += float(-lp[torch.arange(len(y)), y].sum()); n += len(y)
        finally:
            for h in hs:
                h.remove()
        return tot / max(n, 1)

    def succ_nll(ablate):
        hs = ablate_hooks(ablate); tot = 0.0; n = 0
        try:
            with torch.no_grad():
                for seq, target in succ_seqs:
                    lp = F.log_softmax(model(input_ids=torch.tensor([seq], device=dev)).logits[0, -1].float(), -1)
                    tot += float(-lp[target]); n += 1
        finally:
            for h in hs:
                h.remove()
        return tot / max(n, 1)

    def ioi_ld(ablate):
        hs = ablate_hooks(ablate); lds = []
        try:
            with torch.no_grad():
                for idsq, io, s in ioi:
                    lg = model(input_ids=torch.tensor([idsq], device=dev)).logits[0, -1].float()
                    lds.append(float(lg[io] - lg[s]))
        finally:
            for h in hs:
                h.remove()
        return float(np.mean(lds))
    TASKS = ["generic", "induction", "copy_names", "successor", "ioi"]
    metric = {"generic": lm_nll, "induction": lambda ab: copy_nll(ab, ind_seqs),
              "copy_names": lambda ab: copy_nll(ab, name_seqs), "successor": succ_nll, "ioi": ioi_ld}
    base = {t: metric[t](set()) for t in TASKS}

    def effect(t, val):
        return (base["ioi"] - val) / (abs(base["ioi"]) + 1e-9) if t == "ioi" else (val - base[t]) / (base[t] + 1e-9)

    # ============================ B. CAUSAL × TASKS ============================
    op_eff = {t: effect(t, metric[t](set(op_heads))) for t in TASKS}
    rand_eff = {t: [] for t in TASKS}
    for r in range(4):
        rh = [(int(i) // H, int(i) % H) for i in rng.choice(NH, len(op_heads), replace=False)]
        for t in TASKS:
            rand_eff[t].append(effect(t, metric[t](set(rh))))
    rmean = {t: float(np.mean(rand_eff[t])) for t in TASKS}; rstd = {t: float(np.std(rand_eff[t])) for t in TASKS}
    thr = {t: max(0.05, rmean[t] + 2 * rstd[t]) for t in TASKS}
    serves = [t for t in TASKS if op_eff[t] > thr[t]]

    # ============================ C. CHANNELS (K / V) ============================
    # reader B's KEY (match) vs VALUE (move) dependence on each upstream head (GPT-2 c_attn machinery).
    Wc = tr.h[LB].attn.c_proj
    channel = {"reader": name_of(reader), "note": "reader in layer 0 — no upstream; channel skipped" if LB == 0 else ""}
    if LB > 0 and op in BEHAV_OPS:
        upstream = [(L, h) for L in range(LB) for h in range(H)][: args.max_upstream]

        def head_contrib(L, captured, h):
            x = torch.zeros_like(captured); x[..., h * hd:(h + 1) * hd] = captured[..., h * hd:(h + 1) * hd]
            return tr.h[L].attn.c_proj(x) - tr.h[L].attn.c_proj(torch.zeros_like(captured[..., :1, :]))

        def b_value_out(inp_normed, attnB):                                        # B's residual write for a value-input (move channel)
            v = tr.h[LB].attn.c_attn(inp_normed)[..., 2 * d:3 * d]
            headout = attnB.to(v.dtype) @ v[0, :, hB * hd:(hB + 1) * hd]
            x = torch.zeros((1, headout.shape[0], d), dtype=v.dtype, device=v.device); x[0, :, hB * hd:(hB + 1) * hd] = headout
            return Wc(x) - Wc(x[:, :1] * 0)
        capk = {}; hooks = [tr.h[LB].register_forward_pre_hook(lambda m, inp: capk.__setitem__("r", inp[0].detach()))]
        for L in range(LB):
            hooks.append(tr.h[L].attn.c_proj.register_forward_pre_hook((lambda L: lambda m, inp: capk.__setitem__(L, inp[0].detach()))(L)))
        clean = 0.0; kpatch = {u: 0.0 for u in upstream}; vtot = 0.0; vpatch = {u: 0.0 for u in upstream}; tot = 0; sane = None
        with torch.no_grad():
            for s in probes:
                capk.clear(); o = model(input_ids=torch.tensor([s], device=dev), output_attentions=True)
                Msk = op_masks(s)[op]; attnB = o.attentions[LB][0, hB]
                clean += float((attnB.float().cpu().numpy() * Msk).sum())
                resid = capk["r"]; Bout = b_value_out(tr.h[LB].ln_1(resid), attnB); vtot += float(torch.linalg.norm(Bout.float()))
                for (La, ha) in upstream:
                    kin = tr.h[LB].ln_1(resid - head_contrib(La, capk[La], ha))
                    ksl = tr.h[LB].attn.c_attn(kin)[..., d:2 * d]

                    def hook(m, inp, o2, _k=ksl):
                        o2 = o2.clone(); o2[..., d:2 * d] = _k; return o2
                    hkk = tr.h[LB].attn.c_attn.register_forward_hook(hook)
                    try:
                        ap = model(input_ids=torch.tensor([s], device=dev), output_attentions=True).attentions[LB][0, hB]
                        kpatch[(La, ha)] += float((ap.float().cpu().numpy() * Msk).sum())
                    finally:
                        hkk.remove()
                    vpatch[(La, ha)] += float(torch.linalg.norm((Bout - b_value_out(kin, attnB)).float()))
                tot += 1
                if sane is None:
                    kz = tr.h[LB].ln_1(resid); ksl = tr.h[LB].attn.c_attn(kz)[..., d:2 * d]

                    def hz(m, inp, o2, _k=ksl):
                        o2 = o2.clone(); o2[..., d:2 * d] = _k; return o2
                    hkk = tr.h[LB].attn.c_attn.register_forward_hook(hz)
                    az = model(input_ids=torch.tensor([s], device=dev), output_attentions=True).attentions[LB][0, hB]
                    hkk.remove(); sane = float(np.abs(az.float().cpu().numpy() - attnB.float().cpu().numpy()).max())
        for h in hooks:
            h.remove()
        clean /= max(tot, 1)
        krows = sorted([{"head": name_of(La * H + ha), "collapse": (clean - kpatch[(La, ha)] / max(tot, 1)) / clean if clean > 1e-6 else 0.0}
                        for (La, ha) in upstream], key=lambda r: -r["collapse"])
        vrows = sorted([{"head": name_of(La * H + ha), "dvout": vpatch[(La, ha)] / max(vtot, 1e-9)} for (La, ha) in upstream], key=lambda r: -r["dvout"])
        kmed = float(np.median([r["collapse"] for r in krows]))
        channel = {"reader": name_of(reader), "sanity": sane, "key_top": krows[0], "key_top_is_prevtok_head": krows[0]["head"] == name_of(prevtok_head),
                   "key_median": kmed, "key_concentration": krows[0]["collapse"] / (kmed + 1e-9), "value_top": vrows[0],
                   "value_median": float(np.median([r["dvout"] for r in vrows])), "key_rows": krows[:5], "value_rows": vrows[:5]}

    # ============================ D. COMPOSITION (in / out edges) ============================
    Wk = [tr.h[L].attn.c_attn.weight.detach().cpu().numpy().astype(np.float64)[:, d:2 * d] for L in range(nL)]
    Wv = [tr.h[L].attn.c_attn.weight.detach().cpu().numpy().astype(np.float64)[:, 2 * d:3 * d] for L in range(nL)]
    Wo = [tr.h[L].attn.c_proj.weight.detach().cpu().numpy().astype(np.float64) for L in range(nL)]
    OV = np.zeros((NH, d, d)); WK = np.zeros((NH, d, hd)); WV = np.zeros((NH, d, hd))
    for L in range(nL):
        for h in range(H):
            i = L * H + h; sl = slice(h * hd, (h + 1) * hd)
            OV[i] = Wv[L][:, sl] @ Wo[L][sl, :]; WK[i] = Wk[L][:, sl]; WV[i] = Wv[L][:, sl]
    umw = np.linalg.svd(OV.transpose(0, 2, 1).reshape(-1, d), full_matrices=False)[2][0]   # remove shared mean-write dir
    OV = np.einsum("nij,jk->nik", OV, np.eye(d) - np.outer(umw, umw)); ovn = np.linalg.norm(OV.reshape(NH, -1), axis=1) + 1e-9

    def comp_into(port, B):                                                        # earlier heads A whose OV feeds B's <port>
        Wp = {"K": WK, "V": WV}[port]; pn = np.linalg.norm(Wp[B]) + 1e-9
        sc = [(name_of(a), float(np.linalg.norm(OV[a] @ Wp[B]) / (ovn[a] * pn))) for a in range(NH) if a // H < B // H]
        return sorted(sc, key=lambda r: -r[1])[:6]

    def comp_outof(port, A):                                                       # later heads B whose <port> A's OV feeds
        Wp = {"K": WK, "V": WV}[port]
        sc = [(name_of(b), float(np.linalg.norm(OV[A] @ Wp[b]) / (ovn[A] * (np.linalg.norm(Wp[b]) + 1e-9)))) for b in range(NH) if b // H > A // H]
        return sorted(sc, key=lambda r: -r[1])[:6]
    composition = {"in_key": comp_into("K", reader), "in_value": comp_into("V", reader),
                   "out_value": comp_outof("V", reader), "out_key": comp_outof("K", reader)}

    # ============================ E. REDUNDANCY / ABLATION CURVE ============================
    pt = spec["primary"]; mt = metric[pt]; base_pt = base[pt]
    solo = sorted([(name_of(L * H + h), effect(pt, mt(set([(L, h)])))) for (L, h) in op_heads], key=lambda r: -r[1])
    order_heads = [(int(n.split(".")[0]), int(n.split(".")[1])) for n, _ in solo]
    curve = []; acc = []
    for (L, h) in order_heads:
        acc.append((L, h)); curve.append({"after": name_of(L * H + h), "n": len(acc), "effect": effect(pt, mt(set(acc)))})
    full = effect(pt, mt(set(op_heads)))
    loo = [{"head": name_of(L * H + h), "marginal": full - effect(pt, mt(set(op_heads) - {(L, h)}))} for (L, h) in op_heads]
    max_solo = max((e for _, e in solo), default=0.0)
    redundancy = {"primary_task": pt, "full_op_effect": full, "max_single_head_effect": max_solo,
                  "population_gap": full - max_solo, "bottleneck": full <= 1.4 * max_solo and max_solo > 0.1,
                  "solo": solo, "cumulative_curve": curve, "leave_one_out": loo}

    # ============================ assemble + print ============================
    res = {"op": op, "spec": spec, "model": args.pretrained, "prev_token_head": name_of(prevtok_head),
           "A_identity": {"note": id_note, "op_heads": [name_of(i) for i in op_heads_idx], "ranked": ident},
           "B_causal": {"tasks": TASKS, "op_effects": op_eff, "random_mean": rmean, "threshold": thr, "serves": serves, "baselines": base},
           "C_channels": channel, "D_composition": composition, "E_redundancy": redundancy}

    sep = "=" * 96
    print(f"\n{sep}\nDEEP DOSSIER — operator '{op}' ({spec['kind']}): {spec['desc']}\n  model {args.pretrained}  |  prev-token head {name_of(prevtok_head)}\n{sep}")
    print(f"[A. IDENTITY] {id_note}")
    for r in ident:
        sig = f"signal {r['signal']:.3f}" if r["signal"] is not None else "DLA-defined"
        lit = " (in lit)" if r.get("in_lit") else ""
        print(f"    {r['head']:>6}  {sig}  depth {r['depth']:.2f}{lit}")
    print(f"    -> op heads: {[name_of(i) for i in op_heads_idx]}")
    print("\n[B. CAUSAL x TASKS] ablate the op's heads; + = hurts task; * = beyond random control")
    print("    " + " | ".join(f"{t} {op_eff[t]:+.2f}{'*' if op_eff[t] > thr[t] else ' '}" for t in TASKS) + f"   -> serves: {serves or 'none'}")
    if channel.get("note") and "skipped" in channel.get("note", ""):
        print(f"\n[C. CHANNELS] {channel['note']}")
    elif "key_top" in channel:
        kt = channel["key_top"]; vt = channel["value_top"]
        tag = " (=prev-token head)" if channel["key_top_is_prevtok_head"] else ""
        print(f"\n[C. CHANNELS] reader {channel['reader']}  (sanity {channel['sanity']:.1e})")
        print(f"    KEY/match : remove {kt['head']}{tag} from key -> collapse {kt['collapse']:+.0%}  (median {channel['key_median']:+.0%}, concentration {channel['key_concentration']:.1f}x)")
        print(f"    VALUE/move: top mover {vt['head']} dV-out {vt['dvout']:.2f}  (median {channel['value_median']:.2f})")
    else:
        print("\n[C. CHANNELS] (output op — key/value match-patch N/A; carried by OV->unembedding, see D out-edges)")
    print("\n[D. COMPOSITION] local call-graph of the reader " + channel.get("reader", name_of(reader)))
    print("    IN  -> key  : " + ", ".join(f"{n}({s:.3f})" for n, s in composition["in_key"][:4]))
    print("    IN  -> value: " + ", ".join(f"{n}({s:.3f})" for n, s in composition["in_value"][:4]))
    print("    OUT -> value: " + ", ".join(f"{n}({s:.3f})" for n, s in composition["out_value"][:4]))
    print("\n[E. REDUNDANCY] primary task '" + pt + f"'  (baseline {base_pt:+.3f})")
    print("    per-head solo effect: " + ", ".join(f"{n}({e:+.2f})" for n, e in solo))
    print("    cumulative: " + " -> ".join(f"{c['n']}h {c['effect']:+.2f}" for c in curve))
    btxt = "BOTTLENECK (one head ~= whole op)" if redundancy["bottleneck"] else f"DISTRIBUTED population (full {full:+.2f} >> best single {max_solo:+.2f}, gap {redundancy['population_gap']:+.2f})"
    print(f"    -> {btxt}")

    # ============================ F. CROSS-MODEL (behavioural signal survival) ============================
    cross = []
    if not args.no_cross and op in BEHAV_OPS:
        from transformers import AutoModelForCausalLM

        def behaviour_signal(mid):
            m = AutoModelForCausalLM.from_pretrained(mid, attn_implementation="eager", dtype=torch.bfloat16).eval().to(dev)
            V = m.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
            sg = [(lambda s: s + s)([int(x) for x in rng.integers(lo, hi, args.probe_len)]) for _ in range(min(args.probe_seqs, 24))]
            nl2 = m.config.num_hidden_layers; rope = hasattr(m, "model")
            best = 0.0; t1 = t2 = n1 = n2 = 0.0
            with torch.no_grad():
                for s in sg:
                    out = m(input_ids=torch.tensor([s], device=dev), output_attentions=True)
                    Msk = op_masks(s)[op]; L = len(s) // 2
                    lp = F.log_softmax(out.logits[0].float(), -1)
                    for pos in range(1, L - 1):
                        t1 += float(-lp[pos, s[pos + 1]]); n1 += 1
                    for pos in range(L, 2 * L - 1):
                        t2 += float(-lp[pos, s[pos + 1]]); n2 += 1
                    for L2 in range(nl2):
                        at = out.attentions[L2][0].float().cpu().numpy()
                        best = max(best, float((at * Msk[None]).sum((1, 2)).max() / max(Msk.sum(), 1)))
            gtot = (t1 / max(n1, 1)) - (t2 / max(n2, 1))
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
            return {"model": mid.split("/")[-1], "rope": rope, "behaviour_signal": best, "gain": gtot}
        cross.append({"model": args.pretrained, "behaviour_signal": float(mass[reader]), "gain": None})
        for mid in [m.strip() for m in args.cross_models.split(",") if m.strip()]:
            try:
                cross.append(behaviour_signal(mid))
            except Exception as e:  # pragma: no cover
                cross.append({"model": mid, "error": str(e)})
        res["F_cross_model"] = cross
        print("\n[F. CROSS-MODEL] behavioural signal (max head mass on the op's pattern) + induction gain, other architectures:")
        for c in cross:
            if "error" in c:
                print(f"    {c['model']:>16}: [skip] {c['error'][:60]}")
            else:
                g = f"gain {c['gain']:+.2f}" if c.get("gain") is not None else "gain n/a"
                print(f"    {c['model']:>16}: signal {c['behaviour_signal']:.3f}  {g}")

    res["G_sae_operands"] = "NOT RUN — needs a SAE (sae_lens / Gemma Scope); the op's feature-space read/write operands are the next layer."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2, default=float))

    # ============================ figure ============================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.0))
        aB, aC, aD, aE = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
        cB = ["#d62728" if op_eff[t] > thr[t] else "#bbbbbb" for t in TASKS]
        aB.bar(range(len(TASKS)), [op_eff[t] for t in TASKS], color=cB, edgecolor="k")
        aB.plot(range(len(TASKS)), [thr[t] for t in TASKS], "k_", ms=18, mew=2, label="random-control bar")
        aB.set_xticks(range(len(TASKS))); aB.set_xticklabels(TASKS, fontsize=8, rotation=12); aB.axhline(0, color="k", lw=0.5)
        aB.set_title(f"B. causal: which programs need '{op}'", fontsize=10); aB.set_ylabel("ablation damage"); aB.legend(fontsize=7)
        if "key_top" in channel:
            kr = channel["key_rows"]; vr = channel["value_rows"]
            aC.barh(range(len(kr)), [r["collapse"] for r in kr], color="#1f77b4", edgecolor="k", label="KEY/match collapse")
            aC.set_yticks(range(len(kr))); aC.set_yticklabels([r["head"] for r in kr], fontsize=7); aC.invert_yaxis()
            aC.set_title(f"C. channels: key-collapse (top movers dV-out {vr[0]['dvout']:.2f})", fontsize=10); aC.set_xlabel("attention collapse"); aC.legend(fontsize=7)
        else:
            aC.text(0.5, 0.5, "C. channels N/A\n(output op: OV->unembedding)", ha="center", va="center"); aC.axis("off")
        ik = composition["in_key"][:5][::-1]
        aD.barh(range(len(ik)), [s for _, s in ik], color="#2ca02c", edgecolor="k")
        aD.set_yticks(range(len(ik))); aD.set_yticklabels([n for n, _ in ik], fontsize=7)
        aD.set_title(f"D. composition IN->key of {channel.get('reader', name_of(reader))}", fontsize=10); aD.set_xlabel("weight K-composition")
        ns = [c["n"] for c in curve]; es = [c["effect"] for c in curve]
        aE.plot(ns, es, "o-", color="#9467bd"); aE.axhline(max_solo, color="#d62728", ls=":", label=f"best single head {max_solo:+.2f}")
        aE.set_xlabel("op heads ablated (cumulative)"); aE.set_ylabel(f"'{pt}' damage"); aE.legend(fontsize=7)
        aE.set_title("E. redundancy: bottleneck vs population", fontsize=10)
        fig.suptitle(f"Operator dossier — '{op}' ({spec['kind']}) on {args.pretrained}: identity / causal / channels / composition / redundancy", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); args.fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig, dpi=130); print(f"\n[fig] {args.fig}")
    except Exception as e:  # pragma: no cover
        print(f"[fig] skipped: {e}")
    print(f"[done] {args.output}")
    return res


if __name__ == "__main__":
    main()
