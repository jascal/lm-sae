"""Cross-model CIRCUIT dossier — the operator-dossier battery, lifted to composed circuits, on the ResidualVM.

`operator_dossier_xmodel.py` gives every *operator* a deep cross-model causal dossier (identity + necessity +
channel + redundancy). The circuit catalog had only the defining-edge liveness per circuit — no equivalent
per-circuit necessity / sufficiency / redundancy battery. This closes that parity gap, and it is built ON the
unified `ResidualVM` debugger (its `find_heads` / `ablate_heads` / `nll`), dogfooding the tool the whole program
asked for.

For each cross-model circuit (a reader-op fed by a writer-op) and each model, two clean next-token metrics the VM
supports natively — **induction-NLL** (in-context copy, on repeated-random) and **generic-NLL** (general LM, on
prose) — drive a uniform battery:

  IDENTITY    — VM.find_heads locates the circuit's reader heads + writer heads behaviourally (+ depth, mass);
  NECESSITY   — ablate the circuit's heads (and reader-only / writer-only) -> Δ induction-NLL + Δ generic-NLL
                (is the circuit load-bearing for in-context copy? for general LM?);
  SUFFICIENCY — keep ONLY the circuit's heads, mean-ablate every other head (MLPs intact) -> reconstruction
                coverage of induction-NLL + generic-NLL (a small head-set that reconstructs = executable decompile);
  REDUNDANCY  — reader-head solo-vs-cumulative on induction-NLL -> bottleneck (one head carries it) vs distributed;
  EDGE        — the defining composition edge (key collapse + z + value ΔV-out + concentration), HARVESTED from the
                committed `circuit_content_patch` run (no recompute).

Circuits: induction (induction <- prevtok), positional_broadcast (prevtok <- sink), duplicate (duplicate reader).
Output: runs/disassembly/circuits/dossier_summary.json (merge-safe across models). `circuit_catalog_doc.py` turns
it into a "Cross-model causal dossier" section on each circuit page.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# circuit -> (reader_op, writer_op | None, behaviour). The reader op names the circuit; the writer is the upstream
# stage whose edge defines it (None where the writer is layer-0 / not a clean single op, e.g. duplicate).
CIRCUITS = {
    "induction": {"reader": "induction", "writer": "prevtok",
                  "desc": "prev-token head --K--> induction head (the in-context-copy macro)"},
    "positional_broadcast": {"reader": "prevtok", "writer": "sink",
                             "desc": "early sink/write-hub --K--> prev-token head's key (absolute-position broadcast)"},
    "duplicate": {"reader": "duplicate", "writer": None,
                  "desc": "same-token reader (duplicate-token detection; IOI initiator)"},
}


def gpt2_rep(pids, rng):
    """Repeated-random token sampler for GPT-2 — draw from the frequent prose tokens (matches operator_dossier)."""
    cnt = {}
    for t in pids:
        cnt[t] = cnt.get(t, 0) + 1
    vocab = [t for t, _ in sorted(cnt.items(), key=lambda r: -r[1])[:400]]
    return lambda L: [int(vocab[i]) for i in rng.integers(0, len(vocab), L)]


def dossier_one_model(vm, model_id, args):
    rng = np.random.default_rng(args.seed)
    import urllib.request
    prose = urllib.request.urlopen(urllib.request.Request(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        headers={"User-Agent": "Mozilla/5.0"}), timeout=15).read().decode("utf-8", "ignore")[:160000]
    pids = vm.tok(prose)["input_ids"]
    chunks = [pids[i:i + args.ctx] for i in range(0, len(pids), args.ctx) if len(pids[i:i + args.ctx]) >= 8][: args.chunks]
    vm.fit_means(chunks)
    # repeated-random probes (induction substrate)
    if "gpt2" in model_id.lower():
        rep = gpt2_rep(pids, rng)
    else:
        V = vm.model.config.vocab_size; lo, hi = int(0.02 * V), int(0.4 * V)
        rep = lambda L: [int(x) for x in rng.integers(lo, hi, L)]            # noqa: E731
    rep_seqs = [(lambda s: s + s)(rep(args.rep_len)) for _ in range(args.probes)]
    id_rep = rep_seqs[: args.id_probes]            # content ops (induction/duplicate) read repeated-random
    id_prose = chunks[: args.id_probes]            # positional/addressing ops (prevtok/sink) read prose

    H = vm.H; nL = vm.nL
    all_heads = [(L, h) for L in range(nL) for h in range(H)]

    def ind_target(s):
        L = len(s) // 2
        return [(p, s[p + 1]) for p in range(L, 2 * L - 1)]

    def gen_target(s):
        return [(i, s[i + 1]) for i in range(len(s) - 1)]

    def ind_nll(ablate=()):
        with vm.ablate_heads(set(ablate)):
            return vm.nll(rep_seqs, ind_target)

    def gen_nll(ablate=()):
        with vm.ablate_heads(set(ablate)):
            return vm.nll(chunks, gen_target)

    base_ind = vm.nll(rep_seqs, ind_target); base_gen = vm.nll(chunks, gen_target)
    floor_ind = ind_nll(all_heads); floor_gen = gen_nll(all_heads)

    def cov(keep, floor, base, kind):
        keep_nll = (ind_nll if kind == "ind" else gen_nll)([hh for hh in all_heads if hh not in keep])
        return (floor - keep_nll) / (floor - base + 1e-9), keep_nll

    # locate the universal behavioural ops once (reused across circuits sharing a reader/writer)
    located = {}
    for op in {"induction", "prevtok", "duplicate", "sink"}:
        seqs_for = id_rep if op in ("induction", "duplicate") else id_prose
        n_top = 2 if op in ("prevtok", "sink") else 4
        heads, mass = vm.find_heads(seqs_for, op, top=n_top, min_mass=args.min_mass)
        located[op] = {"heads": heads, "mass": mass}

    edge_harvest = load_edge_harvest(args.content_patch, model_id)

    out = {}
    for cname, spec in CIRCUITS.items():
        reader_op = spec["reader"]; writer_op = spec["writer"]
        reader_heads = located[reader_op]["heads"]; rmass = located[reader_op]["mass"]
        writer_heads = located[writer_op]["heads"] if writer_op else []
        circuit_heads = list({*reader_heads, *writer_heads})
        # NECESSITY
        nec = {
            "circuit": {"ind": ind_nll(circuit_heads) - base_ind, "gen": gen_nll(circuit_heads) - base_gen},
            "reader": {"ind": ind_nll(reader_heads) - base_ind, "gen": gen_nll(reader_heads) - base_gen},
        }
        if writer_heads:
            nec["writer"] = {"ind": ind_nll(writer_heads) - base_ind, "gen": gen_nll(writer_heads) - base_gen}
        # SUFFICIENCY (keep only the circuit heads)
        ci, ki = cov(set(circuit_heads), floor_ind, base_ind, "ind")
        cg, kg = cov(set(circuit_heads), floor_gen, base_gen, "gen")
        suff = {"ind_coverage": ci, "gen_coverage": cg, "n_heads": len(circuit_heads),
                "ind_keep_nll": ki, "gen_keep_nll": kg}
        # REDUNDANCY (reader-head solo vs cumulative on induction-NLL)
        solo = sorted([(f"{L}.{h}", ind_nll([(L, h)]) - base_ind) for (L, h) in reader_heads], key=lambda r: -r[1])
        acc = []; curve = []
        for nm, _ in solo:
            acc.append(tuple(int(x) for x in nm.split(".")))
            curve.append({"n": len(acc), "ind": ind_nll(acc) - base_ind})
        max_solo = max((e for _, e in solo), default=0.0); full = nec["reader"]["ind"]
        redundancy = {"solo": solo, "curve": curve, "max_solo": max_solo, "full": full,
                      "bottleneck": bool(full <= 1.4 * max_solo and max_solo > 0.1)}

        def name_depth(heads, mass):
            if not heads:
                return None, None, None
            top = heads[0]; i = top[0] * H + top[1]
            return f"{top[0]}.{top[1]}", (top[0] / (nL - 1) if nL > 1 else 0.0), float(mass[i])
        rtop, rdepth, rmassv = name_depth(reader_heads, rmass)
        wtop, wdepth, wmassv = name_depth(writer_heads, located[writer_op]["mass"]) if writer_op else (None, None, None)
        out[cname] = {
            "desc": spec["desc"], "reader_op": reader_op, "writer_op": writer_op,
            "reader_heads": [f"{L}.{h}" for L, h in reader_heads], "writer_heads": [f"{L}.{h}" for L, h in writer_heads],
            "reader_top": rtop, "reader_depth": rdepth, "reader_mass": rmassv,
            "writer_top": wtop, "writer_depth": wdepth, "writer_mass": wmassv,
            "necessity": nec, "sufficiency": suff, "redundancy": redundancy,
            "edge": edge_harvest.get(cname),
        }
    return {"model": model_id.split("/")[-1], "rope": not vm.is_gpt2, "n_layers": nL, "n_heads": H,
            "base_induction_nll": base_ind, "base_generic_nll": base_gen,
            "floor_induction_nll": floor_ind, "floor_generic_nll": floor_gen, "circuits": out}


# circuit -> the content-patch reader key the edge lives under (the defining composition edge).
EDGE_KEY = {"induction": "induction", "positional_broadcast": "prevtok", "duplicate": "duplicate"}


def load_edge_harvest(path, model_id):
    """Harvest the defining-edge channel (key collapse / z / value / concentration) from the committed content-patch run."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    short = model_id.split("/")[-1]
    rec = next((r for r in d.get("results", []) if r.get("model", "").split("/")[-1] == short), None)
    if not rec or "circuits" not in rec:
        return {}
    out = {}
    for cname, ckey in EDGE_KEY.items():
        c = rec["circuits"].get(ckey)
        if not c or "top_collapse" not in c:
            continue
        med = c.get("median_collapse", 0.0) or 0.0
        out[cname] = {
            "reader": c.get("reader"), "writer": c.get("key_top_head"),
            "key_collapse": c.get("top_collapse"), "key_z": c.get("top_z"),
            "is_prevtok_writer": c.get("top_is_prevtok_head"), "is_sink_writer": c.get("top_is_sink"),
            "value_mover": c.get("value_top_head"), "value_dvout": c.get("value_top_dvout"),
            "concentration": (c.get("top_collapse", 0.0) / (abs(med) + 1e-9)) if c.get("top_collapse") else 0.0,
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="gpt2,gpt2-medium,gpt2-large,google/gemma-2-2b,unsloth/Llama-3.2-1B,Qwen/Qwen2.5-1.5B")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--probes", type=int, default=20)
    p.add_argument("--id-probes", type=int, default=16)
    p.add_argument("--rep-len", type=int, default=22)
    p.add_argument("--min-mass", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--content-patch", type=Path, default=Path("runs/gemma/circuit_content_patch_summary.json"))
    p.add_argument("--outdir", type=Path, default=Path("runs/disassembly/circuits"))
    args = p.parse_args(argv)

    import torch
    from residual_vm import ResidualVM
    dev = args.device if torch.cuda.is_available() else "cpu"
    results = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {mid} ===")
        try:
            vm = ResidualVM(mid, device=dev)
            r = dossier_one_model(vm, mid, args)
            del vm
            if dev == "cuda":
                torch.cuda.empty_cache()
            results.append(r)
            for cname, c in r["circuits"].items():
                nec = c["necessity"]["circuit"]; suff = c["sufficiency"]; red = c["redundancy"]
                print(f"  {cname:>20}: reader {c['reader_top']} writer {c['writer_top']} | "
                      f"necessity Δind {nec['ind']:+.2f} Δgen {nec['gen']:+.2f} | "
                      f"suff ind {suff['ind_coverage']:+.0%} gen {suff['gen_coverage']:+.0%} | "
                      f"{'bottleneck' if red['bottleneck'] else 'distributed'}")
        except Exception as e:  # pragma: no cover
            print(f"  [skip] {e}"); results.append({"model": mid.split("/")[-1], "error": str(e)})

    args.outdir.mkdir(parents=True, exist_ok=True)
    sumpath = args.outdir / "dossier_summary.json"
    prior = json.loads(sumpath.read_text()).get("results", []) if sumpath.exists() else []
    done = {r["model"] for r in results}
    merged = results + [r for r in prior if r.get("model") not in done]
    out = {"experiment": "cross-model circuit dossier (necessity / sufficiency / redundancy + harvested edge, on ResidualVM)",
           "circuits": list(CIRCUITS), "results": merged}
    sumpath.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[done] {len([r for r in results if 'circuits' in r])} models → {sumpath}")
    return out


if __name__ == "__main__":
    main()
