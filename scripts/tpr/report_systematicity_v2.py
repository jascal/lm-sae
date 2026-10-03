"""Render the v2 results section from runs/tpr/systematicity_v2_*_summary.json (no numbers by hand).

    .venv/bin/python scripts/tpr/report_systematicity_v2.py > /tmp/results.md
"""
import json
from pathlib import Path

RUNS = Path("runs/tpr")


def fmt(cell):
    if cell is None:
        return "—"
    acc, n = cell
    return f"{acc:.3f} (n={n})"


def model_section(path):
    s = json.loads(path.read_text())
    lines = [f"### {s['model']} (layers {s['layers']}, primary {s['primary_layer']})", ""]
    for fam in ("LIST", "SVO"):
        F = s[fam]
        lines += [f"**{fam}**: test strata n_w = 0/1/2/≥3: {F['test_strata']}.", "",
                  "| layer | arm | n_w = 0 | n_w = 1 | n_w = 2 | n_w ≥ 3 |", "|---|---|---|---|---|---|"]
        for layer, L in F["layers"].items():
            star = "*" if int(layer) == s["primary_layer"] else ""
            lam = L["tpr"]["lambda"]
            for arm, cells in (("real", L["real"]), (f"tpr (λ={lam:g}, 3-seed mean)", L["tpr"]["by_stratum"]),
                               ("atomic", L["atomic"])):
                lines.append(f"| {layer}{star} | {arm} | " + " | ".join(fmt(cells.get(str(k))) for k in range(4)) + " |")
        P = F["layers"][str(s["primary_layer"])]
        st, pa = P["P2_strict"], P["P2_paper"]
        lines += ["", f"- **P2-strict** (primary layer): n_w=0 {st['acc_nw0']:.3f}, n_w≥1 {st['acc_nw_ge1']:.3f} → "
                  f"**{'PASS' if st['passes'] else 'FAIL'}**"]
        detail = "; ".join(f"n_w={m}: {v['hits']}/{v['n']} vs 1/{m}! = {v['baseline']:.3f}, p = {v['p']:.2g}"
                           for m, v in pa["strata"].items()) or "no stratum with n_w ≥ 2 and ≥ 30 cases"
        lines.append(f"- **P2-paper**: {detail} → **{'PASS' if pa['passes'] else 'FAIL'}**")
        lines.append(f"- TPR seed range by stratum: {P['tpr']['by_stratum_seed_range']}; "
                     f"λ selection (seen-pair validation): {P['tpr']['val_acc_by_lambda']}")
        c = P["conjunction"]
        lines += ["", "| SAE width | eligible | conjunctive | permutation null | planted additive | ReLU-additive (diag.) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for w, v in c.items():
            def f3(x):
                return "—" if x is None else f"{x:.3f}"
            lines.append(f"| {w} | {v['n_eligible']} | {f3(v['conjunctive'])} | {f3(v['null'])} | "
                         f"{f3(v['planted_additive'])} | {f3(v['relu_additive_diagnostic'])} |")
        p4 = P["P4_prime"]
        verdict = "PASS" if p4["passes"] else ("VOID (non-specific)" if not p4["specific"] else "FAIL")
        lines += ["", f"- **P4′**: specific={p4['specific']}, passes={p4['passes']}, refuting={p4['refuting']} → "
                  f"**{verdict}**", ""]
    return s, lines


def verdicts(gpt2):
    def fam(f):
        return gpt2[f]["layers"][str(gpt2["primary_layer"])]
    strict = all(fam(f)["P2_strict"]["passes"] for f in ("LIST", "SVO"))
    paper_fail = all(not fam(f)["P2_paper"]["passes"] for f in ("LIST", "SVO"))
    binding = "established" if strict else ("refuted" if paper_fail else "partial / open")
    p4 = [fam(f)["P4_prime"] for f in ("LIST", "SVO")]
    conj = "established" if all(p["passes"] for p in p4) else ("refuted" if all(p["refuting"] for p in p4) else "open")
    return binding, conj


def main():
    out = ["## Results", ""]
    summaries = {}
    for name in ("gpt2", "Qwen2.5-0.5B"):
        path = RUNS / f"systematicity_v2_{name}_summary.json"
        if path.exists():
            s, lines = model_section(path)
            summaries[name] = s
            out += lines
    if "gpt2" in summaries:
        b, c = verdicts(summaries["gpt2"])
        out += ["### Verdicts (GPT-2 small, primary layer, by the pre-registered rules)", "",
                f"- Systematic binding: **{b}**.", f"- Conjunctive SAE latents: **{c}**."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
