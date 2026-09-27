"""Print the theory-comparison tables from ug_theory.py summaries."""
import json
import sys

for path in sys.argv[1:]:
    r = json.load(open(path))
    th = list(r["own_grouping"])
    print(f"\n{r['model']}{' (random init)' if r.get('random_init') else ''} L{r['layer']} src={r['src']} "
          f"langs={len(r['langs'])}")
    print("  own grouping (beyond word order + centrality), pooled, 95% CI:")
    for t in th:
        v = r["own_grouping"][t]
        print(f"    {t:6s} {v['value']:.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print("  paired differences:")
    for k, v in r["own_grouping_diff"].items():
        sig = "*" if v["ci95"][0] > 0 or v["ci95"][1] < 0 else " "
        print(f"    {k:13s} {v['value']:+.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] {sig}")
    print("  encompassing: probe trained on B carries A-grouping beyond B (pooled)")
    print("    " + "B\\A".ljust(8) + "".join(a.ljust(8) for a in th))
    for b in th:
        print("    " + b.ljust(8) + "".join("   -    " if a == b else
                                           f"{r['pooled'][b]['encompass'][f'{a}|{b}']:+.3f}  " for a in th))
