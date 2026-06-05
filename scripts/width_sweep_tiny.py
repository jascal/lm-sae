"""N1-width on the tiny LM: forge tax vs SAE over-completeness.

Is the tiny-LM cov95 forge tax EMERGENT (bio regime — roughly constant across SAE
width) or OVER-COMPLETENESS-driven (econ regime — forged cov95 degrades as the SAE
gets wider)? Runs the whole loop (train SAE -> forge -> forged cov95) at several
widths on the same tiny GPT and tabulates host vs forged cov95 vs over-completeness.

Caveat (same as econ's N1-width): each width is a separately-trained SAE at fixed
k, so this conflates trainability/sparsity-ratio with over-completeness — suggestive,
not airtight.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import whole_loop_tiny as wl  # noqa: E402

WIDTHS = [128, 256, 512, 1024, 2048]


def main():
    rows = []
    for w in WIDTHS:
        print(f"\n===== SAE width {w} =====", flush=True)
        out = wl.main(["--width", str(w), "--k", "24", "--steps", "600",
                       "--output", f"runs/wl_w{w}_summary.json"])
        h, f = out["host"], out["forged"]
        rows.append({
            "width": w, "over_complete": out["over_complete"],
            "host_cov95": h["all"]["cov95"], "forged_cov95": f["all"]["cov95"],
            "host_token": h.get("token", {}).get("cov95"),
            "forged_token": f.get("token", {}).get("cov95"),
            "host_mauc": h["all"]["mauc"], "forged_mauc": f["all"]["mauc"],
            "retained_cov95": round(f["all"]["cov95"] / max(h["all"]["cov95"], 1e-9), 3),
        })
    Path("runs/width_sweep_tiny_summary.json").write_text(
        json.dumps({"experiment": "N1-width on tiny LM", "k": 24, "d_model": 128, "sweep": rows},
                   indent=2, default=float))
    print("\n=== TINY-LM N1-WIDTH: forge tax vs over-completeness ===")
    print(f"{'width':>6} {'oc':>5} {'host':>6} {'forged':>7} {'retain':>7} "
          f"{'h_tok':>6} {'f_tok':>6} {'h_mAUC':>7} {'f_mAUC':>7}")
    for r in rows:
        print(f"{r['width']:>6} {r['over_complete']:>4}x {r['host_cov95']:>6.3f} "
              f"{r['forged_cov95']:>7.3f} {r['retained_cov95']:>7.2f} "
              f"{r['host_token']:>6.2f} {r['forged_token']:>6.2f} "
              f"{r['host_mauc']:>7.3f} {r['forged_mauc']:>7.3f}")


if __name__ == "__main__":
    main()
