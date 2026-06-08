"""The decompilable fraction vs model scale — how much of an LLM is a small Python program, as a function of size?

Runs the pylm decompile→validate loop across a controlled model ladder (Pythia 14m→1.4b: one GPT-NeoX architecture,
same data, six sizes) + GPT-2 for reference. For each host: capture the model into a flat store (`capture.py`), then
measure pylm↔model top-1 agreement (the decompilable fraction) on held-out corpus (`validate.py`). The scaling laws
predict the decompilable fraction FALLS with size — smaller models are more n-gram/induction-like, bigger ones carry
more irreducible composition (the entangled core / forge tax grows with capability). Pure orchestration; pylm itself
stays neural-net-free.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture  # noqa: E402
import validate  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="EleutherAI/pythia-14m,EleutherAI/pythia-70m,EleutherAI/pythia-160m,"
                                       "EleutherAI/pythia-410m,EleutherAI/pythia-1b,EleutherAI/pythia-1.4b,gpt2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-eval", type=int, default=3000)
    p.add_argument("--outdir", type=Path, default=Path("runs/pylm"))
    args = p.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        tag = mid.split("/")[-1]
        store = f"pylm/store_{tag}.json"; ids = f"pylm/holdout_{tag}.json"
        print(f"\n##### {mid} #####")
        try:
            capture.main(["--model", mid, "--out", store, "--ids-out", ids, "--device", args.device])
            r = validate.main(["--store", store, "--ids", ids, "--model", mid, "--device", args.device,
                               "--n-eval", str(args.n_eval), "--out", str(args.outdir / f"validate_{tag}.json")])
            rows.append({"model": tag, "pylm_corpus_top1": r["pylm_corpus_top1"],
                         "model_corpus_top1": r.get("model_corpus_top1"),
                         "decompilable_fraction": r.get("decompilable_fraction"),
                         "pylm_over_model_acc": r.get("pylm_over_model_acc"), "program_loc": r["program_loc"]})
        except Exception as e:  # pragma: no cover
            import traceback; traceback.print_exc(); print(f"  [skip] {e}"); rows.append({"model": tag, "error": str(e)})

    (args.outdir / "ladder_summary.json").write_text(json.dumps({"experiment": "pylm decompilable fraction vs model scale", "results": rows}, indent=2, default=float))
    print("\n========== pylm decompilable fraction vs scale ==========")
    print(f"{'model':>16} {'pylm-acc':>9} {'model-acc':>10} {'decompilable':>13} {'pylm/model':>11}")
    for r in rows:
        if "decompilable_fraction" not in r or r["decompilable_fraction"] is None:
            print(f"{r['model']:>16}   {r.get('error', 'n/a')[:40]}"); continue
        print(f"{r['model']:>16} {r['pylm_corpus_top1']:>8.1%} {r['model_corpus_top1']:>9.1%} "
              f"{r['decompilable_fraction']:>12.1%} {r['pylm_over_model_acc']:>10.0%}")
    print(f"[done] → {args.outdir / 'ladder_summary.json'}")


if __name__ == "__main__":
    main()
