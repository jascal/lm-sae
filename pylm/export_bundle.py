"""Write a pylm model as a **fieldrun bundle** — the flat, mmap-friendly format the Rust runtime (`fieldrun`) consumes.

The numpy kernels read `.npz` (a zip of .npy) directly; the Rust runtime wants something it can mmap with no zip/npy
parsing. The *fieldrun bundle format* (spec: `fieldrun/FORMAT.md`) is that lm-sae -> fieldrun contract: a JSON manifest
(`<name>.fieldrun.json`) naming the format/version/arch/config and, for each weight array, its dtype, shape, and byte
offset/length into a single raw little-endian blob (`<name>.fieldrun.bin`). The kernel-specific config int vector and
float vector go in the manifest; weights are raw f32 in manifest order (this first cut). fp16/int8 bundles (the
in-RAM-precision path) extend the `dtype` field with a sibling `<name>__scale` array later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FORMAT = "fieldrun-bundle"
VERSION = 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=Path("pylm/weights_gpt2.npz"))
    p.add_argument("--arch", default="gpt2", help="gpt2 · rope · gemma — which kernel the Rust side runs")
    p.add_argument("--store", type=Path, default=None, help="retrieval store.json to embed (makes the bundle whole)")
    p.add_argument("--out", type=Path, default=Path("pylm/gpt2"), help="stem → <stem>.fieldrun.json + <stem>.fieldrun.bin")
    args = p.parse_args(argv)

    raw = dict(np.load(args.npz))
    cfg_key = "config" if "config" in raw else "cfg_i"
    manifest = {"format": FORMAT, "version": VERSION, "arch": args.arch,
                "config": [int(x) for x in raw[cfg_key]], "arrays": []}
    if "cfg_f" in raw:
        manifest["config_f"] = [float(x) for x in raw["cfg_f"]]
    if args.store is not None:                                        # embed the retrieval tier → one self-contained model
        manifest["store"] = json.loads(args.store.read_text())

    blob = bytearray(); offset = 0
    for name, a in raw.items():
        if name in ("config", "cfg_i", "cfg_f"):
            continue
        if name.endswith("__scale") or name.endswith("__rowscale"):
            raise SystemExit(f"int8 bundles not supported yet (saw {name}); export an fp32 or fp16 npz")
        if a.dtype == np.float16:                                     # preserve fp16 (the in-RAM-precision path) ...
            a = np.ascontiguousarray(a, dtype="<f2"); dt = "f16"
        else:                                                         # ... else raw little-endian f32
            a = np.ascontiguousarray(a, dtype="<f4"); dt = "f32"
        b = a.tobytes(); blob += b
        manifest["arrays"].append({"name": name, "dtype": dt, "shape": list(a.shape),
                                   "offset": offset, "bytes": len(b)})
        offset += len(b)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    stem = str(args.out)
    Path(stem + ".fieldrun.json").write_text(json.dumps(manifest))
    Path(stem + ".fieldrun.bin").write_bytes(bytes(blob))
    mb = len(blob) / 1e6; has_store = "store" in manifest
    dts = sorted({a["dtype"] for a in manifest["arrays"]})
    print(f"[fieldrun-bundle v{VERSION}] {args.npz.name} → {stem}.fieldrun.json + .bin "
          f"({len(manifest['arrays'])} arrays, {mb:.0f} MB {'/'.join(dts)}, arch={args.arch}, "
          f"store={'embedded' if has_store else 'no'})")


if __name__ == "__main__":
    main()
