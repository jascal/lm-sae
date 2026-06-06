"""Locate (or download) a Gemma Scope SAE `params.npz` for a layer — portable across machines.

Replaces a previously hardcoded `~/.cache/huggingface/...` glob. Resolution order:
  1. `scope_path` (the --scope-path arg): an explicit local directory or glob (offline / custom cache);
  2. the local Hugging Face hub cache (HF_HUB_CACHE / $HF_HOME) — no network if already downloaded;
  3. download from the Hub (`google/gemma-scope-2b-pt-res`) via huggingface_hub.
Returns the filesystem path to a `params.npz`. Gemma Scope ships several `average_l0_*` per layer;
we take the lowest-l0 (sparsest) by default unless `l0` is given.
"""
from __future__ import annotations

import glob
from pathlib import Path

REPO = "google/gemma-scope-2b-pt-res"


def _rel(layer, width):
    return f"layer_{layer}/width_{width}/average_l0_*/params.npz"


def _pick(files):
    # deterministic: lowest average_l0 first
    def l0(f):
        try:
            return int(f.split("average_l0_")[1].split("/")[0])
        except Exception:
            return 1 << 30
    return sorted(files, key=l0)[0]


def scope_npz(layer, width="16k", repo=REPO, scope_path=None, l0=None):
    """Path to a Gemma Scope params.npz for `layer` (cached → downloaded as needed)."""
    rel = _rel(layer, width)
    # 1. explicit local path / glob
    if scope_path:
        sp = str(scope_path)
        cands = glob.glob(str(Path(sp) / "**" / rel), recursive=True) or glob.glob(sp)
        if cands:
            return _pick(cands)
    # 2. local HF hub cache (portable — resolve the real cache dir, never a hardcoded home)
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        repo_dir = "models--" + repo.replace("/", "--")
        cands = glob.glob(str(Path(HF_HUB_CACHE) / repo_dir / "**" / rel), recursive=True)
        if cands:
            return _pick(cands)
    except Exception:
        pass
    # 3. download from the Hub
    from huggingface_hub import HfApi, hf_hub_download
    prefix = f"layer_{layer}/width_{width}/average_l0_"
    files = [f for f in HfApi().list_repo_files(repo) if f.startswith(prefix) and f.endswith("params.npz")]
    if l0 is not None:
        files = [f for f in files if f"average_l0_{l0}/" in f] or files
    if not files:
        raise FileNotFoundError(f"no Gemma Scope SAE for layer {layer} width {width} in {repo}")
    return hf_hub_download(repo_id=repo, filename=_pick(files))
