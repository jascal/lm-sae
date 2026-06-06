"""Forge a host transformer into its feature-native basis — standalone (saeforge public API only).

This is the lm-sae-local equivalent of bio-sae's forge_capability_eval._forge, lifted here so
lm-sae has NO bio-sae dependency. Given a saeforge FeatureBasis, it projects the host model's
weights into the basis and returns the runnable forged torch module (+ the loaded host), via the
public saeforge pipeline (SubspaceProjector -> adapter -> NativeModel.from_projected_weights).
"""
from __future__ import annotations


def forge_native(basis, host_model: str = "gpt2", device: str = "cpu", scale_boost="auto",
                 dtype: str = "float32"):
    """Forge `host_model` with `basis`; return (forged_torch_module, host). forward_mode=native_in_basis."""
    from saeforge import SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel
    from saeforge.utils.host_loader import load_host_for_forge

    projector = SubspaceProjector(basis=basis, scale_boost=scale_boost)
    host = load_host_for_forge(host_model)
    adapter = adapter_for(host)
    weights = projector.project_module(host, attention_width="host")
    config = adapter.build_native_config(host, basis.n_features)
    config.forward_mode = "native_in_basis"
    model = NativeModel.from_projected_weights(config, weights)
    model._move(dtype=dtype, device=device)
    return model.torch_module, host
