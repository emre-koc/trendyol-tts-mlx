from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .mlx_voxcpm2_layers import Linear, _load_manifest_and_tensors


class MiniCPMRMSNorm:
    """MLX equivalent of VoxCPM MiniCPMRMSNorm.

    Upstream formula:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        hidden = hidden * rsqrt(variance + eps)
        return hidden * weight
    """

    def __init__(self, weight: mx.array, eps: float = 1e-6):
        self.weight = weight
        self.eps = float(eps)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, name: str) -> "MiniCPMRMSNorm":
        manifest, tensors = _load_manifest_and_tensors(manifest_path)
        parameter_entries = {entry["name"]: entry for entry in manifest.get("parameters", [])}
        key = f"{name}.weight"
        if key not in parameter_entries:
            raise KeyError(f"RMSNorm parameter {key!r} not found in manifest")
        eps = float(manifest.get("encoder_rms_norm_eps", 1e-6))
        return cls(tensors[parameter_entries[key]["mlx_tensor"]], eps=eps)

    def _call_mx(self, hidden: np.ndarray | mx.array) -> mx.array:
        x = mx.array(hidden) if not isinstance(hidden, mx.array) else hidden
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        y = x * mx.rsqrt(variance + self.eps)
        return y * self.weight

    def __call__(self, hidden: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(hidden)), dtype=np.float32)


class MiniCPMMLP:
    """MLX equivalent of VoxCPM MiniCPMMLP.

    Upstream formula:
        down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, gate_proj: Linear, up_proj: Linear, down_proj: Linear):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, name: str) -> "MiniCPMMLP":
        return cls(
            gate_proj=Linear.from_manifest(manifest_path, f"{name}.gate_proj"),
            up_proj=Linear.from_manifest(manifest_path, f"{name}.up_proj"),
            down_proj=Linear.from_manifest(manifest_path, f"{name}.down_proj"),
        )

    def _call_mx(self, hidden: np.ndarray | mx.array) -> mx.array:
        gate = self.gate_proj._call_mx(hidden)
        gate = gate * mx.sigmoid(gate)
        up = self.up_proj._call_mx(hidden)
        return self.down_proj._call_mx(gate * up)

    def __call__(self, hidden: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(hidden)), dtype=np.float32)
