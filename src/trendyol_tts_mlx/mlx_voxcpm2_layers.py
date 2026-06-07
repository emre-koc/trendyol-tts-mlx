from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


def _load_manifest_and_tensors(manifest_path: str | Path) -> tuple[dict[str, Any], dict[str, mx.array]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    return manifest, mx.load(manifest["weights"])


def _linear_entries_by_name(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in manifest["linear_layers"]}


@dataclass(frozen=True)
class LinearSpec:
    name: str
    in_features: int
    out_features: int
    mlx_weight: str
    mlx_bias: str | None = None

    @classmethod
    def from_manifest_entry(cls, entry: dict[str, Any]) -> "LinearSpec":
        if entry["kind"] != "linear":
            raise ValueError(f"expected linear entry, got {entry['kind']} for {entry['name']}")
        return cls(
            name=entry["name"],
            in_features=int(entry["in_features"]),
            out_features=int(entry["out_features"]),
            mlx_weight=entry["mlx_weight"],
            mlx_bias=entry.get("mlx_bias"),
        )


class Linear:
    """Minimal MLX linear layer using PyTorch weight layout `[out, in]`."""

    def __init__(self, spec: LinearSpec, tensors: dict[str, mx.array]):
        self.spec = spec
        self.weight = tensors[spec.mlx_weight]
        self.bias = tensors[spec.mlx_bias] if spec.mlx_bias is not None else None

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, name: str) -> "Linear":
        manifest, tensors = _load_manifest_and_tensors(manifest_path)
        entries = _linear_entries_by_name(manifest)
        if name not in entries:
            raise KeyError(f"linear layer {name!r} not found in manifest")
        return cls(LinearSpec.from_manifest_entry(entries[name]), tensors)

    def _call_mx(self, x: np.ndarray | mx.array) -> mx.array:
        y = mx.matmul(mx.array(x) if not isinstance(x, mx.array) else x, mx.transpose(self.weight))
        if self.bias is not None:
            y = y + self.bias
        return y

    def __call__(self, x: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(x)), dtype=np.float32)


class ScalarQuantizationLayer:
    """MLX equivalent of VoxCPM2 ScalarQuantizationLayer eval/inference path."""

    def __init__(self, in_proj: Linear, out_proj: Linear, scale: int):
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.scale = int(scale)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "ScalarQuantizationLayer":
        manifest, tensors = _load_manifest_and_tensors(manifest_path)
        linear_entries = _linear_entries_by_name(manifest)
        return cls(
            in_proj=Linear(LinearSpec.from_manifest_entry(linear_entries["fsq_layer.in_proj"]), tensors),
            out_proj=Linear(LinearSpec.from_manifest_entry(linear_entries["fsq_layer.out_proj"]), tensors),
            scale=int(manifest["scalar_quantization_scale"]),
        )

    def _call_mx(self, hidden: np.ndarray | mx.array) -> mx.array:
        x = self.in_proj._call_mx(hidden)
        x = mx.tanh(x)
        x = mx.round(x * self.scale) / self.scale
        return self.out_proj._call_mx(x)

    def __call__(self, hidden: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(hidden)), dtype=np.float32)


class StopPredictor:
    """MLX equivalent of `stop_head(SiLU(stop_proj(lm_hidden)))`."""

    def __init__(self, stop_proj: Linear, stop_head: Linear):
        self.stop_proj = stop_proj
        self.stop_head = stop_head

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "StopPredictor":
        manifest, tensors = _load_manifest_and_tensors(manifest_path)
        linear_entries = _linear_entries_by_name(manifest)
        return cls(
            stop_proj=Linear(LinearSpec.from_manifest_entry(linear_entries["stop_proj"]), tensors),
            stop_head=Linear(LinearSpec.from_manifest_entry(linear_entries["stop_head"]), tensors),
        )

    def _call_mx(self, lm_hidden: np.ndarray | mx.array) -> mx.array:
        x = self.stop_proj._call_mx(lm_hidden)
        x = x * mx.sigmoid(x)
        return self.stop_head._call_mx(x)

    def __call__(self, lm_hidden: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(lm_hidden)), dtype=np.float32)
