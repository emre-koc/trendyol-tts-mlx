from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass(frozen=True)
class Conv1dSpec:
    name: str
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    dilation: int
    groups: int
    causal_padding: int
    output_padding: int
    mlx_weight: str
    mlx_bias: str | None = None

    @classmethod
    def from_manifest_entry(cls, entry: dict[str, Any]) -> "Conv1dSpec":
        if entry["kind"] != "conv1d":
            raise ValueError(f"expected conv1d entry, got {entry['kind']} for {entry['name']}")
        return cls(
            name=entry["name"],
            in_channels=int(entry["in_channels"]),
            out_channels=int(entry["out_channels"]),
            kernel_size=int(entry["kernel_size"]),
            stride=int(entry["stride"]),
            dilation=int(entry["dilation"]),
            groups=int(entry["groups"]),
            causal_padding=int(entry["causal_padding"]),
            output_padding=int(entry.get("output_padding", 0)),
            mlx_weight=entry["mlx_weight"],
            mlx_bias=entry.get("mlx_bias"),
        )


class CausalConv1d:
    """MLX equivalent of VoxCPM AudioVAEV2 CausalConv1d.

    Public tensors use the upstream PyTorch shape `NCT`. Internally MLX uses
    `NLC`, so this wrapper handles layout conversion and causal left padding.
    """

    def __init__(self, spec: Conv1dSpec, tensors: dict[str, mx.array]):
        self.spec = spec
        self.conv = nn.Conv1d(
            spec.in_channels,
            spec.out_channels,
            spec.kernel_size,
            stride=spec.stride,
            padding=0,
            dilation=spec.dilation,
            groups=spec.groups,
            bias=spec.mlx_bias is not None,
        )
        self.conv.weight = tensors[spec.mlx_weight]
        if spec.mlx_bias is not None:
            self.conv.bias = tensors[spec.mlx_bias]

    def __call__(self, x_nct: np.ndarray | mx.array) -> np.ndarray:
        x = mx.array(x_nct) if not isinstance(x_nct, mx.array) else x_nct
        # NCT -> NLC
        x = mx.transpose(x, (0, 2, 1))
        left_pad = self.spec.causal_padding * 2 - self.spec.output_padding
        if left_pad:
            x = mx.pad(x, [(0, 0), (left_pad, 0), (0, 0)])
        y = self.conv(x)
        # NLC -> NCT
        return np.array(mx.transpose(y, (0, 2, 1)))


class AudioVAEDecoderStem:
    """First two AudioVAE decoder causal convs implemented in MLX.

    This covers `decoder.model.0` and `decoder.model.1` for the Trendyol/VoxCPM2
    AudioVAE decoder. It is a deliberately narrow parity milestone on the path
    to the full decoder graph.
    """

    def __init__(self, layers: list[CausalConv1d]):
        if len(layers) != 2:
            raise ValueError("AudioVAEDecoderStem requires exactly two CausalConv1d layers")
        self.layers = layers

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "AudioVAEDecoderStem":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        tensors = mx.load(manifest["weights"])
        conv_entries = {entry["name"]: entry for entry in manifest["conv_layers"]}
        specs = [
            Conv1dSpec.from_manifest_entry(conv_entries["decoder.model.0"]),
            Conv1dSpec.from_manifest_entry(conv_entries["decoder.model.1"]),
        ]
        return cls([CausalConv1d(spec, tensors) for spec in specs])

    def __call__(self, z_nct: np.ndarray | mx.array) -> np.ndarray:
        x: np.ndarray | mx.array = z_nct
        for layer in self.layers:
            x = layer(x)
        return np.asarray(x, dtype=np.float32)
