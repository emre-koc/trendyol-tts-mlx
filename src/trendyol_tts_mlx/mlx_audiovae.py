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
        if entry["kind"] not in {"conv1d", "conv_transpose1d"}:
            raise ValueError(f"expected conv entry, got {entry['kind']} for {entry['name']}")
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

    def _call_mx(self, x_nct: np.ndarray | mx.array) -> mx.array:
        x = mx.array(x_nct) if not isinstance(x_nct, mx.array) else x_nct
        # NCT -> NLC
        x = mx.transpose(x, (0, 2, 1))
        left_pad = self.spec.causal_padding * 2 - self.spec.output_padding
        if left_pad:
            x = mx.pad(x, [(0, 0), (left_pad, 0), (0, 0)])
        y = self.conv(x)
        # NLC -> NCT
        return mx.transpose(y, (0, 2, 1))

    def __call__(self, x_nct: np.ndarray | mx.array) -> np.ndarray:
        return np.array(self._call_mx(x_nct))


class CausalTransposeConv1d:
    """MLX equivalent of VoxCPM AudioVAEV2 CausalTransposeConv1d."""

    def __init__(self, spec: Conv1dSpec, tensors: dict[str, mx.array]):
        self.spec = spec
        self.conv = nn.ConvTranspose1d(
            spec.in_channels,
            spec.out_channels,
            spec.kernel_size,
            stride=spec.stride,
            padding=0,
            dilation=spec.dilation,
            output_padding=0,
            bias=spec.mlx_bias is not None,
        )
        self.conv.weight = tensors[spec.mlx_weight]
        if spec.mlx_bias is not None:
            self.conv.bias = tensors[spec.mlx_bias]

    def _call_mx(self, x_nct: np.ndarray | mx.array) -> mx.array:
        x = mx.array(x_nct) if not isinstance(x_nct, mx.array) else x_nct
        x = mx.transpose(x, (0, 2, 1))
        y = self.conv(x)
        crop = self.spec.causal_padding * 2 - self.spec.output_padding
        if crop:
            y = y[:, :-crop, :]
        return mx.transpose(y, (0, 2, 1))

    def __call__(self, x_nct: np.ndarray | mx.array) -> np.ndarray:
        return np.array(self._call_mx(x_nct))


class Snake1d:
    def __init__(self, alpha: mx.array):
        self.alpha = alpha

    def _call_mx(self, x_nct: np.ndarray | mx.array) -> mx.array:
        x = mx.array(x_nct) if not isinstance(x_nct, mx.array) else x_nct
        return x + (1.0 / (self.alpha + 1e-9)) * mx.square(mx.sin(self.alpha * x))

    def __call__(self, x_nct: np.ndarray | mx.array) -> np.ndarray:
        return np.array(self._call_mx(x_nct))


class CausalResidualUnit:
    def __init__(self, snake_0: Snake1d, conv_1: CausalConv1d, snake_2: Snake1d, conv_3: CausalConv1d):
        self.snake_0 = snake_0
        self.conv_1 = conv_1
        self.snake_2 = snake_2
        self.conv_3 = conv_3

    def _call_mx(self, x_nct: np.ndarray | mx.array) -> mx.array:
        x = mx.array(x_nct) if not isinstance(x_nct, mx.array) else x_nct
        y = self.snake_0._call_mx(x)
        y = self.conv_1._call_mx(y)
        y = self.snake_2._call_mx(y)
        y = self.conv_3._call_mx(y)
        return x + y

    def __call__(self, x_nct: np.ndarray | mx.array) -> np.ndarray:
        return np.array(self._call_mx(x_nct))


class AudioVAEFirstDecoderBlock:
    """First full decoder upsampling block (`decoder.model.2`) in MLX."""

    def __init__(
        self,
        pre_snake: Snake1d,
        upsample: CausalTransposeConv1d,
        residual_units: list[CausalResidualUnit],
    ):
        if len(residual_units) != 3:
            raise ValueError("AudioVAEFirstDecoderBlock requires exactly three residual units")
        self.pre_snake = pre_snake
        self.upsample = upsample
        self.residual_units = residual_units

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "AudioVAEFirstDecoderBlock":
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        tensors = mx.load(manifest["weights"])
        conv_entries = {entry["name"]: entry for entry in manifest["conv_layers"]}

        def conv(name: str) -> CausalConv1d:
            return CausalConv1d(Conv1dSpec.from_manifest_entry(conv_entries[name]), tensors)

        def snake(name: str) -> Snake1d:
            return Snake1d(tensors[name])

        def residual(prefix: str) -> CausalResidualUnit:
            return CausalResidualUnit(
                snake(f"{prefix}.block.0.alpha"),
                conv(f"{prefix}.block.1"),
                snake(f"{prefix}.block.2.alpha"),
                conv(f"{prefix}.block.3"),
            )

        upsample_spec = Conv1dSpec.from_manifest_entry(conv_entries["decoder.model.2.block.1"])
        return cls(
            pre_snake=snake("decoder.model.2.block.0.alpha"),
            upsample=CausalTransposeConv1d(upsample_spec, tensors),
            residual_units=[
                residual("decoder.model.2.block.2"),
                residual("decoder.model.2.block.3"),
                residual("decoder.model.2.block.4"),
            ],
        )

    def _call_mx(self, x_nct: np.ndarray | mx.array) -> mx.array:
        x = self.pre_snake._call_mx(x_nct)
        x = self.upsample._call_mx(x)
        for unit in self.residual_units:
            x = unit._call_mx(x)
        return x

    def __call__(self, x_nct: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(x_nct)), dtype=np.float32)


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
