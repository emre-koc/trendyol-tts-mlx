#!/usr/bin/env python3
"""Parity check for the first exported AudioVAE decoder conv in MLX.

This checks one real layer end-to-end:
PyTorch CausalConv1d(weight_norm resolved) vs mlx.nn.Conv1d with exported layout
and explicit causal left padding. It is intentionally small; the next port step
is extending this pattern to the whole AudioVAE decoder graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from huggingface_hub import snapshot_download

from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2


def load_torch_vae(model_id: str) -> AudioVAEV2:
    model_dir = Path(snapshot_download(model_id, local_files_only=True))
    cfg = json.loads((model_dir / "config.json").read_text())
    vae = AudioVAEV2(config=AudioVAEConfigV2(**cfg["audio_vae_config"]))
    ckpt = torch.load(model_dir / "audiovae.pth", map_location="cpu", weights_only=True)
    vae.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    return vae.eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Trendyol/Trendyol-TTS")
    parser.add_argument("--manifest", default="models/Trendyol-TTS-mlx/audiovae_decoder_manifest.json")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    first = manifest["conv_layers"][0]
    weights = mx.load(manifest["weights"])

    torch_vae = load_torch_vae(args.model_id)
    torch_layer = dict(torch_vae.named_modules())[first["name"]]

    rng = np.random.default_rng(42)
    x_nct = rng.standard_normal((1, first["in_channels"], 9), dtype=np.float32)
    with torch.no_grad():
        y_torch = torch_layer(torch.from_numpy(x_nct)).detach().cpu().numpy()

    # MLX Conv1d consumes NLC. CausalConv1d pads left by padding*2-output_padding.
    x_nlc = mx.array(np.transpose(x_nct, (0, 2, 1)))
    left_pad = first["causal_padding"] * 2 - first.get("output_padding", 0)
    x_nlc = mx.pad(x_nlc, [(0, 0), (left_pad, 0), (0, 0)])

    conv = nn.Conv1d(
        first["in_channels"],
        first["out_channels"],
        first["kernel_size"],
        stride=first["stride"],
        padding=0,
        dilation=first["dilation"],
        groups=first["groups"],
        bias="mlx_bias" in first,
    )
    conv.weight = weights[first["mlx_weight"]]
    if "mlx_bias" in first:
        conv.bias = weights[first["mlx_bias"]]
    y_mlx_nlc = conv(x_nlc)
    y_mlx = np.transpose(np.array(y_mlx_nlc), (0, 2, 1))

    diff = np.abs(y_torch - y_mlx)
    result = {
        "layer": first["name"],
        "torch_shape": list(y_torch.shape),
        "mlx_shape": list(y_mlx.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
    }
    print(json.dumps(result, indent=2))
    if result["max_abs_error"] > 1e-4:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
