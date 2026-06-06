#!/usr/bin/env python3
"""Export Trendyol/VoxCPM2 AudioVAE decoder weights in MLX-friendly layout.

This is the first concrete pure-MLX porting step: resolve PyTorch weight_norm
(`weight_g`/`weight_v`) into actual convolution kernels and store them in the
layout expected by `mlx.nn`:

- PyTorch Conv1d:          [out_channels, in_channels/groups, kernel]
- PyTorch ConvTranspose1d: [in_channels, out_channels/groups, kernel]
- MLX Conv/ConvTranspose:  [out_channels, kernel, in_channels/groups]

The full VoxCPM2 pipeline still uses the MPS backend; these exported decoder
weights are the artifact that enables implementing AudioVAE decode in MLX next.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download

from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2


def _load_config(model_dir: Path) -> AudioVAEConfigV2:
    cfg = json.loads((model_dir / "config.json").read_text())
    return AudioVAEConfigV2(**cfg["audio_vae_config"])


def _load_torch_vae(model_dir: Path) -> AudioVAEV2:
    config = _load_config(model_dir)
    vae = AudioVAEV2(config=config)
    checkpoint = torch.load(model_dir / "audiovae.pth", map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = vae.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"warning: missing={len(missing)} unexpected={len(unexpected)}")
    vae.eval()
    return vae


def _conv_kind(module: torch.nn.Module) -> str | None:
    if isinstance(module, torch.nn.ConvTranspose1d):
        return "conv_transpose1d"
    if isinstance(module, torch.nn.Conv1d):
        return "conv1d"
    return None


def _resolved_torch_weight(module: torch.nn.Module) -> torch.Tensor:
    # Old torch.nn.utils.weight_norm stores checkpoint weights as weight_g/weight_v
    # and refreshes module.weight in a pre-forward hook. Resolve directly here so
    # export does not depend on having run a dummy forward first.
    if hasattr(module, "weight_g") and hasattr(module, "weight_v"):
        return torch._weight_norm(module.weight_v.detach().cpu().float(), module.weight_g.detach().cpu().float(), 0)
    return module.weight.detach().cpu().float()


def _to_mlx_conv_weight(module: torch.nn.Module) -> mx.array:
    w = _resolved_torch_weight(module)
    if isinstance(module, torch.nn.ConvTranspose1d):
        # torch ConvTranspose1d: [in, out/groups, k] -> mlx: [out/groups, k, in]
        w = w.permute(1, 2, 0).contiguous()
    elif isinstance(module, torch.nn.Conv1d):
        # torch Conv1d: [out, in/groups, k] -> mlx: [out, k, in/groups]
        w = w.permute(0, 2, 1).contiguous()
    else:
        raise TypeError(f"not a conv module: {type(module)!r}")
    return mx.array(w.numpy())


def export(model_id: str, out_dir: Path, local_files_only: bool = False) -> dict[str, Any]:
    model_dir = Path(snapshot_download(model_id, local_files_only=local_files_only))
    vae = _load_torch_vae(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, mx.array] = {}
    layers: list[dict[str, Any]] = []

    for name, module in vae.named_modules():
        if not name.startswith("decoder"):
            continue
        kind = _conv_kind(module)
        if kind is None:
            continue
        weight_name = f"{name}.weight"
        tensors[weight_name] = _to_mlx_conv_weight(module)
        causal_padding = int(getattr(module, "_CausalConv1d__padding", getattr(module, "_CausalTransposeConv1d__padding", module.padding[0])))
        causal_output_padding = int(getattr(module, "_CausalConv1d__output_padding", getattr(module, "_CausalTransposeConv1d__output_padding", module.output_padding[0] if hasattr(module, "output_padding") else 0)))
        entry: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "in_channels": int(module.in_channels),
            "out_channels": int(module.out_channels),
            "kernel_size": int(module.kernel_size[0]),
            "stride": int(module.stride[0]),
            "dilation": int(module.dilation[0]),
            "groups": int(module.groups),
            "torch_padding": int(module.padding[0]),
            "causal_padding": causal_padding,
            "output_padding": causal_output_padding,
            "mlx_weight": weight_name,
            "mlx_weight_shape": list(tensors[weight_name].shape),
        }
        if module.bias is not None:
            bias_name = f"{name}.bias"
            tensors[bias_name] = mx.array(module.bias.detach().cpu().float().numpy())
            entry["mlx_bias"] = bias_name
            entry["mlx_bias_shape"] = list(tensors[bias_name].shape)
        layers.append(entry)

    # Export Snake alpha tensors and sample-rate conditioning embeddings unchanged.
    for name, tensor in vae.state_dict().items():
        if not name.startswith("decoder"):
            continue
        if name.endswith(".alpha") or "sr_cond_model" in name or name.endswith("sr_bin_boundaries"):
            tensors[name] = mx.array(tensor.detach().cpu().float().numpy())

    weights_path = out_dir / "audiovae_decoder_mlx.safetensors"
    mx.save_safetensors(str(weights_path), tensors)

    config = _load_config(model_dir)
    manifest = {
        "source_model": model_id,
        "source_snapshot": str(model_dir),
        "weights": str(weights_path),
        "sample_rate": config.sample_rate,
        "out_sample_rate": config.out_sample_rate,
        "latent_dim": config.latent_dim,
        "decoder_dim": config.decoder_dim,
        "decoder_rates": config.decoder_rates,
        "decode_chunk_size": int(torch.prod(torch.tensor(config.decoder_rates)).item()),
        "num_exported_tensors": len(tensors),
        "num_conv_layers": len(layers),
        "conv_layers": layers,
        "notes": [
            "Weights are resolved after PyTorch weight_norm and transposed for mlx.nn Conv1d/ConvTranspose1d.",
            "Causal padding/cropping is not encoded in the weights; MLX decoder implementation must mirror AudioVAEV2 CausalConv1d/CausalTransposeConv1d.",
        ],
    }
    manifest_path = out_dir / "audiovae_decoder_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(manifest_path), "weights": str(weights_path), "conv_layers": len(layers), "tensors": len(tensors)}, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Trendyol/Trendyol-TTS")
    parser.add_argument("--out-dir", default="models/Trendyol-TTS-mlx")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    export(args.model_id, Path(args.out_dir), args.local_files_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
