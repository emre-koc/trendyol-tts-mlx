#!/usr/bin/env python3
"""Export small VoxCPM2 non-transformer layers in MLX-friendly layout.

This targets the next upstream slice after the AudioVAE decoder: projection and
quantization layers around the latent-producing path. Linear weights are stored
in PyTorch/MLX-compatible row-major form and applied as `x @ weight.T + bias`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from huggingface_hub import snapshot_download
from safetensors import safe_open


LINEAR_PREFIXES = [
    "fsq_layer.in_proj",
    "fsq_layer.out_proj",
    "enc_to_lm_proj",
    "lm_to_dit_proj",
    "res_to_dit_proj",
    "fusion_concat_proj",
    "stop_proj",
    "stop_head",
    "feat_encoder.in_proj",
    "feat_encoder.encoder.layers.0.mlp.gate_proj",
    "feat_encoder.encoder.layers.0.mlp.up_proj",
    "feat_encoder.encoder.layers.0.mlp.down_proj",
    "feat_encoder.encoder.layers.0.self_attn.q_proj",
    "feat_encoder.encoder.layers.0.self_attn.k_proj",
    "feat_encoder.encoder.layers.0.self_attn.v_proj",
    "feat_encoder.encoder.layers.0.self_attn.o_proj",
]

PARAMETER_KEYS = [
    "feat_encoder.special_token",
    "feat_encoder.encoder.layers.0.input_layernorm.weight",
    "feat_encoder.encoder.layers.0.post_attention_layernorm.weight",
    "feat_encoder.encoder.norm.weight",
]


def _load_config(model_dir: Path) -> dict[str, Any]:
    return json.loads((model_dir / "config.json").read_text())


def export(model_id: str, out_dir: Path, local_files_only: bool = False) -> dict[str, Any]:
    model_dir = Path(snapshot_download(model_id, local_files_only=local_files_only))
    cfg = _load_config(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, mx.array] = {}
    linear_layers: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    weights_path = model_dir / "model.safetensors"
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for prefix in LINEAR_PREFIXES:
            weight_key = f"{prefix}.weight"
            bias_key = f"{prefix}.bias"
            if weight_key not in keys:
                raise KeyError(f"missing expected tensor {weight_key}")
            weight = f.get_tensor(weight_key).float().contiguous()
            tensors[weight_key] = mx.array(weight.numpy())
            entry: dict[str, Any] = {
                "name": prefix,
                "kind": "linear",
                "in_features": int(weight.shape[1]),
                "out_features": int(weight.shape[0]),
                "mlx_weight": weight_key,
                "mlx_weight_shape": list(weight.shape),
            }
            if bias_key in keys:
                bias = f.get_tensor(bias_key).float().contiguous()
                tensors[bias_key] = mx.array(bias.numpy())
                entry["mlx_bias"] = bias_key
                entry["mlx_bias_shape"] = list(bias.shape)
            linear_layers.append(entry)

        for key in PARAMETER_KEYS:
            if key not in keys:
                raise KeyError(f"missing expected tensor {key}")
            tensor = f.get_tensor(key).float().contiguous()
            tensors[key] = mx.array(tensor.numpy())
            parameters.append(
                {
                    "name": key,
                    "kind": "parameter",
                    "mlx_tensor": key,
                    "mlx_tensor_shape": list(tensor.shape),
                }
            )

    mlx_weights_path = out_dir / "voxcpm2_layers_mlx.safetensors"
    mx.save_safetensors(str(mlx_weights_path), tensors)

    manifest = {
        "source_model": model_id,
        "source_snapshot": str(model_dir),
        "weights": str(mlx_weights_path),
        "lm_hidden_size": cfg["lm_config"]["hidden_size"],
        "dit_hidden_size": cfg["dit_config"]["hidden_dim"],
        "encoder_hidden_size": cfg["encoder_config"]["hidden_dim"],
        "encoder_num_attention_heads": cfg["encoder_config"]["num_heads"],
        "encoder_num_key_value_heads": cfg["lm_config"]["num_key_value_heads"],
        "encoder_kv_channels": cfg["encoder_config"]["kv_channels"],
        "scalar_quantization_latent_dim": cfg["scalar_quantization_latent_dim"],
        "scalar_quantization_scale": cfg["scalar_quantization_scale"],
        "num_exported_tensors": len(tensors),
        "linear_layers": linear_layers,
        "parameters": parameters,
        "notes": [
            "Linear weights keep PyTorch layout [out_features, in_features]; MLX applies x @ weight.T + bias.",
            "ScalarQuantizationLayer eval path is tanh -> round(hidden * scale) / scale -> out_proj.",
        ],
    }
    manifest_path = out_dir / "voxcpm2_layers_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "weights": str(mlx_weights_path),
                "linear_layers": len(linear_layers),
                "tensors": len(tensors),
            },
            indent=2,
        )
    )
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
