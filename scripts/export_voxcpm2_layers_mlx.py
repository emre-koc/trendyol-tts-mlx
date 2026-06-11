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


BASE_LINEAR_PREFIXES = [
    "fsq_layer.in_proj",
    "fsq_layer.out_proj",
    "enc_to_lm_proj",
    "lm_to_dit_proj",
    "res_to_dit_proj",
    "fusion_concat_proj",
    "stop_proj",
    "stop_head",
    "feat_encoder.in_proj",
    "feat_decoder.estimator.in_proj",
    "feat_decoder.estimator.cond_proj",
    "feat_decoder.estimator.out_proj",
    "feat_decoder.estimator.time_mlp.linear_1",
    "feat_decoder.estimator.time_mlp.linear_2",
    "feat_decoder.estimator.delta_time_mlp.linear_1",
    "feat_decoder.estimator.delta_time_mlp.linear_2",
]

LAYER_LINEAR_SUFFIXES = [
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]

BASE_PARAMETER_KEYS = [
    "base_lm.embed_tokens.weight",
    "base_lm.norm.weight",
    "residual_lm.norm.weight",
    "feat_encoder.special_token",
    "feat_encoder.encoder.norm.weight",
    "feat_decoder.estimator.decoder.norm.weight",
]

LAYER_PARAMETER_SUFFIXES = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
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
    encoder_layer_count = int(cfg["encoder_config"]["num_layers"])
    dit_layer_count = int(cfg["dit_config"]["num_layers"])
    lm_export_layer_count = int(cfg["lm_config"]["num_hidden_layers"])
    residual_lm_export_layer_count = int(cfg["residual_lm_num_layers"])
    linear_prefixes = (
        BASE_LINEAR_PREFIXES
        + [
            f"base_lm.layers.{layer_idx}.{suffix}"
            for layer_idx in range(lm_export_layer_count)
            for suffix in LAYER_LINEAR_SUFFIXES
        ]
        + [
            f"residual_lm.layers.{layer_idx}.{suffix}"
            for layer_idx in range(residual_lm_export_layer_count)
            for suffix in LAYER_LINEAR_SUFFIXES
        ]
        + [
            f"feat_encoder.encoder.layers.{layer_idx}.{suffix}"
            for layer_idx in range(encoder_layer_count)
            for suffix in LAYER_LINEAR_SUFFIXES
        ]
        + [
            f"feat_decoder.estimator.decoder.layers.{layer_idx}.{suffix}"
            for layer_idx in range(dit_layer_count)
            for suffix in LAYER_LINEAR_SUFFIXES
        ]
    )
    parameter_keys = (
        BASE_PARAMETER_KEYS
        + [
            f"base_lm.layers.{layer_idx}.{suffix}"
            for layer_idx in range(lm_export_layer_count)
            for suffix in LAYER_PARAMETER_SUFFIXES
        ]
        + [
            f"residual_lm.layers.{layer_idx}.{suffix}"
            for layer_idx in range(residual_lm_export_layer_count)
            for suffix in LAYER_PARAMETER_SUFFIXES
        ]
        + [
            f"feat_encoder.encoder.layers.{layer_idx}.{suffix}"
            for layer_idx in range(encoder_layer_count)
            for suffix in LAYER_PARAMETER_SUFFIXES
        ]
        + [
            f"feat_decoder.estimator.decoder.layers.{layer_idx}.{suffix}"
            for layer_idx in range(dit_layer_count)
            for suffix in LAYER_PARAMETER_SUFFIXES
        ]
    )
    weights_path = model_dir / "model.safetensors"
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for prefix in linear_prefixes:
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

        for key in parameter_keys:
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
        "lm_vocab_size": cfg["lm_config"]["vocab_size"],
        "lm_scale_emb": cfg["lm_config"].get("scale_emb", 1.0),
        "lm_use_mup": cfg["lm_config"].get("use_mup", False),
        "lm_rms_norm_eps": cfg["lm_config"]["rms_norm_eps"],
        "lm_num_attention_heads": cfg["lm_config"]["num_attention_heads"],
        "lm_num_key_value_heads": cfg["lm_config"]["num_key_value_heads"],
        "lm_kv_channels": cfg["lm_config"].get("kv_channels"),
        "lm_max_position_embeddings": cfg["lm_config"]["max_position_embeddings"],
        "lm_rope_theta": cfg["lm_config"]["rope_theta"],
        "lm_rope_scaling": cfg["lm_config"]["rope_scaling"],
        "lm_scale_depth": cfg["lm_config"]["scale_depth"],
        "lm_num_hidden_layers": cfg["lm_config"]["num_hidden_layers"],
        "lm_no_rope": cfg["lm_config"].get("no_rope", False),
        "lm_exported_layer_count": lm_export_layer_count,
        "residual_lm_hidden_size": cfg["lm_config"]["hidden_size"],
        "residual_lm_num_attention_heads": cfg["lm_config"]["num_attention_heads"],
        "residual_lm_num_key_value_heads": cfg["lm_config"]["num_key_value_heads"],
        "residual_lm_kv_channels": cfg["lm_config"].get("kv_channels"),
        "residual_lm_rms_norm_eps": cfg["lm_config"]["rms_norm_eps"],
        "residual_lm_scale_depth": cfg["lm_config"]["scale_depth"],
        "residual_lm_num_hidden_layers": cfg["residual_lm_num_layers"],
        "residual_lm_no_rope": cfg["residual_lm_no_rope"],
        "residual_lm_use_mup": cfg["lm_config"].get("use_mup", False),
        "residual_lm_exported_layer_count": residual_lm_export_layer_count,
        "dit_hidden_size": cfg["dit_config"]["hidden_dim"],
        "dit_num_attention_heads": cfg["dit_config"]["num_heads"],
        "dit_num_key_value_heads": cfg["lm_config"]["num_key_value_heads"],
        "dit_kv_channels": cfg["dit_config"].get("kv_channels"),
        "dit_max_position_embeddings": cfg["lm_config"]["max_position_embeddings"],
        "dit_rope_theta": cfg["lm_config"]["rope_theta"],
        "dit_rope_scaling": cfg["lm_config"]["rope_scaling"],
        "dit_rms_norm_eps": cfg["lm_config"]["rms_norm_eps"],
        "dit_scale_depth": cfg["lm_config"]["scale_depth"],
        "dit_num_hidden_layers": cfg["dit_config"]["num_layers"],
        "dit_use_mup": False,
        "dit_in_channels": cfg["audio_vae_config"]["latent_dim"],
        "dit_patch_size": cfg["patch_size"],
        "dit_sigma_min": cfg["dit_config"]["cfm_config"].get("sigma_min", 1e-6),
        "dit_solver": cfg["dit_config"]["cfm_config"].get("solver", "euler"),
        "dit_inference_cfg_rate": cfg["dit_config"]["cfm_config"].get("inference_cfg_rate", 2.0),
        "dit_t_scheduler": cfg["dit_config"]["cfm_config"].get("t_scheduler", "log-norm"),
        "dit_mean_mode": cfg["dit_config"].get("mean_mode", False),
        "dit_exported_layer_count": dit_layer_count,
        "encoder_hidden_size": cfg["encoder_config"]["hidden_dim"],
        "encoder_num_attention_heads": cfg["encoder_config"]["num_heads"],
        "encoder_num_key_value_heads": cfg["lm_config"]["num_key_value_heads"],
        "encoder_kv_channels": cfg["encoder_config"]["kv_channels"],
        "encoder_max_position_embeddings": cfg["lm_config"]["max_position_embeddings"],
        "encoder_rope_theta": cfg["lm_config"]["rope_theta"],
        "encoder_rope_scaling": cfg["lm_config"]["rope_scaling"],
        "encoder_rms_norm_eps": cfg["lm_config"]["rms_norm_eps"],
        "encoder_scale_depth": cfg["lm_config"]["scale_depth"],
        "encoder_num_hidden_layers": cfg["encoder_config"]["num_layers"],
        "encoder_use_mup": cfg["lm_config"]["use_mup"],
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
