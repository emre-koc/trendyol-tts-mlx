from __future__ import annotations

import subprocess
import sys
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from voxcpm.modules.locenc.local_encoder import VoxCPMLocEnc as TorchVoxCPMLocEnc


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/Trendyol-TTS-mlx/voxcpm2_layers_manifest.json"


def ensure_exported_layers() -> None:
    needs_export = True
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        linear_names = {entry["name"] for entry in manifest.get("linear_layers", [])}
        parameter_names = {entry["name"] for entry in manifest.get("parameters", [])}
        needs_export = "feat_encoder.in_proj" not in linear_names or "feat_encoder.special_token" not in parameter_names
    if not needs_export:
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_voxcpm2_layers_mlx.py"), "--local-files-only"],
        cwd=ROOT,
        check=True,
    )


def local_encoder_config() -> SimpleNamespace:
    model_dir = Path(snapshot_download("Trendyol/Trendyol-TTS", local_files_only=True))
    cfg = json.loads((model_dir / "config.json").read_text())
    lm_cfg = cfg["lm_config"]
    enc_cfg = cfg["encoder_config"]
    return SimpleNamespace(
        hidden_size=enc_cfg["hidden_dim"],
        intermediate_size=enc_cfg["ffn_dim"],
        num_attention_heads=enc_cfg["num_heads"],
        num_key_value_heads=lm_cfg["num_key_value_heads"],
        max_position_embeddings=lm_cfg["max_position_embeddings"],
        kv_channels=enc_cfg["kv_channels"],
        rope_theta=lm_cfg["rope_theta"],
        rope_scaling=SimpleNamespace(**lm_cfg["rope_scaling"]),
        rms_norm_eps=lm_cfg["rms_norm_eps"],
        scale_depth=lm_cfg["scale_depth"],
        num_hidden_layers=enc_cfg["num_layers"],
        use_mup=lm_cfg["use_mup"],
        vocab_size=0,
        no_rope=lm_cfg.get("no_rope", False),
    )


def torch_locenc_input_projection(x: np.ndarray) -> np.ndarray:
    model_dir = Path(snapshot_download("Trendyol/Trendyol-TTS", local_files_only=True))
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        weight = f.get_tensor("feat_encoder.in_proj.weight")
        bias = f.get_tensor("feat_encoder.in_proj.bias")
        special_token = f.get_tensor("feat_encoder.special_token")

    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=True).eval()
    layer.weight.data.copy_(weight)
    layer.bias.data.copy_(bias)

    x_torch = torch.from_numpy(x)
    with torch.no_grad():
        projected = layer(x_torch)
        bsz, timesteps, _, _ = projected.shape
        special_tokens = special_token.expand(bsz, timesteps, 1, -1)
        y = torch.cat([special_tokens, projected], dim=2)
        y = y.reshape(bsz * timesteps, y.shape[2], y.shape[3])
    return y.detach().cpu().numpy()


def load_torch_locenc() -> TorchVoxCPMLocEnc:
    model_dir = Path(snapshot_download("Trendyol/Trendyol-TTS", local_files_only=True))
    locenc = TorchVoxCPMLocEnc(local_encoder_config()).eval()
    state = {}
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        for key in locenc.state_dict().keys():
            state[key] = f.get_tensor(f"feat_encoder.{key}")
    locenc.load_state_dict(state)
    return locenc


def torch_full_locenc(x: np.ndarray) -> np.ndarray:
    locenc = load_torch_locenc()
    with torch.no_grad():
        return locenc(torch.from_numpy(x)).detach().cpu().numpy()


def torch_linear_from_safetensors(name: str) -> torch.nn.Linear:
    model_dir = Path(snapshot_download("Trendyol/Trendyol-TTS", local_files_only=True))
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        weight = f.get_tensor(f"{name}.weight")
        bias = f.get_tensor(f"{name}.bias") if f"{name}.bias" in f.keys() else None
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None).eval()
    layer.weight.data.copy_(weight)
    if bias is not None:
        layer.bias.data.copy_(bias)
    return layer


def torch_locenc_to_lm_projection(x: np.ndarray) -> np.ndarray:
    locenc = load_torch_locenc()
    enc_to_lm_proj = torch_linear_from_safetensors("enc_to_lm_proj")
    with torch.no_grad():
        locenc_output = locenc(torch.from_numpy(x))
        return enc_to_lm_proj(locenc_output).detach().cpu().numpy()


class TestMLXVoxCPMLocEnc(unittest.TestCase):
    def test_input_projection_and_special_token_match_pytorch(self) -> None:
        from trendyol_tts_mlx.mlx_locenc import LocEncInputProjection

        ensure_exported_layers()
        rng = np.random.default_rng(2029)
        audio_feats = rng.standard_normal((2, 3, 4, 64), dtype=np.float32)

        y_torch = torch_locenc_input_projection(audio_feats)
        y_mlx = LocEncInputProjection.from_manifest(MANIFEST)(audio_feats)

        self.assertEqual(y_torch.shape, y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-5, atol=1e-5)

    def test_full_locenc_matches_pytorch_uncausal_cls_output(self) -> None:
        from trendyol_tts_mlx.mlx_locenc import VoxCPMLocEnc

        ensure_exported_layers()
        rng = np.random.default_rng(2037)
        audio_feats = rng.standard_normal((1, 2, 4, 64), dtype=np.float32)

        y_torch = torch_full_locenc(audio_feats)
        y_mlx = VoxCPMLocEnc.from_manifest(MANIFEST)(audio_feats)

        self.assertEqual(y_torch.shape, y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-4, atol=1e-4)

    def test_locenc_to_lm_projection_matches_pytorch(self) -> None:
        from trendyol_tts_mlx.mlx_locenc import VoxCPMLocEncToLMProjection

        ensure_exported_layers()
        rng = np.random.default_rng(2038)
        audio_feats = rng.standard_normal((1, 2, 4, 64), dtype=np.float32)

        y_torch = torch_locenc_to_lm_projection(audio_feats)
        y_mlx = VoxCPMLocEncToLMProjection.from_manifest(MANIFEST)(audio_feats)

        self.assertEqual(y_torch.shape, y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
