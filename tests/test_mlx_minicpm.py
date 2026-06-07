from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from voxcpm.modules.minicpm4.model import MiniCPMMLP, MiniCPMRMSNorm


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/Trendyol-TTS-mlx/voxcpm2_layers_manifest.json"
RMSNORM_CASES = [
    "feat_encoder.encoder.layers.0.input_layernorm",
    "feat_encoder.encoder.layers.0.post_attention_layernorm",
    "feat_encoder.encoder.norm",
]
MLP_CASE = "feat_encoder.encoder.layers.0.mlp"


def ensure_exported_layers() -> None:
    needs_export = True
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        parameter_names = {entry["name"] for entry in manifest.get("parameters", [])}
        linear_names = {entry["name"] for entry in manifest.get("linear_layers", [])}
        needs_export = any(f"{name}.weight" not in parameter_names for name in RMSNORM_CASES) or any(
            f"{MLP_CASE}.{proj}" not in linear_names for proj in ("gate_proj", "up_proj", "down_proj")
        )
    if not needs_export:
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_voxcpm2_layers_mlx.py"), "--local-files-only"],
        cwd=ROOT,
        check=True,
    )


def load_torch_rmsnorm(name: str) -> MiniCPMRMSNorm:
    model_dir = Path(snapshot_download("Trendyol/Trendyol-TTS", local_files_only=True))
    cfg = json.loads((model_dir / "config.json").read_text())["encoder_config"]
    norm = MiniCPMRMSNorm(cfg["hidden_dim"], eps=cfg.get("rms_norm_eps", 1e-6)).eval()
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        norm.weight.data.copy_(f.get_tensor(f"{name}.weight"))
    return norm


def load_torch_mlp(name: str) -> MiniCPMMLP:
    model_dir = Path(snapshot_download("Trendyol/Trendyol-TTS", local_files_only=True))
    cfg = json.loads((model_dir / "config.json").read_text())["encoder_config"]
    mlp = MiniCPMMLP(SimpleNamespace(hidden_size=cfg["hidden_dim"], intermediate_size=cfg["ffn_dim"])).eval()
    state = {}
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        for key in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            state[key] = f.get_tensor(f"{name}.{key}")
    mlp.load_state_dict(state)
    return mlp


class TestMLXMiniCPM(unittest.TestCase):
    def test_rmsnorm_layers_match_pytorch(self) -> None:
        from trendyol_tts_mlx.mlx_minicpm import MiniCPMRMSNorm as MLXMiniCPMRMSNorm

        ensure_exported_layers()
        rng = np.random.default_rng(2030)
        hidden = rng.standard_normal((2, 5, 1024), dtype=np.float32)

        for name in RMSNORM_CASES:
            with self.subTest(name=name):
                torch_norm = load_torch_rmsnorm(name)
                with torch.no_grad():
                    y_torch = torch_norm(torch.from_numpy(hidden)).detach().cpu().numpy()

                mlx_norm = MLXMiniCPMRMSNorm.from_manifest(MANIFEST, name)
                y_mlx = mlx_norm(hidden)

                self.assertEqual(y_torch.shape, y_mlx.shape)
                np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-5, atol=1e-5)

    def test_first_mlp_matches_pytorch(self) -> None:
        from trendyol_tts_mlx.mlx_minicpm import MiniCPMMLP as MLXMiniCPMMLP

        ensure_exported_layers()
        rng = np.random.default_rng(2031)
        hidden = rng.standard_normal((2, 5, 1024), dtype=np.float32)

        torch_mlp = load_torch_mlp(MLP_CASE)
        with torch.no_grad():
            y_torch = torch_mlp(torch.from_numpy(hidden)).detach().cpu().numpy()

        mlx_mlp = MLXMiniCPMMLP.from_manifest(MANIFEST, MLP_CASE)
        y_mlx = mlx_mlp(hidden)

        self.assertEqual(y_torch.shape, y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
