from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from voxcpm.modules.minicpm4.model import MiniCPMRMSNorm


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/Trendyol-TTS-mlx/voxcpm2_layers_manifest.json"
RMSNORM_CASES = [
    "feat_encoder.encoder.layers.0.input_layernorm",
    "feat_encoder.encoder.layers.0.post_attention_layernorm",
    "feat_encoder.encoder.norm",
]


def ensure_exported_layers() -> None:
    needs_export = True
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        parameter_names = {entry["name"] for entry in manifest.get("parameters", [])}
        needs_export = any(f"{name}.weight" not in parameter_names for name in RMSNORM_CASES)
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


if __name__ == "__main__":
    unittest.main()
