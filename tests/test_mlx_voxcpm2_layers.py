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

from voxcpm.modules.layers import ScalarQuantizationLayer as TorchScalarQuantizationLayer


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/Trendyol-TTS-mlx/voxcpm2_layers_manifest.json"


def ensure_exported_layers() -> None:
    if MANIFEST.exists():
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_voxcpm2_layers_mlx.py"), "--local-files-only"],
        cwd=ROOT,
        check=True,
    )


def load_torch_fsq() -> TorchScalarQuantizationLayer:
    model_dir = Path(snapshot_download("Trendyol/Trendyol-TTS", local_files_only=True))
    cfg = json.loads((model_dir / "config.json").read_text())
    layer = TorchScalarQuantizationLayer(
        cfg["lm_config"]["hidden_size"],
        cfg["lm_config"]["hidden_size"],
        cfg["scalar_quantization_latent_dim"],
        cfg["scalar_quantization_scale"],
    )
    state = {}
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("fsq_layer."):
                state[key.removeprefix("fsq_layer.")] = f.get_tensor(key)
    layer.load_state_dict(state)
    return layer.eval()


class TestMLXVoxCPM2Layers(unittest.TestCase):
    def test_scalar_quantization_layer_matches_pytorch_eval(self) -> None:
        from trendyol_tts_mlx.mlx_voxcpm2_layers import ScalarQuantizationLayer

        ensure_exported_layers()
        rng = np.random.default_rng(2026)
        hidden = rng.standard_normal((2, 3, 2048), dtype=np.float32)

        torch_fsq = load_torch_fsq()
        with torch.no_grad():
            y_torch = torch_fsq(torch.from_numpy(hidden)).detach().cpu().numpy()

        mlx_fsq = ScalarQuantizationLayer.from_manifest(MANIFEST)
        y_mlx = mlx_fsq(hidden)

        self.assertEqual(y_torch.shape, y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
