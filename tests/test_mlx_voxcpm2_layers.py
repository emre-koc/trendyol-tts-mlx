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
PROJECTION_CASES = [
    ("enc_to_lm_proj", (2, 3, 1024)),
    ("lm_to_dit_proj", (2, 3, 2048)),
    ("res_to_dit_proj", (2, 3, 2048)),
    ("fusion_concat_proj", (2, 3, 4096)),
]


def ensure_exported_layers() -> None:
    if MANIFEST.exists():
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_voxcpm2_layers_mlx.py"), "--local-files-only"],
        cwd=ROOT,
        check=True,
    )


def load_model_dir() -> Path:
    return Path(snapshot_download("Trendyol/Trendyol-TTS", local_files_only=True))


def load_torch_fsq() -> TorchScalarQuantizationLayer:
    model_dir = load_model_dir()
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


def torch_linear_from_safetensors(name: str) -> torch.nn.Linear:
    model_dir = load_model_dir()
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        weight = f.get_tensor(f"{name}.weight")
        bias = f.get_tensor(f"{name}.bias") if f"{name}.bias" in f.keys() else None
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    layer.weight.data.copy_(weight)
    if bias is not None:
        layer.bias.data.copy_(bias)
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

    def test_projection_layers_match_pytorch_linear(self) -> None:
        from trendyol_tts_mlx.mlx_voxcpm2_layers import Linear

        ensure_exported_layers()
        rng = np.random.default_rng(2027)

        for name, shape in PROJECTION_CASES:
            with self.subTest(name=name):
                x = rng.standard_normal(shape, dtype=np.float32)
                torch_layer = torch_linear_from_safetensors(name)
                with torch.no_grad():
                    y_torch = torch_layer(torch.from_numpy(x)).detach().cpu().numpy()

                mlx_layer = Linear.from_manifest(MANIFEST, name)
                y_mlx = mlx_layer(x)

                self.assertEqual(y_torch.shape, y_mlx.shape)
                np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-5, atol=1e-5)

    def test_stop_predictor_matches_pytorch_path(self) -> None:
        from trendyol_tts_mlx.mlx_voxcpm2_layers import StopPredictor

        ensure_exported_layers()
        rng = np.random.default_rng(2028)
        lm_hidden = rng.standard_normal((2, 3, 2048), dtype=np.float32)

        torch_stop_proj = torch_linear_from_safetensors("stop_proj")
        torch_stop_head = torch_linear_from_safetensors("stop_head")
        with torch.no_grad():
            hidden = torch_stop_proj(torch.from_numpy(lm_hidden))
            y_torch = torch_stop_head(torch.nn.functional.silu(hidden)).detach().cpu().numpy()

        mlx_stop = StopPredictor.from_manifest(MANIFEST)
        y_mlx = mlx_stop(lm_hidden)

        self.assertEqual(y_torch.shape, y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
