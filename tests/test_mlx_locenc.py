from __future__ import annotations

import subprocess
import sys
import unittest
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


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


if __name__ == "__main__":
    unittest.main()
