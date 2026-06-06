from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/Trendyol-TTS-mlx/audiovae_decoder_manifest.json"


def ensure_exported_decoder() -> None:
    if MANIFEST.exists():
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_audiovae_mlx.py"), "--local-files-only"],
        cwd=ROOT,
        check=True,
    )


def load_torch_vae() -> AudioVAEV2:
    model_dir = Path(snapshot_download("Trendyol/Trendyol-TTS", local_files_only=True))
    cfg = json.loads((model_dir / "config.json").read_text())
    vae = AudioVAEV2(config=AudioVAEConfigV2(**cfg["audio_vae_config"]))
    ckpt = torch.load(model_dir / "audiovae.pth", map_location="cpu", weights_only=True)
    vae.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    return vae.eval()


class TestMLXAudioVAEDecoder(unittest.TestCase):
    def test_decoder_stem_matches_pytorch_first_two_causal_convs(self) -> None:
        ensure_exported_decoder()
        from trendyol_tts_mlx.mlx_audiovae import AudioVAEDecoderStem

        rng = np.random.default_rng(123)
        z_nct = rng.standard_normal((1, 64, 7), dtype=np.float32)

        torch_vae = load_torch_vae()
        with torch.no_grad():
            x = torch.from_numpy(z_nct)
            y_torch = torch_vae.decoder.model[1](torch_vae.decoder.model[0](x)).detach().cpu().numpy()

        stem = AudioVAEDecoderStem.from_manifest(MANIFEST)
        y_mlx = stem(z_nct)

        self.assertEqual(y_torch.shape, y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
