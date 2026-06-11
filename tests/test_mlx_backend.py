from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from test_mlx_lm import load_torch_lm_model, load_torch_residual_lm_model, torch_base_lm_token_embedding
from test_mlx_locenc import torch_locenc_to_lm_projection
from test_mlx_voxcpm2_layers import load_torch_fsq, torch_linear_from_safetensors

MANIFEST = ROOT / "models/Trendyol-TTS-mlx/audiovae_decoder_manifest.json"
VOXCPM2_MANIFEST = ROOT / "models/Trendyol-TTS-mlx/voxcpm2_layers_manifest.json"


def ensure_exported_decoder() -> None:
    if MANIFEST.exists():
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_audiovae_mlx.py"), "--local-files-only"],
        cwd=ROOT,
        check=True,
    )


def ensure_exported_voxcpm2_layers() -> None:
    if VOXCPM2_MANIFEST.exists():
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_voxcpm2_layers_mlx.py"), "--local-files-only"],
        cwd=ROOT,
        check=True,
    )


class TestMLXBackend(unittest.TestCase):
    def test_mlx_audio_vae_decoder_returns_numpy_shape_contract(self) -> None:
        ensure_exported_decoder()
        from trendyol_tts_mlx.mlx_audiovae import AudioVAEDecoder

        rng = np.random.default_rng(4040)
        latent = rng.standard_normal((1, 64, 2), dtype=np.float32)

        decoder = AudioVAEDecoder.from_manifest(MANIFEST)
        wav = decoder.decode(latent, sample_rate=48000)

        self.assertIsInstance(wav, np.ndarray)
        self.assertEqual(wav.dtype, np.float32)
        self.assertEqual(wav.shape[0], 1)
        self.assertEqual(wav.shape[1], 1)
        self.assertGreater(wav.shape[2], 0)
        self.assertTrue(np.isfinite(wav).all())

    def test_config_from_env_accepts_mlx_backend(self) -> None:
        from trendyol_tts_mlx.backend import config_from_env

        cfg = config_from_env(backend="mlx")

        self.assertEqual(cfg.backend, "mlx")

    def test_mlx_generation_prefill_matches_pytorch_until_dit_boundary(self) -> None:
        from trendyol_tts_mlx.mlx_backend import MLXGenerationPrefill

        ensure_exported_voxcpm2_layers()
        rng = np.random.default_rng(5050)
        text_tokens = np.array([[101, 42, 102]], dtype=np.int64)
        text_mask = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
        feat_mask = 1.0 - text_mask
        audio_feats = rng.standard_normal((1, 3, 4, 64), dtype=np.float32)

        feat_embed = torch.from_numpy(torch_locenc_to_lm_projection(audio_feats))
        text_embed = torch.from_numpy(torch_base_lm_token_embedding(text_tokens))
        text_mask_t = torch.from_numpy(text_mask)
        feat_mask_t = torch.from_numpy(feat_mask)
        combined_embed = text_mask_t.unsqueeze(-1) * text_embed + feat_mask_t.unsqueeze(-1) * feat_embed

        base_lm = load_torch_lm_model()
        residual_lm = load_torch_residual_lm_model()
        fsq = load_torch_fsq()
        fusion_concat_proj = torch_linear_from_safetensors("fusion_concat_proj")
        lm_to_dit_proj = torch_linear_from_safetensors("lm_to_dit_proj")
        res_to_dit_proj = torch_linear_from_safetensors("res_to_dit_proj")
        stop_proj = torch_linear_from_safetensors("stop_proj")
        stop_head = torch_linear_from_safetensors("stop_head")
        with torch.no_grad():
            enc_outputs, _ = base_lm(inputs_embeds=combined_embed, is_causal=True)
            enc_outputs = fsq(enc_outputs) * feat_mask_t.unsqueeze(-1) + enc_outputs * text_mask_t.unsqueeze(-1)
            lm_hidden = enc_outputs[:, -1, :]
            residual_inputs = fusion_concat_proj(torch.cat((enc_outputs, feat_mask_t.unsqueeze(-1) * feat_embed), dim=-1))
            residual_outputs, _ = residual_lm(inputs_embeds=residual_inputs, is_causal=True)
            residual_hidden = residual_outputs[:, -1, :]
            dit_hidden = torch.cat((lm_to_dit_proj(lm_hidden), res_to_dit_proj(residual_hidden)), dim=-1)
            stop_logits = stop_head(torch.nn.functional.silu(stop_proj(lm_hidden)))

        state = MLXGenerationPrefill.from_manifest(VOXCPM2_MANIFEST).prefill(
            text_tokens=text_tokens,
            text_mask=text_mask,
            audio_feats=audio_feats,
            feat_mask=feat_mask,
            max_length=8,
        )

        np.testing.assert_allclose(state.lm_hidden, lm_hidden.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(state.residual_hidden, residual_hidden.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(state.dit_hidden, dit_hidden.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(state.stop_logits, stop_logits.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
    def test_mlx_dit_estimator_matches_pytorch_slice(self) -> None:
        from trendyol_tts_mlx.mlx_dit import MLXVoxCPMLocDiT

        ensure_exported_voxcpm2_layers()
        rng = np.random.default_rng(6060)
        x = rng.standard_normal((1, 64, 4), dtype=np.float32)
        mu = rng.standard_normal((1, 2048), dtype=np.float32)
        t = np.array([0.5], dtype=np.float32)
        dt = np.array([0.0], dtype=np.float32)
        cond = rng.standard_normal((1, 64, 4), dtype=np.float32)

        from voxcpm.core import VoxCPM

        tts_model = VoxCPM.from_pretrained(
            hf_model_id="Trendyol/Trendyol-TTS",
            load_denoiser=False,
            local_files_only=True,
            optimize=False,
            device="cpu",
        ).tts_model
        dtype = next(tts_model.feat_decoder.estimator.parameters()).dtype
        with torch.no_grad():
            expected = (
                tts_model.feat_decoder.estimator(
                    torch.from_numpy(x).to(dtype),
                    torch.from_numpy(mu).to(dtype),
                    torch.from_numpy(t).to(dtype),
                    torch.from_numpy(cond).to(dtype),
                    torch.from_numpy(dt).to(dtype),
                )
                .float()
                .detach()
                .cpu()
                .numpy()
            )

        actual = MLXVoxCPMLocDiT.from_manifest(VOXCPM2_MANIFEST)(x, mu, t, cond, dt)

        np.testing.assert_allclose(actual, expected, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
