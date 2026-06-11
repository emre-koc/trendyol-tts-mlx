from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .mlx_minicpm import MiniCPMEncoder
from .mlx_voxcpm2_layers import Linear, _load_manifest_and_tensors


class LocEncInputProjection:
    """MLX equivalent of the VoxCPMLocEnc pre-encoder input projection.

    Upstream path:
        x = in_proj(x)                         # [B, T, P, 64] -> [B, T, P, 1024]
        x = cat([special_token.expand(...), x], dim=2)
        x = rearrange(x, "b t p c -> (b t) p c")
    """

    def __init__(self, in_proj: Linear, special_token: mx.array):
        self.in_proj = in_proj
        self.special_token = special_token

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "LocEncInputProjection":
        manifest, tensors = _load_manifest_and_tensors(manifest_path)
        parameter_entries = {entry["name"]: entry for entry in manifest.get("parameters", [])}
        special_entry = parameter_entries["feat_encoder.special_token"]
        return cls(
            in_proj=Linear.from_manifest(manifest_path, "feat_encoder.in_proj"),
            special_token=tensors[special_entry["mlx_tensor"]],
        )

    def _call_mx(self, audio_feats: np.ndarray | mx.array) -> mx.array:
        x = self.in_proj._call_mx(audio_feats)
        bsz, timesteps, patches, hidden = x.shape
        special_tokens = mx.broadcast_to(self.special_token, (bsz, timesteps, 1, hidden))
        x = mx.concatenate([special_tokens, x], axis=2)
        return mx.reshape(x, (bsz * timesteps, patches + 1, hidden))

    def __call__(self, audio_feats: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(audio_feats)), dtype=np.float32)


class VoxCPMLocEnc:
    """MLX equivalent of VoxCPMLocEnc.

    Upstream path:
        projected = LocEncInputProjection(audio_feats)  # [B*T, P+1, C]
        outputs, _ = encoder(projected, is_causal=False)
        cls_output = outputs[:, 0, :]
        return rearrange(cls_output, "(b t) c -> b t c", b=B)
    """

    def __init__(self, input_projection: LocEncInputProjection, encoder: MiniCPMEncoder):
        self.input_projection = input_projection
        self.encoder = encoder

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "VoxCPMLocEnc":
        return cls(
            input_projection=LocEncInputProjection.from_manifest(manifest_path),
            encoder=MiniCPMEncoder.from_manifest(manifest_path),
        )

    def _call_mx(self, audio_feats: np.ndarray | mx.array) -> mx.array:
        x = mx.array(audio_feats) if not isinstance(audio_feats, mx.array) else audio_feats
        bsz, timesteps = x.shape[0], x.shape[1]
        encoded_inputs = self.input_projection._call_mx(x)
        outputs, _ = self.encoder._call_mx(encoded_inputs, is_causal=False)
        cls_output = outputs[:, 0, :]
        return mx.reshape(cls_output, (bsz, timesteps, cls_output.shape[-1]))

    def __call__(self, audio_feats: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(audio_feats)), dtype=np.float32)


class VoxCPMLocEncToLMProjection:
    """MLX bridge from local audio features into LM hidden space.

    This mirrors the upstream partial path:
        enc_to_lm_proj(feat_encoder(audio_feats))
    producing `[B, T, lm_hidden_size]` from audio features shaped `[B, T, P, 64]`.
    """

    def __init__(self, locenc: VoxCPMLocEnc, enc_to_lm_proj: Linear):
        self.locenc = locenc
        self.enc_to_lm_proj = enc_to_lm_proj

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "VoxCPMLocEncToLMProjection":
        return cls(
            locenc=VoxCPMLocEnc.from_manifest(manifest_path),
            enc_to_lm_proj=Linear.from_manifest(manifest_path, "enc_to_lm_proj"),
        )

    def _call_mx(self, audio_feats: np.ndarray | mx.array) -> mx.array:
        locenc_output = self.locenc._call_mx(audio_feats)
        return self.enc_to_lm_proj._call_mx(locenc_output)

    def __call__(self, audio_feats: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(audio_feats)), dtype=np.float32)
