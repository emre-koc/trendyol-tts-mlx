from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

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
