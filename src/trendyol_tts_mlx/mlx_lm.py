from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .mlx_voxcpm2_layers import _load_manifest_and_tensors
from .mlx_minicpm import MiniCPMDecoderLayer, MiniCPMRMSNorm


class MiniCPMTokenEmbedding:
    """MLX token embedding for VoxCPM2 `base_lm.embed_tokens`.

    Mirrors the upstream forward slice:
        text_embed = base_lm.embed_tokens(text_tokens) * scale_emb

    `scale_emb` is applied only when `use_mup` is enabled, matching VoxCPM2Model.forward.
    """

    def __init__(self, weight: mx.array, scale_emb: float = 1.0):
        self.weight = weight
        self.scale_emb = float(scale_emb)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, name: str = "base_lm.embed_tokens") -> "MiniCPMTokenEmbedding":
        manifest, tensors = _load_manifest_and_tensors(manifest_path)
        parameter_entries = {entry["name"]: entry for entry in manifest.get("parameters", [])}
        key = f"{name}.weight"
        if key not in parameter_entries:
            raise KeyError(f"token embedding parameter {key!r} not found in manifest")
        scale_emb = float(manifest.get("lm_scale_emb", 1.0)) if manifest.get("lm_use_mup", False) else 1.0
        return cls(tensors[parameter_entries[key]["mlx_tensor"]], scale_emb=scale_emb)

    def _call_mx(self, token_ids: np.ndarray | mx.array) -> mx.array:
        ids = mx.array(token_ids) if not isinstance(token_ids, mx.array) else token_ids
        return self.weight[ids] * self.scale_emb

    def __call__(self, token_ids: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(token_ids)), dtype=np.float32)


class MLXStaticKVCache:
    """Static per-layer KV cache for MiniCPM autoregressive MLX forward_step."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_key_value_heads: int,
        max_length: int,
        head_dim: int,
    ):
        self.num_layers = int(num_layers)
        self.max_length = int(max_length)
        self.current_length = 0
        self.kv_cache = [
            (
                mx.zeros((batch_size, num_key_value_heads, max_length, head_dim), dtype=mx.float32),
                mx.zeros((batch_size, num_key_value_heads, max_length, head_dim), dtype=mx.float32),
            )
            for _ in range(num_layers)
        ]

    def get_layer_cache(self, layer_idx: int) -> tuple[mx.array, mx.array]:
        return self.kv_cache[layer_idx]

    def set_layer_cache(self, layer_idx: int, cache: tuple[mx.array, mx.array]) -> None:
        self.kv_cache[layer_idx] = cache

    def fill_caches(self, kv_caches: list[tuple[np.ndarray, np.ndarray]] | list[tuple[mx.array, mx.array]]) -> None:
        if not kv_caches:
            self.current_length = 0
            return
        self.current_length = int(kv_caches[0][0].shape[2])
        for idx, (key, value) in enumerate(kv_caches):
            key_mx = mx.array(key) if not isinstance(key, mx.array) else key
            value_mx = mx.array(value) if not isinstance(value, mx.array) else value
            key_cache, value_cache = self.kv_cache[idx]
            key_cache = mx.slice_update(
                key_cache,
                key_mx,
                start_indices=mx.array([0], dtype=mx.int32),
                axes=(2,),
            )
            value_cache = mx.slice_update(
                value_cache,
                value_mx,
                start_indices=mx.array([0], dtype=mx.int32),
                axes=(2,),
            )
            self.kv_cache[idx] = (key_cache, value_cache)

    def step(self) -> int:
        if self.current_length >= self.max_length:
            raise ValueError("KV cache is full")
        ret = self.current_length
        self.current_length += 1
        return ret


class MiniCPMBaseLM:
    """MLX full-stack VoxCPM2 `base_lm` forward for embedded prompts.

    This mirrors upstream MiniCPMModel.forward over all exported decoder layers:
    position ids -> decoder layers with causal attention -> final RMSNorm.
    """

    def __init__(self, layers: list[MiniCPMDecoderLayer], norm: MiniCPMRMSNorm, manifest: dict):
        self.layers = layers
        self.norm = norm
        self.manifest = manifest
        self.kv_cache: MLXStaticKVCache | None = None

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "MiniCPMBaseLM":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
        num_layers = int(manifest["lm_exported_layer_count"])
        expected_layers = int(manifest.get("lm_num_hidden_layers", num_layers))
        if num_layers != expected_layers:
            raise ValueError(f"base_lm stack requires all {expected_layers} layers, found {num_layers}")
        return cls(
            layers=[
                MiniCPMDecoderLayer.from_manifest(manifest_path, scope="base_lm", layer_idx=idx)
                for idx in range(num_layers)
            ],
            norm=MiniCPMRMSNorm.from_manifest(manifest_path, "base_lm.norm"),
            manifest=manifest,
        )

    def _call_mx(
        self, inputs_embeds: np.ndarray | mx.array, is_causal: bool = True
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        hidden = mx.array(inputs_embeds) if not isinstance(inputs_embeds, mx.array) else inputs_embeds
        position_ids = mx.arange(hidden.shape[1], dtype=mx.int64)
        next_decoder_cache = []
        for layer in self.layers:
            hidden, this_cache = layer._call_mx(hidden, position_ids=position_ids, is_causal=is_causal)
            next_decoder_cache.append(this_cache)
        hidden = self.norm._call_mx(hidden)
        return hidden, next_decoder_cache

    def setup_cache(self, batch_size: int, max_length: int) -> None:
        hidden_size = int(self.manifest["lm_hidden_size"])
        num_heads = int(self.manifest["lm_num_attention_heads"])
        head_dim = int(self.manifest.get("lm_kv_channels") or hidden_size // num_heads)
        self.kv_cache = MLXStaticKVCache(
            num_layers=len(self.layers),
            batch_size=batch_size,
            num_key_value_heads=int(self.manifest["lm_num_key_value_heads"]),
            max_length=max_length,
            head_dim=head_dim,
        )

    def fill_caches(self, kv_caches: list[tuple[np.ndarray, np.ndarray]] | list[tuple[mx.array, mx.array]]) -> None:
        if self.kv_cache is None:
            batch_size = int(kv_caches[0][0].shape[0]) if kv_caches else 1
            current_length = int(kv_caches[0][0].shape[2]) if kv_caches else 0
            self.setup_cache(batch_size=batch_size, max_length=max(current_length, 1))
        assert self.kv_cache is not None
        self.kv_cache.fill_caches(kv_caches)

    def cache_step(self) -> int:
        if self.kv_cache is None:
            raise AssertionError("KV cache is not setup")
        return self.kv_cache.step()

    def forward_step_mx(
        self,
        inputs_embeds: np.ndarray | mx.array,
        position_id: int | np.ndarray | mx.array,
        *,
        evaluate: bool = True,
    ) -> mx.array:
        """Run one cached autoregressive step and keep the result on MLX.

        `forward_step()` remains the NumPy public compatibility wrapper for
        parity tests, but strict MLX generation uses this method to avoid a
        CPU synchronization and a NumPy→MLX conversion on every generated token.
        """
        if self.kv_cache is None:
            raise AssertionError("KV cache is not setup")
        hidden = mx.array(inputs_embeds) if not isinstance(inputs_embeds, mx.array) else inputs_embeds
        for idx, layer in enumerate(self.layers):
            hidden, updated_cache = layer._forward_step_mx(
                hidden,
                position_id=position_id,
                kv_cache=self.kv_cache.get_layer_cache(idx),
            )
            self.kv_cache.set_layer_cache(idx, updated_cache)
        hidden = self.norm._call_mx(hidden)
        if evaluate:
            mx.eval(hidden)
        return hidden

    def forward_step(self, inputs_embeds: np.ndarray | mx.array, position_id: int | np.ndarray | mx.array) -> np.ndarray:
        hidden = self.forward_step_mx(inputs_embeds, position_id)
        return np.asarray(np.array(hidden), dtype=np.float32)

    def __call__(
        self, inputs_embeds: np.ndarray | mx.array, is_causal: bool = True
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        output, cache = self._call_mx(inputs_embeds, is_causal=is_causal)
        return (
            np.asarray(np.array(output), dtype=np.float32),
            [
                (
                    np.asarray(np.array(key_states), dtype=np.float32),
                    np.asarray(np.array(value_states), dtype=np.float32),
                )
                for key_states, value_states in cache
            ],
        )


class MiniCPMResidualLM(MiniCPMBaseLM):
    """MLX full-stack VoxCPM2 `residual_lm` forward for embedded residual inputs."""

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "MiniCPMResidualLM":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
        num_layers = int(manifest["residual_lm_exported_layer_count"])
        expected_layers = int(manifest.get("residual_lm_num_hidden_layers", num_layers))
        if num_layers != expected_layers:
            raise ValueError(f"residual_lm stack requires all {expected_layers} layers, found {num_layers}")
        return cls(
            layers=[
                MiniCPMDecoderLayer.from_manifest(manifest_path, scope="residual_lm", layer_idx=idx)
                for idx in range(num_layers)
            ],
            norm=MiniCPMRMSNorm.from_manifest(manifest_path, "residual_lm.norm"),
            manifest=manifest,
        )
