from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from .mlx_minicpm import MiniCPMDecoderLayer, MiniCPMRMSNorm
from .mlx_voxcpm2_layers import Linear, _load_manifest_and_tensors


def _to_mx(x: np.ndarray | mx.array) -> mx.array:
    return x if isinstance(x, mx.array) else mx.array(x)


class SinusoidalPosEmb:
    """MLX equivalent of VoxCPM loc-DiT SinusoidalPosEmb."""

    def __init__(self, dim: int):
        self.dim = int(dim)
        if self.dim % 2 != 0:
            raise ValueError("SinusoidalPosEmb requires dim to be even")

    def _call_mx(self, x: np.ndarray | mx.array, scale: float = 1000.0) -> mx.array:
        x_mx = _to_mx(x).astype(mx.float32)
        if x_mx.ndim < 1:
            x_mx = mx.expand_dims(x_mx, 0)
        half_dim = self.dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        freqs = mx.exp(mx.arange(half_dim, dtype=x_mx.dtype) * -emb_scale)
        emb = scale * x_mx[:, None] * freqs[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class TimestepEmbedding:
    def __init__(self, linear_1: Linear, linear_2: Linear):
        self.linear_1 = linear_1
        self.linear_2 = linear_2

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, name: str) -> "TimestepEmbedding":
        return cls(
            linear_1=Linear.from_manifest(manifest_path, f"{name}.linear_1"),
            linear_2=Linear.from_manifest(manifest_path, f"{name}.linear_2"),
        )

    def _call_mx(self, sample: np.ndarray | mx.array) -> mx.array:
        x = self.linear_1._call_mx(sample)
        x = x * mx.sigmoid(x)
        return self.linear_2._call_mx(x)


class MiniCPMDiTDecoder:
    """MLX MiniCPM stack used inside VoxCPM loc-DiT."""

    def __init__(self, layers: list[MiniCPMDecoderLayer], norm: MiniCPMRMSNorm):
        self.layers = layers
        self.norm = norm
        self._profile_context: Callable[[str, int | None], Any] | None = None
        self._profile_iteration: int | None = None

    def set_profile_context(
        self,
        profile_context: Callable[[str, int | None], Any] | None,
        *,
        iteration: int | None = None,
    ) -> None:
        self._profile_context = profile_context
        self._profile_iteration = iteration
        for idx, layer in enumerate(self.layers):
            layer.set_profile_context(profile_context, prefix=f"dit_decoder_layer_{idx}", iteration=iteration)

    def _profile(self, stage: str):
        if self._profile_context is None:
            return contextlib.nullcontext()
        return self._profile_context(stage, self._profile_iteration)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "MiniCPMDiTDecoder":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
        num_layers = int(manifest["dit_exported_layer_count"])
        expected_layers = int(manifest.get("dit_num_hidden_layers", num_layers))
        if num_layers != expected_layers:
            raise ValueError(f"DiT stack requires all {expected_layers} layers, found {num_layers}")
        return cls(
            layers=[MiniCPMDecoderLayer.from_manifest(manifest_path, scope="dit", layer_idx=idx) for idx in range(num_layers)],
            norm=MiniCPMRMSNorm.from_manifest(manifest_path, "feat_decoder.estimator.decoder.norm"),
        )

    def _call_mx(self, inputs_embeds: np.ndarray | mx.array, is_causal: bool = False) -> mx.array:
        hidden = _to_mx(inputs_embeds)
        position_ids = mx.arange(hidden.shape[1], dtype=mx.int64)
        if self._profile_context is None:
            for layer in self.layers:
                hidden, _ = layer._call_mx(hidden, position_ids=position_ids, is_causal=is_causal)
            return self.norm._call_mx(hidden)
        for idx, layer in enumerate(self.layers):
            with self._profile(f"dit_decoder_layer_{idx}.total"):
                hidden, _ = layer._call_mx(hidden, position_ids=position_ids, is_causal=is_causal)
        with self._profile("dit_decoder_norm"):
            return self.norm._call_mx(hidden)


class MLXVoxCPMLocDiT:
    """Pure-MLX VoxCPM loc-DiT estimator."""

    def __init__(
        self,
        in_proj: Linear,
        cond_proj: Linear,
        out_proj: Linear,
        time_embeddings: SinusoidalPosEmb,
        time_mlp: TimestepEmbedding,
        delta_time_mlp: TimestepEmbedding,
        decoder: MiniCPMDiTDecoder,
    ):
        self.in_proj = in_proj
        self.cond_proj = cond_proj
        self.out_proj = out_proj
        self.time_embeddings = time_embeddings
        self.time_mlp = time_mlp
        self.delta_time_mlp = delta_time_mlp
        self.decoder = decoder
        self._profile_context: Callable[[str, int | None], Any] | None = None
        self._profile_iteration: int | None = None

    def set_profile_context(
        self,
        profile_context: Callable[[str, int | None], Any] | None,
        *,
        iteration: int | None = None,
    ) -> None:
        self._profile_context = profile_context
        self._profile_iteration = iteration
        self.decoder.set_profile_context(profile_context, iteration=iteration)

    def _profile(self, stage: str):
        if self._profile_context is None:
            return contextlib.nullcontext()
        return self._profile_context(stage, self._profile_iteration)

    def _profile_eval(self, *arrays: mx.array) -> None:
        if self._profile_context is not None and arrays:
            mx.eval(*arrays)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "MLXVoxCPMLocDiT":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
        return cls(
            in_proj=Linear.from_manifest(manifest_path, "feat_decoder.estimator.in_proj"),
            cond_proj=Linear.from_manifest(manifest_path, "feat_decoder.estimator.cond_proj"),
            out_proj=Linear.from_manifest(manifest_path, "feat_decoder.estimator.out_proj"),
            time_embeddings=SinusoidalPosEmb(int(manifest["dit_hidden_size"])),
            time_mlp=TimestepEmbedding.from_manifest(manifest_path, "feat_decoder.estimator.time_mlp"),
            delta_time_mlp=TimestepEmbedding.from_manifest(manifest_path, "feat_decoder.estimator.delta_time_mlp"),
            decoder=MiniCPMDiTDecoder.from_manifest(manifest_path),
        )

    def prepare_static_inputs_mx(self, mu: np.ndarray | mx.array, cond: np.ndarray | mx.array) -> dict[str, mx.array | int]:
        if self._profile_context is None:
            cond_mx = mx.transpose(_to_mx(cond), (0, 2, 1))
            cond_mx = self.cond_proj._call_mx(cond_mx)
            x_hidden_dim = cond_mx.shape[-1]
            mu_tokens = mx.reshape(_to_mx(mu), (cond_mx.shape[0], -1, x_hidden_dim))
            return {
                "cond_mx": cond_mx,
                "mu_tokens": mu_tokens,
                "prefix": int(cond_mx.shape[1]),
                "mu_token_count": int(mu_tokens.shape[1]),
            }
        with self._profile("dit_estimator.cond_proj"):
            cond_mx = mx.transpose(_to_mx(cond), (0, 2, 1))
            cond_mx = self.cond_proj._call_mx(cond_mx)
            self._profile_eval(cond_mx)
        x_hidden_dim = cond_mx.shape[-1]
        with self._profile("dit_estimator.mu_tokens"):
            mu_tokens = mx.reshape(_to_mx(mu), (cond_mx.shape[0], -1, x_hidden_dim))
            self._profile_eval(mu_tokens)
        return {
            "cond_mx": cond_mx,
            "mu_tokens": mu_tokens,
            "prefix": int(cond_mx.shape[1]),
            "mu_token_count": int(mu_tokens.shape[1]),
        }

    def prepare_timestep_conditioning_mx(
        self,
        t: np.ndarray | mx.array,
        dt: np.ndarray | mx.array,
        dtype: mx.Dtype,
    ) -> mx.array:
        if self._profile_context is None:
            t_emb = self.time_mlp._call_mx(self.time_embeddings._call_mx(t)).astype(dtype)
            dt_emb = self.delta_time_mlp._call_mx(self.time_embeddings._call_mx(dt)).astype(dtype)
            return t_emb + dt_emb
        with self._profile("dit_estimator.timestep_embedding"):
            t_emb = self.time_mlp._call_mx(self.time_embeddings._call_mx(t)).astype(dtype)
            dt_emb = self.delta_time_mlp._call_mx(self.time_embeddings._call_mx(dt)).astype(dtype)
            total = t_emb + dt_emb
            self._profile_eval(total)
            return total

    def _call_with_static_inputs_mx(
        self,
        x: np.ndarray | mx.array,
        t: np.ndarray | mx.array,
        dt: np.ndarray | mx.array,
        static_inputs: dict[str, mx.array | int],
        timestep_conditioning: mx.array | None = None,
    ) -> mx.array:
        if self._profile_context is None:
            x_mx = mx.transpose(_to_mx(x), (0, 2, 1))
            x_mx = self.in_proj._call_mx(x_mx)
            cond_mx = static_inputs["cond_mx"]
            mu_tokens = static_inputs["mu_tokens"]
            if not isinstance(cond_mx, mx.array) or not isinstance(mu_tokens, mx.array):
                raise TypeError("precomputed DiT static inputs are invalid")
            prefix = int(static_inputs["prefix"])
            mu_token_count = int(static_inputs["mu_token_count"])
            if timestep_conditioning is None:
                t_total = self.prepare_timestep_conditioning_mx(t, dt, x_mx.dtype)
            else:
                t_total = timestep_conditioning.astype(x_mx.dtype)
            hidden = mx.concatenate([mu_tokens, t_total[:, None, :], cond_mx, x_mx], axis=1)
            hidden = self.decoder._call_mx(hidden, is_causal=False)
            hidden = hidden[:, prefix + mu_token_count + 1 :, :]
            hidden = self.out_proj._call_mx(hidden)
            return mx.transpose(hidden, (0, 2, 1))
        with self._profile("dit_estimator.in_proj"):
            x_mx = mx.transpose(_to_mx(x), (0, 2, 1))
            x_mx = self.in_proj._call_mx(x_mx)
            self._profile_eval(x_mx)
        cond_mx = static_inputs["cond_mx"]
        mu_tokens = static_inputs["mu_tokens"]
        if not isinstance(cond_mx, mx.array) or not isinstance(mu_tokens, mx.array):
            raise TypeError("precomputed DiT static inputs are invalid")
        prefix = int(static_inputs["prefix"])
        mu_token_count = int(static_inputs["mu_token_count"])

        if timestep_conditioning is None:
            t_total = self.prepare_timestep_conditioning_mx(t, dt, x_mx.dtype)
        else:
            t_total = timestep_conditioning.astype(x_mx.dtype)

        with self._profile("dit_estimator.concat"):
            hidden = mx.concatenate([mu_tokens, t_total[:, None, :], cond_mx, x_mx], axis=1)
            self._profile_eval(hidden)
        with self._profile("dit_estimator.decoder"):
            hidden = self.decoder._call_mx(hidden, is_causal=False)
            self._profile_eval(hidden)
        with self._profile("dit_estimator.out_proj"):
            hidden = hidden[:, prefix + mu_token_count + 1 :, :]
            hidden = self.out_proj._call_mx(hidden)
            output = mx.transpose(hidden, (0, 2, 1))
            self._profile_eval(output)
            return output

    def _call_mx(
        self,
        x: np.ndarray | mx.array,
        mu: np.ndarray | mx.array,
        t: np.ndarray | mx.array,
        cond: np.ndarray | mx.array,
        dt: np.ndarray | mx.array,
    ) -> mx.array:
        static_inputs = self.prepare_static_inputs_mx(mu, cond)
        return self._call_with_static_inputs_mx(x, t, dt, static_inputs)

    def __call__(self, x, mu, t, cond, dt) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(x, mu, t, cond, dt)), dtype=np.float32)


class MLXUnifiedCFM:
    """Pure-MLX Euler sampler for VoxCPM's UnifiedCFM inference path."""

    def __init__(self, estimator: MLXVoxCPMLocDiT, in_channels: int, mean_mode: bool = False):
        self.estimator = estimator
        self.in_channels = int(in_channels)
        self.mean_mode = bool(mean_mode)
        self._profile_context: Callable[[str, int | None], Any] | None = None
        self._profile_iteration: int | None = None

    def set_profile_context(
        self,
        profile_context: Callable[[str, int | None], Any] | None,
        *,
        iteration: int | None = None,
    ) -> None:
        self._profile_context = profile_context
        self._profile_iteration = iteration
        if hasattr(self.estimator, "set_profile_context"):
            self.estimator.set_profile_context(profile_context, iteration=iteration)

    def _profile(self, stage: str):
        if self._profile_context is None:
            return contextlib.nullcontext()
        return self._profile_context(stage, self._profile_iteration)

    def _profile_eval(self, *arrays: mx.array) -> None:
        if self._profile_context is not None and arrays:
            mx.eval(*arrays)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "MLXUnifiedCFM":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
        return cls(
            estimator=MLXVoxCPMLocDiT.from_manifest(manifest_path),
            in_channels=int(manifest["dit_in_channels"]),
            mean_mode=bool(manifest.get("dit_mean_mode", False)),
        )

    def _optimized_scale_mx(self, positive_flat: mx.array, negative_flat: mx.array) -> mx.array:
        dot_product = mx.sum(positive_flat * negative_flat, axis=1, keepdims=True)
        squared_norm = mx.sum(mx.square(negative_flat), axis=1, keepdims=True) + 1e-8
        return dot_product / squared_norm

    def _solve_euler_mx(
        self,
        x: mx.array,
        t_span: mx.array,
        mu: mx.array,
        cond: mx.array,
        cfg_value: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> mx.array:
        t = t_span[0]
        dt = t_span[0] - t_span[1]
        zero_init_steps = max(1, int(t_span.shape[0] * 0.04))
        b = x.shape[0]
        if self._profile_context is None:
            mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
            cond_in = mx.concatenate([cond, cond], axis=0)
            static_inputs = self.estimator.prepare_static_inputs_mx(mu_in, cond_in)
            timestep_conditioning: list[mx.array | None] = [None] * int(t_span.shape[0])
            for step in range(1, int(t_span.shape[0])):
                step_dt = t_span[step - 1] - t_span[step]
                t_in = mx.full((2 * b,), t_span[step - 1], dtype=x.dtype)
                dt_in = mx.full((2 * b,), step_dt, dtype=x.dtype)
                if not self.mean_mode:
                    dt_in = mx.zeros_like(dt_in)
                timestep_conditioning[step] = self.estimator.prepare_timestep_conditioning_mx(t_in, dt_in, x.dtype)
            for step in range(1, int(t_span.shape[0])):
                if use_cfg_zero_star and step <= zero_init_steps:
                    dphi_dt = mx.zeros_like(x)
                else:
                    x_in = mx.concatenate([x, x], axis=0)
                    t_in = mx.full((2 * b,), t, dtype=x.dtype)
                    dt_in = mx.full((2 * b,), dt, dtype=x.dtype)
                    if not self.mean_mode:
                        dt_in = mx.zeros_like(dt_in)
                    est = self.estimator._call_with_static_inputs_mx(
                        x_in,
                        t_in,
                        dt_in,
                        static_inputs,
                        timestep_conditioning=timestep_conditioning[step],
                    )
                    dphi_dt, cfg_dphi_dt = est[:b], est[b:]
                    if use_cfg_zero_star:
                        positive_flat = mx.reshape(dphi_dt, (b, -1))
                        negative_flat = mx.reshape(cfg_dphi_dt, (b, -1))
                        st_star = self._optimized_scale_mx(positive_flat, negative_flat)
                        st_star = mx.reshape(st_star, (b,) + (1,) * (dphi_dt.ndim - 1))
                    else:
                        st_star = 1.0
                    dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)
                x = x - dt * dphi_dt
                t = t - dt
                if step < int(t_span.shape[0]) - 1:
                    dt = t - t_span[step + 1]
            return x
        with self._profile("dit_cfm.static_inputs"):
            mu_in = mx.concatenate([mu, mx.zeros_like(mu)], axis=0)
            cond_in = mx.concatenate([cond, cond], axis=0)
            static_inputs = self.estimator.prepare_static_inputs_mx(mu_in, cond_in)
        timestep_conditioning: list[mx.array | None] = [None] * int(t_span.shape[0])
        for step in range(1, int(t_span.shape[0])):
            step_dt = t_span[step - 1] - t_span[step]
            t_in = mx.full((2 * b,), t_span[step - 1], dtype=x.dtype)
            dt_in = mx.full((2 * b,), step_dt, dtype=x.dtype)
            if not self.mean_mode:
                dt_in = mx.zeros_like(dt_in)
            with self._profile("dit_cfm.timestep_conditioning"):
                timestep_conditioning[step] = self.estimator.prepare_timestep_conditioning_mx(t_in, dt_in, x.dtype)
        for step in range(1, int(t_span.shape[0])):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = mx.zeros_like(x)
            else:
                x_in = mx.concatenate([x, x], axis=0)
                t_in = mx.full((2 * b,), t, dtype=x.dtype)
                dt_in = mx.full((2 * b,), dt, dtype=x.dtype)
                if not self.mean_mode:
                    dt_in = mx.zeros_like(dt_in)
                with self._profile("dit_cfm.estimator_step"):
                    est = self.estimator._call_with_static_inputs_mx(
                        x_in,
                        t_in,
                        dt_in,
                        static_inputs,
                        timestep_conditioning=timestep_conditioning[step],
                    )
                    self._profile_eval(est)
                dphi_dt, cfg_dphi_dt = est[:b], est[b:]
                if use_cfg_zero_star:
                    positive_flat = mx.reshape(dphi_dt, (b, -1))
                    negative_flat = mx.reshape(cfg_dphi_dt, (b, -1))
                    st_star = self._optimized_scale_mx(positive_flat, negative_flat)
                    st_star = mx.reshape(st_star, (b,) + (1,) * (dphi_dt.ndim - 1))
                else:
                    st_star = 1.0
                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)
            with self._profile("dit_cfm.euler_update"):
                x = x - dt * dphi_dt
                self._profile_eval(x)
            t = t - dt
            if step < int(t_span.shape[0]) - 1:
                dt = t - t_span[step + 1]
        return x

    def _call_mx(
        self,
        mu: np.ndarray | mx.array,
        n_timesteps: int,
        patch_size: int,
        cond: np.ndarray | mx.array,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
        noise: np.ndarray | mx.array | None = None,
        profile_iteration: int | None = None,
    ) -> mx.array:
        previous_iteration = self._profile_iteration
        if self._profile_context is not None:
            self.set_profile_context(self._profile_context, iteration=profile_iteration)
        mu_mx = _to_mx(mu).astype(mx.float32)
        cond_mx = _to_mx(cond).astype(mx.float32)
        b = mu_mx.shape[0]
        if noise is None:
            z = mx.random.normal((b, self.in_channels, int(patch_size))).astype(mu_mx.dtype) * float(temperature)
        else:
            z = _to_mx(noise).astype(mu_mx.dtype) * float(temperature)
        if self._profile_context is None:
            t_span = mx.linspace(1.0, 0.0, int(n_timesteps) + 1, dtype=mu_mx.dtype)
            t_span = t_span + sway_sampling_coef * (mx.cos(math.pi / 2 * t_span) - 1 + t_span)
            return self._solve_euler_mx(z, t_span, mu_mx, cond_mx, cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)
        with self._profile("dit_cfm.t_span"):
            t_span = mx.linspace(1.0, 0.0, int(n_timesteps) + 1, dtype=mu_mx.dtype)
            t_span = t_span + sway_sampling_coef * (mx.cos(math.pi / 2 * t_span) - 1 + t_span)
            self._profile_eval(t_span)
        try:
            return self._solve_euler_mx(z, t_span, mu_mx, cond_mx, cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)
        finally:
            if self._profile_context is not None:
                self.set_profile_context(self._profile_context, iteration=previous_iteration)

    def __call__(self, mu, n_timesteps: int, patch_size: int, cond, temperature: float = 1.0, cfg_value: float = 1.0) -> np.ndarray:
        return np.asarray(
            np.array(self._call_mx(mu, n_timesteps=n_timesteps, patch_size=patch_size, cond=cond, temperature=temperature, cfg_value=cfg_value)),
            dtype=np.float32,
        )
