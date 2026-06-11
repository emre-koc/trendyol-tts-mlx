from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import contextlib
import math

import mlx.core as mx
import numpy as np

from .mlx_voxcpm2_layers import Linear, _load_manifest_and_tensors


class MiniCPMRMSNorm:
    """MLX equivalent of VoxCPM MiniCPMRMSNorm.

    Upstream formula:
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        hidden = hidden * rsqrt(variance + eps)
        return hidden * weight
    """

    def __init__(self, weight: mx.array, eps: float = 1e-6):
        self.weight = weight
        self.eps = float(eps)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, name: str) -> "MiniCPMRMSNorm":
        manifest, tensors = _load_manifest_and_tensors(manifest_path)
        parameter_entries = {entry["name"]: entry for entry in manifest.get("parameters", [])}
        key = f"{name}.weight"
        if key not in parameter_entries:
            raise KeyError(f"RMSNorm parameter {key!r} not found in manifest")
        if name.startswith("base_lm."):
            eps = float(manifest.get("lm_rms_norm_eps", manifest.get("encoder_rms_norm_eps", 1e-6)))
        elif name.startswith("residual_lm."):
            eps = float(manifest.get("residual_lm_rms_norm_eps", manifest.get("lm_rms_norm_eps", 1e-6)))
        elif name.startswith("feat_decoder.estimator.decoder."):
            eps = float(manifest.get("dit_rms_norm_eps", manifest.get("lm_rms_norm_eps", 1e-6)))
        else:
            eps = float(manifest.get("encoder_rms_norm_eps", 1e-6))
        return cls(tensors[parameter_entries[key]["mlx_tensor"]], eps=eps)

    def _call_mx(self, hidden: np.ndarray | mx.array) -> mx.array:
        x = mx.array(hidden) if not isinstance(hidden, mx.array) else hidden
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        y = x * mx.rsqrt(variance + self.eps)
        return y * self.weight

    def __call__(self, hidden: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(hidden)), dtype=np.float32)


class MiniCPMMLP:
    """MLX equivalent of VoxCPM MiniCPMMLP.

    Upstream formula:
        down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, gate_proj: Linear, up_proj: Linear, down_proj: Linear):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, name: str) -> "MiniCPMMLP":
        return cls(
            gate_proj=Linear.from_manifest(manifest_path, f"{name}.gate_proj"),
            up_proj=Linear.from_manifest(manifest_path, f"{name}.up_proj"),
            down_proj=Linear.from_manifest(manifest_path, f"{name}.down_proj"),
        )

    def _call_mx(self, hidden: np.ndarray | mx.array) -> mx.array:
        gate = self.gate_proj._call_mx(hidden)
        gate = gate * mx.sigmoid(gate)
        up = self.up_proj._call_mx(hidden)
        return self.down_proj._call_mx(gate * up)

    def __call__(self, hidden: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._call_mx(hidden)), dtype=np.float32)


def rotate_half(x: np.ndarray | mx.array) -> mx.array:
    x_mx = mx.array(x) if not isinstance(x, mx.array) else x
    half = x_mx.shape[-1] // 2
    x1 = x_mx[..., :half]
    x2 = x_mx[..., half:]
    return mx.concatenate((-x2, x1), axis=-1)


def _ensure_rotary_broadcast(cos_or_sin: np.ndarray | mx.array) -> mx.array:
    value = mx.array(cos_or_sin) if not isinstance(cos_or_sin, mx.array) else cos_or_sin
    if value.ndim == 2:
        return value[None, None, :, :]
    if value.ndim == 3:
        return value[:, None, :, :]
    return value


def _apply_rotary_pos_emb_mx(
    q: np.ndarray | mx.array, k: np.ndarray | mx.array, cos: np.ndarray | mx.array, sin: np.ndarray | mx.array
) -> tuple[mx.array, mx.array]:
    q_mx = mx.array(q) if not isinstance(q, mx.array) else q
    k_mx = mx.array(k) if not isinstance(k, mx.array) else k
    cos_mx = _ensure_rotary_broadcast(cos)
    sin_mx = _ensure_rotary_broadcast(sin)
    q32 = q_mx.astype(mx.float32)
    k32 = k_mx.astype(mx.float32)
    q_embed = (q32 * cos_mx) + (rotate_half(q32) * sin_mx)
    k_embed = (k32 * cos_mx) + (rotate_half(k32) * sin_mx)
    return q_embed.astype(q_mx.dtype), k_embed.astype(k_mx.dtype)


def apply_rotary_pos_emb(
    q: np.ndarray | mx.array, k: np.ndarray | mx.array, cos: np.ndarray | mx.array, sin: np.ndarray | mx.array
) -> tuple[np.ndarray, np.ndarray]:
    q_embed, k_embed = _apply_rotary_pos_emb_mx(q, k, cos, sin)
    return np.asarray(np.array(q_embed), dtype=np.float32), np.asarray(np.array(k_embed), dtype=np.float32)


def _position_id_to_int(position_id: int | np.integer | np.ndarray | mx.array) -> int:
    """Convert a scalar position id without forcing MLX/NumPy roundtrip for Python ints."""
    if isinstance(position_id, (int, np.integer)):
        return int(position_id)
    return int(np.asarray(np.array(position_id)).reshape(-1)[0])


class MiniCPMLongRoPE:
    """MLX equivalent of upstream MiniCPMLongRoPE cache and lookup."""

    def __init__(
        self,
        dim: int,
        base: float,
        max_position_embeddings: int,
        short_factor: list[float],
        long_factor: list[float],
        original_max_position_embeddings: int,
    ):
        self.dim = int(dim)
        self.base = float(base)
        self.max_position_embeddings = int(max_position_embeddings)
        self.short_factor = [float(v) for v in short_factor]
        self.long_factor = [float(v) for v in long_factor]
        self.original_max_position_embeddings = int(original_max_position_embeddings)
        scale = self.max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
        self.cos_cached, self.sin_cached = self._set_cos_sin_cache(self.max_position_embeddings)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, scope: str = "encoder") -> "MiniCPMLongRoPE":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
        if scope in {"lm", "dit"}:
            prefix = "dit" if scope == "dit" else "lm"
            rope_scaling = manifest[f"{prefix}_rope_scaling"]
            hidden_size = manifest[f"{prefix}_hidden_size"]
            num_heads = manifest[f"{prefix}_num_attention_heads"]
            kv_channels = manifest.get(f"{prefix}_kv_channels")
            return cls(
                dim=kv_channels or hidden_size // num_heads,
                base=manifest[f"{prefix}_rope_theta"],
                max_position_embeddings=manifest[f"{prefix}_max_position_embeddings"],
                short_factor=rope_scaling["short_factor"],
                long_factor=rope_scaling["long_factor"],
                original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            )
        rope_scaling = manifest["encoder_rope_scaling"]
        return cls(
            dim=manifest.get("encoder_kv_channels")
            or manifest["encoder_hidden_size"] // manifest["encoder_num_attention_heads"],
            base=manifest["encoder_rope_theta"],
            max_position_embeddings=manifest["encoder_max_position_embeddings"],
            short_factor=rope_scaling["short_factor"],
            long_factor=rope_scaling["long_factor"],
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
        )

    def _set_cos_sin_cache(self, seq_len: int) -> tuple[mx.array, mx.array]:
        t = mx.arange(seq_len, dtype=mx.float32)
        factors = self.long_factor if seq_len > self.original_max_position_embeddings else self.short_factor
        ext_factors = mx.array(factors, dtype=mx.float32)
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim))
        freqs = mx.multiply(mx.outer(t, 1.0 / ext_factors), inv_freq)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        return mx.cos(emb) * self.scaling_factor, mx.sin(emb) * self.scaling_factor

    def _call_mx(self, position_ids: np.ndarray | mx.array) -> tuple[mx.array, mx.array]:
        ids = mx.array(position_ids) if not isinstance(position_ids, mx.array) else position_ids
        return self.cos_cached[ids], self.sin_cached[ids]

    def __call__(self, position_ids: np.ndarray | mx.array) -> tuple[np.ndarray, np.ndarray]:
        cos, sin = self._call_mx(position_ids)
        return np.asarray(np.array(cos), dtype=np.float32), np.asarray(np.array(sin), dtype=np.float32)


class MiniCPMAttentionProjections:
    """MLX projection/reshape slice of VoxCPM MiniCPMAttention.

    Covers q/k/v linear projections and reshapes before RoPE/SDPA, plus o_proj
    after SDPA. It intentionally does not implement RoPE or attention yet.
    """

    def __init__(
        self,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.num_heads = int(num_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, name: str) -> "MiniCPMAttentionProjections":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
        if name.startswith("feat_decoder.estimator.decoder."):
            hidden_size = manifest["dit_hidden_size"]
            num_heads = manifest["dit_num_attention_heads"]
            num_key_value_heads = manifest["dit_num_key_value_heads"]
            head_dim = int(manifest.get("dit_kv_channels") or hidden_size // num_heads)
        elif name.startswith("base_lm.") or name.startswith("residual_lm."):
            if name.startswith("residual_lm."):
                hidden_size = manifest["residual_lm_hidden_size"]
                num_heads = manifest["residual_lm_num_attention_heads"]
                num_key_value_heads = manifest["residual_lm_num_key_value_heads"]
                head_dim = int(manifest.get("residual_lm_kv_channels") or hidden_size // num_heads)
            else:
                hidden_size = manifest["lm_hidden_size"]
                num_heads = manifest["lm_num_attention_heads"]
                num_key_value_heads = manifest["lm_num_key_value_heads"]
                head_dim = int(manifest.get("lm_kv_channels") or hidden_size // num_heads)
        else:
            num_heads = manifest["encoder_num_attention_heads"]
            num_key_value_heads = manifest["encoder_num_key_value_heads"]
            head_dim = int(
                manifest.get("encoder_kv_channels")
                or manifest["encoder_hidden_size"] // manifest["encoder_num_attention_heads"]
            )
        return cls(
            q_proj=Linear.from_manifest(manifest_path, f"{name}.q_proj"),
            k_proj=Linear.from_manifest(manifest_path, f"{name}.k_proj"),
            v_proj=Linear.from_manifest(manifest_path, f"{name}.v_proj"),
            o_proj=Linear.from_manifest(manifest_path, f"{name}.o_proj"),
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )

    def _reshape_heads(self, x: mx.array, num_heads: int) -> mx.array:
        bsz, seq_len = x.shape[0], x.shape[1]
        x = mx.reshape(x, (bsz, seq_len, num_heads, self.head_dim))
        return mx.transpose(x, (0, 2, 1, 3))

    def _project_qkv_mx(self, hidden: np.ndarray | mx.array) -> tuple[mx.array, mx.array, mx.array]:
        q = self._reshape_heads(self.q_proj._call_mx(hidden), self.num_heads)
        k = self._reshape_heads(self.k_proj._call_mx(hidden), self.num_key_value_heads)
        v = self._reshape_heads(self.v_proj._call_mx(hidden), self.num_key_value_heads)
        return q, k, v

    def project_qkv(self, hidden: np.ndarray | mx.array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q, k, v = self._project_qkv_mx(hidden)
        return (
            np.asarray(np.array(q), dtype=np.float32),
            np.asarray(np.array(k), dtype=np.float32),
            np.asarray(np.array(v), dtype=np.float32),
        )

    def _output_projection_mx(self, attn_output: np.ndarray | mx.array) -> mx.array:
        return self.o_proj._call_mx(attn_output)

    def output_projection(self, attn_output: np.ndarray | mx.array) -> np.ndarray:
        return np.asarray(np.array(self._output_projection_mx(attn_output)), dtype=np.float32)


class MiniCPMAttention:
    """MLX forward slice of VoxCPM MiniCPMAttention for full-sequence inference."""

    def __init__(self, projections: MiniCPMAttentionProjections, rope: MiniCPMLongRoPE | None):
        self.projections = projections
        self.rope = rope
        self.num_heads = projections.num_heads
        self.num_key_value_heads = projections.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = projections.head_dim

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, name: str) -> "MiniCPMAttention":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
        if name.startswith("residual_lm.") and manifest.get("residual_lm_no_rope", False):
            rope = None
        else:
            if name.startswith("feat_decoder.estimator.decoder."):
                scope = "dit"
            else:
                scope = "lm" if name.startswith("base_lm.") or name.startswith("residual_lm.") else "encoder"
            rope = MiniCPMLongRoPE.from_manifest(manifest_path, scope=scope)
        return cls(
            projections=MiniCPMAttentionProjections.from_manifest(manifest_path, name),
            rope=rope,
        )

    def _scaled_dot_product_attention_mx(
        self, query: mx.array, key: mx.array, value: mx.array, is_causal: bool = False
    ) -> mx.array:
        if self.num_key_value_groups != 1:
            key = mx.repeat(key, self.num_key_value_groups, axis=1)
            value = mx.repeat(value, self.num_key_value_groups, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = mx.matmul(query.astype(mx.float32), mx.transpose(key.astype(mx.float32), (0, 1, 3, 2))) * scale
        if is_causal:
            q_len, k_len = scores.shape[-2], scores.shape[-1]
            mask = mx.triu(mx.ones((q_len, k_len), dtype=mx.bool_), k=1)
            scores = mx.where(mask[None, None, :, :], mx.array(-float("inf"), dtype=mx.float32), scores)
        probs = mx.softmax(scores, axis=-1)
        return mx.matmul(probs, value.astype(mx.float32))

    def _forward_step_mx(
        self,
        hidden_states: np.ndarray | mx.array,
        position_id: int | np.ndarray | mx.array,
        kv_cache: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        hidden = mx.array(hidden_states) if not isinstance(hidden_states, mx.array) else hidden_states
        if hidden.ndim != 2:
            raise ValueError(f"forward_step expects [batch, hidden], got shape {hidden.shape}")
        bsz = hidden.shape[0]
        pos = _position_id_to_int(position_id)

        query_states = self.projections.q_proj._call_mx(hidden)
        key_states = self.projections.k_proj._call_mx(hidden)
        value_states = self.projections.v_proj._call_mx(hidden)
        query_states = mx.transpose(mx.reshape(query_states, (bsz, 1, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        key_states = mx.transpose(mx.reshape(key_states, (bsz, 1, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))
        value_states = mx.transpose(mx.reshape(value_states, (bsz, 1, self.num_key_value_heads, self.head_dim)), (0, 2, 1, 3))

        if self.rope is not None:
            cos, sin = self.rope._call_mx(mx.array([pos], dtype=mx.int64))
            query_states, key_states = _apply_rotary_pos_emb_mx(query_states, key_states, cos, sin)

        key_cache, value_cache = kv_cache
        key_cache = mx.concatenate([key_cache[:, :, :pos, :], key_states, key_cache[:, :, pos + 1 :, :]], axis=2)
        value_cache = mx.concatenate(
            [value_cache[:, :, :pos, :], value_states, value_cache[:, :, pos + 1 :, :]], axis=2
        )
        key_context = key_cache[:, :, : pos + 1, :]
        value_context = value_cache[:, :, : pos + 1, :]

        attn_output = self._scaled_dot_product_attention_mx(query_states, key_context, value_context, is_causal=False)
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (bsz, self.num_heads * self.head_dim))
        attn_output = self.projections._output_projection_mx(attn_output)
        return attn_output, (key_cache, value_cache)

    def _call_mx(
        self, hidden_states: np.ndarray | mx.array, position_ids: np.ndarray | mx.array, is_causal: bool = False
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        hidden = mx.array(hidden_states) if not isinstance(hidden_states, mx.array) else hidden_states
        bsz, q_len = hidden.shape[0], hidden.shape[1]
        query_states, key_states, value_states = self.projections._project_qkv_mx(hidden)
        if self.rope is not None:
            cos, sin = self.rope._call_mx(position_ids)
            query_states, key_states = _apply_rotary_pos_emb_mx(query_states, key_states, cos, sin)
        past_key_value = (key_states, value_states)

        attn_output = self._scaled_dot_product_attention_mx(query_states, key_states, value_states, is_causal=is_causal)
        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (bsz, q_len, self.num_heads * self.head_dim))
        attn_output = self.projections._output_projection_mx(attn_output)
        return attn_output, past_key_value

    def __call__(
        self, hidden_states: np.ndarray | mx.array, position_ids: np.ndarray | mx.array, is_causal: bool = False
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        output, (key_states, value_states) = self._call_mx(hidden_states, position_ids, is_causal=is_causal)
        return (
            np.asarray(np.array(output), dtype=np.float32),
            (
                np.asarray(np.array(key_states), dtype=np.float32),
                np.asarray(np.array(value_states), dtype=np.float32),
            ),
        )


class MiniCPMDecoderLayer:
    """MLX equivalent of one VoxCPM local-encoder MiniCPMDecoderLayer."""

    def __init__(
        self,
        input_layernorm: MiniCPMRMSNorm,
        self_attn: MiniCPMAttention,
        post_attention_layernorm: MiniCPMRMSNorm,
        mlp: MiniCPMMLP,
        scale_depth: float,
        num_hidden_layers: int,
        use_mup: bool,
    ):
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp
        self.scale_depth = float(scale_depth)
        self.num_hidden_layers = int(num_hidden_layers)
        self.use_mup = bool(use_mup)
        self._profile_context: Callable[[str, int | None], Any] | None = None
        self._profile_prefix: str | None = None
        self._profile_iteration: int | None = None

    def set_profile_context(
        self,
        profile_context: Callable[[str, int | None], Any] | None,
        *,
        prefix: str,
        iteration: int | None = None,
    ) -> None:
        self._profile_context = profile_context
        self._profile_prefix = prefix
        self._profile_iteration = iteration

    def _profile(self, suffix: str):
        if self._profile_context is None or self._profile_prefix is None:
            return contextlib.nullcontext()
        return self._profile_context(f"{self._profile_prefix}.{suffix}", self._profile_iteration)

    def _profile_eval(self, *arrays: mx.array) -> None:
        if self._profile_context is not None and arrays:
            mx.eval(*arrays)

    @classmethod
    def from_manifest(
        cls, manifest_path: str | Path, scope: str = "encoder", layer_idx: int = 0
    ) -> "MiniCPMDecoderLayer":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
        if scope == "dit":
            prefix = f"feat_decoder.estimator.decoder.layers.{layer_idx}"
            return cls(
                input_layernorm=MiniCPMRMSNorm.from_manifest(manifest_path, f"{prefix}.input_layernorm"),
                self_attn=MiniCPMAttention.from_manifest(manifest_path, f"{prefix}.self_attn"),
                post_attention_layernorm=MiniCPMRMSNorm.from_manifest(manifest_path, f"{prefix}.post_attention_layernorm"),
                mlp=MiniCPMMLP.from_manifest(manifest_path, f"{prefix}.mlp"),
                scale_depth=manifest.get("dit_scale_depth", 1.0),
                num_hidden_layers=manifest.get("dit_num_hidden_layers", 1),
                use_mup=manifest.get("dit_use_mup", False),
            )
        if scope == "base_lm" or scope == "residual_lm":
            prefix = f"{scope}.layers.{layer_idx}"
            if scope == "residual_lm":
                scale_depth = manifest.get("residual_lm_scale_depth", manifest.get("lm_scale_depth", 1.0))
                num_hidden_layers = manifest.get("residual_lm_num_hidden_layers", manifest.get("residual_lm_exported_layer_count", 1))
                use_mup = manifest.get("residual_lm_use_mup", manifest.get("lm_use_mup", False))
            else:
                scale_depth = manifest.get("lm_scale_depth", 1.0)
                num_hidden_layers = manifest.get("lm_num_hidden_layers", manifest.get("lm_exported_layer_count", 1))
                use_mup = manifest.get("lm_use_mup", False)
            return cls(
                input_layernorm=MiniCPMRMSNorm.from_manifest(manifest_path, f"{prefix}.input_layernorm"),
                self_attn=MiniCPMAttention.from_manifest(manifest_path, f"{prefix}.self_attn"),
                post_attention_layernorm=MiniCPMRMSNorm.from_manifest(manifest_path, f"{prefix}.post_attention_layernorm"),
                mlp=MiniCPMMLP.from_manifest(manifest_path, f"{prefix}.mlp"),
                scale_depth=scale_depth,
                num_hidden_layers=num_hidden_layers,
                use_mup=use_mup,
            )
        prefix = f"feat_encoder.encoder.layers.{layer_idx}"
        return cls(
            input_layernorm=MiniCPMRMSNorm.from_manifest(manifest_path, f"{prefix}.input_layernorm"),
            self_attn=MiniCPMAttention.from_manifest(manifest_path, f"{prefix}.self_attn"),
            post_attention_layernorm=MiniCPMRMSNorm.from_manifest(manifest_path, f"{prefix}.post_attention_layernorm"),
            mlp=MiniCPMMLP.from_manifest(manifest_path, f"{prefix}.mlp"),
            scale_depth=manifest.get("encoder_scale_depth", 1.0),
            num_hidden_layers=manifest.get("encoder_num_hidden_layers", 1),
            use_mup=manifest.get("encoder_use_mup", False),
        )

    def _residual_add_mx(self, residual: mx.array, hidden: mx.array) -> mx.array:
        if self.use_mup:
            hidden = hidden * (self.scale_depth / math.sqrt(self.num_hidden_layers))
        return residual + hidden

    def _forward_step_mx(
        self,
        hidden_states: np.ndarray | mx.array,
        position_id: int | np.ndarray | mx.array,
        kv_cache: tuple[mx.array, mx.array],
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        hidden = mx.array(hidden_states) if not isinstance(hidden_states, mx.array) else hidden_states
        residual = hidden
        hidden = self.input_layernorm._call_mx(hidden)
        hidden, updated_cache = self.self_attn._forward_step_mx(hidden, position_id=position_id, kv_cache=kv_cache)
        hidden = self._residual_add_mx(residual, hidden)

        residual = hidden
        hidden = self.post_attention_layernorm._call_mx(hidden)
        hidden = self.mlp._call_mx(hidden)
        hidden = self._residual_add_mx(residual, hidden)
        return hidden, updated_cache

    def _call_mx(
        self, hidden_states: np.ndarray | mx.array, position_ids: np.ndarray | mx.array, is_causal: bool = False
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        hidden = mx.array(hidden_states) if not isinstance(hidden_states, mx.array) else hidden_states
        if self._profile_context is None:
            residual = hidden
            hidden = self.input_layernorm._call_mx(hidden)
            hidden, present_key_value = self.self_attn._call_mx(hidden, position_ids=position_ids, is_causal=is_causal)
            hidden = self._residual_add_mx(residual, hidden)

            residual = hidden
            hidden = self.post_attention_layernorm._call_mx(hidden)
            hidden = self.mlp._call_mx(hidden)
            hidden = self._residual_add_mx(residual, hidden)
            return hidden, present_key_value
        residual = hidden
        hidden = self.input_layernorm._call_mx(hidden)
        with self._profile("attention"):
            hidden, present_key_value = self.self_attn._call_mx(hidden, position_ids=position_ids, is_causal=is_causal)
            self._profile_eval(hidden)
        hidden = self._residual_add_mx(residual, hidden)

        residual = hidden
        hidden = self.post_attention_layernorm._call_mx(hidden)
        with self._profile("mlp"):
            hidden = self.mlp._call_mx(hidden)
            self._profile_eval(hidden)
        hidden = self._residual_add_mx(residual, hidden)
        return hidden, present_key_value

    def __call__(
        self, hidden_states: np.ndarray | mx.array, position_ids: np.ndarray | mx.array, is_causal: bool = False
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        output, (key_states, value_states) = self._call_mx(hidden_states, position_ids, is_causal=is_causal)
        return (
            np.asarray(np.array(output), dtype=np.float32),
            (
                np.asarray(np.array(key_states), dtype=np.float32),
                np.asarray(np.array(value_states), dtype=np.float32),
            ),
        )


class MiniCPMEncoder:
    """MLX equivalent of the VoxCPM local-encoder MiniCPMModel stack."""

    def __init__(self, layers: list[MiniCPMDecoderLayer], norm: MiniCPMRMSNorm):
        self.layers = layers
        self.norm = norm

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "MiniCPMEncoder":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
        num_layers = int(manifest["encoder_num_hidden_layers"])
        return cls(
            layers=[MiniCPMDecoderLayer.from_manifest(manifest_path, layer_idx=idx) for idx in range(num_layers)],
            norm=MiniCPMRMSNorm.from_manifest(manifest_path, "feat_encoder.encoder.norm"),
        )

    def _call_mx(
        self, inputs_embeds: np.ndarray | mx.array, is_causal: bool = False
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        hidden = mx.array(inputs_embeds) if not isinstance(inputs_embeds, mx.array) else inputs_embeds
        position_ids = mx.arange(hidden.shape[1], dtype=mx.int64)
        next_decoder_cache = []
        for layer in self.layers:
            hidden, this_cache = layer._call_mx(hidden, position_ids=position_ids, is_causal=is_causal)
            next_decoder_cache.append(this_cache)
        hidden = self.norm._call_mx(hidden)
        return hidden, next_decoder_cache

    def __call__(
        self, inputs_embeds: np.ndarray | mx.array, is_causal: bool = False
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
