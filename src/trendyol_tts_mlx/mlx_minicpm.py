from __future__ import annotations

from pathlib import Path
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
    def from_manifest(cls, manifest_path: str | Path) -> "MiniCPMLongRoPE":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
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
        head_dim = int(
            manifest.get("encoder_kv_channels")
            or manifest["encoder_hidden_size"] // manifest["encoder_num_attention_heads"]
        )
        return cls(
            q_proj=Linear.from_manifest(manifest_path, f"{name}.q_proj"),
            k_proj=Linear.from_manifest(manifest_path, f"{name}.k_proj"),
            v_proj=Linear.from_manifest(manifest_path, f"{name}.v_proj"),
            o_proj=Linear.from_manifest(manifest_path, f"{name}.o_proj"),
            num_heads=manifest["encoder_num_attention_heads"],
            num_key_value_heads=manifest["encoder_num_key_value_heads"],
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

    def __init__(self, projections: MiniCPMAttentionProjections, rope: MiniCPMLongRoPE):
        self.projections = projections
        self.rope = rope
        self.num_heads = projections.num_heads
        self.num_key_value_heads = projections.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = projections.head_dim

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, name: str) -> "MiniCPMAttention":
        return cls(
            projections=MiniCPMAttentionProjections.from_manifest(manifest_path, name),
            rope=MiniCPMLongRoPE.from_manifest(manifest_path),
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

    def _call_mx(
        self, hidden_states: np.ndarray | mx.array, position_ids: np.ndarray | mx.array, is_causal: bool = False
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        hidden = mx.array(hidden_states) if not isinstance(hidden_states, mx.array) else hidden_states
        bsz, q_len = hidden.shape[0], hidden.shape[1]
        query_states, key_states, value_states = self.projections._project_qkv_mx(hidden)
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

    @classmethod
    def from_manifest(cls, manifest_path: str | Path, layer_idx: int = 0) -> "MiniCPMDecoderLayer":
        manifest, _ = _load_manifest_and_tensors(manifest_path)
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

    def _call_mx(
        self, hidden_states: np.ndarray | mx.array, position_ids: np.ndarray | mx.array, is_causal: bool = False
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        hidden = mx.array(hidden_states) if not isinstance(hidden_states, mx.array) else hidden_states
        residual = hidden
        hidden = self.input_layernorm._call_mx(hidden)
        hidden, present_key_value = self.self_attn._call_mx(hidden, position_ids=position_ids, is_causal=is_causal)
        hidden = self._residual_add_mx(residual, hidden)

        residual = hidden
        hidden = self.post_attention_layernorm._call_mx(hidden)
        hidden = self.mlp._call_mx(hidden)
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
