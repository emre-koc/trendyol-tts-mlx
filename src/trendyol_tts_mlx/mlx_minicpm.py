from __future__ import annotations

from pathlib import Path

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
