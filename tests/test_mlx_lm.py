from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from voxcpm.modules.minicpm4.model import MiniCPMAttention, MiniCPMDecoderLayer, MiniCPMLongRoPE, MiniCPMMLP, MiniCPMModel, MiniCPMRMSNorm


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/Trendyol-TTS-mlx/voxcpm2_layers_manifest.json"


def ensure_exported_layers() -> None:
    needs_export = True
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        parameter_names = {entry["name"] for entry in manifest.get("parameters", [])}
        linear_names = {entry["name"] for entry in manifest.get("linear_layers", [])}
        expected_lm_layers = int(manifest.get("lm_num_hidden_layers", 28))
        last_layer = expected_lm_layers - 1
        needs_export = (
            "base_lm.embed_tokens.weight" not in parameter_names
            or "base_lm.norm.weight" not in parameter_names
            or "residual_lm.norm.weight" not in parameter_names
            or "base_lm.layers.0.input_layernorm.weight" not in parameter_names
            or f"base_lm.layers.{last_layer}.input_layernorm.weight" not in parameter_names
            or "residual_lm.layers.0.input_layernorm.weight" not in parameter_names
            or "base_lm.layers.0.self_attn.q_proj" not in linear_names
            or f"base_lm.layers.{last_layer}.self_attn.q_proj" not in linear_names
            or "residual_lm.layers.0.self_attn.q_proj" not in linear_names
            or manifest.get("lm_num_attention_heads") is None
            or manifest.get("lm_exported_layer_count") != expected_lm_layers
            or manifest.get("residual_lm_exported_layer_count") != manifest.get("residual_lm_num_hidden_layers")
            or any(
                f"base_lm.layers.0.mlp.{proj}" not in linear_names
                for proj in ("gate_proj", "up_proj", "down_proj")
            )
            or any(
                f"base_lm.layers.{last_layer}.mlp.{proj}" not in linear_names
                for proj in ("gate_proj", "up_proj", "down_proj")
            )
        )
    if not needs_export:
        return
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_voxcpm2_layers_mlx.py"), "--local-files-only"],
        cwd=ROOT,
        check=True,
    )


def load_model_dir() -> Path:
    return Path(snapshot_download("Trendyol/Trendyol-TTS", local_files_only=True))


def torch_base_lm_token_embedding(token_ids: np.ndarray) -> np.ndarray:
    model_dir = load_model_dir()
    cfg = json.loads((model_dir / "config.json").read_text())["lm_config"]
    scale_emb = float(cfg.get("scale_emb", 1.0)) if cfg.get("use_mup", False) else 1.0
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        weight = f.get_tensor("base_lm.embed_tokens.weight")
    embedding = torch.nn.Embedding(weight.shape[0], weight.shape[1]).eval()
    embedding.weight.data.copy_(weight)
    with torch.no_grad():
        return (embedding(torch.from_numpy(token_ids).long()) * scale_emb).detach().cpu().numpy()


def lm_config() -> SimpleNamespace:
    model_dir = load_model_dir()
    cfg = json.loads((model_dir / "config.json").read_text())["lm_config"]
    return SimpleNamespace(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        kv_channels=cfg["kv_channels"],
        rope_theta=cfg["rope_theta"],
        rope_scaling=SimpleNamespace(**cfg["rope_scaling"]),
        rms_norm_eps=cfg["rms_norm_eps"],
        scale_depth=cfg["scale_depth"],
        num_hidden_layers=cfg["num_hidden_layers"],
        use_mup=cfg["use_mup"],
        vocab_size=cfg["vocab_size"],
        no_rope=cfg.get("no_rope", False),
    )


def residual_lm_config() -> SimpleNamespace:
    cfg = lm_config()
    root_cfg = json.loads((load_model_dir() / "config.json").read_text())
    cfg.num_hidden_layers = root_cfg["residual_lm_num_layers"]
    cfg.vocab_size = 0
    cfg.no_rope = root_cfg["residual_lm_no_rope"]
    return cfg


def load_torch_lm_rmsnorm(name: str) -> MiniCPMRMSNorm:
    cfg = lm_config()
    norm = MiniCPMRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).eval()
    model_dir = load_model_dir()
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        norm.weight.data.copy_(f.get_tensor(f"{name}.weight"))
    return norm


def load_torch_lm_mlp(name: str) -> MiniCPMMLP:
    mlp = MiniCPMMLP(lm_config()).eval()
    state = {}
    model_dir = load_model_dir()
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        for key in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            state[key] = f.get_tensor(f"{name}.{key}")
    mlp.load_state_dict(state)
    return mlp


def load_torch_lm_attention(name: str, layer_idx: int = 0) -> MiniCPMAttention:
    attn = MiniCPMAttention(lm_config(), layer_idx=layer_idx).eval()
    state = {}
    model_dir = load_model_dir()
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        for key in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):
            state[key] = f.get_tensor(f"{name}.{key}")
    attn.load_state_dict(state)
    return attn


def load_torch_lm_decoder_layer(layer_idx: int = 0) -> MiniCPMDecoderLayer:
    layer = MiniCPMDecoderLayer(lm_config(), layer_idx=layer_idx).eval()
    prefix = f"base_lm.layers.{layer_idx}"
    state = {}
    model_dir = load_model_dir()
    suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        for suffix in suffixes:
            state[suffix] = f.get_tensor(f"{prefix}.{suffix}")
    layer.load_state_dict(state)
    return layer


def torch_lm_position_emb(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    rope = MiniCPMLongRoPE(lm_config())
    position_ids = torch.arange(seq_len, dtype=torch.long)
    return rope(position_ids)


def load_torch_lm_model() -> MiniCPMModel:
    model = MiniCPMModel(lm_config()).eval()
    state = {}
    model_dir = load_model_dir()
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("base_lm."):
                state[key[len("base_lm."):]] = f.get_tensor(key)
    model.load_state_dict(state)
    return model


def load_torch_residual_lm_model() -> MiniCPMModel:
    model = MiniCPMModel(residual_lm_config()).eval()
    state = {}
    model_dir = load_model_dir()
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("residual_lm."):
                state[key[len("residual_lm."):]] = f.get_tensor(key)
    model.load_state_dict(state)
    return model


class TestMLXBaseLM(unittest.TestCase):
    def test_token_embedding_matches_pytorch_with_scale(self) -> None:
        from trendyol_tts_mlx.mlx_lm import MiniCPMTokenEmbedding

        ensure_exported_layers()
        token_ids = np.array([[101, 102, 103, 104], [1, 42, 734, 2048]], dtype=np.int64)

        y_torch = torch_base_lm_token_embedding(token_ids)
        y_mlx = MiniCPMTokenEmbedding.from_manifest(MANIFEST, "base_lm.embed_tokens")(token_ids)

        self.assertEqual(y_torch.shape, y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-5, atol=1e-5)

    def test_first_lm_rmsnorm_matches_pytorch(self) -> None:
        from trendyol_tts_mlx.mlx_minicpm import MiniCPMRMSNorm as MLXMiniCPMRMSNorm

        ensure_exported_layers()
        rng = np.random.default_rng(2039)
        hidden = rng.standard_normal((2, 4, 2048), dtype=np.float32)
        name = "base_lm.layers.0.input_layernorm"

        torch_norm = load_torch_lm_rmsnorm(name)
        with torch.no_grad():
            y_torch = torch_norm(torch.from_numpy(hidden)).detach().cpu().numpy()

        y_mlx = MLXMiniCPMRMSNorm.from_manifest(MANIFEST, name)(hidden)

        self.assertEqual(y_torch.shape, y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-5, atol=1e-5)

    def test_first_lm_mlp_matches_pytorch(self) -> None:
        from trendyol_tts_mlx.mlx_minicpm import MiniCPMMLP as MLXMiniCPMMLP

        ensure_exported_layers()
        rng = np.random.default_rng(2040)
        hidden = rng.standard_normal((2, 4, 2048), dtype=np.float32)
        name = "base_lm.layers.0.mlp"

        torch_mlp = load_torch_lm_mlp(name)
        with torch.no_grad():
            y_torch = torch_mlp(torch.from_numpy(hidden)).detach().cpu().numpy()

        y_mlx = MLXMiniCPMMLP.from_manifest(MANIFEST, name)(hidden)

        self.assertEqual(y_torch.shape, y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch, rtol=5e-5, atol=1e-4)

    def test_first_lm_self_attention_matches_pytorch(self) -> None:
        from trendyol_tts_mlx.mlx_minicpm import MiniCPMAttention as MLXMiniCPMAttention

        ensure_exported_layers()
        rng = np.random.default_rng(2041)
        hidden = rng.standard_normal((1, 4, 2048), dtype=np.float32)
        name = "base_lm.layers.0.self_attn"

        torch_attn = load_torch_lm_attention(name)
        position_emb = torch_lm_position_emb(hidden.shape[1])
        with torch.no_grad():
            y_torch, (k_torch, v_torch) = torch_attn(
                torch.from_numpy(hidden), position_emb=position_emb, is_causal=True
            )

        y_mlx, (k_mlx, v_mlx) = MLXMiniCPMAttention.from_manifest(MANIFEST, name)(
            hidden, np.arange(hidden.shape[1], dtype=np.int64), is_causal=True
        )

        self.assertEqual(tuple(y_torch.shape), y_mlx.shape)
        self.assertEqual(tuple(k_torch.shape), k_mlx.shape)
        self.assertEqual(tuple(v_torch.shape), v_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(k_mlx, k_torch.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(v_mlx, v_torch.detach().cpu().numpy(), rtol=1e-5, atol=1e-5)

    def test_first_lm_decoder_layer_matches_pytorch(self) -> None:
        from trendyol_tts_mlx.mlx_minicpm import MiniCPMDecoderLayer as MLXMiniCPMDecoderLayer

        ensure_exported_layers()
        rng = np.random.default_rng(2042)
        hidden = rng.standard_normal((1, 4, 2048), dtype=np.float32)

        torch_layer = load_torch_lm_decoder_layer(layer_idx=0)
        position_emb = torch_lm_position_emb(hidden.shape[1])
        with torch.no_grad():
            y_torch, _ = torch_layer(torch.from_numpy(hidden), position_emb=position_emb, is_causal=True)

        y_mlx, _ = MLXMiniCPMDecoderLayer.from_manifest(MANIFEST, "base_lm", layer_idx=0)(
            hidden, np.arange(hidden.shape[1], dtype=np.int64), is_causal=True
        )

        self.assertEqual(tuple(y_torch.shape), y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch.detach().cpu().numpy(), rtol=2e-4, atol=2e-4)
    def test_last_lm_decoder_layer_matches_pytorch(self) -> None:
        from trendyol_tts_mlx.mlx_minicpm import MiniCPMDecoderLayer as MLXMiniCPMDecoderLayer

        ensure_exported_layers()
        layer_idx = lm_config().num_hidden_layers - 1
        rng = np.random.default_rng(2043)
        hidden = rng.standard_normal((1, 4, 2048), dtype=np.float32)

        torch_layer = load_torch_lm_decoder_layer(layer_idx=layer_idx)
        position_emb = torch_lm_position_emb(hidden.shape[1])
        with torch.no_grad():
            y_torch, _ = torch_layer(torch.from_numpy(hidden), position_emb=position_emb, is_causal=True)

        y_mlx, _ = MLXMiniCPMDecoderLayer.from_manifest(MANIFEST, "base_lm", layer_idx=layer_idx)(
            hidden, np.arange(hidden.shape[1], dtype=np.int64), is_causal=True
        )

        self.assertEqual(tuple(y_torch.shape), y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch.detach().cpu().numpy(), rtol=3e-4, atol=3e-4)
    def test_base_lm_stack_matches_pytorch(self) -> None:
        from trendyol_tts_mlx.mlx_lm import MiniCPMBaseLM

        ensure_exported_layers()
        rng = np.random.default_rng(2044)
        inputs_embeds = rng.standard_normal((1, 3, 2048), dtype=np.float32)

        torch_model = load_torch_lm_model()
        with torch.no_grad():
            y_torch, torch_cache = torch_model(torch.from_numpy(inputs_embeds), is_causal=True)

        y_mlx, mlx_cache = MiniCPMBaseLM.from_manifest(MANIFEST)(inputs_embeds, is_causal=True)

        self.assertEqual(tuple(y_torch.shape), y_mlx.shape)
        self.assertEqual(len(torch_cache), len(mlx_cache))
        self.assertEqual(tuple(torch_cache[-1][0].shape), mlx_cache[-1][0].shape)
        self.assertEqual(tuple(torch_cache[-1][1].shape), mlx_cache[-1][1].shape)
        np.testing.assert_allclose(y_mlx, y_torch.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)

    def test_base_lm_forward_step_matches_pytorch_after_prefill(self) -> None:
        from trendyol_tts_mlx.mlx_lm import MiniCPMBaseLM

        ensure_exported_layers()
        rng = np.random.default_rng(2045)
        prefill_embeds = rng.standard_normal((1, 3, 2048), dtype=np.float32)
        next_embed = rng.standard_normal((1, 2048), dtype=np.float32)
        max_length = 8

        torch_model = load_torch_lm_model()
        torch_model.setup_cache(batch_size=1, max_length=max_length, device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad():
            _, torch_cache = torch_model(torch.from_numpy(prefill_embeds), is_causal=True)
            torch_model.kv_cache.fill_caches(torch_cache)
            y_torch = torch_model.forward_step(
                torch.from_numpy(next_embed),
                torch.tensor([torch_model.kv_cache.step()], dtype=torch.long),
            )

        mlx_model = MiniCPMBaseLM.from_manifest(MANIFEST)
        _, mlx_prefill_cache = mlx_model(prefill_embeds, is_causal=True)
        mlx_model.setup_cache(batch_size=1, max_length=max_length)
        mlx_model.fill_caches(mlx_prefill_cache)
        y_mlx = mlx_model.forward_step(next_embed, mlx_model.cache_step())

        self.assertEqual(tuple(y_torch.shape), y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
    def test_residual_lm_stack_matches_pytorch_without_rope(self) -> None:
        from trendyol_tts_mlx.mlx_lm import MiniCPMResidualLM

        ensure_exported_layers()
        rng = np.random.default_rng(2046)
        inputs_embeds = rng.standard_normal((1, 3, 2048), dtype=np.float32)

        torch_model = load_torch_residual_lm_model()
        with torch.no_grad():
            y_torch, torch_cache = torch_model(torch.from_numpy(inputs_embeds), is_causal=True)

        y_mlx, mlx_cache = MiniCPMResidualLM.from_manifest(MANIFEST)(inputs_embeds, is_causal=True)

        self.assertEqual(tuple(y_torch.shape), y_mlx.shape)
        self.assertEqual(len(torch_cache), len(mlx_cache))
        self.assertEqual(tuple(torch_cache[-1][0].shape), mlx_cache[-1][0].shape)
        self.assertEqual(tuple(torch_cache[-1][1].shape), mlx_cache[-1][1].shape)
        np.testing.assert_allclose(y_mlx, y_torch.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)

    def test_residual_lm_forward_step_matches_pytorch_after_prefill_without_rope(self) -> None:
        from trendyol_tts_mlx.mlx_lm import MiniCPMResidualLM

        ensure_exported_layers()
        rng = np.random.default_rng(2047)
        prefill_embeds = rng.standard_normal((1, 3, 2048), dtype=np.float32)
        next_embed = rng.standard_normal((1, 2048), dtype=np.float32)
        max_length = 8

        torch_model = load_torch_residual_lm_model()
        torch_model.setup_cache(batch_size=1, max_length=max_length, device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad():
            _, torch_cache = torch_model(torch.from_numpy(prefill_embeds), is_causal=True)
            torch_model.kv_cache.fill_caches(torch_cache)
            y_torch = torch_model.forward_step(
                torch.from_numpy(next_embed),
                torch.tensor([torch_model.kv_cache.step()], dtype=torch.long),
            )

        mlx_model = MiniCPMResidualLM.from_manifest(MANIFEST)
        _, mlx_prefill_cache = mlx_model(prefill_embeds, is_causal=True)
        mlx_model.setup_cache(batch_size=1, max_length=max_length)
        mlx_model.fill_caches(mlx_prefill_cache)
        y_mlx = mlx_model.forward_step(next_embed, mlx_model.cache_step())

        self.assertEqual(tuple(y_torch.shape), y_mlx.shape)
        np.testing.assert_allclose(y_mlx, y_torch.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
