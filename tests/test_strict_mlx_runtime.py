from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _all_imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_mlx_backend_has_no_torch_imports() -> None:
    assert "torch" not in _all_imported_names(ROOT / "src/trendyol_tts_mlx/mlx_backend.py")


def test_mlx_load_uses_native_runtime_without_voxcpm_from_pretrained() -> None:
    from trendyol_tts_mlx.backend import SynthesisConfig, TrendyolTTS

    fake_native = Mock()
    fake_native.sample_rate = 48_000
    with patch("trendyol_tts_mlx.backend.TrendyolMLXModel.from_pretrained", return_value=fake_native) as native_loader:
        with patch("voxcpm.core.VoxCPM.from_pretrained") as upstream_loader:
            tts = TrendyolTTS(SynthesisConfig(backend="mlx", device="mps", local_files_only=True)).load()

    native_loader.assert_called_once()
    upstream_loader.assert_not_called()
    assert tts.model is fake_native
    assert tts.sample_rate == 48_000


def test_mlx_load_passes_configured_artifact_dir_to_native_runtime() -> None:
    from trendyol_tts_mlx.backend import SynthesisConfig, TrendyolTTS

    fake_native = Mock()
    fake_native.sample_rate = 48_000
    with patch("trendyol_tts_mlx.backend.TrendyolMLXModel.from_pretrained", return_value=fake_native) as native_loader:
        TrendyolTTS(
            SynthesisConfig(
                backend="mlx",
                device="mps",
                local_files_only=True,
                mlx_artifact_dir="/tmp/mlx-artifacts",
            )
        ).load()

    assert native_loader.call_args.kwargs["artifact_dir"] == "/tmp/mlx-artifacts"


def test_config_from_env_reads_mlx_artifact_dir(monkeypatch) -> None:
    from trendyol_tts_mlx.backend import config_from_env

    monkeypatch.setenv("MLX_ARTIFACT_DIR", "/tmp/trendyol-artifacts")

    cfg = config_from_env(backend="mlx")

    assert cfg.mlx_artifact_dir == "/tmp/trendyol-artifacts"


def test_native_mlx_generate_text_only_returns_numpy_audio(monkeypatch) -> None:
    from trendyol_tts_mlx.mlx_backend import TrendyolMLXModel

    model = TrendyolMLXModel.__new__(TrendyolMLXModel)
    model.sample_rate = 48_000
    model.patch_size = 4
    model.latent_dim = 64
    model.audio_start_token = 101
    model._decode_chunk_size = 320
    model.text_tokenizer = lambda text: [11, 22, 33]
    model.inference = Mock()
    model.audio_vae = Mock()
    model.inference.infer_arrays.return_value = (
        np.zeros((1, 64, 8), dtype=np.float32),
        np.zeros((2, 4, 64), dtype=np.float32),
        0,
    )
    model.audio_vae.decode.return_value = np.ones((1, 1, 128), dtype=np.float32) * 0.25

    audio = model.generate("Merhaba", cfg_value=2.0, inference_timesteps=10, max_len=40)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.shape == (128,)
    np.testing.assert_allclose(audio[:3], 0.25)
    kwargs = model.inference.infer_arrays.call_args.kwargs
    assert kwargs["text_tokens"].dtype == np.int64
    assert kwargs["text_tokens"].tolist() == [[11, 22, 33, 101]]
    assert kwargs["feat"].shape == (1, 4, 4, 64)
    assert kwargs["text_mask"].tolist() == [[1.0, 1.0, 1.0, 1.0]]
    assert kwargs["feat_mask"].tolist() == [[0.0, 0.0, 0.0, 0.0]]


def test_native_mlx_rejects_prompt_audio_until_encoder_is_ported() -> None:
    from trendyol_tts_mlx.mlx_backend import TrendyolMLXModel

    model = TrendyolMLXModel.__new__(TrendyolMLXModel)

    try:
        model.generate("Merhaba", prompt_wav_path="prompt.wav", prompt_text="Selam")
    except NotImplementedError as exc:
        assert "prompt audio" in str(exc).lower()
    else:
        raise AssertionError("prompt audio should be explicitly unsupported in strict MLX mode")
