from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL_ID = "Trendyol/Trendyol-TTS"


@dataclass(frozen=True)
class SynthesisConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "mps"
    optimize: bool = False
    seed: int = 42
    cfg_value: float = 2.0
    inference_timesteps: int = 16
    max_len: int = 4096
    normalize: bool = True
    denoise: bool = False
    cache_dir: str | None = None
    local_files_only: bool = False


class TrendyolTTS:
    """Apple Silicon runner for Trendyol/Trendyol-TTS.

    The public project is MLX-ready, but the production backend here delegates to
    upstream VoxCPM on PyTorch MPS. A pure MLX VoxCPM2 implementation can replace
    this class later without changing the CLI/app surface.
    """

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self._model = None
        self.sample_rate = 48000

    def load(self) -> "TrendyolTTS":
        # Import lazily so `trendyol-tts-check` can show actionable dependency failures.
        import torch
        from voxcpm.core import VoxCPM

        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

        if self.config.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but torch MPS is not available on this machine")

        # VoxCPM 2.0.3 auto-adjusts bfloat16 -> float32 on MPS for stability.
        self._model = VoxCPM.from_pretrained(
            hf_model_id=self.config.model_id,
            load_denoiser=False,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
            optimize=self.config.optimize,
            device=self.config.device,
        )
        self.sample_rate = int(getattr(self._model.tts_model, "sample_rate", 48000))
        return self

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            raise ValueError("Text is empty")
        audio = self.model.generate(
            text=text,
            cfg_value=self.config.cfg_value,
            inference_timesteps=self.config.inference_timesteps,
            max_len=self.config.max_len,
            normalize=self.config.normalize,
            denoise=self.config.denoise,
        )
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        return audio, self.sample_rate


def config_from_env(**overrides) -> SynthesisConfig:
    def env_bool(name: str, default: bool) -> bool:
        v = os.getenv(name)
        if v is None:
            return default
        return v.strip().lower() in {"1", "true", "yes", "on"}

    values = dict(
        model_id=os.getenv("MODEL_ID", DEFAULT_MODEL_ID),
        device=os.getenv("DEVICE", "mps"),
        optimize=env_bool("OPTIMIZE_MODEL", False),
        seed=int(os.getenv("INFERENCE_SEED", "42")),
        cfg_value=float(os.getenv("CFG_VALUE", "2.0")),
        inference_timesteps=int(os.getenv("INFERENCE_TIMESTEPS", "16")),
        max_len=int(os.getenv("MAX_LEN", "4096")),
        normalize=env_bool("NORMALIZE", True),
        denoise=env_bool("DENOISE", False),
        cache_dir=os.getenv("HF_HOME") or None,
        local_files_only=env_bool("LOCAL_FILES_ONLY", False),
    )
    values.update({k: v for k, v in overrides.items() if v is not None})
    return SynthesisConfig(**values)


def ensure_parent(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
