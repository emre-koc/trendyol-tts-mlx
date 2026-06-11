from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np

from .mlx_audiovae import AudioVAEDecoder
from .mlx_dit import MLXUnifiedCFM
from .mlx_lm import MiniCPMBaseLM, MiniCPMResidualLM, MiniCPMTokenEmbedding
from .mlx_locenc import VoxCPMLocEncToLMProjection
from .mlx_voxcpm2_layers import Linear, ScalarQuantizationLayer, StopPredictor


@dataclass(frozen=True)
class GenerationProfileEvent:
    stage: str
    seconds: float
    iteration: int | None = None


@contextlib.contextmanager
def generation_stage_timer(
    events: list[GenerationProfileEvent],
    stage: str,
    iteration: int | None = None,
) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        events.append(
            GenerationProfileEvent(
                stage=stage,
                seconds=round(float(time.perf_counter() - start), 6),
                iteration=iteration,
            )
        )


def _aggregate_generation_stage_stats(events: list[GenerationProfileEvent]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = {}
    for event in events:
        grouped.setdefault(event.stage, []).append(float(event.seconds))
    return {
        stage: {
            "count": len(values),
            "total_seconds": round(float(sum(values)), 6),
            "mean_seconds": round(float(sum(values) / len(values)), 6),
            "max_seconds": round(float(max(values)), 6),
        }
        for stage, values in sorted(grouped.items())
    }


def aggregate_generation_profile(events: list[GenerationProfileEvent]) -> dict[str, object]:
    steady_state_events = [event for event in events if event.iteration is None or event.iteration > 0]
    return {
        "total_internal_profile_seconds": round(float(sum(event.seconds for event in events)), 6),
        "events": [asdict(event) for event in events],
        "stages": _aggregate_generation_stage_stats(events),
        "steady_state_stages": _aggregate_generation_stage_stats(steady_state_events),
    }


def write_generation_profile_csv(events: list[GenerationProfileEvent], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "seconds", "iteration"])
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def _feat_seq_to_latent_mx(pred_feat_seq: list[mx.array], batch: int, patch_size: int) -> mx.array:
    seq = mx.concatenate(pred_feat_seq, axis=1)  # [B, T, P, D]
    latent = mx.transpose(seq, (0, 3, 1, 2))
    return mx.reshape(latent, (batch, seq.shape[3], seq.shape[1] * patch_size))


def _generated_feat_seq_to_np(pred_feat_seq: list[mx.array], context_len: int) -> np.ndarray:
    """Materialize generated feature patches once after the AR loop.

    The strict MLX generation loop used to grow a second `generated_feat` tensor
    with `mx.concatenate` on every token. That adds O(n²)-style copying pressure
    to the autoregressive loop. Keep the per-step list for the latent decoder and
    only concatenate the post-context slice when the badcase-length heuristic needs
    a NumPy shape.
    """
    generated_seq = pred_feat_seq[int(context_len) :]
    if not generated_seq:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(np.array(mx.concatenate(generated_seq, axis=1).squeeze(0)), dtype=np.float32)


class MLXPureInference:
    """Pure-MLX replacement for VoxCPM2Model._inference.

    The upstream wrapper still prepares text/audio tensors and decodes the final
    tensor contract, but the generation loop itself (local encoder, base LM,
    residual LM, DiT/CFM, FSQ, projections, and stop predictor) runs in MLX.
    """

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.profile_enabled = False
        self.profile_events: list[GenerationProfileEvent] = []
        self.feat_encoder = VoxCPMLocEncToLMProjection.from_manifest(self.manifest_path)
        self.token_embedding = MiniCPMTokenEmbedding.from_manifest(self.manifest_path)
        self.base_lm = MiniCPMBaseLM.from_manifest(self.manifest_path)
        self.fsq_layer = ScalarQuantizationLayer.from_manifest(self.manifest_path)
        self.residual_lm = MiniCPMResidualLM.from_manifest(self.manifest_path)
        self.fusion_concat_proj = Linear.from_manifest(self.manifest_path, "fusion_concat_proj")
        self.lm_to_dit_proj = Linear.from_manifest(self.manifest_path, "lm_to_dit_proj")
        self.res_to_dit_proj = Linear.from_manifest(self.manifest_path, "res_to_dit_proj")
        self.stop_predictor = StopPredictor.from_manifest(self.manifest_path)
        self.feat_decoder = MLXUnifiedCFM.from_manifest(self.manifest_path)

    def enable_profile(self, enabled: bool = True, *, clear: bool = True) -> None:
        self.profile_enabled = bool(enabled)
        if clear:
            self.profile_events.clear()
        self.feat_decoder.set_profile_context(self._profile if self.profile_enabled else None)

    def profile_summary(self) -> dict[str, object]:
        return aggregate_generation_profile(self.profile_events)

    def _profile(self, stage: str, iteration: int | None = None):
        if not self.profile_enabled:
            return contextlib.nullcontext()
        return generation_stage_timer(self.profile_events, stage, iteration=iteration)

    def infer_arrays(
        self,
        *,
        text_tokens: np.ndarray,
        text_mask: np.ndarray,
        feat: np.ndarray,
        feat_mask: np.ndarray,
        patch_size: int,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if streaming:
            raise NotImplementedError("pure MLX backend currently supports non-streaming generation only")

        feat_np = np.asarray(feat, dtype=np.float32)
        text_np = np.asarray(text_tokens, dtype=np.int64)
        text_mask_np = np.asarray(text_mask, dtype=np.float32)
        feat_mask_np = np.asarray(feat_mask, dtype=np.float32)
        bsz = int(feat_np.shape[0])
        patch_size = int(patch_size)

        text_mask_mx = mx.array(text_mask_np, dtype=mx.float32)
        feat_mask_mx = mx.array(feat_mask_np, dtype=mx.float32)
        with self._profile("feat_encoder_prefill"):
            feat_embed = self.feat_encoder._call_mx(feat_np)
            if self.profile_enabled:
                mx.eval(feat_embed)
        with self._profile("token_embedding"):
            text_embed = self.token_embedding._call_mx(text_np)
            if self.profile_enabled:
                mx.eval(text_embed)
        with self._profile("combine_embeddings"):
            combined_embed = text_mask_mx[..., None] * text_embed + feat_mask_mx[..., None] * feat_embed
            if self.profile_enabled:
                mx.eval(combined_embed)

        prefix_feat_cond = mx.array(feat_np[:, -1, :, :], dtype=mx.float32)
        pred_feat_seq: list[mx.array] = []
        has_continuation_audio = bool(np.asarray(feat_mask_np[0, -1]).item() == 1)
        context_len = 0
        if has_continuation_audio:
            audio_indices = np.nonzero(feat_mask_np[0])[0]
            context_len = min(streaming_prefix_len - 1, len(audio_indices))
            last_audio_indices = audio_indices[-context_len:] if context_len > 0 else []
            pred_feat_seq = [mx.array(feat_np[:, idx : idx + 1, :, :], dtype=mx.float32) for idx in last_audio_indices]

        with self._profile("base_lm_prefill"):
            enc_outputs, base_cache = self.base_lm._call_mx(combined_embed, is_causal=True)
            if self.profile_enabled:
                mx.eval(enc_outputs, *[item for cache in base_cache for item in cache])
        with self._profile("base_lm_cache_fill"):
            self.base_lm.setup_cache(batch_size=bsz, max_length=int(max_len) + int(combined_embed.shape[1]) + 4)
            self.base_lm.fill_caches(base_cache)
        with self._profile("fsq_prefill"):
            enc_outputs = self.fsq_layer._call_mx(enc_outputs) * feat_mask_mx[..., None] + enc_outputs * text_mask_mx[..., None]
            if self.profile_enabled:
                mx.eval(enc_outputs)
        lm_hidden = enc_outputs[:, -1, :]

        with self._profile("fusion_prefill"):
            residual_inputs = self.fusion_concat_proj._call_mx(
                mx.concatenate([enc_outputs, feat_mask_mx[..., None] * feat_embed], axis=-1)
            )
            if self.profile_enabled:
                mx.eval(residual_inputs)
        with self._profile("residual_lm_prefill"):
            residual_outputs, residual_cache = self.residual_lm._call_mx(residual_inputs, is_causal=True)
            if self.profile_enabled:
                mx.eval(residual_outputs, *[item for cache in residual_cache for item in cache])
        with self._profile("residual_lm_cache_fill"):
            self.residual_lm.setup_cache(batch_size=bsz, max_length=int(max_len) + int(residual_inputs.shape[1]) + 4)
            self.residual_lm.fill_caches(residual_cache)
        residual_hidden = residual_outputs[:, -1, :]

        for i in range(int(max_len)):
            with self._profile("dit_projection", i):
                dit_hidden = mx.concatenate(
                    [self.lm_to_dit_proj._call_mx(lm_hidden), self.res_to_dit_proj._call_mx(residual_hidden)],
                    axis=-1,
                )
                if self.profile_enabled:
                    mx.eval(dit_hidden)
            with self._profile("cond_transpose", i):
                cond = mx.transpose(prefix_feat_cond, (0, 2, 1))
            with self._profile("dit_cfm_sample", i):
                pred_feat = self.feat_decoder._call_mx(
                    mu=dit_hidden,
                    n_timesteps=int(inference_timesteps),
                    patch_size=patch_size,
                    cond=cond,
                    cfg_value=float(cfg_value),
                    profile_iteration=i,
                )
                if self.profile_enabled:
                    mx.eval(pred_feat)
            with self._profile("pred_feat_transpose", i):
                pred_feat = mx.transpose(pred_feat, (0, 2, 1))  # [B, P, D]
            with self._profile("feat_encoder_step", i):
                curr_embed = self.feat_encoder._call_mx(pred_feat[:, None, :, :])
                if self.profile_enabled:
                    mx.eval(curr_embed)
            pred_feat_seq.append(pred_feat[:, None, :, :])
            prefix_feat_cond = pred_feat

            with self._profile("stop_predictor", i):
                stop_logits = self.stop_predictor._call_mx(lm_hidden)
                stop_flag = int(np.argmax(np.asarray(np.array(stop_logits[0]))))
            if i > min_len and stop_flag == 1:
                break

            with self._profile("base_lm_step", i):
                lm_hidden = self.fsq_layer._call_mx(
                    self.base_lm.forward_step_mx(
                        curr_embed[:, 0, :],
                        position_id=self.base_lm.cache_step(),
                        evaluate=False,
                    )
                )
                if self.profile_enabled:
                    mx.eval(lm_hidden)
            with self._profile("fusion_step", i):
                curr_residual_input = self.fusion_concat_proj._call_mx(
                    mx.concatenate([lm_hidden, curr_embed[:, 0, :]], axis=-1)
                )
                if self.profile_enabled:
                    mx.eval(curr_residual_input)
            with self._profile("residual_lm_step", i):
                residual_hidden = self.residual_lm.forward_step_mx(
                    curr_residual_input, position_id=self.residual_lm.cache_step()
                )

        latent_pred = _feat_seq_to_latent_mx(pred_feat_seq, bsz, patch_size)
        latent_np = np.asarray(np.array(latent_pred), dtype=np.float32)
        generated_np = _generated_feat_seq_to_np(pred_feat_seq, context_len)
        return latent_np, generated_np, context_len


class _CharTokenizerWrapper:
    def __init__(self, base_tokenizer):
        self.tokenizer = base_tokenizer
        vocab = getattr(base_tokenizer, "vocab", None) or base_tokenizer.get_vocab()
        self.multichar_tokens = {
            token
            for token in vocab.keys()
            if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
        }

    def tokenize(self, text: str, **kwargs) -> list[str]:
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text)}")
        tokens = self.tokenizer.tokenize(text, **kwargs)
        processed: list[str] = []
        for token in tokens:
            clean_token = token.replace("▁", "")
            if clean_token in self.multichar_tokens:
                processed.extend(list(clean_token))
            else:
                processed.append(token)
        return processed

    def __call__(self, text: str, **kwargs) -> list[int]:
        tokens = self.tokenize(text, **kwargs)
        return self.tokenizer.convert_tokens_to_ids(tokens)


class TrendyolMLXModel:
    """Strict full-MLX text-only VoxCPM2 runtime.

    This runtime owns the public generation wrapper and instantiates MLX modules
    directly from tokenizer/config assets plus exported safetensors manifests. It
    does not allocate an upstream PyTorch VoxCPM/VoxCPM2 model object.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        artifact_dir: str | Path,
        text_tokenizer,
        inference: MLXPureInference,
        audio_vae: AudioVAEDecoder,
        sample_rate: int = 48000,
        patch_size: int = 4,
        latent_dim: int = 64,
        decode_chunk_size: int | None = None,
    ):
        self.model_path = Path(model_path)
        self.artifact_dir = Path(artifact_dir)
        self.text_tokenizer = text_tokenizer
        self.inference = inference
        self.audio_vae = audio_vae
        self.sample_rate = int(sample_rate)
        self.patch_size = int(patch_size)
        self.latent_dim = int(latent_dim)
        self._decode_chunk_size = int(decode_chunk_size or patch_size)
        self.audio_start_token = 101
        self.audio_end_token = 102
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104
        self.text_normalizer = None

    def enable_generation_profile(self, enabled: bool = True, *, clear: bool = True) -> None:
        self.inference.enable_profile(enabled, clear=clear)

    @property
    def generation_profile_events(self) -> list[GenerationProfileEvent]:
        return self.inference.profile_events

    def generation_profile_summary(self) -> dict[str, object]:
        return self.inference.profile_summary()

    def _profile(self, stage: str):
        if not isinstance(self.inference, MLXPureInference):
            return contextlib.nullcontext()
        return self.inference._profile(stage)

    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        artifact_dir: str | Path | None = None,
    ) -> "TrendyolMLXModel":
        if os.path.isdir(hf_model_id):
            model_path = Path(hf_model_id)
        else:
            from huggingface_hub import snapshot_download

            model_path = Path(
                snapshot_download(repo_id=hf_model_id, cache_dir=cache_dir, local_files_only=local_files_only)
            )
        artifact_path = Path(artifact_dir) if artifact_dir is not None else Path(__file__).resolve().parents[2] / "models/Trendyol-TTS-mlx"
        voxcpm2_manifest = artifact_path / "voxcpm2_layers_manifest.json"
        audio_manifest = artifact_path / "audiovae_decoder_manifest.json"
        if not voxcpm2_manifest.exists():
            raise FileNotFoundError(f"MLX VoxCPM2 manifest not found: {voxcpm2_manifest}")
        if not audio_manifest.exists():
            raise FileNotFoundError(f"MLX AudioVAE manifest not found: {audio_manifest}")

        with open(model_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        audio_config = config.get("audio_vae_config", {})

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        text_tokenizer = _CharTokenizerWrapper(tokenizer)
        return cls(
            model_path=model_path,
            artifact_dir=artifact_path,
            text_tokenizer=text_tokenizer,
            inference=MLXPureInference(voxcpm2_manifest),
            audio_vae=AudioVAEDecoder.from_manifest(audio_manifest),
            sample_rate=int(audio_config.get("out_sample_rate", 48000)),
            patch_size=int(config.get("patch_size", 4)),
            latent_dim=int(audio_config.get("latent_dim", config.get("feat_dim", 64))),
            decode_chunk_size=int(audio_config.get("decode_chunk_size", config.get("patch_size", 4))),
        )

    def generate(
        self,
        text: str,
        prompt_wav_path: str | None = None,
        prompt_text: str | None = None,
        reference_wav_path: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        normalize: bool = False,
        denoise: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
    ) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")
        if prompt_wav_path is not None or prompt_text is not None or reference_wav_path is not None:
            raise NotImplementedError("prompt audio/reference audio is not supported in strict MLX mode until AudioVAE encode is ported")
        if denoise:
            raise NotImplementedError("denoiser/enhancer is not supported in strict MLX mode")
        if streaming:
            raise NotImplementedError("streaming is not supported in strict MLX mode yet")

        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        if normalize:
            from voxcpm.utils.text_normalize import TextNormalizer

            if self.text_normalizer is None:
                self.text_normalizer = TextNormalizer()
            text = self.text_normalizer.normalize(text)

        text_ids = list(self.text_tokenizer(text))
        target_text_length = len(text_ids)
        text_ids.append(self.audio_start_token)
        text_tokens = np.asarray([text_ids], dtype=np.int64)
        seq_len = int(text_tokens.shape[1])
        feat = np.zeros((1, seq_len, self.patch_size, self.latent_dim), dtype=np.float32)
        text_mask = np.ones((1, seq_len), dtype=np.float32)
        feat_mask = np.zeros((1, seq_len), dtype=np.float32)

        attempts = int(retry_badcase_max_times) if retry_badcase else 1
        attempts = max(1, attempts)
        last_result: tuple[np.ndarray, np.ndarray, int] | None = None
        effective_max_len = min(int(target_text_length * retry_badcase_ratio_threshold + 10), int(max_len))
        for _ in range(attempts):
            with self._profile("infer_arrays_total"):
                last_result = self.inference.infer_arrays(
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                    feat=feat,
                    feat_mask=feat_mask,
                    patch_size=self.patch_size,
                    min_len=min_len,
                    max_len=effective_max_len,
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    streaming=False,
                )
            latent_pred, pred_audio_feat, context_len = last_result
            if not retry_badcase or pred_audio_feat.shape[0] < target_text_length * retry_badcase_ratio_threshold:
                break
        if last_result is None:
            raise RuntimeError("MLX generation did not produce a result")
        latent_pred, _pred_audio_feat, context_len = last_result
        with self._profile("audio_vae_decode"):
            decode_audio = self.audio_vae.decode(latent_pred, sample_rate=self.sample_rate)
        decode_patch_len = self.patch_size * self._decode_chunk_size
        if context_len > 0:
            decode_audio = decode_audio[..., decode_patch_len * context_len :]
        return np.asarray(decode_audio, dtype=np.float32).reshape(-1)


@dataclass(frozen=True)
class MLXGenerationPrefillState:
    """Pure-MLX state at the VoxCPM2 inference loop's DiT boundary."""

    lm_hidden: np.ndarray
    residual_hidden: np.ndarray
    dit_hidden: np.ndarray
    stop_logits: np.ndarray


class MLXGenerationPrefill:
    """Pure-MLX VoxCPM2 inference prefill through base/residual LMs.

    This intentionally stops at the DiT/CFM boundary. It mirrors the upstream
    `_inference` prefill path up to the values consumed by the diffusion
    decoder: `lm_hidden`, `residual_hidden`, concatenated `dit_hidden`, and the
    stop-token logits for the current language-model state.
    """

    def __init__(
        self,
        feat_encoder: VoxCPMLocEncToLMProjection,
        token_embedding: MiniCPMTokenEmbedding,
        base_lm: MiniCPMBaseLM,
        fsq_layer: ScalarQuantizationLayer,
        residual_lm: MiniCPMResidualLM,
        fusion_concat_proj: Linear,
        lm_to_dit_proj: Linear,
        res_to_dit_proj: Linear,
        stop_predictor: StopPredictor,
    ):
        self.feat_encoder = feat_encoder
        self.token_embedding = token_embedding
        self.base_lm = base_lm
        self.fsq_layer = fsq_layer
        self.residual_lm = residual_lm
        self.fusion_concat_proj = fusion_concat_proj
        self.lm_to_dit_proj = lm_to_dit_proj
        self.res_to_dit_proj = res_to_dit_proj
        self.stop_predictor = stop_predictor

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "MLXGenerationPrefill":
        return cls(
            feat_encoder=VoxCPMLocEncToLMProjection.from_manifest(manifest_path),
            token_embedding=MiniCPMTokenEmbedding.from_manifest(manifest_path),
            base_lm=MiniCPMBaseLM.from_manifest(manifest_path),
            fsq_layer=ScalarQuantizationLayer.from_manifest(manifest_path),
            residual_lm=MiniCPMResidualLM.from_manifest(manifest_path),
            fusion_concat_proj=Linear.from_manifest(manifest_path, "fusion_concat_proj"),
            lm_to_dit_proj=Linear.from_manifest(manifest_path, "lm_to_dit_proj"),
            res_to_dit_proj=Linear.from_manifest(manifest_path, "res_to_dit_proj"),
            stop_predictor=StopPredictor.from_manifest(manifest_path),
        )

    def prefill(
        self,
        text_tokens: np.ndarray,
        text_mask: np.ndarray,
        audio_feats: np.ndarray,
        feat_mask: np.ndarray,
        max_length: int,
    ) -> MLXGenerationPrefillState:
        text_mask_mx = mx.array(text_mask, dtype=mx.float32)
        feat_mask_mx = mx.array(feat_mask, dtype=mx.float32)

        feat_embed = self.feat_encoder._call_mx(audio_feats)
        text_embed = self.token_embedding._call_mx(text_tokens)
        combined_embed = text_mask_mx[..., None] * text_embed + feat_mask_mx[..., None] * feat_embed

        enc_outputs, base_cache = self.base_lm._call_mx(combined_embed, is_causal=True)
        self.base_lm.setup_cache(batch_size=int(combined_embed.shape[0]), max_length=max_length)
        self.base_lm.fill_caches(base_cache)

        enc_outputs = self.fsq_layer._call_mx(enc_outputs) * feat_mask_mx[..., None] + enc_outputs * text_mask_mx[..., None]
        lm_hidden = enc_outputs[:, -1, :]

        residual_inputs = self.fusion_concat_proj._call_mx(
            mx.concatenate([enc_outputs, feat_mask_mx[..., None] * feat_embed], axis=-1)
        )
        residual_outputs, residual_cache = self.residual_lm._call_mx(residual_inputs, is_causal=True)
        self.residual_lm.setup_cache(batch_size=int(residual_inputs.shape[0]), max_length=max_length)
        self.residual_lm.fill_caches(residual_cache)
        residual_hidden = residual_outputs[:, -1, :]

        dit_hidden = mx.concatenate(
            [self.lm_to_dit_proj._call_mx(lm_hidden), self.res_to_dit_proj._call_mx(residual_hidden)],
            axis=-1,
        )
        stop_logits = self.stop_predictor._call_mx(lm_hidden)

        return MLXGenerationPrefillState(
            lm_hidden=np.asarray(np.array(lm_hidden), dtype=np.float32),
            residual_hidden=np.asarray(np.array(residual_hidden), dtype=np.float32),
            dit_hidden=np.asarray(np.array(dit_hidden), dtype=np.float32),
            stop_logits=np.asarray(np.array(stop_logits), dtype=np.float32),
        )
