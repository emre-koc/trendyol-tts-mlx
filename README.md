# Trendyol TTS on Mac M1 / Apple Silicon

This is a dedicated macOS project for running [`Trendyol/Trendyol-TTS`](https://huggingface.co/Trendyol/Trendyol-TTS) locally on Apple Silicon.

Important status:

- The upstream model is **VoxCPM2**, not a normal Llama/Qwen text model.
- It contains a transformer LM, residual LM, local encoder, DiT/CFM decoder, and AudioVAE vocoder.
- `--backend mlx` is now a strict text-only MLX runtime: it instantiates the local encoder, base LM, residual LM, DiT/CFM sampler, FSQ/projection/stop layers, and AudioVAE decode directly from `config.json`, tokenizer assets, and exported MLX safetensors manifests.
- The MLX runtime does **not** call `VoxCPM.from_pretrained(...)` and does not allocate the upstream PyTorch VoxCPM/VoxCPM2 model object. The upstream `voxcpm==2.0.3` package remains installed for the optional `--backend mps`/`--backend torch` fallback and text-normalization utility.
- Prompt audio/reference audio, streaming, and denoiser/enhancer paths are explicitly unsupported in strict MLX mode until those feature paths are ported to MLX.

## Quick start

Use Python 3.11 on this Mac:

```bash
cd /Users/fobem/trendyol-tts-mlx
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If Hugging Face rate limits or auth is needed, put your token in a project-local file, not chat. The `.huggingface_token` file is gitignored:

```bash
printf 'hf_xxx...' > .huggingface_token
chmod 600 .huggingface_token
export HF_TOKEN="$(cat .huggingface_token)"
```

Check the environment:

```bash
trendyol-tts-check
```

Verified on this Mac M1/MPS:

```text
mlx: ok, default_device=Device(gpu, 0)
torch mps available: True
```

First run downloads about 4.8 GB of model files into the Hugging Face cache.

Generate a WAV with the MLX backend:

```bash
trendyol-tts \
  --text "Merhaba, Trendyol TTS modelinden Türkçe bir ses örneği dinliyorsunuz." \
  --out out.wav \
  --device mps \
  --backend mlx \
  --cfg 2.0 \
  --steps 16 \
  --max-len 4096
```

`--backend mlx` uses native MLX for the VoxCPM2 local encoder, base LM, residual LM, DiT/CFM latent sampler, FSQ/projection/stop layers, and AudioVAE/vocoder decode. It loads tokenizer/config files plus MLX safetensors manifests directly and avoids the upstream PyTorch VoxCPM model object. Use `--backend mps` or `--backend torch` for the original upstream PyTorch fallback.

Launch Gradio:

```bash
trendyol-tts-app --host 0.0.0.0 --port 7860 --device mps
```

## Docker command mapping

Original CUDA Docker:

```bash
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all \
 -e HF_TOKEN="YOUR_VALUE_HERE" \
 -e MODEL_ID="AlicanKiraz0/Trendyol-TTS" \
 -e REQUIRE_CUDA="1" \
 -e OPTIMIZE_MODEL="0" \
 -e INFERENCE_SEED="42" \
 registry.hf.space/alicankiraz0-trendyol-tts-space:latest python app.py
```

Mac native equivalent:

```bash
cd /Users/fobem/trendyol-tts-mlx
source .venv/bin/activate
MODEL_ID="Trendyol/Trendyol-TTS" \
INFERENCE_SEED="42" \
DEVICE="mps" \
BACKEND="mlx" \
OPTIMIZE_MODEL="0" \
trendyol-tts-app --host 0.0.0.0 --port 7860
```

Do **not** set `REQUIRE_CUDA=1` on Mac.

## Pure MLX port progress

The first pure-MLX artifact is implemented: AudioVAE decoder weight export, plus an MLX implementation of the full decoder/vocoder graph from latent audio tokens to waveform.

```bash
cd /Users/fobem/trendyol-tts-mlx
source .venv/bin/activate
python scripts/export_audiovae_mlx.py --local-files-only
python scripts/check_mlx_first_conv_parity.py
python -m unittest tests.test_mlx_audiovae_decoder -v
```

Verified result for the first real decoder `CausalConv1d` layer:

```text
max_abs_error: 2.384185791015625e-07
mean_abs_error: 8.980957311166549e-09
```

Verified MLX AudioVAE decoder milestones:

- `AudioVAEDecoderStem` (`decoder.model.0` + `decoder.model.1`) matches upstream PyTorch.
- `AudioVAEDecoderBlock` matches individual upsampling blocks such as `decoder.model.2` and `decoder.model.3`.
- `AudioVAEDecoder` matches the full upstream decoder model sequence (`decoder.model.0` through `decoder.model.10`, including final `Tanh`) for deterministic latent inputs.

The next upstream VoxCPM2 layer slice is also started:

- `scripts/export_voxcpm2_layers_mlx.py` exports non-transformer projection weights from `model.safetensors`.
- `ScalarQuantizationLayer` (`fsq_layer`) is implemented in MLX and parity-tested against the upstream PyTorch eval path.
- `Linear.from_manifest(...)` loads the standalone projection layers (`enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj`, `fusion_concat_proj`) and parity-tests them against upstream PyTorch `nn.Linear`.
- `StopPredictor` implements and parity-tests the stop path: `stop_head(SiLU(stop_proj(lm_hidden)))`.
- `LocEncInputProjection` implements and parity-tests the first VoxCPMLocEnc slice: `feat_encoder.in_proj`, special-token prepend, and `b t p c -> (b t) p c` reshape.
- `MiniCPMRMSNorm` implements and parity-tests the first local-encoder MiniCPM normalization slices (`layers.0.input_layernorm`, `layers.0.post_attention_layernorm`, and final `encoder.norm`).
- `MiniCPMMLP` implements and parity-tests the first local-encoder feed-forward block: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`.
- `MiniCPMAttentionProjections` implements and parity-tests the first local-encoder attention projection/reshape slice (`q_proj`, `k_proj`, `v_proj`, `o_proj`) before RoPE/SDPA.
- `MiniCPMLongRoPE` and `apply_rotary_pos_emb` implement and parity-test the local-encoder LongRoPE cache/rotary application.
- `MiniCPMAttention` implements and parity-tests the first local-encoder full attention path with RoPE + grouped-query SDPA for `is_causal=False`.
- `MiniCPMDecoderLayer` implements and parity-tests the first complete local-encoder transformer layer: input RMSNorm + attention residual + post-attention RMSNorm + MLP residual.
- `MiniCPMEncoder` implements and parity-tests the full 12-layer local-encoder transformer stack plus final norm for `is_causal=False`.
- `VoxCPMLocEnc` now wraps the MLX input projection and full MiniCPM encoder, returning the upstream CLS output reshaped from `[B*T, C]` back to `[B, T, C]`, and is parity-tested against upstream PyTorch.
- `VoxCPMLocEncToLMProjection` now bridges local audio encoder features into LM hidden space with `enc_to_lm_proj(feat_encoder(audio_feats))`, and is parity-tested against upstream PyTorch.
- `MiniCPMTokenEmbedding` implements the base LM text-token embedding lookup with upstream `scale_emb` semantics, and is parity-tested against `base_lm.embed_tokens`.
- The first `base_lm.layers.0` RMSNorm and MLP slices are exported into the manifest and parity-tested against upstream PyTorch using the shared MLX MiniCPM components with LM-specific RMSNorm epsilon metadata.
- `base_lm.layers.0.self_attn` now uses LM-specific manifest metadata (`lm_num_attention_heads`, `lm_num_key_value_heads`, `lm_kv_channels`, RoPE config) and parity-tests the first causal LM self-attention path against upstream PyTorch.
- All 28 `base_lm.layers.*` decoder layers are now exported into the MLX manifest.
- `MiniCPMDecoderLayer.from_manifest(..., scope="base_lm")` builds arbitrary base-LM decoder layers; both layer 0 and layer 27 are parity-tested against upstream PyTorch.
- `MiniCPMBaseLM` stacks the full 28-layer base LM plus final `base_lm.norm`, caches manifest/tensor loading to avoid repeated safetensors opens, and parity-tests full-stack embedded-prompt forward output/cache shapes against upstream `MiniCPMModel`.
- `MiniCPMBaseLM.forward_step` now implements autoregressive single-token/patch stepping with an MLX static KV cache; after prefill, the MLX step output is parity-tested against upstream PyTorch `MiniCPMModel.forward_step`.
- `MiniCPMResidualLM` now exports and stacks the full 8-layer `residual_lm` path with upstream-compatible `no_rope=True` behavior, and its full-stack causal forward pass is parity-tested against upstream PyTorch.
- `MiniCPMResidualLM.forward_step` is covered by the same prefill+step parity pattern as the base LM, including the residual-LM no-RoPE path.
- `MLXGenerationPrefill` wires the MLX local encoder, text embedding, base LM, FSQ layer, residual LM, stop predictor, and DiT projection linears through the upstream `_inference` prefill boundary. Its `lm_hidden`, `residual_hidden`, `dit_hidden`, and `stop_logits` are parity-tested against the PyTorch path.

Generated artifacts, intentionally ignored by git:

- `models/Trendyol-TTS-mlx/audiovae_decoder_mlx.safetensors`
- `models/Trendyol-TTS-mlx/audiovae_decoder_manifest.json`
- `models/Trendyol-TTS-mlx/voxcpm2_layers_mlx.safetensors`
- `models/Trendyol-TTS-mlx/voxcpm2_layers_manifest.json`

## Strict MLX runtime notes

The strict MLX runtime is text-only by design for now:

1. `TrendyolMLXModel.from_pretrained(...)` resolves the HF snapshot, reads `config.json`, loads tokenizer assets, and constructs MLX modules from `models/Trendyol-TTS-mlx/*_manifest.json`.
2. The public `generate(text, ...)` wrapper now owns text cleanup, optional text normalization, token/mask construction, MLX inference, AudioVAE decode, and NumPy audio output.
3. `prompt_wav_path`, `prompt_text`, `reference_wav_path`, `streaming=True`, and `denoise=True` raise `NotImplementedError` in MLX mode. These are intentionally out of scope for the text-only product path; see `TODO.md` for deferred work.
4. For packaged/non-editable deployments, set `MLX_ARTIFACT_DIR=/path/to/Trendyol-TTS-mlx` or pass `--mlx-artifact-dir /path/to/Trendyol-TTS-mlx` so the large MLX safetensors artifacts can live outside the Python wheel.
5. `scripts/check_strict_mlx_runtime.py` verifies static runtime boundaries: no `torch` import in `mlx_backend.py`, no `VoxCPM.from_pretrained(...)`, a native `TrendyolMLXModel.generate`, and explicit unsupported-path errors.

## Benchmarking

Run backend benchmarks in separate processes to avoid one backend's allocated memory affecting the next backend:

```bash
cd /Users/fobem/trendyol-tts-mlx
source .venv/bin/activate

python scripts/benchmark_backends.py \
  --backend mlx \
  --local-files-only \
  --text "Merhaba, bu MLX ve MPS performans karşılaştırmasıdır." \
  --seed 123 --steps 10 --max-len 40 \
  --output-dir /tmp/trendyol_tts_bench \
  --csv /tmp/trendyol_tts_bench/mlx_results.csv \
  --json /tmp/trendyol_tts_bench/mlx_results.json

python scripts/benchmark_backends.py \
  --backend mps \
  --local-files-only \
  --text "Merhaba, bu MLX ve MPS performans karşılaştırmasıdır." \
  --seed 123 --steps 10 --max-len 40 \
  --output-dir /tmp/trendyol_tts_bench \
  --csv /tmp/trendyol_tts_bench/mps_results.csv \
  --json /tmp/trendyol_tts_bench/mps_results.json
```

Verified M1 benchmark, same text/seed/steps/max-len, each backend in a fresh process:

- MLX: startup/load 3.460s, synthesis 19.919s, total latency 23.379s, peak RSS 4724 MB, generated 184320 frames / 3.84s audio, 7884 frames/s, audio realtime factor 0.164x.
- MPS/PyTorch: startup/load 87.597s, synthesis 41.720s, total latency 129.318s, peak RSS 9335 MB, generated 192000 frames / 4.0s audio, 1485 frames/s, audio realtime factor 0.031x.
- Audio stats: MLX RMS 0.181, peak 0.972, clipping 0; MPS RMS 0.168, peak 0.910, clipping 0.

Observed deltas: strict MLX startup is about 25.3x faster, total latency about 5.5x faster, synthesis about 2.1x faster, and peak RSS about 49% lower than the MPS/PyTorch fallback on this benchmark.

## Quality validation and profiling

Broader MLX quality matrix:

```bash
python scripts/validate_quality_matrix.py \
  --backend mlx \
  --local-files-only \
  --text short_tr="Merhaba, bugün MLX ses kalitesini doğruluyoruz." \
  --text numbers_tr="Saat on iki otuzda iki yüz elli üç sipariş hazır olacak." \
  --seed 123 \
  --steps 16 --steps 24 \
  --max-len 80 \
  --output-dir /tmp/trendyol_tts_quality \
  --json /tmp/trendyol_tts_quality/quality_results.json \
  --csv /tmp/trendyol_tts_quality/quality_results.csv
```

Latest M1 MLX-only validation matrix generated 4 WAV files with zero clipping. Summary: mean RMS 0.174655, mean peak 0.986203, total latency 120.970s across 4 runs.

Runtime profiling:

```bash
python scripts/profile_mlx_runtime.py \
  --local-files-only \
  --text "Merhaba, MLX performans profilini çıkarıyoruz." \
  --seed 123 --steps 16 --max-len 80 \
  --json /tmp/trendyol_tts_profile/profile_after.json \
  --cprofile /tmp/trendyol_tts_profile/profile_after.prof
```

Profiling showed the strict MLX synthesis path improved from 32.75s to 25.27s for the profiled 16-step sample after keeping cached LM `forward_step` results on MLX and evaluating with `mx.eval(...)` instead of roundtripping through NumPy each generated token. Avoiding `mx.eval(...)` caused MLX lazy-graph buildup and degraded latency, so keep explicit evaluation at the cached step seam.

Keep the `trendyol-tts` CLI as the public API while extending strict MLX feature coverage.
