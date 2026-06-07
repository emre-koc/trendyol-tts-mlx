# Trendyol TTS on Mac M1 / Apple Silicon

This is a dedicated macOS project for running [`Trendyol/Trendyol-TTS`](https://huggingface.co/Trendyol/Trendyol-TTS) locally on Apple Silicon.

Important status:

- The upstream model is **VoxCPM2**, not a normal Llama/Qwen text model.
- It contains a transformer LM, residual LM, local encoder, DiT/CFM decoder, and AudioVAE vocoder.
- A true pure-MLX port requires reimplementing those VoxCPM2 modules in `mlx.nn` and converting both `model.safetensors` and `audiovae.pth`.
- This project is therefore set up as a Mac/MLX project, but the currently working inference backend uses upstream `voxcpm==2.0.3` on PyTorch MPS, because that is the available runnable Apple Silicon backend today.
- `mlx` is installed and checked so the project is ready for incremental pure-MLX module replacement.

## Quick start

Use Python 3.11 on this Mac:

```bash
cd /Users/fobem/trendyol-tts-mlx
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If Hugging Face rate limits or auth is needed, put your token in a file, not chat:

```bash
mkdir -p ~/.cache/huggingface
printf 'hf_xxx...' > ~/.hf_token
chmod 600 ~/.hf_token
export HF_TOKEN="$(cat ~/.hf_token)"
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

Generate a WAV:

```bash
trendyol-tts \
  --text "Merhaba, Trendyol TTS modelinden Türkçe bir ses örneği dinliyorsunuz." \
  --out out.wav \
  --device mps \
  --cfg 2.0 \
  --steps 16 \
  --max-len 4096
```

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

Generated artifacts, intentionally ignored by git:

- `models/Trendyol-TTS-mlx/audiovae_decoder_mlx.safetensors`
- `models/Trendyol-TTS-mlx/audiovae_decoder_manifest.json`
- `models/Trendyol-TTS-mlx/voxcpm2_layers_mlx.safetensors`
- `models/Trendyol-TTS-mlx/voxcpm2_layers_manifest.json`

## Notes for a future pure MLX port

A full pure-MLX port should replace modules in this order:

1. `AudioVAEV2` decoder only: latent -> waveform, compare against PyTorch.
2. `MiniCPMModel` transformer: token/embed input -> hidden/logits, compare against PyTorch.
3. `VoxCPMLocEnc` encoder.
4. `UnifiedCFM` + `VoxCPMLocDiTV2` decoder/sampler.
5. Full autoregressive pipeline.

Keep the `trendyol-tts` CLI as the public API while swapping backend internals.
