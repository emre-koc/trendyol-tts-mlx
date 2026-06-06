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

## Notes for a future pure MLX port

A full pure-MLX port should replace modules in this order:

1. `AudioVAEV2` decoder only: latent -> waveform, compare against PyTorch.
2. `MiniCPMModel` transformer: token/embed input -> hidden/logits, compare against PyTorch.
3. `VoxCPMLocEnc` encoder.
4. `UnifiedCFM` + `VoxCPMLocDiTV2` decoder/sampler.
5. Full autoregressive pipeline.

Keep the `trendyol-tts` CLI as the public API while swapping backend internals.
