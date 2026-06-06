from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr
import soundfile as sf

from .backend import TrendyolTTS, config_from_env


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gradio app for Trendyol TTS on Apple Silicon")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--device", default=None, choices=["mps", "cpu", "auto"])
    p.add_argument("--model-id", default=None)
    p.add_argument("--cfg", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--max-len", type=int, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = config_from_env(
        model_id=args.model_id,
        device=args.device,
        cfg_value=args.cfg,
        inference_timesteps=args.steps,
        max_len=args.max_len,
    )
    runner = TrendyolTTS(cfg)

    def infer(text: str, cfg_value: float, steps: int):
        nonlocal runner
        # Recreate only if UI changed generation params.
        effective = config_from_env(
            model_id=cfg.model_id,
            device=cfg.device,
            optimize=cfg.optimize,
            seed=cfg.seed,
            cfg_value=float(cfg_value),
            inference_timesteps=int(steps),
            max_len=cfg.max_len,
            normalize=cfg.normalize,
            denoise=cfg.denoise,
            cache_dir=cfg.cache_dir,
            local_files_only=cfg.local_files_only,
        )
        if effective != runner.config:
            runner = TrendyolTTS(effective)
        audio, sr = runner.synthesize(text)
        path = Path(tempfile.mkdtemp(prefix="trendyol_tts_")) / "out.wav"
        sf.write(path, audio, sr)
        return str(path)

    demo = gr.Interface(
        fn=infer,
        inputs=[
            gr.Textbox(label="Text", value="Merhaba, Trendyol TTS modelinden Türkçe bir ses örneği dinliyorsunuz."),
            gr.Slider(label="CFG", minimum=1.0, maximum=3.0, step=0.1, value=cfg.cfg_value),
            gr.Slider(label="Steps", minimum=4, maximum=32, step=1, value=cfg.inference_timesteps),
        ],
        outputs=gr.Audio(label="Generated audio", type="filepath"),
        title="Trendyol TTS - Mac Apple Silicon",
        description="Runs Trendyol/Trendyol-TTS on Apple Silicon using the VoxCPM MPS backend. MLX is installed for incremental pure-MLX porting.",
    )
    demo.launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
