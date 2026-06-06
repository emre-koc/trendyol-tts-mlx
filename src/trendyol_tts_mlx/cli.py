from __future__ import annotations

import argparse

import soundfile as sf

from .backend import TrendyolTTS, config_from_env, ensure_parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate Turkish speech with Trendyol/Trendyol-TTS on Apple Silicon")
    p.add_argument("--text", required=True, help="Turkish text to synthesize")
    p.add_argument("--out", default="out.wav", help="Output wav path")
    p.add_argument("--model-id", default=None, help="HF model id or local model path")
    p.add_argument("--device", default=None, choices=["mps", "cpu", "auto"], help="Runtime device")
    p.add_argument("--cfg", type=float, default=None, help="CFG value; recommended 2.0 or 1.5")
    p.add_argument("--steps", type=int, default=None, help="Inference timesteps; recommended 16")
    p.add_argument("--max-len", type=int, default=None, help="Max autoregressive length")
    p.add_argument("--seed", type=int, default=None, help="Inference seed")
    p.add_argument("--optimize", action="store_true", help="Enable upstream torch.compile warmup; usually off on Mac")
    p.add_argument("--local-files-only", action="store_true", help="Do not download from HF; use cache/local path only")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = config_from_env(
        model_id=args.model_id,
        device=args.device,
        cfg_value=args.cfg,
        inference_timesteps=args.steps,
        max_len=args.max_len,
        seed=args.seed,
        optimize=True if args.optimize else None,
        local_files_only=True if args.local_files_only else None,
    )
    audio, sr = TrendyolTTS(cfg).synthesize(args.text)
    out = ensure_parent(args.out)
    sf.write(out, audio, sr)
    print(f"wrote {out} ({len(audio) / sr:.2f}s @ {sr} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
