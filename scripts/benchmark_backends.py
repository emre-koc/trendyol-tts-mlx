from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports KiB.
    if rss < 10_000_000:
        return int(rss) * 1024
    return int(rss)


def build_result(
    *,
    backend: str,
    text: str,
    seed: int,
    steps: int,
    max_len: int,
    wall_seconds: float,
    load_seconds: float,
    synth_seconds: float,
    audio_samples: int,
    sample_rate: int,
    peak_rss_bytes: int,
    output_path: str | None,
) -> dict[str, Any]:
    audio_seconds = _safe_div(float(audio_samples), float(sample_rate))
    return {
        "backend": backend,
        "text": text,
        "seed": int(seed),
        "steps": int(steps),
        "max_len": int(max_len),
        "latency_seconds": round(float(wall_seconds), 6),
        "load_seconds": round(float(load_seconds), 6),
        "synthesis_seconds": round(float(synth_seconds), 6),
        "audio_seconds": round(audio_seconds, 6),
        "generated_audio_frames": int(audio_samples),
        "sample_rate": int(sample_rate),
        "audio_frames_per_second": round(_safe_div(float(audio_samples), float(wall_seconds)), 6),
        "audio_seconds_per_wall_second": round(_safe_div(audio_seconds, float(wall_seconds)), 6),
        "peak_rss_mb": round(float(peak_rss_bytes) / (1024 * 1024), 3),
        "output_path": output_path,
    }


def benchmark_once(
    *,
    backend: str,
    text: str,
    seed: int,
    steps: int,
    max_len: int,
    model_id: str | None,
    local_files_only: bool,
    output_dir: Path | None,
) -> dict[str, Any]:
    from scipy.io import wavfile
    from trendyol_tts_mlx.backend import SynthesisConfig, TrendyolTTS

    start = time.perf_counter()
    load_start = time.perf_counter()
    cfg = SynthesisConfig(
        model_id=model_id or "Trendyol/Trendyol-TTS",
        backend=backend,
        device="mps" if backend in {"mlx", "mps"} else "cpu",
        seed=seed,
        inference_timesteps=steps,
        max_len=max_len,
        local_files_only=local_files_only,
        normalize=False,
        denoise=False,
    )
    runner = TrendyolTTS(cfg).load()
    load_seconds = time.perf_counter() - load_start

    synth_start = time.perf_counter()
    audio, sample_rate = runner.synthesize(text)
    synth_seconds = time.perf_counter() - synth_start
    wall_seconds = time.perf_counter() - start

    output_path: str | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"benchmark_{backend}_seed{seed}_steps{steps}_max{max_len}.wav"
        wavfile.write(output_file, int(sample_rate), np.asarray(audio, dtype=np.float32))
        output_path = str(output_file)

    return build_result(
        backend=backend,
        text=text,
        seed=seed,
        steps=steps,
        max_len=max_len,
        wall_seconds=wall_seconds,
        load_seconds=load_seconds,
        synth_seconds=synth_seconds,
        audio_samples=int(np.asarray(audio).reshape(-1).shape[0]),
        sample_rate=int(sample_rate),
        peak_rss_bytes=_rss_bytes(),
        output_path=output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Trendyol TTS MLX/MPS backends on Apple Silicon")
    parser.add_argument("--backend", action="append", choices=["mlx", "mps", "torch"], default=None)
    parser.add_argument("--text", default="Merhaba, bu MLX ve MPS performans karşılaştırmasıdır.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, action="append", default=None)
    parser.add_argument("--max-len", type=int, action="append", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/trendyol_tts_bench"))
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = args.backend or ["mlx", "mps"]
    steps_values = args.steps or [10]
    max_len_values = args.max_len or [40]
    results: list[dict[str, Any]] = []
    for backend in backends:
        for steps in steps_values:
            for max_len in max_len_values:
                result = benchmark_once(
                    backend=backend,
                    text=args.text,
                    seed=args.seed,
                    steps=steps,
                    max_len=max_len,
                    model_id=args.model_id,
                    local_files_only=args.local_files_only,
                    output_dir=args.output_dir,
                )
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)

    if args.csv is not None and results:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
