from __future__ import annotations

import argparse
import cProfile
import contextlib
import json
import pstats
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class ProfileEvent:
    name: str
    seconds: float


@contextlib.contextmanager
def stage_timer(events: list[ProfileEvent], name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        events.append(ProfileEvent(name=name, seconds=round(time.perf_counter() - start, 6)))


def aggregate_events(events: list[ProfileEvent]) -> dict[str, object]:
    grouped: dict[str, list[float]] = {}
    for event in events:
        grouped.setdefault(event.name, []).append(float(event.seconds))
    stages = {
        name: {
            "count": len(values),
            "total_seconds": round(float(sum(values)), 6),
            "mean_seconds": round(float(sum(values) / len(values)), 6),
            "max_seconds": round(float(max(values)), 6),
        }
        for name, values in sorted(grouped.items())
    }
    return {
        "total_profiled_seconds": round(float(sum(event.seconds for event in events)), 6),
        "events": [asdict(event) for event in events],
        "stages": stages,
    }


def run_profile(
    *,
    text: str,
    seed: int,
    steps: int,
    max_len: int,
    local_files_only: bool,
    cprofile_path: Path | None,
    internal_profile_json: Path | None = None,
    internal_profile_csv: Path | None = None,
) -> dict[str, object]:
    from trendyol_tts_mlx.backend import SynthesisConfig, TrendyolTTS
    from trendyol_tts_mlx.mlx_backend import aggregate_generation_profile, write_generation_profile_csv

    events: list[ProfileEvent] = []
    profiler = cProfile.Profile()
    profiled_model = None

    def workload() -> tuple[np.ndarray, int]:
        nonlocal profiled_model
        with stage_timer(events, "load_mlx_model"):
            runner = TrendyolTTS(
                SynthesisConfig(
                    backend="mlx",
                    device="mps",
                    seed=seed,
                    inference_timesteps=steps,
                    max_len=max_len,
                    local_files_only=local_files_only,
                    normalize=False,
                    denoise=False,
                )
            ).load()
            if internal_profile_json is not None or internal_profile_csv is not None:
                profiled_model = runner.model
                if profiled_model is None:
                    raise RuntimeError("MLX model did not load")
                profiled_model.enable_generation_profile(True)
        with stage_timer(events, "synthesize_text"):
            return runner.synthesize(text)

    if cprofile_path is not None:
        profiler.enable()
    try:
        with stage_timer(events, "total_wall"):
            audio, sample_rate = workload()
    finally:
        if cprofile_path is not None:
            profiler.disable()
            cprofile_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path = cprofile_path.with_suffix(".txt")
            profiler.dump_stats(str(cprofile_path))
            with stats_path.open("w", encoding="utf-8") as f:
                stats = pstats.Stats(profiler, stream=f).sort_stats("cumtime")
                stats.print_stats(80)

    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    internal_profile = None
    if internal_profile_json is not None or internal_profile_csv is not None:
        if profiled_model is None:
            raise RuntimeError("Internal profile requested but no MLX model was profiled")
        internal_events = list(profiled_model.generation_profile_events)
        internal_profile = aggregate_generation_profile(internal_events)
        if internal_profile_json is not None:
            internal_profile_json.parent.mkdir(parents=True, exist_ok=True)
            internal_profile_json.write_text(json.dumps(internal_profile, ensure_ascii=False, indent=2), encoding="utf-8")
        if internal_profile_csv is not None:
            write_generation_profile_csv(internal_events, internal_profile_csv)
    result = aggregate_events(events)
    result.update(
        {
            "seed": int(seed),
            "steps": int(steps),
            "max_len": int(max_len),
            "audio_samples": int(samples.size),
            "sample_rate": int(sample_rate),
            "audio_seconds": round(float(samples.size / sample_rate), 6) if sample_rate else 0.0,
            "rms": round(float(np.sqrt(np.mean(np.square(samples)))), 6) if samples.size else 0.0,
            "peak": round(float(np.max(np.abs(samples))), 6) if samples.size else 0.0,
            "cprofile_path": str(cprofile_path) if cprofile_path is not None else None,
            "cprofile_text_path": str(cprofile_path.with_suffix(".txt")) if cprofile_path is not None else None,
            "internal_profile_json_path": str(internal_profile_json) if internal_profile_json is not None else None,
            "internal_profile_csv_path": str(internal_profile_csv) if internal_profile_csv is not None else None,
            "internal_profile": internal_profile,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile strict MLX Trendyol TTS runtime")
    p.add_argument("--text", default="Merhaba, MLX performans profilini çıkarıyoruz.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--max-len", type=int, default=80)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--json", type=Path, default=Path("/tmp/trendyol_tts_profile/profile.json"))
    p.add_argument("--cprofile", type=Path, default=Path("/tmp/trendyol_tts_profile/profile.prof"))
    p.add_argument("--internal-profile-json", type=Path, default=None)
    p.add_argument("--internal-profile-csv", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = run_profile(
        text=args.text,
        seed=args.seed,
        steps=args.steps,
        max_len=args.max_len,
        local_files_only=args.local_files_only,
        cprofile_path=args.cprofile,
        internal_profile_json=args.internal_profile_json,
        internal_profile_csv=args.internal_profile_csv,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
