from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_TEXTS: tuple[tuple[str, str], ...] = (
    ("short_tr", "Merhaba, bugün MLX ses kalitesini doğruluyoruz."),
    ("numbers_tr", "Saat on iki otuzda iki yüz elli üç sipariş hazır olacak."),
    ("long_tr", "Trendyol ses modeli, daha uzun cümlelerde de doğal, temiz ve anlaşılır konuşma üretmelidir."),
)


@dataclass(frozen=True)
class AudioStats:
    rms: float
    peak: float
    audio_seconds: float
    clipping_samples: int


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def audio_stats(audio: np.ndarray, sample_rate: int) -> AudioStats:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return AudioStats(rms=0.0, peak=0.0, audio_seconds=0.0, clipping_samples=0)
    abs_samples = np.abs(samples)
    return AudioStats(
        rms=round(float(np.sqrt(np.mean(np.square(samples)))), 6),
        peak=round(float(np.max(abs_samples)), 6),
        audio_seconds=round(_safe_div(float(samples.size), float(sample_rate)), 6),
        clipping_samples=int(np.count_nonzero(abs_samples >= 0.999)),
    )


def build_quality_record(
    *,
    backend: str,
    text_id: str,
    text: str,
    seed: int,
    steps: int,
    max_len: int,
    audio: np.ndarray,
    sample_rate: int,
    latency_seconds: float,
    load_seconds: float,
    synthesis_seconds: float,
    output_path: str | None,
) -> dict[str, Any]:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    stats = audio_stats(samples, sample_rate)
    return {
        "backend": backend,
        "text_id": text_id,
        "text": text,
        "seed": int(seed),
        "steps": int(steps),
        "max_len": int(max_len),
        "sample_rate": int(sample_rate),
        "audio_samples": int(samples.size),
        "audio_seconds": stats.audio_seconds,
        "rms": stats.rms,
        "peak": stats.peak,
        "clipping_samples": stats.clipping_samples,
        "latency_seconds": round(float(latency_seconds), 6),
        "load_seconds": round(float(load_seconds), 6),
        "synthesis_seconds": round(float(synthesis_seconds), 6),
        "output_path": output_path,
    }


def _ratio(a: float, b: float) -> float:
    small = min(abs(float(a)), abs(float(b)))
    large = max(abs(float(a)), abs(float(b)))
    return round(_safe_div(large, small), 6) if small else (0.0 if large == 0 else float("inf"))


def compare_pair(
    *,
    text_id: str,
    seed: int,
    steps: int,
    max_len: int,
    mlx: AudioStats,
    reference: AudioStats,
    rms_ratio_limit: float,
    peak_ratio_limit: float,
    duration_ratio_limit: float,
) -> dict[str, Any]:
    rms_ratio = _ratio(mlx.rms, reference.rms)
    peak_ratio = _ratio(mlx.peak, reference.peak)
    duration_ratio = _ratio(mlx.audio_seconds, reference.audio_seconds)
    reasons: list[str] = []
    if mlx.clipping_samples:
        reasons.append("clipping")
    if rms_ratio > rms_ratio_limit:
        reasons.append("rms_ratio")
    if peak_ratio > peak_ratio_limit:
        reasons.append("peak_ratio")
    if duration_ratio > duration_ratio_limit:
        reasons.append("duration_ratio")
    return {
        "text_id": text_id,
        "seed": int(seed),
        "steps": int(steps),
        "max_len": int(max_len),
        "rms_ratio": rms_ratio,
        "peak_ratio": peak_ratio,
        "duration_ratio": duration_ratio,
        "mlx_clipping_samples": int(mlx.clipping_samples),
        "reference_clipping_samples": int(reference.clipping_samples),
        "passed": not reasons,
        "reasons": reasons,
    }


def summarize_quality_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records_list = list(records)
    by_backend: dict[str, dict[str, Any]] = {}
    for backend in sorted({str(r["backend"]) for r in records_list}):
        subset = [r for r in records_list if r["backend"] == backend]
        by_backend[backend] = {
            "count": len(subset),
            "mean_rms": round(float(np.mean([float(r["rms"]) for r in subset])), 6) if subset else 0.0,
            "mean_peak": round(float(np.mean([float(r["peak"]) for r in subset])), 6) if subset else 0.0,
            "total_latency_seconds": round(float(np.sum([float(r.get("latency_seconds", 0.0)) for r in subset])), 6),
            "total_clipping_samples": int(np.sum([int(r.get("clipping_samples", 0)) for r in subset])),
        }
    return {"total_records": len(records_list), "by_backend": by_backend}


def write_quality_outputs(records: list[dict[str, Any]], *, json_path: Path | None, csv_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if csv_path is not None and records:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def _parse_text_arg(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return list(DEFAULT_TEXTS)
    texts: list[tuple[str, str]] = []
    for idx, value in enumerate(values, start=1):
        if "=" in value:
            text_id, text = value.split("=", 1)
        else:
            text_id, text = f"text_{idx}", value
        texts.append((text_id.strip() or f"text_{idx}", text.strip()))
    return texts


def synthesize_record(*, backend: str, text_id: str, text: str, seed: int, steps: int, max_len: int, local_files_only: bool, output_dir: Path | None) -> dict[str, Any]:
    import soundfile as sf
    from trendyol_tts_mlx.backend import SynthesisConfig, TrendyolTTS

    start = time.perf_counter()
    load_start = time.perf_counter()
    runner = TrendyolTTS(
        SynthesisConfig(
            backend=backend,
            device="mps" if backend in {"mlx", "mps"} else "cpu",
            seed=seed,
            inference_timesteps=steps,
            max_len=max_len,
            local_files_only=local_files_only,
            normalize=False,
            denoise=False,
        )
    ).load()
    load_seconds = time.perf_counter() - load_start
    synth_start = time.perf_counter()
    audio, sample_rate = runner.synthesize(text)
    synth_seconds = time.perf_counter() - synth_start
    latency_seconds = time.perf_counter() - start

    output_path: str | None = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"quality_{backend}_{text_id}_seed{seed}_steps{steps}_max{max_len}.wav"
        sf.write(out, np.asarray(audio, dtype=np.float32), int(sample_rate))
        output_path = str(out)

    return build_quality_record(
        backend=backend,
        text_id=text_id,
        text=text,
        seed=seed,
        steps=steps,
        max_len=max_len,
        audio=audio,
        sample_rate=sample_rate,
        latency_seconds=latency_seconds,
        load_seconds=load_seconds,
        synthesis_seconds=synth_seconds,
        output_path=output_path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a broader Trendyol TTS MLX quality validation matrix")
    p.add_argument("--backend", action="append", choices=["mlx", "mps", "torch"], default=None)
    p.add_argument("--text", action="append", help="Validation text; optionally id=text")
    p.add_argument("--seed", type=int, action="append", default=None)
    p.add_argument("--steps", type=int, action="append", default=None)
    p.add_argument("--max-len", type=int, action="append", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/trendyol_tts_quality"))
    p.add_argument("--json", type=Path, default=Path("/tmp/trendyol_tts_quality/quality_results.json"))
    p.add_argument("--csv", type=Path, default=Path("/tmp/trendyol_tts_quality/quality_results.csv"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    backends = args.backend or ["mlx"]
    texts = _parse_text_arg(args.text)
    seeds = args.seed or [123]
    steps_values = args.steps or [16, 24]
    max_len_values = args.max_len or [80]

    records: list[dict[str, Any]] = []
    for backend in backends:
        for text_id, text in texts:
            for seed in seeds:
                for steps in steps_values:
                    for max_len in max_len_values:
                        record = synthesize_record(
                            backend=backend,
                            text_id=text_id,
                            text=text,
                            seed=seed,
                            steps=steps,
                            max_len=max_len,
                            local_files_only=args.local_files_only,
                            output_dir=args.output_dir,
                        )
                        records.append(record)
                        print(json.dumps(record, ensure_ascii=False), flush=True)

    write_quality_outputs(records, json_path=args.json, csv_path=args.csv)
    print(json.dumps({"summary": summarize_quality_records(records)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
