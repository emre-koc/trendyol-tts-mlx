from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.validate_quality_matrix import (
    AudioStats,
    build_quality_record,
    compare_pair,
    summarize_quality_records,
    write_quality_outputs,
)


def test_build_quality_record_computes_audio_stats() -> None:
    audio = np.asarray([0.0, 0.5, -1.0, 1.0], dtype=np.float32)

    record = build_quality_record(
        backend="mlx",
        text_id="short_tr",
        text="Merhaba.",
        seed=123,
        steps=16,
        max_len=80,
        audio=audio,
        sample_rate=48_000,
        latency_seconds=2.5,
        load_seconds=0.5,
        synthesis_seconds=2.0,
        output_path="/tmp/sample.wav",
    )

    assert record["backend"] == "mlx"
    assert record["text_id"] == "short_tr"
    assert record["audio_samples"] == 4
    assert record["audio_seconds"] == round(4 / 48_000, 6)
    assert record["rms"] == round(float(np.sqrt(np.mean(np.square(audio)))), 6)
    assert record["peak"] == 1.0
    assert record["clipping_samples"] == 2
    assert record["latency_seconds"] == 2.5
    assert record["output_path"] == "/tmp/sample.wav"


def test_compare_pair_computes_quality_ratios_and_pass_flag() -> None:
    mlx = AudioStats(rms=0.18, peak=0.92, audio_seconds=3.9, clipping_samples=0)
    reference = AudioStats(rms=0.17, peak=0.9, audio_seconds=4.0, clipping_samples=0)

    comparison = compare_pair(
        text_id="case1",
        seed=123,
        steps=16,
        max_len=80,
        mlx=mlx,
        reference=reference,
        rms_ratio_limit=1.2,
        peak_ratio_limit=1.2,
        duration_ratio_limit=1.2,
    )

    assert comparison["text_id"] == "case1"
    assert comparison["rms_ratio"] == round(0.18 / 0.17, 6)
    assert comparison["peak_ratio"] == round(0.92 / 0.9, 6)
    assert comparison["duration_ratio"] == round(4.0 / 3.9, 6)
    assert comparison["passed"] is True


def test_compare_pair_fails_on_clipping_or_large_delta() -> None:
    comparison = compare_pair(
        text_id="case2",
        seed=123,
        steps=16,
        max_len=80,
        mlx=AudioStats(rms=0.5, peak=1.0, audio_seconds=1.0, clipping_samples=1),
        reference=AudioStats(rms=0.1, peak=0.2, audio_seconds=3.0, clipping_samples=0),
        rms_ratio_limit=1.2,
        peak_ratio_limit=1.2,
        duration_ratio_limit=1.2,
    )

    assert comparison["passed"] is False
    assert "clipping" in comparison["reasons"]
    assert "rms_ratio" in comparison["reasons"]
    assert "peak_ratio" in comparison["reasons"]
    assert "duration_ratio" in comparison["reasons"]


def test_summarize_and_write_quality_outputs(tmp_path: Path) -> None:
    records = [
        {"backend": "mlx", "text_id": "a", "seed": 1, "steps": 16, "max_len": 80, "rms": 0.1, "peak": 0.5, "audio_seconds": 1.0, "clipping_samples": 0},
        {"backend": "mps", "text_id": "a", "seed": 1, "steps": 16, "max_len": 80, "rms": 0.11, "peak": 0.55, "audio_seconds": 1.1, "clipping_samples": 0},
    ]

    summary = summarize_quality_records(records)
    assert summary["total_records"] == 2
    assert summary["by_backend"]["mlx"]["count"] == 1
    assert summary["by_backend"]["mps"]["mean_rms"] == 0.11

    json_path = tmp_path / "quality.json"
    csv_path = tmp_path / "quality.csv"
    write_quality_outputs(records, json_path=json_path, csv_path=csv_path)

    assert json.loads(json_path.read_text())[0]["backend"] == "mlx"
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["backend"] == "mlx"
