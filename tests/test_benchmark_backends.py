from __future__ import annotations

from scripts.benchmark_backends import build_result


def test_build_result_computes_latency_memory_and_throughput_metrics() -> None:
    result = build_result(
        backend="mlx",
        text="Merhaba test.",
        seed=123,
        steps=10,
        max_len=40,
        wall_seconds=4.0,
        load_seconds=1.0,
        synth_seconds=3.0,
        audio_samples=96_000,
        sample_rate=48_000,
        peak_rss_bytes=2_147_483_648,
        output_path="/tmp/out.wav",
    )

    assert result["backend"] == "mlx"
    assert result["latency_seconds"] == 4.0
    assert result["load_seconds"] == 1.0
    assert result["synthesis_seconds"] == 3.0
    assert result["audio_seconds"] == 2.0
    assert result["generated_audio_frames"] == 96_000
    assert result["audio_frames_per_second"] == 24_000.0
    assert result["audio_seconds_per_wall_second"] == 0.5
    assert result["peak_rss_mb"] == 2048.0
    assert result["output_path"] == "/tmp/out.wav"


def test_build_result_handles_zero_latency_without_division_by_zero() -> None:
    result = build_result(
        backend="mps",
        text="x",
        seed=1,
        steps=1,
        max_len=1,
        wall_seconds=0.0,
        load_seconds=0.0,
        synth_seconds=0.0,
        audio_samples=0,
        sample_rate=48_000,
        peak_rss_bytes=0,
        output_path=None,
    )

    assert result["audio_frames_per_second"] == 0.0
    assert result["audio_seconds_per_wall_second"] == 0.0
    assert result["peak_rss_mb"] == 0.0
