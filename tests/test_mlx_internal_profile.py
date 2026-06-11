from __future__ import annotations

import json
from pathlib import Path

from trendyol_tts_mlx.mlx_backend import (
    GenerationProfileEvent,
    aggregate_generation_profile,
    generation_stage_timer,
    write_generation_profile_csv,
)


def test_generation_stage_timer_records_stage_iteration_and_elapsed_seconds() -> None:
    events: list[GenerationProfileEvent] = []

    with generation_stage_timer(events, "base_lm_step", iteration=3):
        sum(range(1000))

    assert len(events) == 1
    assert events[0].stage == "base_lm_step"
    assert events[0].iteration == 3
    assert events[0].seconds >= 0.0


def test_aggregate_generation_profile_groups_duplicate_stages() -> None:
    events = [
        GenerationProfileEvent(stage="base_lm_step", seconds=1.0, iteration=0),
        GenerationProfileEvent(stage="residual_lm_step", seconds=2.0, iteration=0),
        GenerationProfileEvent(stage="base_lm_step", seconds=3.0, iteration=1),
    ]

    result = aggregate_generation_profile(events)

    assert result["total_internal_profile_seconds"] == 6.0
    assert result["stages"]["base_lm_step"]["count"] == 2
    assert result["stages"]["base_lm_step"]["total_seconds"] == 4.0
    assert result["stages"]["base_lm_step"]["mean_seconds"] == 2.0
    assert result["steady_state_stages"]["base_lm_step"]["count"] == 1
    assert result["steady_state_stages"]["base_lm_step"]["total_seconds"] == 3.0
    assert result["events"][0] == {"stage": "base_lm_step", "seconds": 1.0, "iteration": 0}


def test_write_generation_profile_csv_writes_machine_readable_rows(tmp_path: Path) -> None:
    events = [GenerationProfileEvent(stage="base_lm_step", seconds=1.25, iteration=4)]
    csv_path = tmp_path / "internal_profile.csv"

    write_generation_profile_csv(events, csv_path)

    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "stage,seconds,iteration",
        "base_lm_step,1.25,4",
    ]


def test_profile_runtime_exposes_internal_profile_output_arguments() -> None:
    import inspect

    from scripts.profile_mlx_runtime import run_profile

    sig = inspect.signature(run_profile)

    assert "internal_profile_json" in sig.parameters
    assert "internal_profile_csv" in sig.parameters
