from __future__ import annotations

from pathlib import Path

from scripts.profile_mlx_runtime import ProfileEvent, aggregate_events, stage_timer


def test_stage_timer_records_elapsed_seconds() -> None:
    events: list[ProfileEvent] = []

    with stage_timer(events, "load"):
        sum(range(1000))

    assert len(events) == 1
    assert events[0].name == "load"
    assert events[0].seconds >= 0.0


def test_aggregate_events_groups_duplicate_stage_names() -> None:
    events = [
        ProfileEvent(name="ar_loop", seconds=1.0),
        ProfileEvent(name="dit_cfm", seconds=2.0),
        ProfileEvent(name="ar_loop", seconds=3.0),
    ]

    result = aggregate_events(events)

    assert result["total_profiled_seconds"] == 6.0
    assert result["stages"]["ar_loop"]["count"] == 2
    assert result["stages"]["ar_loop"]["total_seconds"] == 4.0
    assert result["stages"]["ar_loop"]["mean_seconds"] == 2.0
    assert result["stages"]["dit_cfm"]["count"] == 1
