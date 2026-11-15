"""Foreground research supervisor: checkpoints, single owner, quotas, restart."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_news_research_worker import files

from tris_market_event_intelligence.news_research_supervisor import (
    read_supervisor_status,
    run_supervised_news_research,
)


def execute(paths, tmp_path, **options):
    return run_supervised_news_research(
        cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
        status_path=tmp_path / "status.json", lock_path=tmp_path / "worker.lock",
        **options,
    )


def test_checkpoint_resume_without_duplicate_experiments(tmp_path: Path) -> None:
    paths = files(tmp_path)
    original = [x.read_bytes() for x in paths[:3]]
    first = execute(paths, tmp_path, max_iterations=2)
    assert first["status"] == "ITERATION_LIMIT_REACHED"
    assert first["newResearchExperiments"] == 2
    assert not (tmp_path / "worker.lock").exists()
    later = execute(paths, tmp_path, max_iterations=10)
    assert later["status"] == "NO_NEW_EXPERIMENTS"
    assert later["newResearchExperiments"] == 3
    again = execute(paths, tmp_path, max_iterations=10)
    assert again["newResearchExperiments"] == 0
    assert again["status"] == "NO_NEW_EXPERIMENTS"
    state = read_supervisor_status(tmp_path / "status.json")
    assert state["statusFresh"] is True
    assert state["processLivenessVerified"] is False
    assert state["marketPricesUsed"] is False
    assert state["modelPromotionAttempted"] is False
    assert [x.read_bytes() for x in paths[:3]] == original


def test_cpu_and_memory_safety_checks_stop_before_training(tmp_path: Path) -> None:
    paths = files(tmp_path)
    cpu = execute(paths, tmp_path, sample=lambda: (30.0, 1.0),
                  max_cpu_seconds=20)
    # Constant process CPU measurements do not consume a relative run budget.
    assert cpu["newResearchExperiments"] == 5
    memory = execute(paths, tmp_path, sample=lambda: (0.0, 200.0),
                     max_peak_rss_mb=100)
    assert memory["status"] == "MEMORY_THRESHOLD_REACHED"
    assert memory["attemptedIterations"] == 0


def test_stop_file_prevents_start(tmp_path: Path) -> None:
    paths = files(tmp_path)
    stop = tmp_path / "STOP"
    stop.write_text("stop")
    state = execute(paths, tmp_path, stop_file=stop)
    assert state["status"] == "STOP_REQUESTED"
    assert state["attemptedIterations"] == 0
    assert not paths[3].exists()


def test_existing_lock_fails_closed(tmp_path: Path) -> None:
    paths = files(tmp_path)
    lock = tmp_path / "worker.lock"
    lock.write_text("previous process may be alive")
    with pytest.raises(ValueError, match="lock exists"):
        execute(paths, tmp_path)
    assert lock.exists()


def test_error_creates_failure_checkpoint_and_releases_lock(
    tmp_path: Path, monkeypatch,
) -> None:
    paths = files(tmp_path)
    from tris_market_event_intelligence import news_research_supervisor

    def broken(**kwargs):
        raise ValueError("input no longer matches")

    monkeypatch.setattr(news_research_supervisor, "run_news_research_loop", broken)
    with pytest.raises(ValueError, match="input no longer"):
        execute(paths, tmp_path)
    assert not (tmp_path / "worker.lock").exists()
    report = read_supervisor_status(tmp_path / "status.json")
    assert report["status"] == "FAILED_REQUIRES_INSPECTION"
    assert report["errorType"] == "ValueError"


def test_stale_status_does_not_claim_worker_liveness(tmp_path: Path) -> None:
    paths = files(tmp_path)
    execute(paths, tmp_path, max_iterations=1)
    report = read_supervisor_status(
        tmp_path / "status.json",
        now=datetime.now(UTC) + timedelta(hours=2),
    )
    assert report["statusFresh"] is False
    assert report["processLivenessVerified"] is False


@pytest.mark.parametrize("options", [
    {"max_iterations": 0}, {"max_seconds": 0},
    {"max_cpu_seconds": float("nan")}, {"max_cpu_seconds": 0},
    {"max_peak_rss_mb": -1}, {"max_peak_rss_mb": float("inf")},
    {"max_iterations": True},
])
def test_invalid_limits_fail_before_status(tmp_path: Path, options) -> None:
    paths = files(tmp_path)
    with pytest.raises(ValueError):
        execute(paths, tmp_path, **options)
    assert not (tmp_path / "status.json").exists()
    assert not (tmp_path / "worker.lock").exists()


def test_invalid_existing_status_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"schemaVersion": "other"}))
    with pytest.raises(ValueError, match="Invalid"):
        read_supervisor_status(path)
