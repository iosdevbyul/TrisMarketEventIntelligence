"""OS CPU quota and process-group RSS guard for NEWS-only research."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from test_news_research_worker import files

from tris_market_event_intelligence.news_research_process_guard import (
    inspect_guard_status,
    run_guarded_news_research,
)


def guarded(paths, tmp_path: Path, **kwargs):
    kwargs.setdefault("poll_seconds", 0.05)
    return run_guarded_news_research(
        cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
        status_path=tmp_path / "guard-status.json",
        lock_path=tmp_path / "guard.lock",
        **kwargs,
    )


def test_guarded_original_research_runs_without_promoting(tmp_path: Path) -> None:
    paths = files(tmp_path)
    originals = [p.read_bytes() for p in paths[:3]]
    report = guarded(paths, tmp_path, max_cpu_seconds=15,
                     max_wall_seconds=20, max_rss_mb=256)
    assert report["status"] == "COMPLETED"
    assert report["childExitCode"] == 0
    assert report["modelPromoted"] is False
    assert report["stockPricesUsed"] is False
    assert report["rssIsHardKernelLimit"] is False
    assert report["maxCpuSecondsHardRlimit"] == 15
    assert not (tmp_path / "guard.lock").exists()
    assert inspect_guard_status(tmp_path / "guard-status.json")["status"] == "COMPLETED"
    assert [p.read_bytes() for p in paths[:3]] == originals


def test_memory_threshold_terminates_running_child(tmp_path: Path) -> None:
    paths = files(tmp_path)
    report = guarded(
        paths, tmp_path, max_wall_seconds=5, max_rss_mb=64,
        runner=[sys.executable, "-c", "import time; time.sleep(10)"],
        rss_reader=lambda pid: 128.0,
    )
    assert report["status"] == "RSS_THRESHOLD_REACHED"
    assert report["childExitCode"] != 0
    assert not (tmp_path / "guard.lock").exists()


def test_wall_time_reaps_blocking_child(tmp_path: Path) -> None:
    paths = files(tmp_path)
    report = guarded(
        paths, tmp_path, max_wall_seconds=1, max_rss_mb=512,
        runner=[sys.executable, "-c", "import time; time.sleep(10)"],
        rss_reader=lambda pid: 6.0,
    )
    assert report["status"] == "WALL_TIME_LIMIT"
    assert report["elapsedWallSeconds"] < 4


def test_missing_memory_monitor_fails_closed(tmp_path: Path) -> None:
    paths = files(tmp_path)
    report = guarded(
        paths, tmp_path, max_wall_seconds=5, max_rss_mb=128,
        runner=[sys.executable, "-c", "import time; time.sleep(10)"],
        rss_reader=lambda pid: None,
    )
    assert report["status"] == "RSS_MONITOR_UNAVAILABLE"
    assert not (tmp_path / "guard.lock").exists()


def test_child_failure_is_reported_not_success(tmp_path: Path) -> None:
    paths = files(tmp_path)
    report = guarded(
        paths, tmp_path, max_wall_seconds=5, max_rss_mb=512,
        runner=[sys.executable, "-c", "raise RuntimeError('fixture failure')"],
        rss_reader=lambda pid: 1.0,
    )
    assert report["status"] == "CHILD_FAILED"
    assert report["childExitCode"] != 0
    assert inspect_guard_status(tmp_path / "guard-status.json")["status"] == (
        "CHILD_FAILED"
    )


def test_existing_lock_is_not_removed(tmp_path: Path) -> None:
    paths = files(tmp_path)
    lock = tmp_path / "guard.lock"
    lock.write_text("a previous run may still exist", encoding="utf-8")
    with pytest.raises(ValueError, match="Existing research lock"):
        guarded(paths, tmp_path)
    assert lock.exists()
    assert not (tmp_path / "guard-status.json").exists()


def test_stop_before_child_start(tmp_path: Path) -> None:
    paths = files(tmp_path)
    stop = tmp_path / "STOP"
    stop.write_text("stop")
    report = guarded(paths, tmp_path, stop_file=stop)
    assert report["status"] == "STOP_REQUESTED"
    assert report["childStarted"] is False
    assert not paths[3].exists()


@pytest.mark.parametrize("config", [
    {"max_cpu_seconds": 0}, {"max_cpu_seconds": True},
    {"max_wall_seconds": 0}, {"max_rss_mb": 0},
    {"max_iterations": 0}, {"max_iterations": 1001},
    {"poll_seconds": float("nan")}, {"poll_seconds": 0.001},
])
def test_invalid_limits_fail_before_start(tmp_path: Path, config: dict) -> None:
    paths = files(tmp_path)
    with pytest.raises(ValueError):
        guarded(paths, tmp_path, **config)
    assert not (tmp_path / "guard.lock").exists()
    assert not (tmp_path / "guard-status.json").exists()


def test_status_is_historical_not_liveness(tmp_path: Path) -> None:
    report = inspect_guard_status(tmp_path / "absent.json")
    assert report["status"] == "NO_STATUS"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schemaVersion": "unknown"}))
    with pytest.raises(ValueError, match="schema"):
        inspect_guard_status(bad)


def test_os_cpu_limit_terminates_busy_subprocess(tmp_path: Path) -> None:
    paths = files(tmp_path)
    report = guarded(
        paths, tmp_path, max_cpu_seconds=1, max_wall_seconds=6,
        max_rss_mb=512,
        runner=[sys.executable, "-c", "while True: pass"],
        rss_reader=lambda pid: 1.0,
    )
    assert report["status"] == "CHILD_FAILED"
    assert report["childExitCode"] != 0
    assert report["maxCpuSecondsHardRlimit"] == 1
    assert os.path.exists(tmp_path / "guard-status.json")


def test_guarded_train_cv_runs_and_resumes(tmp_path: Path) -> None:
    paths = files(tmp_path)
    original_inputs = [p.read_bytes() for p in paths[:3]]
    first = guarded(paths, tmp_path, research_mode="train-cv",
                    max_iterations=2, max_cpu_seconds=15,
                    max_wall_seconds=20, max_rss_mb=256)
    assert first["status"] == "COMPLETED"
    assert first["researchMode"] == "train-cv"
    assert first["modelImprovementValidated"] is False
    assert first["modelPromoted"] is False
    assert first["stockPricesUsed"] is False
    from tris_market_event_intelligence.news_train_cv_exploration import (
        audit_train_exploration,
    )
    count = audit_train_exploration(paths[3])["totalExperiments"]
    assert 0 <= count <= 2
    second = guarded(paths, tmp_path, research_mode="train-cv",
                     max_iterations=2, max_cpu_seconds=15,
                     max_wall_seconds=20, max_rss_mb=256)
    assert second["status"] == "COMPLETED"
    assert count <= audit_train_exploration(paths[3])["totalExperiments"] <= count + 2
    assert [p.read_bytes() for p in paths[:3]] == original_inputs


@pytest.mark.parametrize("mode,iterations", [
    ("invalid", 1), ("train-cv", 19),
])
def test_invalid_train_cv_mode_or_limit_is_rejected_before_lock(
    tmp_path: Path, mode: str, iterations: int,
) -> None:
    paths = files(tmp_path)
    with pytest.raises(ValueError):
        guarded(paths, tmp_path, research_mode=mode, max_iterations=iterations)
    assert not (tmp_path / "guard.lock").exists()
    assert not (tmp_path / "guard-status.json").exists()
