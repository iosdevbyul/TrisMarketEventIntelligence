"""Bounded news worker should not spin, lose stops, or repeat research."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Event

import pytest
from test_news_research_worker import files

from tris_market_event_intelligence import news_research_loop
from tris_market_event_intelligence.news_research_loop import (
    request_research_stop,
    run_news_research_loop,
)
from tris_market_event_intelligence.news_research_worker import audit_research_journal


def loop(
    paths: tuple[Path, Path, Path, Path], **options,
) -> dict[str, object]:
    return run_news_research_loop(
        cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
        interval_seconds=0, **options,
    )


def test_repeat_replay_exhausts_without_duplicate_experiments(tmp_path: Path) -> None:
    paths = files(tmp_path)
    source_before = [item.read_bytes() for item in paths[:3]]
    first = loop(paths, max_iterations=2)
    assert first["status"] == "ITERATION_LIMIT_REACHED"
    assert first["newResearchExperiments"] == 2
    resumed = loop(paths, max_iterations=10)
    assert resumed["status"] == "NO_NEW_EXPERIMENTS"
    assert resumed["newResearchExperiments"] == 3
    assert resumed["attemptedIterations"] == 4
    last = loop(paths, max_iterations=10)
    assert last["status"] == "NO_NEW_EXPERIMENTS"
    assert last["newResearchExperiments"] == 0
    assert last["attemptedIterations"] == 1
    assert audit_research_journal(paths[3])["totalIterations"] == 5
    assert [item.read_bytes() for item in paths[:3]] == source_before


def test_stop_file_before_start_prevents_research(tmp_path: Path) -> None:
    paths = files(tmp_path)
    stop_file = tmp_path / "STOP"
    assert request_research_stop(stop_file)["workerStoppedVerified"] is False
    result = loop(paths, max_iterations=5, stop_file=stop_file)
    assert result["status"] == "STOP_REQUESTED"
    assert result["attemptedIterations"] == 0
    assert not paths[3].exists()


def test_stop_after_first_iteration(tmp_path: Path) -> None:
    paths = files(tmp_path)
    flag = tmp_path / "STOP"

    def signal_stop(seconds: float) -> bool:
        if not flag.exists():
            request_research_stop(flag)
        return False

    result = run_news_research_loop(
        cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
        stop_file=flag, interval_seconds=1.0,
        max_iterations=10, wait=signal_stop,
    )
    assert result["status"] == "STOP_REQUESTED"
    assert result["newResearchExperiments"] == 1
    assert audit_research_journal(paths[3])["totalIterations"] == 1


def test_missing_gold_exits_without_polling_and_model_training(tmp_path: Path) -> None:
    paths = files(tmp_path)
    review = json.loads(paths[2].read_text())
    for entry in review["judgments"][:6]:
        for field in ("importance", "sourceAlignment", "reviewer", "reviewedAt", "reason"):
            entry[field] = None
    paths[2].write_text(json.dumps(review))
    from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout
    from test_news_research_worker import TRAIN_END, VALIDATION_END

    plan = plan_news_holdout(
        paths[0], train_end=TRAIN_END, validation_end=VALIDATION_END,
        review_path=paths[2],
    )
    paths[1].write_text(json.dumps(plan))
    result = loop(paths, max_iterations=100)
    assert result["status"] == "PAUSED_REQUIRES_NEW_HUMAN_GOLD"
    assert result["attemptedIterations"] == 1
    assert result["newResearchExperiments"] == 0
    assert audit_research_journal(paths[3])["totalIterations"] == 0


def test_transient_db_busy_uses_bounded_retry_and_stops(tmp_path: Path, monkeypatch) -> None:
    paths = files(tmp_path)
    calls = 0
    waited = []

    def busy(**kwargs):
        nonlocal calls
        calls += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(news_research_loop, "research_news_once", busy)
    result = run_news_research_loop(
        cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
        interval_seconds=0, max_iterations=3,
        max_backoff_seconds=2,
        wait=lambda seconds: waited.append(seconds) or False,
    )
    assert result["status"] == "TRANSIENT_RETRY_LIMIT_REACHED"
    assert result["attemptedIterations"] == calls == 3
    assert result["retryableResearchLedgerErrors"] == 3
    assert sum(waited) == pytest.approx(3.0)


def test_unknown_operational_error_fails_closed(tmp_path: Path, monkeypatch) -> None:
    paths = files(tmp_path)

    def error(**kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(news_research_loop, "research_news_once", error)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        loop(paths, max_iterations=2)


def test_explicit_event_is_cooperative(tmp_path: Path) -> None:
    paths = files(tmp_path)
    event = Event()
    event.set()
    result = loop(paths, max_iterations=10, stop_event=event)
    assert result["status"] == "STOP_REQUESTED"
    assert result["attemptedIterations"] == 0


@pytest.mark.parametrize("options", [
    {"max_iterations": 0}, {"max_iterations": True},
    {"max_iterations": 1001}, {"max_seconds": 0}, {"max_seconds": 86401},
    {"max_seconds": True}, {"interval_seconds": -1},
    {"interval_seconds": float("nan")},
    {"max_backoff_seconds": float("inf")}, {"max_backoff_seconds": 0},
])
def test_invalid_resource_limits_rejected(
    tmp_path: Path, options: dict[str, object],
) -> None:
    paths = files(tmp_path)
    with pytest.raises(ValueError):
        loop(paths, **options)
    assert not paths[3].exists()


def test_stop_flag_is_create_only_and_rejects_symlink(tmp_path: Path) -> None:
    flag = tmp_path / "STOP"
    request_research_stop(flag)
    with pytest.raises(ValueError, match="new"):
        request_research_stop(flag)
    symlink = tmp_path / "STOP-LINK"
    symlink.symlink_to(flag)
    with pytest.raises(ValueError, match="new"):
        request_research_stop(symlink)
