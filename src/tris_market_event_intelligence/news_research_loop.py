"""Explicit, bounded Mac-safe NEWS research loop.

The existing one-iteration engine remains authoritative for provenance,
train/validation isolation, immutable experiment deduplication and metrics.
This runner never starts itself, installs launchd, or promotes any model.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from pathlib import Path
from threading import Event
from typing import Callable

from tris_market_event_intelligence.news_research_worker import research_news_once

MAX_RUN_SECONDS = 86400
MAX_ITERATIONS = 1000


def _check_stop_file(path: Path | None) -> bool:
    if path is None:
        return False
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("Stop signal must be an ordinary local file")
    return path.exists()


def _wait(
    seconds: float, *, stop_file: Path | None, stop_event: Event,
    wait: Callable[[float], bool],
) -> bool:
    """Poll so stop-file requests work even while waiting a long backoff."""
    remaining = seconds
    while remaining > 0:
        if stop_event.is_set() or _check_stop_file(stop_file):
            return True
        step = min(remaining, 0.25)
        if wait(step):
            return True
        remaining -= step
    return stop_event.is_set() or _check_stop_file(stop_file)


def run_news_research_loop(
    *, cohort_path: Path, plan_path: Path, review_path: Path,
    journal_path: Path, max_iterations: int = 5, max_seconds: int = 3600,
    interval_seconds: float = 1.0, max_backoff_seconds: float = 60.0,
    stop_file: Path | None = None, stop_event: Event | None = None,
    wait: Callable[[float], bool] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> dict[str, object]:
    """Run a finite, cooperative set of news-only candidate experiments.

    A previously completed candidate is not rerun, thanks to the existing
    SQLite experiment fingerprint. Exhausted experiment space immediately
    exits. Missing human gold pauses with backoff; no fake learning is done.
    """
    if type(max_iterations) is not int or not 1 <= max_iterations <= MAX_ITERATIONS:
        raise ValueError("max_iterations must be 1..1000")
    if type(max_seconds) is not int or not 1 <= max_seconds <= MAX_RUN_SECONDS:
        raise ValueError("max_seconds must be 1..86400")
    if (
        type(interval_seconds) not in (int, float)
        or not math.isfinite(interval_seconds)
        or not 0.0 <= interval_seconds <= 3600
    ):
        raise ValueError("interval_seconds must be finite and within 0..3600")
    if (
        type(max_backoff_seconds) not in (int, float)
        or not math.isfinite(max_backoff_seconds)
        or not 1.0 <= max_backoff_seconds <= 3600
    ):
        raise ValueError("max_backoff_seconds must be finite and within 1..3600")
    if stop_file is not None:
        _check_stop_file(stop_file)
    event = stop_event if stop_event is not None else Event()
    wait_fn = wait if wait is not None else event.wait
    timer = monotonic if monotonic is not None else time.monotonic
    start = timer()
    attempts = 0
    new_experiments = 0
    blocked = 0
    retryable_db_errors = 0
    last_iteration_status: str | None = None
    status = "ITERATION_LIMIT_REACHED"
    while attempts < max_iterations:
        if event.is_set() or _check_stop_file(stop_file):
            status = "STOP_REQUESTED"
            break
        if timer() - start >= max_seconds:
            status = "TIME_BUDGET_EXHAUSTED"
            break
        try:
            outcome = research_news_once(
                cohort_path=cohort_path, plan_path=plan_path,
                review_path=review_path, journal_path=journal_path,
            )
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            attempts += 1
            retryable_db_errors += 1
            last_iteration_status = "TRANSIENT_RESEARCH_LEDGER_BUSY"
            remaining = max_seconds - (timer() - start)
            if remaining <= 0:
                status = "TIME_BUDGET_EXHAUSTED"
                break
            if attempts == max_iterations:
                status = "TRANSIENT_RETRY_LIMIT_REACHED"
                break
            backoff = min(float(max_backoff_seconds), 2 ** min(retryable_db_errors - 1, 10))
            if _wait(
                min(backoff, remaining), stop_file=stop_file, stop_event=event,
                wait=wait_fn,
            ):
                status = "STOP_REQUESTED"
                break
            continue
        attempts += 1
        retryable_db_errors = 0
        last_iteration_status = outcome["status"]
        if last_iteration_status == "EXPERIMENT_GRID_EXHAUSTED":
            status = "NO_NEW_EXPERIMENTS"
            break
        if last_iteration_status == "BLOCKED_INSUFFICIENT_HUMAN_GOLD":
            blocked += 1
            # No useful research can happen on this unchanged, immutable
            # cohort/review/plan. Do not poll a fixed input indefinitely.
            status = "PAUSED_REQUIRES_NEW_HUMAN_GOLD"
            break
        if last_iteration_status != "DESCRIPTIVE_RESEARCH_ONLY":
            raise ValueError("Unexpected one-iteration research outcome")
        new_experiments += 1
        if attempts == max_iterations:
            break
        remaining = max_seconds - (timer() - start)
        if remaining <= 0:
            status = "TIME_BUDGET_EXHAUSTED"
            break
        delay = min(float(interval_seconds), remaining)
        if _wait(delay, stop_file=stop_file, stop_event=event, wait=wait_fn):
            status = "STOP_REQUESTED"
            break
    return {
        "schemaVersion": "news_research_loop_v1",
        "status": status,
        "attemptedIterations": attempts,
        "newResearchExperiments": new_experiments,
        "blockedForHumanGold": blocked,
        "retryableResearchLedgerErrors": retryable_db_errors,
        "lastIterationStatus": last_iteration_status,
        "maxIterations": max_iterations,
        "maxRunSeconds": max_seconds,
        "modelPromotionAttempted": False,
        "productionCollectorModified": False,
        "testSetEvaluated": False,
        "marketPricesUsed": False,
        "modelImprovementValidated": False,
    }


def request_research_stop(path: Path) -> dict[str, object]:
    """Atomically create a stop flag; does not signal unrelated processes."""
    if path.is_symlink() or path.exists() or not path.parent.is_dir():
        raise ValueError("Stop file must be new and within an existing local directory")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"schemaVersion": "news_research_stop_v1", "stop": True}, stream)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {"status": "STOP_FILE_CREATED", "workerStoppedVerified": False}
