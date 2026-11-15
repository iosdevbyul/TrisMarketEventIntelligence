"""Opt-in, finite NEWS-only research supervisor with local status and quotas.

Does not start a daemon, install launchd, change production model, or read
market prices. CPU/memory accounting is sampled at iteration boundaries;
it cannot kill a blocking operation midway through an iteration.
"""
from __future__ import annotations

import json
import math
import os
import resource
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_research_loop import (
    _check_stop_file,
    run_news_research_loop,
)

SCHEMA = "news_research_supervisor_status_v1"
TERMINAL = {
    "NO_NEW_EXPERIMENTS", "PAUSED_REQUIRES_NEW_HUMAN_GOLD",
    "STOP_REQUESTED", "TIME_BUDGET_EXHAUSTED",
    "TRANSIENT_RETRY_LIMIT_REACHED",
}
MAX_STATUS_BYTES = 32768


def process_resources() -> tuple[float, float]:
    """Current cumulative process CPU seconds and peak resident MB."""
    value = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(value.ru_maxrss)
    # macOS ru_maxrss uses bytes; Linux uses KiB.
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    return float(value.ru_utime + value.ru_stime), rss_mb


def _valid_file(path: Path, *, allow_absent: bool = False) -> None:
    if path.is_symlink() or path.is_dir() or not path.parent.is_dir() or (
        not allow_absent and not path.is_file()
    ):
        raise ValueError("Expected private ordinary file in an existing directory")


def _write_status(path: Path, data: dict[str, object]) -> None:
    _valid_file(path, allow_absent=True)
    payload = (json.dumps(data, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if len(payload) > MAX_STATUS_BYTES:
        raise ValueError("Research supervisor status exceeds size limit")
    temporary = path.with_name(path.name + ".pending")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("Pending checkpoint exists; refusing to overwrite")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_supervisor_status(path: Path, *, now: datetime | None = None,
                           stale_after_seconds: int = 600) -> dict[str, object]:
    """Read a bounded status record, never infer that a process is running."""
    if type(stale_after_seconds) is not int or not 1 <= stale_after_seconds <= 86400:
        raise ValueError("Invalid status stale threshold")
    if path.is_symlink() or not path.is_file():
        return {"schemaVersion": SCHEMA, "status": "NO_STATUS"}
    if path.stat().st_size > MAX_STATUS_BYTES:
        raise ValueError("Status file too large")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schemaVersion") != SCHEMA:
        raise ValueError("Invalid research supervisor status")
    current = now or datetime.now(UTC)
    stamp = datetime.fromisoformat(data["updatedAt"])
    if current.utcoffset() is None or stamp.utcoffset() is None:
        raise ValueError("Status timestamp must have timezone")
    age = (current - stamp).total_seconds()
    if age < 0:
        raise ValueError("Status timestamp is in the future")
    return {
        **data,
        "statusFresh": age <= stale_after_seconds,
        "statusAgeSeconds": round(age, 3),
        "processLivenessVerified": False,
    }


def run_supervised_news_research(
    *, cohort_path: Path, plan_path: Path, review_path: Path,
    journal_path: Path, status_path: Path, lock_path: Path,
    stop_file: Path | None = None, max_iterations: int = 5,
    max_seconds: int = 3600, max_cpu_seconds: float = 600,
    max_peak_rss_mb: float = 1024,
    sample: Callable[[], tuple[float, float]] = process_resources,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    """Single owner, bounded experiments, atomic checkpoint after each result."""
    if type(max_iterations) is not int or not 1 <= max_iterations <= 1000:
        raise ValueError("Invalid max iterations")
    if type(max_seconds) is not int or not 1 <= max_seconds <= 86400:
        raise ValueError("Invalid wall-time limit")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 1 <= v <= cap
           for v, cap in ((max_cpu_seconds, 86400), (max_peak_rss_mb, 131072))):
        raise ValueError("Invalid CPU or memory sampling threshold")
    _valid_file(status_path, allow_absent=True)
    if status_path.with_name(status_path.name + ".pending").exists():
        raise ValueError("Unreconciled pending status checkpoint")
    _valid_file(lock_path, allow_absent=True)
    if lock_path.exists():
        raise ValueError("Research supervisor lock exists: inspect old process first")
    if stop_file is not None:
        _check_stop_file(stop_file)
    flags = os.O_WRONLY | os.O_EXCL | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    start = monotonic()
    cpu_start = sample()[0]
    attempts = 0
    new_experiments = 0
    outcome_status = "NO_ITERATIONS"
    error_type = None

    def checkpoint(status: str) -> dict[str, object]:
        cpu_total, rss_mb = sample()
        return {
            "schemaVersion": SCHEMA,
            "status": status,
            "updatedAt": datetime.now(UTC).isoformat(),
            "attemptedIterations": attempts,
            "newResearchExperiments": new_experiments,
            "lastResearchStatus": outcome_status,
            "elapsedSeconds": round(max(0, monotonic() - start), 3),
            "cpuSecondsThisRun": round(max(0, cpu_total - cpu_start), 3),
            "peakResidentMbObserved": round(max(0, rss_mb), 3),
            "maxCpuSeconds": max_cpu_seconds,
            "maxPeakResidentMb": max_peak_rss_mb,
            "errorType": error_type,
            "supervisorIsDaemon": False,
            "processLivenessVerified": False,
            "modelPromotionAttempted": False,
            "marketPricesUsed": False,
            "productionDatabaseModified": False,
        }
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump({"pid": os.getpid(), "startedAt": datetime.now(UTC).isoformat()}, file)
            file.write("\n")
        _write_status(status_path, checkpoint("RUNNING_CHECKPOINT_ONLY"))
        while attempts < max_iterations:
            cpu_now, rss_now = sample()
            if stop_file is not None and _check_stop_file(stop_file):
                status = "STOP_REQUESTED"
                break
            if monotonic() - start >= max_seconds:
                status = "TIME_BUDGET_EXHAUSTED"
                break
            if cpu_now - cpu_start >= max_cpu_seconds:
                status = "CPU_BUDGET_REACHED"
                break
            if rss_now >= max_peak_rss_mb:
                status = "MEMORY_THRESHOLD_REACHED"
                break
            remaining = max(1, math.ceil(max_seconds - (monotonic() - start)))
            run = run_news_research_loop(
                cohort_path=cohort_path, plan_path=plan_path,
                review_path=review_path, journal_path=journal_path,
                max_iterations=1, max_seconds=remaining,
                interval_seconds=0, stop_file=stop_file,
            )
            attempts += run["attemptedIterations"]
            new_experiments += run["newResearchExperiments"]
            outcome_status = run["status"]
            _write_status(status_path, checkpoint("RUNNING_CHECKPOINT_ONLY"))
            if outcome_status in TERMINAL or (
                run["attemptedIterations"] == 0
            ):
                status = outcome_status
                break
        else:
            status = "ITERATION_LIMIT_REACHED"
        final = checkpoint(status)
        _write_status(status_path, final)
        return final
    except Exception as exc:
        error_type = type(exc).__name__
        _write_status(status_path, checkpoint("FAILED_REQUIRES_INSPECTION"))
        raise
    finally:
        lock_path.unlink(missing_ok=True)
