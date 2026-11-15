"""Foreground UNIX subprocess guard for a single NEWS-only research run.

RLIMIT_CPU is enforced by the OS on the child. Memory is polled via ps and
terminated if observed above its budget; sampling is NOT a kernel hard
memory limit. Neither this module nor its CLI enables launchd.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_research_loop import _check_stop_file
from tris_market_event_intelligence.news_research_supervisor import _write_status

SCHEMA = "news_research_process_guard_v1"


def _rss_mb(pid: int) -> float | None:
    """Best-effort RSS of the direct child; ps reports KiB on macOS/Linux."""
    result = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True, text=True, timeout=2, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return int(result.stdout.strip()) / 1024
    except ValueError as exc:
        raise ValueError("Unexpected ps resident memory measurement") from exc


def _stop_child(process: subprocess.Popen[bytes]) -> None:
    """Terminate the child's process group; never signal the supervisor."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


def run_guarded_news_research(
    *, cohort_path: Path, plan_path: Path, review_path: Path,
    journal_path: Path, status_path: Path, lock_path: Path,
    stop_file: Path | None = None, max_cpu_seconds: int = 30,
    max_wall_seconds: int = 300, max_rss_mb: int = 1024,
    max_iterations: int = 1, poll_seconds: float = 0.25,
    runner: list[str] | None = None,
    rss_reader: Callable[[int], float | None] = _rss_mb,
) -> dict[str, object]:
    """Run a finite isolated child, preserving its existing SQLite journal.

    Optional runner injection is for synthetic CI only. The CLI always uses
    the unmodified existing news research CLI with private NEWS inputs.
    """
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        raise ValueError("UNIX CPU and process-group controls required")
    for value, ceiling, name in (
        (max_cpu_seconds, 86400, "CPU"),
        (max_wall_seconds, 86400, "wall"),
        (max_rss_mb, 131072, "RSS"),
        (max_iterations, 1000, "iterations"),
    ):
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError(f"Invalid {name} limit")
    if type(poll_seconds) not in (int, float) or not math.isfinite(
        poll_seconds
    ) or not 0.05 <= poll_seconds <= 2:
        raise ValueError("Polling interval must be 0.05..2 seconds")
    for path in (cohort_path, plan_path, review_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Private ordinary research source file required")
    if journal_path.is_symlink() or journal_path.is_dir() or (
        not journal_path.parent.is_dir()
    ):
        raise ValueError("Private research journal parent must exist")
    for path in (status_path, lock_path):
        if path.is_symlink() or path.is_dir() or not path.parent.is_dir():
            raise ValueError("Private status and lock directories required")
    if lock_path.exists():
        raise ValueError("Existing research lock: inspect process before clearing")
    if status_path.with_name(status_path.name + ".pending").exists():
        raise ValueError("Pending status checkpoint must be inspected")
    if stop_file is not None and _check_stop_file(stop_file):
        return {
            "schemaVersion": SCHEMA, "status": "STOP_REQUESTED",
            "childStarted": False, "productionChanged": False,
            "modelPromoted": False, "stockPricesUsed": False,
        }
    command = runner if runner is not None else [
        sys.executable, "-m",
        "tris_market_event_intelligence.news_research_loop_cli",
        "--run", "--cohort", str(cohort_path),
        "--holdout", str(plan_path), "--review", str(review_path),
        "--journal", str(journal_path),
        "--max-iterations", str(max_iterations),
        "--max-seconds", str(max_wall_seconds),
        "--interval-seconds", "0",
    ]
    if runner is None and stop_file is not None:
        command.extend(["--stop-file", str(stop_file)])
    if not command or any(not isinstance(part, str) for part in command):
        raise ValueError("Invalid research child command")
    flags = os.O_EXCL | os.O_CREAT | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    lock_inode = os.fstat(fd).st_ino
    process = None
    status = "FAILED_REQUIRES_INSPECTION"
    max_seen = 0.0
    child_exit = None
    start = time.monotonic()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "createdAt": datetime.now(UTC).isoformat()}, stream)
            stream.write("\n")
        _write_status(status_path, {
            "schemaVersion": SCHEMA, "status": "STARTING",
            "updatedAt": datetime.now(UTC).isoformat(),
            "modelPromoted": False, "stockPricesUsed": False,
        })
        process = subprocess.Popen(
            [
                sys.executable, "-m",
                "tris_market_event_intelligence.news_research_limit_exec",
                "--cpu-seconds", str(max_cpu_seconds), "--", *command,
            ],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
            close_fds=True,
        )
        while True:
            child_exit = process.poll()
            if child_exit is not None:
                status = "COMPLETED" if child_exit == 0 else "CHILD_FAILED"
                break
            if stop_file is not None and _check_stop_file(stop_file):
                status = "STOP_REQUESTED"
                break
            if time.monotonic() - start >= max_wall_seconds:
                status = "WALL_TIME_LIMIT"
                break
            current = rss_reader(process.pid)
            if current is None:
                if process.poll() is None:
                    status = "RSS_MONITOR_UNAVAILABLE"
                    break
                continue
            if not math.isfinite(current) or current < 0:
                raise ValueError("Invalid child resident-memory reading")
            max_seen = max(max_seen, current)
            if current >= max_rss_mb:
                status = "RSS_THRESHOLD_REACHED"
                break
            time.sleep(poll_seconds)
        if status in {"STOP_REQUESTED", "WALL_TIME_LIMIT", "RSS_THRESHOLD_REACHED", "RSS_MONITOR_UNAVAILABLE"}:
            _stop_child(process)
        child_exit = process.wait(timeout=3)
        report = {
            "schemaVersion": SCHEMA, "status": status,
            "updatedAt": datetime.now(UTC).isoformat(),
            "childExitCode": child_exit,
            "childStarted": True,
            "maxObservedChildRssMb": round(max_seen, 3),
            "maxCpuSecondsHardRlimit": max_cpu_seconds,
            "maxWallSeconds": max_wall_seconds,
            "maxRssMbSampled": max_rss_mb,
            "rssIsHardKernelLimit": False,
            "elapsedWallSeconds": round(time.monotonic() - start, 3),
            "productionChanged": False,
            "modelPromoted": False,
            "stockPricesUsed": False,
        }
        _write_status(status_path, report)
        return report
    except Exception as exc:
        if process is not None:
            _stop_child(process)
        _write_status(status_path, {
            "schemaVersion": SCHEMA, "status": "FAILED_REQUIRES_INSPECTION",
            "updatedAt": datetime.now(UTC).isoformat(),
            "errorType": type(exc).__name__, "modelPromoted": False,
            "stockPricesUsed": False,
        })
        raise
    finally:
        if (
            lock_path.is_file() and not lock_path.is_symlink()
            and lock_path.stat().st_ino == lock_inode
        ):
            lock_path.unlink()


def inspect_guard_status(path: Path) -> dict[str, object]:
    """Status is historical evidence, not proof that a process is live."""
    if path.is_symlink() or not path.is_file():
        return {"schemaVersion": SCHEMA, "status": "NO_STATUS"}
    if path.stat().st_size > 32768:
        raise ValueError("Research guard status too large")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schemaVersion") != SCHEMA:
        raise ValueError("Invalid guarded research status schema")
    return {**data, "processLivenessVerified": False}
