"""Bounded read-only assessment of recent collection cycle history."""

from __future__ import annotations

import json
from pathlib import Path

VALID_STATUSES = frozenset({"healthy", "degraded", "failed"})


def recent_runs(path: Path, *, max_entries: int = 100, max_bytes: int = 262144) -> list[dict[str, object]]:
    """Read recent complete JSONL rows within a fixed byte and item budget.

    A partial trailing line is ignored because the producer may have crashed
    while writing it. Older data beyond the byte budget is intentionally not
    used to infer a recovery.
    """
    if not 1 <= max_entries <= 1000 or not 1 <= max_bytes <= 2_000_000:
        raise ValueError("Invalid history read bounds")
    with path.open("rb") as stream:
        stream.seek(0, 2)
        length = stream.tell()
        start = max(0, length - max_bytes)
        stream.seek(start)
        data = stream.read(max_bytes)
    if start:
        first_newline = data.find(b"\n")
        data = data[first_newline + 1 :] if first_newline >= 0 else b""
    if not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1] if b"\n" in data else b""
    rows = []
    for line in reversed(data.splitlines()):
        if not line:
            continue
        try:
            row = json.loads(line)
        except (UnicodeError, ValueError):
            break
        if not isinstance(row, dict) or row.get("status") not in VALID_STATUSES:
            break
        rows.append(row)
        if len(rows) >= max_entries:
            break
    return rows


def assess_recovery(path: Path, *, failure_threshold: int = 3) -> dict[str, object]:
    if failure_threshold < 1:
        raise ValueError("failure_threshold must be positive")
    if not path.is_file():
        return {
            "level": "ok", "reason": "history_not_recorded_yet",
            "observed_runs": 0, "consecutive_failures": 0,
            "consecutive_degraded": 0, "recovered": False,
        }
    try:
        rows = recent_runs(path)
    except OSError:
        return {"level": "warning", "reason": "history_unreadable"}
    if not rows:
        return {"level": "warning", "reason": "history_incomplete"}
    failures = 0
    degraded = 0
    for row in rows:
        if row["status"] != "failed":
            break
        failures += 1
    for row in rows:
        if row["status"] != "degraded":
            break
        degraded += 1
    recovered = rows[0]["status"] == "healthy" and any(
        row["status"] != "healthy" for row in rows[1:]
    )
    if failures >= failure_threshold:
        level, reason = "error", "repeated_failure"
    elif failures > 0:
        level, reason = "warning", "recent_failure"
    elif degraded > 0:
        level, reason = "warning", "consecutive_degraded"
    elif recovered:
        level, reason = "ok", "recovered"
    else:
        level, reason = "ok", "no_recent_failures"
    return {
        "level": level, "reason": reason,
        "observed_runs": len(rows), "consecutive_failures": failures,
        "consecutive_degraded": degraded, "recovered": recovered,
        "last_cycle_status": rows[0]["status"],
    }
