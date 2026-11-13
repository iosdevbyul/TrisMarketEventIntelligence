"""Offline operational diagnostics; never starts a collector or modifies SQLite."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def diagnose(
    db_path: Path,
    status_path: Path,
    *,
    now: datetime | None = None,
    stale_after_minutes: int = 90,
    warning_bytes: int = 512 * 1024 * 1024,
) -> dict[str, object]:
    """Report evidence only; no network, schema initialization, or DB writes."""
    if stale_after_minutes < 1 or warning_bytes < 1:
        raise ValueError("Thresholds must be positive")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone aware")
    checks: dict[str, dict[str, object]] = {}
    if not status_path.is_file():
        checks["last_run"] = {"level": "error", "reason": "missing_status"}
    else:
        try:
            content = json.loads(status_path.read_text(encoding="utf-8"))
            if not isinstance(content, dict):
                raise ValueError("Invalid status object")
            finished = datetime.fromisoformat(content["finished_at"].replace("Z", "+00:00"))
            if finished.tzinfo is None:
                raise ValueError("Naive timestamp")
            age = (now - finished).total_seconds()
            status = content.get("status")
            if status not in {"healthy", "degraded", "failed"}:
                level, reason = "error", "invalid_status"
            elif age < -300:
                level, reason = "error", "future_timestamp"
            elif age > stale_after_minutes * 60:
                level, reason = "error", "stale_run"
            elif status == "failed":
                level, reason = "error", "last_run_failed"
            elif status == "degraded":
                level, reason = "warning", "last_run_degraded"
            else:
                level, reason = "ok", "recent_healthy"
            checks["last_run"] = {
                "level": level, "reason": reason,
                "last_status": status, "age_seconds": round(age),
            }
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            checks["last_run"] = {"level": "error", "reason": "invalid_status_json"}
    if not db_path.is_file():
        checks["database"] = {"level": "error", "reason": "missing_database"}
    else:
        size = db_path.stat().st_size
        try:
            # Use SQLite's read-only connection; no implicit creation or repair.
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5) as db:
                db.execute("PRAGMA query_only=ON")
                integrity = db.execute("PRAGMA quick_check").fetchone()
                if not integrity or integrity[0] != "ok":
                    level, reason = "error", "integrity_failed"
                elif size >= warning_bytes:
                    level, reason = "warning", "database_large"
                else:
                    level, reason = "ok", "database_valid"
        except sqlite3.Error:
            level, reason = "error", "database_unreadable"
        checks["database"] = {
            "level": level, "reason": reason, "bytes": size,
        }
    levels = {value["level"] for value in checks.values()}
    overall = "error" if "error" in levels else "warning" if "warning" in levels else "ok"
    return {
        "status": overall, "checks": checks,
        "stale_after_minutes": stale_after_minutes,
        "warning_bytes": warning_bytes,
        "network_requests": 0, "modified_database": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check offline macOS market-event health")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--status", type=Path, default=base / "last-run.json")
    parser.add_argument("--stale-after-minutes", type=int, default=90)
    parser.add_argument("--warning-mb", type=int, default=512)
    args = parser.parse_args()
    try:
        result = diagnose(
            args.db.expanduser(), args.status.expanduser(),
            stale_after_minutes=args.stale_after_minutes,
            warning_bytes=args.warning_mb * 1024 * 1024,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "error":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
