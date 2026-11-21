"""Read-only audit of continuously scheduled news acquisition on a Mac.

Checks whether the installed LaunchAgent is loaded, whether its latest cycle
used full-text acquisition, and whether the publisher-body backlog is moving.
Does not launch jobs, request the network, or change the production database.
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

SOURCES = (
    "fed_monetary", "fed_all", "bok_all", "bok_statistics",
    "bok_corrections", "bbc_business", "bbc_technology", "bbc_world",
)
LABEL = "com.tris.market-event-intelligence-integrated"


def stamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else None


def audit(db_path: Path, status_path: Path, interval: int = 900) -> dict:
    now = datetime.now(UTC)
    errors: list[str] = []
    if not db_path.is_file() or db_path.is_symlink():
        return {"status": "UNVERIFIED", "errors": ["DB_UNAVAILABLE"]}
    if not status_path.is_file() or status_path.is_symlink():
        errors.append("STATUS_FILE_UNAVAILABLE")
        last_cycle = {}
    else:
        try:
            last_cycle = json.loads(status_path.read_text(encoding="utf-8"))
            if not isinstance(last_cycle, dict):
                raise TypeError("Invalid status JSON root")
        except (OSError, ValueError, TypeError):
            last_cycle = {}
            errors.append("STATUS_FILE_INVALID")
    completed = stamp(last_cycle.get("finished_at"))
    elapsed = int((now - completed).total_seconds()) if completed else None
    if elapsed is None or elapsed < -60 or elapsed > max(interval * 3, 1800):
        errors.append("LAST_CYCLE_STALE")
    if last_cycle.get("status") not in {"healthy", "degraded"}:
        errors.append("LAST_CYCLE_NOT_SUCCESSFUL")
    collector = last_cycle.get("collection") or {}
    if not isinstance(collector, dict):
        collector = {}
    if not collector.get("fulltext_acquisition_enabled"):
        errors.append("FULLTEXT_NOT_ENABLED_IN_LAST_CYCLE")
    if not isinstance(collector.get("article_fulltext"), dict):
        errors.append("FULLTEXT_CYCLE_REPORT_MISSING")
    agent_path = (
        Path.home() / "Library" / "LaunchAgents" / (LABEL + ".plist")
    )
    if not agent_path.is_file() or agent_path.is_symlink():
        errors.append("INTEGRATED_LAUNCH_AGENT_MISSING")
        configured_interval = None
        configured_sources: list[str] = []
    else:
        try:
            with agent_path.open("rb") as handle:
                config = plistlib.load(handle)
            configured_interval = config.get("StartInterval")
            command = config.get("ProgramArguments", [])
            selected = command[command.index("--sources") + 1]
            configured_sources = selected.split(",")
            if config.get("Label") != LABEL:
                errors.append("LAUNCH_AGENT_LABEL_MISMATCH")
            if set(configured_sources) != set(SOURCES):
                errors.append("LAUNCH_AGENT_SOURCE_MISMATCH")
        except (OSError, ValueError, IndexError, TypeError, KeyError):
            configured_interval = None
            configured_sources = []
            errors.append("LAUNCH_AGENT_CONFIG_INVALID")
    try:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"],
            capture_output=True, text=True, timeout=8, check=False,
        )
        loaded = result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        loaded = False
    if not loaded:
        errors.append("INTEGRATED_LAUNCH_AGENT_NOT_LOADED")
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "article_full_text" not in tables:
            errors.append("FULLTEXT_TABLE_MISSING")
        cutoff = (now - timedelta(hours=24)).isoformat()
        per_source = []
        for source in SOURCES:
            total, recent = db.execute(
                "SELECT COUNT(*), SUM(CASE WHEN detected_at >= ? THEN 1 ELSE 0 END) "
                "FROM articles WHERE source=?", (cutoff, source),
            ).fetchone()
            states = {}
            pending = total
            if "article_full_text" in tables:
                states = dict(db.execute(
                    "SELECT COALESCE(f.status,'NOT_ATTEMPTED'), COUNT(*) "
                    "FROM articles a LEFT JOIN article_full_text f "
                    "ON f.article_id=a.id WHERE a.source=? "
                    "GROUP BY COALESCE(f.status,'NOT_ATTEMPTED')", (source,),
                ))
                pending = states.get("NOT_ATTEMPTED", 0) + states.get("INCOMPLETE", 0)
            per_source.append({
                "source": source, "discoveredTotal": total,
                "discoveredLast24h": recent or 0,
                "fulltextStates": states, "notCollectedOrFailed": pending,
            })
        if "article_full_text" in tables:
            backlog = db.execute(
                "SELECT COUNT(*) FROM articles a LEFT JOIN article_full_text f "
                "ON f.article_id=a.id WHERE a.source IN "
                "(?,?,?,?,?,?,?,?) AND "
                "(f.article_id IS NULL OR f.status='INCOMPLETE')",
                SOURCES,
            ).fetchone()[0]
            errors_by_code = dict(db.execute(
                "SELECT COALESCE(f.error_code,'UNKNOWN'), COUNT(*) "
                "FROM article_full_text f JOIN articles a ON a.id=f.article_id "
                "WHERE a.source IN (?,?,?,?,?,?,?,?) AND f.status='INCOMPLETE' "
                "GROUP BY COALESCE(f.error_code,'UNKNOWN')",
                SOURCES,
            ))
        else:
            backlog = sum(row["notCollectedOrFailed"] for row in per_source)
            errors_by_code = {}
    if backlog:
        errors.append("ARTICLE_BODY_BACKLOG_PRESENT")
    return {
        "schemaVersion": "continuous_fulltext_audit_v1",
        "status": "HEALTHY" if not errors else "REQUIRES_ATTENTION",
        "checkedAt": now.isoformat(),
        "lastCycleStatus": last_cycle.get("status"),
        "lastCycleAgeSeconds": elapsed,
        "configuredIntervalSeconds": configured_interval,
        "configuredSources": configured_sources,
        "launchAgentLoaded": loaded,
        "fulltextEnabledInLastCycle": collector.get(
            "fulltext_acquisition_enabled", False
        ),
        "lastCycleFulltext": collector.get("article_fulltext"),
        "bodyBacklog": backlog,
        "incompleteErrorCodes": errors_by_code,
        "sources": per_source,
        "errors": errors,
        "productionDatabaseModified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--status-file", type=Path, default=base / "integrated-last-run.json")
    parser.add_argument("--interval", type=int, default=900)
    args = parser.parse_args()
    result = audit(args.db.expanduser(), args.status_file.expanduser(), args.interval)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "HEALTHY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
