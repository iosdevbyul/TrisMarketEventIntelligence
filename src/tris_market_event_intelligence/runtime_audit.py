"""Read-only audit of real macOS LaunchAgents, RSS history and localhost API."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from tris_market_event_intelligence.api_launchd_config import API_LABEL
from tris_market_event_intelligence.launchd_config import LABEL
from tris_market_event_intelligence.operational_diagnostics import diagnose
from tris_market_event_intelligence.operational_history import history_path_for
from tris_market_event_intelligence.operational_recovery import recent_runs

LOCAL_API = "http://127.0.0.1:8100"


def _launchctl(label: str, user_id: int) -> dict[str, object]:
    try:
        command = ["launchctl", "print", f"gui/{user_id}/{label}"]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"level": "error", "reason": "launchctl_unavailable"}
    if completed.returncode:
        return {"level": "error", "reason": "agent_not_loaded"}
    output = completed.stdout
    state = re.search(r"^\s*state = (.+)$", output, re.MULTILINE)
    count = re.search(r"^\s*runs = (\d+)$", output, re.MULTILINE)
    exit_code = re.search(r"^\s*last exit code = (.+)$", output, re.MULTILINE)
    # Nested launchctl coalition states also exist. The first occurrence is
    # the process state; do not mistake 'not running' for a failure on one-shots.
    if not state or not count or not exit_code:
        return {"level": "error", "reason": "launchctl_format_unrecognized"}
    return {
        "level": "ok", "reason": "agent_loaded",
        "state": state.group(1).strip(), "runs": int(count.group(1)),
        "last_exit_code": exit_code.group(1).strip(),
    }


def _api_probe(port: int) -> dict[str, object]:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/ready/claims", timeout=3
        ) as response:
            if response.status != 200:
                return {"level": "error", "reason": "claims_not_ready"}
            body = response.read(4096)
        if json.loads(body) != {"status": "ready"}:
            return {"level": "error", "reason": "claims_invalid_response"}
    except (OSError, ValueError, urllib.error.URLError):
        return {"level": "error", "reason": "api_unreachable_or_not_ready"}
    return {"level": "ok", "reason": "claims_ready"}


def audit(
    db_path: Path,
    status_path: Path,
    *,
    user_id: int,
    port: int = 8100,
    minimum_runs: int = 2,
) -> dict[str, object]:
    """Read-only, bounded snapshot. Does not infer wake/reboot success."""
    if user_id < 0 or not 1 <= port <= 65535 or minimum_runs < 2:
        raise ValueError("Invalid audit parameters")
    base = diagnose(db_path, status_path)
    checks = dict(base["checks"])
    collector = _launchctl(LABEL, user_id)
    api_agent = _launchctl(API_LABEL, user_id)
    if collector["level"] == "ok":
        if collector["runs"] < minimum_runs:
            collector["level"], collector["reason"] = "warning", "insufficient_launches"
        elif collector["last_exit_code"] != "0":
            collector["level"], collector["reason"] = "warning", "last_exit_not_success"
    if api_agent["level"] == "ok" and api_agent["state"] != "running":
        api_agent["level"], api_agent["reason"] = "error", "api_agent_not_running"
    checks["collector_agent"] = collector
    checks["api_agent"] = api_agent
    checks["api_ready"] = _api_probe(port)

    path = history_path_for(status_path)
    if not path.is_file():
        checks["recurrence"] = {
            "level": "warning", "reason": "run_history_missing", "observed_runs": 0,
        }
    else:
        try:
            rows = recent_runs(path)
        except OSError:
            rows = []
        # A cycle can finish with no new articles; that still counts.
        observed = len(rows)
        if observed < minimum_runs:
            level, reason = "warning", "insufficient_history"
        elif len({row.get("finished_at") for row in rows}) < minimum_runs:
            level, reason = "warning", "timestamps_not_distinct"
        else:
            level, reason = "ok", "multiple_completed_cycles"
        checks["recurrence"] = {
            "level": level, "reason": reason, "observed_runs": observed,
        }
    levels = {item["level"] for item in checks.values()}
    overall = "error" if "error" in levels else "warning" if "warning" in levels else "ok"
    return {
        "status": overall, "checks": checks, "minimum_runs": minimum_runs,
        "modified_database": False, "modified_launchd": False,
        "collector_started": False, "api_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit macOS news services without modifying them")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--status", type=Path, default=base / "last-run.json")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--minimum-runs", type=int, default=2)
    args = parser.parse_args()
    import os

    try:
        result = audit(
            args.db.expanduser(), args.status.expanduser(),
            user_id=os.getuid(), port=args.port, minimum_runs=args.minimum_runs,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] == "error":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
