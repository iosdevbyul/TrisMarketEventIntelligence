"""Generate an unprivileged, opt-in launchd LaunchAgent plist for macOS.

Never invoke launchctl from Python or automatically install a background job.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import sys
from pathlib import Path

LABEL = "com.tris.market-event-intelligence"


def launchd_configuration(
    python_path: Path, *, db_path: Path, status_path: Path,
    interval_seconds: int = 1800,
) -> dict[str, object]:
    if interval_seconds < 900 or interval_seconds > 86400:
        raise ValueError("Interval must be 900..86400 seconds")
    paths = (python_path, db_path, status_path)
    if any(not path.is_absolute() for path in paths):
        raise ValueError("All launchd paths must be absolute")
    logs = Path.home() / "Library" / "Logs" / "TrisMarketEventIntelligence"
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(python_path), "-m", "tris_market_event_intelligence.cycle_cli",
            "--db", str(db_path), "--status", str(status_path),
        ],
        "StartInterval": interval_seconds,
        "RunAtLoad": False,
        "StandardOutPath": str(logs / "stdout.log"),
        "StandardErrorPath": str(logs / "stderr.log"),
        "ProcessType": "Background",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate opt-in macOS launchd config")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    target = Path.home() / "Library" / "LaunchAgents" / (LABEL + ".plist")
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--status", type=Path, default=base / "last-run.json")
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument(
        "--write", action="store_true",
        help="Write plist only; does not load or activate a LaunchAgent",
    )
    args = parser.parse_args()
    try:
        config = launchd_configuration(
            Path(sys.executable).resolve(), db_path=args.db.expanduser().resolve(),
            status_path=args.status.expanduser().resolve(),
            interval_seconds=args.interval,
        )
    except ValueError as exc:
        parser.error(str(exc))
    payload = plistlib.dumps(config)
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        os.chmod(target, 0o600)
        print(f"Plist created but NOT activated: {target}")
    else:
        print(payload.decode())
    print("Activation requires a separate explicit launchctl bootstrap command.")


if __name__ == "__main__":
    main()
