"""Build an opt-in, separate macOS LaunchAgent for the local read-only API.

Never activate, unload, or alter the existing news collection LaunchAgent.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import sys
from pathlib import Path

API_LABEL = "com.tris.market-event-intelligence-api"


def api_launchd_configuration(
    python_path: Path,
    *,
    db_path: Path,
    port: int = 8100,
) -> dict[str, object]:
    if not python_path.is_absolute() or not python_path.is_file():
        raise ValueError("Python executable path must exist and be absolute")
    if not db_path.is_absolute():
        raise ValueError("Database path must be absolute")
    if not 1 <= port <= 65535:
        raise ValueError("Port must be 1..65535")
    logs = Path.home() / "Library" / "Logs" / "TrisMarketEventIntelligence"
    return {
        "Label": API_LABEL,
        "ProgramArguments": [
            str(python_path), "-m", "tris_market_event_intelligence.api_service",
            "--db", str(db_path), "--port", str(port),
        ],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Background",
        "StandardOutPath": str(logs / "api-stdout.log"),
        "StandardErrorPath": str(logs / "api-stderr.log"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate opt-in local API LaunchAgent plist")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    target = Path.home() / "Library" / "LaunchAgents" / (API_LABEL + ".plist")
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--write", action="store_true",
        help="Write plist only, without installing or enabling the API service",
    )
    args = parser.parse_args()
    try:
        cfg = api_launchd_configuration(
            Path(sys.executable).resolve(),
            db_path=args.db.expanduser().resolve(), port=args.port,
        )
    except ValueError as exc:
        parser.error(str(exc))
    payload = plistlib.dumps(cfg)
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        os.chmod(target, 0o600)
        print(f"API plist created but NOT activated: {target}")
    else:
        print(payload.decode())
    print("To activate: run an explicit launchctl bootstrap command on the Mac.")


if __name__ == "__main__":
    main()
