"""Preview/create an opt-in launchd agent for the automatic Gemma pipeline.

Configuration creation never activates a background process. Do not run this
agent's collector concurrently with the legacy scheduled RSS collector.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import sys
from pathlib import Path

LABEL = "com.tris.market-event-intelligence-gemma-auto"


def configuration(*, python: Path, db: Path, model: str, interval: int,
                  skip_collection: bool = False) -> dict:
    from tris_market_event_intelligence.gemma_original_article_batch_cli import (
        MODEL_PATTERN,
    )

    if not MODEL_PATTERN.fullmatch(model):
        raise ValueError("Invalid model")
    if not 300 <= interval <= 86400:
        raise ValueError("Interval must be 300..86400")
    if not python.is_absolute() or not db.is_absolute() or db.is_symlink():
        raise ValueError("Absolute paths and non-symlink DB required")
    args = [
        str(python), "-m", "tris_market_event_intelligence.automatic_news_pipeline",
        "--db", str(db), "--model", model, "--watch", "--interval", str(interval),
    ]
    if skip_collection:
        args.append("--skip-collection")
    logs = Path.home() / "Library/Logs/TrisMarketEventIntelligence"
    return {
        "Label": LABEL, "ProgramArguments": args,
        "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(logs / "gemma-auto-stdout.log"),
        "StandardErrorPath": str(logs / "gemma-auto-stderr.log"),
        "ProcessType": "Background",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure opt-in Gemma auto-run")
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--interval", type=int, default=900)
    parser.add_argument("--skip-collection", action="store_true",
                        help="Existing scheduled collector owns news ingestion")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    db = args.db.expanduser().absolute()
    if not db.is_file():
        parser.error("Existing news database required")
    try:
        plist = configuration(python=Path(sys.executable).resolve(), db=db,
                              model=args.model, interval=args.interval,
                              skip_collection=args.skip_collection)
    except ValueError as exc:
        parser.error(str(exc))
    destination = Path.home() / "Library/LaunchAgents" / (LABEL + ".plist")
    if args.write:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(plistlib.dumps(plist))
        print(f"Created (NOT activated): {destination}")
        print("Before launchctl bootstrap, stop any competing collector or use "
              "--skip-collection if your existing collector is already active.")
    else:
        print(plistlib.dumps(plist).decode())


if __name__ == "__main__":
    main()
