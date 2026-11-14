"""Offline no-write event lifecycle latency audit CLI."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.event_lifecycle_latency import audit_event_lifecycle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit first-receipt to event extraction and local model checkpoint",
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--as-of", default=None, help="Timezone-aware ISO audit time")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        observed = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
        report = audit_event_lifecycle(args.db, as_of=observed)
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False))
        else:
            with args.output.open("x", encoding="utf-8") as file:
                json.dump(report, file, ensure_ascii=False, indent=2)
                file.write("\n")
            print(json.dumps({
                "status": "audit_only", "report": str(args.output),
                "events": report["economic_events"],
                "checkpoints": report["model_checkpoints"],
            }))
    except (OSError, ValueError, TypeError, sqlite3.Error) as error:
        parser.exit(2, f"event-lifecycle-latency: {error}\n")


if __name__ == "__main__":
    main()
