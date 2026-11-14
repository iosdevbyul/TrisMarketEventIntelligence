"""Inspect a previously selected 1–3-event Ollama pilot without rerunning it."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.impact_pilot_report import (
    parse_event_ids,
    pilot_outcomes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only per-event pilot outcome report")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument(
        "--db", type=Path,
        default=Path(os.getenv("MARKET_EVENT_DB", str(base / "news.sqlite3"))),
    )
    parser.add_argument("--event-ids", required=True, type=str)
    parser.add_argument(
        "--model-id", default=os.getenv("MARKET_EVENT_LLM_MODEL", "gemma2:latest")
    )
    parser.add_argument("--prompt-version", default=None)
    args = parser.parse_args()
    try:
        ids = parse_event_ids(args.event_ids)
        opts = {"event_ids": ids, "model_id": args.model_id}
        if args.prompt_version is not None:
            opts["prompt_version"] = args.prompt_version
        report = pilot_outcomes(args.db.expanduser(), **opts)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
