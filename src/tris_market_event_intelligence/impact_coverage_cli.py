"""Inspect real-event impact coverage without starting Ollama or changing SQLite."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.impact_coverage import impact_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only impact analysis coverage")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument(
        "--db", type=Path,
        default=Path(os.getenv("MARKET_EVENT_DB", str(base / "news.sqlite3"))),
    )
    parser.add_argument(
        "--model-id", default=os.getenv("MARKET_EVENT_LLM_MODEL", "gemma2:latest"),
    )
    parser.add_argument("--prompt-version", default=None)
    parser.add_argument("--sample-limit", type=int, default=10)
    args = parser.parse_args()
    options = {"model_id": args.model_id, "sample_limit": args.sample_limit}
    if args.prompt_version:
        options["prompt_version"] = args.prompt_version
    try:
        report = impact_coverage(args.db.expanduser(), **options)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
