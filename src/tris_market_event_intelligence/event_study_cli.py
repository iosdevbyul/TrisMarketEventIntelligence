"""Local read-only CSV adapter for future DonghakStockVision price exports.

No repository writes, network access, API token or external ML dependency.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path

from tris_market_event_intelligence.event_study import (
    DailyClose,
    PublishedEvent,
    study_events,
)


def _rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV missing required headers: {', '.join(sorted(required))}")
        return list(reader)


def study_csv(
    events_path: Path, prices_path: Path, *, benchmark: str,
) -> dict[str, object]:
    events = [
        PublishedEvent(
            event_id=int(item["event_id"]), symbol=item["symbol"].strip(),
            published_at=datetime.fromisoformat(item["published_at"].strip()),
        )
        for item in _rows(events_path, {"event_id", "symbol", "published_at"})
    ]
    closes = [
        DailyClose(
            symbol=item["symbol"].strip(),
            day=date.fromisoformat(item["date"].strip()),
            adjusted_close=float(item["adjusted_close"]),
        )
        for item in _rows(prices_path, {"symbol", "date", "adjusted_close"})
    ]
    results = study_events(events, closes, benchmark_symbol=benchmark)
    return {
        "schema_version": "0.1", "calculation": "close_to_close_market_adjusted",
        "price_basis": "adjusted_close",
        "publication_policy": "first_shared_close_strictly_after_local_publication_date",
        "benchmark_symbol": benchmark, "quality_verified": False,
        "causal_effect_estimated": False,
        "label_count": sum(
            label.status == "COMPLETED" for event in results for label in event.labels
        ),
        "items": [event.as_dict() for event in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline historical event-study labels")
    parser.add_argument("--events", type=Path, required=True, help="Event CSV")
    parser.add_argument("--prices", type=Path, required=True, help="Adjusted price CSV")
    parser.add_argument("--benchmark", required=True, help="Market benchmark symbol")
    parser.add_argument("--output", type=Path, default=None, help="Optional new JSON output file")
    args = parser.parse_args()
    try:
        result = study_csv(args.events, args.prices, benchmark=args.benchmark.strip())
        if args.output is not None:
            # Never overwrite a label dataset by accident.
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({
                "status": "ok", "output": str(args.output),
                "events": len(result["items"]), "labels": result["label_count"],
            }))
        else:
            print(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, OverflowError) as exc:
        parser.exit(2, f"event-study: {exc}\n")


if __name__ == "__main__":
    main()
