"""Offline CSV boundary for KR/US minute-label research, no external requests."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from tris_market_event_intelligence.intraday_event_study import (
    IntradayNews,
    MinuteClose,
    study_intraday,
)


def _csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(required))}")
        return list(reader)


def intraday_csv(news_path: Path, prices_path: Path) -> dict[str, object]:
    """Read only explicitly adjusted, minute-end prices and timestamped news."""
    news = [
        IntradayNews(
            event_id=int(row["event_id"]),
            market=row["market"].strip(),
            symbol=row["symbol"].strip(),
            benchmark_symbol=row["benchmark_symbol"].strip(),
            published_at=datetime.fromisoformat(row["published_at"].strip()),
            received_at=datetime.fromisoformat(row["received_at"].strip()),
            available_at=datetime.fromisoformat(row["available_at"].strip()),
        )
        for row in _csv_rows(news_path, {
            "event_id", "market", "symbol", "benchmark_symbol",
            "published_at", "received_at", "available_at",
        })
    ]
    prices = []
    for row in _csv_rows(prices_path, {
        "market", "symbol", "close_at", "adjusted_close", "adjustment",
    }):
        if row["adjustment"].strip() != "adjusted":
            raise ValueError("Unadjusted or unknown intraday bars cannot form return labels")
        prices.append(MinuteClose(
            market=row["market"].strip(), symbol=row["symbol"].strip(),
            close_at=datetime.fromisoformat(row["close_at"].strip()),
            adjusted_close=float(row["adjusted_close"]),
        ))
    results = study_intraday(news, prices)
    return {
        "schema_version": "intraday_event_study_v1",
        "markets": ["XKRX", "XNYS", "XNAS"],
        "horizons_minutes": [1, 5, 15, 30, 60],
        "price_basis": "adjusted_minute_close",
        "entry_policy": "first_completed_minute_strictly_after_available_at",
        "market_hours": "regular_only; timezone-aware; holiday_calendar_unverified",
        "benchmark": "same_market_exact_minute",
        "quality_verified": False,
        "causal_effect_estimated": False,
        "prediction_model_trained": False,
        "completed_label_count": sum(
            label.status == "COMPLETED" for row in results for label in row.labels
        ),
        "items": [row.as_dict() for row in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline KR/US intraday news event study")
    parser.add_argument("--news", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = intraday_csv(args.news, args.prices)
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False))
        else:
            # Refuse existing artifacts, never overwrite prior experiment results.
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(json.dumps({
                "status": "ok", "output": str(args.output),
                "events": len(report["items"]),
                "completed_labels": report["completed_label_count"],
            }))
    except (ValueError, OSError, OverflowError) as exc:
        parser.exit(2, f"intraday-study: {exc}\n")


if __name__ == "__main__":
    main()
