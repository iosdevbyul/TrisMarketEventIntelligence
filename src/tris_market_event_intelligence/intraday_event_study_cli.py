"""Offline CSV boundary for KR/US minute-label research, no external requests."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path

from tris_market_event_intelligence.event_ticker_screen import (
    EventTickerLink,
    screen_intraday_links,
)
from tris_market_event_intelligence.exchange_sessions import (
    SessionCalendar,
    SessionDay,
    TradingHalt,
)
from tris_market_event_intelligence.intraday_event_study import (
    IntradayNews,
    MinuteClose,
    study_intraday,
)
from tris_market_event_intelligence.walkforward_dataset import build_walkforward_dataset


def _csv_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(required))}")
        return list(reader)


def _calendar_csv(
    sessions_path: Path | None, halts_path: Path | None,
) -> SessionCalendar | None:
    if sessions_path is None:
        if halts_path is not None:
            raise ValueError("--halts requires an explicit --sessions file")
        return None
    sessions = []
    for row in _csv_rows(
        sessions_path, {"market", "date", "status", "open_at", "close_at"},
    ):
        status = row["status"].strip()
        sessions.append(SessionDay(
            market=row["market"].strip(), day=date.fromisoformat(row["date"].strip()),
            status=status,
            open_at=datetime.fromisoformat(row["open_at"].strip()) if row["open_at"].strip() else None,
            close_at=(
                datetime.fromisoformat(row["close_at"].strip())
                if row["close_at"].strip() else None
            ),
        ))
    halts = []
    if halts_path is not None:
        for row in _csv_rows(halts_path, {"market", "symbol", "start_at", "end_at"}):
            halts.append(TradingHalt(
                market=row["market"].strip(), symbol=row["symbol"].strip(),
                start_at=datetime.fromisoformat(row["start_at"].strip()),
                end_at=datetime.fromisoformat(row["end_at"].strip()),
            ))
    return SessionCalendar(sessions, halts)


def intraday_csv(
    news_path: Path, prices_path: Path, *,
    sessions_path: Path | None = None, halts_path: Path | None = None,
    links_path: Path | None = None,
    split_at: datetime | None = None, embargo_minutes: int = 60,
) -> dict[str, object]:
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
    calendar = _calendar_csv(sessions_path, halts_path)
    results = study_intraday(news, prices, calendar=calendar)
    screening = []
    if split_at is not None and (links_path is None or sessions_path is None):
        raise ValueError("Walk-forward split requires explicit --links and --sessions")
    links = []
    if links_path is not None:
        links = [
            EventTickerLink(
                event_id=int(row["event_id"]),
                market=row["market"].strip(), symbol=row["symbol"].strip(),
                benchmark_symbol=row["benchmark_symbol"].strip(),
                source_url=row["source_url"].strip(),
                evidence_seen_at=datetime.fromisoformat(row["evidence_seen_at"].strip()),
                relationship=row["relationship"].strip(),
            )
            for row in _csv_rows(links_path, {
                "event_id", "market", "symbol", "benchmark_symbol",
                "source_url", "evidence_seen_at", "relationship",
            })
        ]
        results, screening = screen_intraday_links(
            news, links, results, horizons=(1, 5, 15, 30, 60),
        )
    split_report = (
        build_walkforward_dataset(
            news, links, results, cutoff=split_at, embargo_minutes=embargo_minutes,
        ) if split_at is not None else None
    )
    return {
        "walkforward_dataset": split_report,
        "schema_version": "intraday_event_study_v1",
        "markets": ["XKRX", "XNYS", "XNAS"],
        "horizons_minutes": [1, 5, 15, 30, 60],
        "price_basis": "adjusted_minute_close",
        "entry_policy": "first_completed_minute_strictly_after_available_at",
        "market_hours": (
            "explicit_supplied_session_days; source_accuracy_not_independently_verified"
            if calendar is not None
            else "legacy_static_weekday_hours; holiday_calendar_unverified"
        ),
        "session_calendar_supplied": calendar is not None,
        "point_in_time_links_supplied": links_path is not None,
        "source_links_independently_verified": False,
        "training_dataset_eligible": False,
        "link_screening": screening,
        "session_calendar_source_verified": False,
        "trading_halts_supplied": halts_path is not None,
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
    parser.add_argument("--sessions", type=Path, default=None,
                        help="Reviewed open/closed market-date CSV; avoids static weekday assumptions")
    parser.add_argument("--halts", type=Path, default=None,
                        help="Optional date-time bounded stock/benchmark halt CSV")
    parser.add_argument("--links", type=Path, default=None,
                        help="Optional point-in-time event-to-ticker evidence CSV")
    parser.add_argument("--split-at", default=None,
                        help="Timezone-aware holdout cutoff for purged research samples")
    parser.add_argument("--embargo-minutes", type=int, default=60,
                        help="Temporal embargo >= max horizon (default: 60)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = intraday_csv(
            args.news, args.prices, sessions_path=args.sessions, halts_path=args.halts,
            links_path=args.links,
            split_at=datetime.fromisoformat(args.split_at) if args.split_at else None,
            embargo_minutes=args.embargo_minutes,
        )
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
