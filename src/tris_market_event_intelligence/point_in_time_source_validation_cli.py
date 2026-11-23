"""Read-only point-in-time market validation on user-supplied source artifacts.

Reuses the legacy hash-pinned minute audit and explicit exchange calendars.
A validated CSV format is NOT independently authenticated market provenance.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

from tris_market_event_intelligence.exchange_sessions import (
    SessionCalendar,
    SessionDay,
    TradingHalt,
    utc,
)
from tris_market_event_intelligence.market_minute_source import audit_market_source
from tris_market_event_intelligence.point_in_time_outcome import (
    EventClock,
    SealedForecast,
    TimestampedPrice,
    observe,
)

MAX_EVENTS = 2000
MAX_SESSIONS = 3000
MAX_BYTES = 8 * 1024 * 1024
EVENT_COLUMNS = {
    "article_id", "family_id", "published_at", "first_public_at",
    "detected_at", "novelty", "source_url", "source_verified",
    "target", "benchmark", "direction", "forecast_at", "horizon_end",
    "horizon_name", "mode", "model_version", "rule_version", "prediction_id",
    "market",
}
SESSION_COLUMNS = {"market", "date", "status", "open_at", "close_at"}


def rows(path: Path, required: set[str], max_rows: int) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Expected existing bounded non-symlink CSV")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV requires headers {sorted(required)}")
        result = []
        for item in reader:
            if len(result) >= max_rows:
                raise ValueError("Input exceeds row limit")
            if None in item or any(value is None for value in item.values()):
                raise ValueError("Malformed CSV")
            result.append(item)
        return result


def _time(value: str) -> datetime:
    return utc(datetime.fromisoformat(value.strip()))


def calendar_from_csv(
    sessions_csv: Path, halts_csv: Path | None,
) -> SessionCalendar:
    sessions = []
    for row in rows(sessions_csv, SESSION_COLUMNS, MAX_SESSIONS):
        sessions.append(SessionDay(
            market=row["market"], day=date.fromisoformat(row["date"]),
            status=row["status"],
            open_at=_time(row["open_at"]) if row["open_at"] else None,
            close_at=_time(row["close_at"]) if row["close_at"] else None,
        ))
    halts = []
    if halts_csv is not None:
        for row in rows(halts_csv, {"market", "symbol", "start_at", "end_at"},
                        MAX_SESSIONS):
            halts.append(TradingHalt(
                market=row["market"], symbol=row["symbol"],
                start_at=_time(row["start_at"]), end_at=_time(row["end_at"]),
            ))
    return SessionCalendar(sessions, halts)


def load_forecasts(path: Path) -> list[tuple[str, SealedForecast]]:
    result = []
    seen: set[str] = set()
    for row in rows(path, EVENT_COLUMNS, MAX_EVENTS):
        if row["source_verified"] not in {"true", "false"}:
            raise ValueError("source_verified must be literal true/false")
        event = EventClock(
            article_id=int(row["article_id"]), family_id=row["family_id"],
            published_at=_time(row["published_at"]),
            first_public_at=_time(row["first_public_at"]) if row["first_public_at"] else None,
            detected_at=_time(row["detected_at"]), novelty=row["novelty"],
            source_url=row["source_url"], source_verified=row["source_verified"] == "true",
        )
        forecast = SealedForecast(
            event=event, target=row["target"], benchmark=row["benchmark"],
            direction=row["direction"], forecast_at=_time(row["forecast_at"]),
            horizon_end=_time(row["horizon_end"]), horizon_name=row["horizon_name"],
            mode=row["mode"], model_version=row["model_version"],
            rule_version=row["rule_version"], prediction_id=row["prediction_id"],
        )
        if forecast.prediction_id in seen:
            raise ValueError("Duplicate prediction ID")
        seen.add(forecast.prediction_id)
        result.append((row["market"], forecast))
    return result


def load_prices(manifest_path: Path) -> tuple[list[TimestampedPrice], dict, dict]:
    audit = audit_market_source(manifest_path)
    if any(row["status"] != "SCHEMA_COMPATIBLE_SOURCE_UNVERIFIED"
           for row in audit["sources"]):
        raise ValueError("Price manifest contains nonadjusted or absent minute bars")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    bars: list[TimestampedPrice] = []
    availability: dict[tuple[str, datetime], datetime] = {}
    provider: dict[str, str] = {}
    market_by_symbol: dict[str, str] = {}
    for source in raw["sources"]:
        market = source["market"]
        filename = source["filename"]
        path = manifest_path.resolve().parent / filename
        for row in rows(path, {
            "market", "symbol", "close_at", "adjusted_close", "adjustment",
            "instrument_kind", "provider", "observed_at",
        }, 100_000):
            symbol = row["symbol"]
            if symbol in market_by_symbol and market_by_symbol[symbol] != market:
                raise ValueError("One instrument cannot silently span exchanges")
            market_by_symbol[symbol] = market
            stamp = _time(row["close_at"])
            available = _time(row["observed_at"])
            if available < stamp:
                raise ValueError("Observed price timestamp precedes bar end")
            if row["adjustment"] != "adjusted":
                raise ValueError("Unadjusted minute price")
            if symbol in provider and provider[symbol] != row["provider"]:
                raise ValueError("Mixed providers for one price series")
            provider[symbol] = row["provider"]
            availability[(symbol, stamp)] = available
            bars.append(TimestampedPrice(
                symbol=symbol, bar_end=stamp, adjusted_price=float(row["adjusted_close"]),
                provider=row["provider"], source_id=filename,
                session_id=stamp.date().isoformat(),
            ))
    return bars, availability, {"audit": audit, "market_by_symbol": market_by_symbol}


def _selected(
    prices: list[TimestampedPrice], symbol: str,
    before: datetime, after: datetime | None = None,
) -> TimestampedPrice | None:
    matching = [
        bar for bar in prices
        if bar.symbol == symbol and utc(bar.bar_end) <= before
        and (after is None or utc(bar.bar_end) > after)
    ]
    return max(matching, key=lambda x: utc(x.bar_end), default=None)


def evaluate_sources(
    predictions_csv: Path, source_manifest: Path, sessions_csv: Path,
    *, halts_csv: Path | None = None,
    max_anchor_age_minutes: int = 5,
    max_exit_age_minutes: int = 5,
) -> dict[str, object]:
    if not 1 <= max_anchor_age_minutes <= 4320 or not 1 <= max_exit_age_minutes <= 4320:
        raise ValueError("Price age windows must be 1..4320 explicit minutes")
    calendar = calendar_from_csv(sessions_csv, halts_csv)
    forecasts = load_forecasts(predictions_csv)
    bars, availability, info = load_prices(source_manifest)
    market_by_symbol = info["market_by_symbol"]
    items = []
    statuses: Counter[str] = Counter()
    for market, prediction in forecasts:
        event = prediction.event
        release = event.verified_cutoff
        if market_by_symbol.get(prediction.target) != market or (
            market_by_symbol.get(prediction.benchmark) != market
        ):
            status = "MARKET_MAPPING_UNVERIFIED"
        elif release is None:
            status = "SOURCE_FIRST_PUBLIC_UNVERIFIED"
        else:
            base_target = _selected(
                bars, prediction.target, release - timedelta(microseconds=1)
            )
            base_market = _selected(
                bars, prediction.benchmark, release - timedelta(microseconds=1)
            )
            end_target = _selected(
                bars, prediction.target, utc(prediction.horizon_end),
                utc(prediction.forecast_at),
            )
            end_market = _selected(
                bars, prediction.benchmark, utc(prediction.horizon_end),
                utc(prediction.forecast_at),
            )
            forecast_target = _selected(bars, prediction.target, utc(prediction.forecast_at))
            forecast_market = _selected(bars, prediction.benchmark, utc(prediction.forecast_at))
            selected = (base_target, base_market, end_target, end_market,
                        forecast_target, forecast_market)
            if any(p is None for p in selected):
                status = "PRICE_BAR_UNAVAILABLE"
            elif any(
                calendar.instant_status(market, p.bar_end) != "IN_SESSION" for p in selected
            ):
                status = "SESSION_UNVERIFIED"
            elif calendar.intersects_halt(
                market, (prediction.target, prediction.benchmark),
                min(p.bar_end for p in selected), max(p.bar_end for p in selected),
            ):
                status = "TRADING_HALT_OVERLAP"
            else:
                status = "READY_FOR_DESCRIPTIVE_EVALUATION"
        output: dict[str, object] = {
            "predictionId": prediction.prediction_id, "familyId": event.family_id,
            "articleId": event.article_id, "novelty": event.novelty, "status": status,
            "predictionMode": prediction.mode, "sourceAttestedByUser": event.source_verified,
            "independentlyVerifiedFirstPublic": False,
            "causalAttribution": "NOT_ESTABLISHED", "learningEligible": False,
        }
        if status == "READY_FOR_DESCRIPTIVE_EVALUATION":
            result = observe(
                prediction, bars, max_anchor_age=timedelta(minutes=max_anchor_age_minutes),
                max_exit_age=timedelta(minutes=max_exit_age_minutes),
                benchmark_tolerance=timedelta(minutes=1),
            )
            output["observation"] = result.as_dict()
            output["status"] = result.status
            # A historical download is not proof that a bar was accessible at
            # the original prediction cutoff. Never promote this replay to live.
            anchor_bars = (forecast_target, forecast_market)
            output["forecastBarsAvailableAtSeal"] = all(
                availability[(p.symbol, utc(p.bar_end))] <= utc(prediction.forecast_at)
                for p in anchor_bars
            )
            output["prospectiveScoreCertified"] = False
        statuses[output["status"]] += 1
        items.append(output)
    return {
        "schemaVersion": "market_learning_source_integration_v1",
        "rows": len(items), "statusCounts": dict(sorted(statuses.items())),
        "items": items, "sourceAudit": info["audit"],
        "exchangeCalendarIndependentlyVerified": False,
        "firstPublicIndependentlyVerified": False,
        "marketSourceIndependentlyVerified": False,
        "eventCausalityEstablished": False,
        "learningPerformanceDemonstrated": False,
        "trainingEligible": False,
        "anchorMaximumAgeMinutes": max_anchor_age_minutes,
        "exitMaximumAgeMinutes": max_exit_age_minutes,
        "independentFamilyCount": len({p.event.family_id for _, p in forecasts}),
        "nonNewArticles": sum(p.event.novelty != "NEW" for _, p in forecasts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only source-pinned market validation")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--halts", type=Path)
    parser.add_argument("--max-anchor-age-minutes", type=int, default=5)
    parser.add_argument("--max-exit-age-minutes", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate_sources(
            args.predictions, args.source_manifest, args.sessions, halts_csv=args.halts,
            max_anchor_age_minutes=args.max_anchor_age_minutes,
            max_exit_age_minutes=args.max_exit_age_minutes,
        )
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({"rows": report["rows"], "statusCounts": report["statusCounts"],
                          "report": str(args.output)}, ensure_ascii=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"market-learning-source-validation: {exc}\n")


if __name__ == "__main__":
    main()
