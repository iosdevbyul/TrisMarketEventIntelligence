"""Conservative all-session SEC event & comparison-day research universe.

Existing, externally validated SEC event packets are the ONLY event evidence.
Unobserved events on other days are UNKNOWN, not genuine negative labels.
The prior-close to future-close outcomes are retrospective and not executable
preopen returns or sealed predictions. No training or model calls happen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from tris_market_event_intelligence.historical_issuer_corroboration_cli import (
    _read,
    _write,
)
from tris_market_event_intelligence.historical_shock_research import (
    NY,
    read_daily_prices,
)

MAX_EVENTS = 500
HORIZONS = (1, 3, 5)
ONE_MILLION = 1_000_000


def _hex_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _verify_event(
    packet: dict, *, source_sha: str,
    ticker: str, benchmark: str,
    expected_stock_sha: str, expected_market_sha: str,
    stock_by_day: dict, benchmark_by_day: dict,
) -> tuple[str, dict]:
    features, outcome = packet.get("features"), packet.get("outcome")
    if (
        packet.get("schemaVersion") != "historical_sec_earnings_price_event_v1"
        or packet.get("status") !=
        "SEC_FILING_PREOPEN_PRICE_OUTCOME_OBSERVED_RETROSPECTIVE"
        or packet.get("ticker") != ticker
        or packet.get("benchmark") != benchmark
        or packet.get("stockPriceSha256") != expected_stock_sha
        or packet.get("benchmarkPriceSha256") != expected_market_sha
        or not _hex_digest(packet.get("secEvidenceSha256"))
        or not _hex_digest(packet.get("corroborationSha256"))
        or not isinstance(features, dict)
        or not isinstance(outcome, dict)
        or packet.get("researchMode") != "RETROSPECTIVE_OUTCOME_ALREADY_KNOWN"
        or packet.get("modelPredictionMadeBeforeOutcome") is not False
        or packet.get("trainingReady") is not False
        or packet.get("causalImpactVerified") is not False
        or packet.get("secAcceptanceDoesNotProvePublisherFirstRelease") is not True
        or packet.get("secAcceptanceDoesNotProveFirstMarketAvailability") is not True
        or packet.get("modelPromotionAllowed") is not False
        or features.get("sourceType") != "SEC_8K_WITH_ISSUER_EXHIBIT"
        or features.get("sourceAvailableByDecisionCutoff") is not True
        or features.get("publisherFirstReleaseTimeVerified") is not False
        or features.get("gkgIssuerLinksVerified") is not False
        or outcome.get("returnWindow") != "PREVIOUS_CLOSE_TO_EVENT_DAY_CLOSE"
        or outcome.get("actualOutcomeUnavailableAtDecision") is not True
        or outcome.get("notAnExecutablePreopenEntryReturn") is not True
    ):
        raise ValueError("Unverified SEC event or source lineage")
    if not isinstance(packet.get("eventId"), str) or not packet["eventId"]:
        raise ValueError("Missing pinned SEC event identifier")
    day = outcome.get("session")
    if not isinstance(day, str) or day not in stock_by_day or day not in benchmark_by_day:
        raise ValueError("Unpriced event day")
    decision = datetime.fromisoformat(packet["decisionAt"])
    accepted = datetime.fromisoformat(features["secAcceptedAt"])
    if (decision.utcoffset() is None or accepted.utcoffset() is None
            or decision.astimezone(NY).date().isoformat() != day
            or decision.astimezone(NY).strftime("%H:%M") != "09:30"
            or accepted >= decision
            or accepted.astimezone(UTC).isoformat() !=
            features["secAcceptedAt"]):
        raise ValueError("Wrong decision cutoff or post-decision SEC event")
    if not isinstance(features.get("filingUrl"), str) or not features["filingUrl"].startswith(
        "https://www.sec.gov/Archives/edgar/data/"
    ):
        raise ValueError("SEC source URL missing")
    if not isinstance(features.get("exhibitUrl"), str) or not features["exhibitUrl"].startswith(
        "https://www.sec.gov/Archives/edgar/data/"
    ):
        raise ValueError("SEC exhibit URL missing")
    if not isinstance(features.get("issuerFacts"), list) or not features["issuerFacts"]:
        raise ValueError("Missing source-attested issuer facts")
    if features.get("previousSession") not in stock_by_day:
        raise ValueError("Missing prior session source price")
    prior_day = features["previousSession"]
    if prior_day not in benchmark_by_day:
        raise ValueError("Prior benchmark price missing")
    prior_stock = stock_by_day[prior_day]
    prior_market = benchmark_by_day[prior_day]
    actual_stock = stock_by_day[day]
    actual_market = benchmark_by_day[day]
    sr = actual_stock.adjusted_close / prior_stock.adjusted_close - 1
    mr = actual_market.adjusted_close / prior_market.adjusted_close - 1
    if (abs(sr - outcome.get("stockReturn", float("inf"))) > 1e-10
            or abs(mr - outcome.get("benchmarkReturn", float("inf"))) > 1e-10
            or abs((sr - mr) - outcome.get("excessReturn", float("inf"))) > 1e-10
            or abs(prior_stock.adjusted_close -
                   features["previousStockAdjustedClose"]) > 1e-10
            or abs(prior_market.adjusted_close -
                   features["previousBenchmarkAdjustedClose"]) > 1e-10):
        raise ValueError("Event source price truth mismatch")
    return day, {
        "eventId": packet["eventId"],
        "packetSha256": source_sha,
        "secAccession": features["secAccession"],
        "secAcceptedAt": features["secAcceptedAt"],
        "secFilingUrl": features["filingUrl"],
        "issuerFacts": features["issuerFacts"],
        "issuerLink": "VERIFIED_AS_SEC_ISSUER_DOCUMENT",
        "secAcceptedBeforeDecision": True,
        "publisherFirstReleaseClock": "UNKNOWN",
        "newsCausalityProven": False,
    }


def build_sec_event_universe(
    *, stock_csv: Path, benchmark_csv: Path, event_paths: list[Path],
    ticker: str = "NVDA", benchmark: str = "SPY",
) -> dict:
    if (not 1 <= len(event_paths) <= MAX_EVENTS
            or not ticker.isascii() or not ticker.isalnum()
            or not benchmark.isascii() or not benchmark.isalnum()
            or ticker == benchmark or len(ticker) > 12 or len(benchmark) > 12):
        raise ValueError("Requires bounded issuer/benchmark/SEC input list")
    stock, stock_sha = read_daily_prices(stock_csv)
    market, market_sha = read_daily_prices(benchmark_csv)
    stock_by_day = {bar.day.isoformat(): bar for bar in stock}
    market_by_day = {bar.day.isoformat(): bar for bar in market}
    stock_positions = {bar.day.isoformat(): index for index, bar in enumerate(stock)}
    market_positions = {bar.day.isoformat(): index for index, bar in enumerate(market)}
    pinned_events = {}
    event_ids: set[str] = set()
    for path in event_paths:
        packet, packet_sha = _read(path)
        day, event = _verify_event(
            packet, source_sha=packet_sha,
            ticker=ticker, benchmark=benchmark,
            expected_stock_sha=stock_sha,
            expected_market_sha=market_sha,
            stock_by_day=stock_by_day, benchmark_by_day=market_by_day,
        )
        if day in pinned_events or event["eventId"] in event_ids:
            raise ValueError("Duplicated SEC event session or eventId")
        pinned_events[day] = event
        event_ids.add(event["eventId"])
    records = []
    for prev, today in pairwise(stock):
        prev_day = prev.day.isoformat()
        day = today.day.isoformat()
        if prev_day not in market_by_day or day not in market_by_day:
            continue
        if market_positions[day] != market_positions[prev_day] + 1:
            continue
        cutoff = datetime.combine(
            today.day, datetime.min.time().replace(hour=9, minute=30),
            tzinfo=NY,
        ).astimezone(UTC).isoformat()
        evidence = pinned_events.get(day)
        # Crucial: no pinned SEC event means UNKNOWN, not negative NEWS gold.
        source_status = (
            "PINNED_SEC_FILING" if evidence else "NO_PINNED_SEC_EVIDENCE"
        )
        horizons = {}
        for horizon in HORIZONS:
            idx = stock_positions[day] + horizon - 1
            if idx >= len(stock):
                continue
            end_day = stock[idx].day.isoformat()
            market_idx = market_positions[day] + horizon - 1
            if market_idx >= len(market) or market[market_idx].day.isoformat() != end_day:
                continue
            if any(
                stock_positions.get(market[k].day.isoformat()) is None
                for k in range(market_positions[day], market_idx + 1)
            ):
                continue
            sr = stock[idx].adjusted_close / prev.adjusted_close - 1
            mr = market[market_idx].adjusted_close / market_by_day[prev_day].adjusted_close - 1
            horizons[str(horizon)] = {
                "targetSession": end_day,
                "stockReturn": sr,
                "benchmarkReturn": mr,
                "excessReturn": sr - mr,
                "directionVersusBenchmark": (
                    "UP" if sr > mr else "DOWN" if sr < mr else "FLAT"
                ),
            }
        if not horizons:
            continue
        records.append({
            "session": day,
            "decisionAt": cutoff,
            "features": {
                "previousSession": prev_day,
                "lastKnownStockAdjustedClose": prev.adjusted_close,
                "lastKnownBenchmarkAdjustedClose": (
                    market_by_day[prev_day].adjusted_close
                ),
                "secEvidenceStatus": source_status,
                "pinnedSecEvents": [evidence] if evidence else [],
                "otherCompanyOrMacroNewsAbsenceVerified": False,
            },
            "outcome": {
                "returnDefinition": "PREVIOUS_CLOSE_TO_TARGET_CLOSE",
                "includesPriceMovementBeforeDecision": True,
                "horizonsTradingSessions": horizons,
            },
            "noSecFilingDoesNotMeanNoNews": True,
            "trainingEligible": False,
            "prospectivePredictionMade": False,
        })
    with_event = sum(
        row["features"]["secEvidenceStatus"] == "PINNED_SEC_FILING"
        for row in records
    )
    if with_event != len(pinned_events) or not records:
        raise ValueError("SEC event cannot be joined to aligned priced sessions")
    return {
        "schemaVersion": "historical_sec_event_universe_v1",
        "status": "MULTI_SESSION_RETROSPECTIVE_SEC_EVIDENCE",
        "ticker": ticker,
        "benchmark": benchmark,
        "stockSha256": stock_sha,
        "benchmarkSha256": market_sha,
        "eventCount": len(pinned_events),
        "sessionCount": len(records),
        "sessionsWithPinnedSecEvent": with_event,
        "sessionsWithoutPinnedSecEvent": len(records) - with_event,
        "nonEventSessionsAreTrueNoNewsControls": False,
        "evidenceCoverageComplete": False,
        "trainingReady": False,
        "forwardPredictionsSealed": False,
        "actualModelSkillEstablished": False,
        "priceCausalityProven": False,
        "productionDbModified": False,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-session SEC event evidence universe")
    parser.add_argument("--stock-csv", required=True, type=Path)
    parser.add_argument("--benchmark-csv", required=True, type=Path)
    parser.add_argument("--event", required=True, action="append", type=Path)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    try:
        if not args.preview and args.output is None:
            raise ValueError("Output file required outside preview")
        report = build_sec_event_universe(
            stock_csv=args.stock_csv, benchmark_csv=args.benchmark_csv,
            event_paths=args.event, ticker=args.symbol,
            benchmark=args.benchmark,
        )
        if not args.preview:
            _write(args.output, report)
        print(json.dumps({
            "status": report["status"],
            "sessions": report["sessionCount"],
            "pinnedEvents": report["eventCount"],
            "unknownEvidenceSessions": report["sessionsWithoutPinnedSecEvent"],
            "trueNoNewsControlDays": 0,
            "trainingReady": False,
            "outputCreated": not args.preview,
        }))
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError):
        parser.exit(2, "historical-sec-universe: INVALID_OR_UNVERIFIED_INPUT\n")


if __name__ == "__main__":
    main()
