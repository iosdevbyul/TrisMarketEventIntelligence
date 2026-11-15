"""SEC-accepted earnings exhibit joined to a historical price outcome.

SEC's 8-K accepted time proves a filing receipt time, NOT the first
publisher release or the exact first moment every investor could trade.
A retrospective shock-study record is not a sealed prospective forecast.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.historical_issuer_corroboration_cli import (
    _read,
    _write,
)
from tris_market_event_intelligence.historical_shock_research import (
    read_daily_prices,
)

NY = ZoneInfo("America/New_York")
FILING = (
    "https://www.sec.gov/Archives/edgar/data/1045810/"
    "0001045810-18-000148-index.htm"
)
EXHIBIT = (
    "https://www.sec.gov/Archives/edgar/data/1045810/"
    "000104581018000148/q3fy19pr.htm"
)


def build_sec_earnings_price_event(
    *, corroboration: dict, corroboration_sha: str, sec: dict,
    sec_sha: str, stock_csv: Path, benchmark_csv: Path,
) -> dict[str, object]:
    if (corroboration.get("schemaVersion") != "historical_source_corrob_review_v1"
            or corroboration.get("status") !=
            "OFFICIAL_DATE_AND_FACTS_CORROBORATED_CLOCK_UNVERIFIED"
            or corroboration.get("ticker") != "NVDA"
            or corroboration.get("shockDate") != "2018-11-16"
            or corroboration.get("priceImpactCausalityVerified") is not False
            or corroboration.get("replayTrainingEligible") is not False
            or corroboration.get("issuerSource", {}).get(
                "publisherFirstPublicationClockVerified"
            ) is not False):
        raise ValueError("Unverified historic corroboration contract required")
    if (sec.get("schemaVersion") != "historical_sec_acceptance_evidence_v1"
            or sec.get("ticker") != "NVDA"
            or sec.get("eventId") != corroboration.get("eventId")
            or sec.get("secAccession") != "0001045810-18-000148"
            or sec.get("secFilingIndex") != FILING
            or sec.get("secExhibit99_1") != EXHIBIT
            or sec.get("formType") != "8-K"
            or sec.get("sourceDocumentType") != "EX-99.1"
            or sec.get("exhibitAttachedToAcceptedFiling") is not True
            or sec.get("acceptanceTimeEvidence") != "SEC_EDGAR_ACCEPTED"
            or sec.get("secAcceptanceIsPublisherInitialRelease") is not False
            or sec.get("secAcceptanceIsVerifiedTradableMarketAvailability")
            is not False
            or sec.get("causalPriceEffectEstablished") is not False):
        raise ValueError("SEC first-publication or issuer evidence mismatch")
    accepted = datetime.fromisoformat(sec["form8kAcceptedEastern"])
    expected = datetime(2018, 11, 15, 16, 21, 48, tzinfo=NY)
    if accepted.utcoffset() is None or accepted != expected:
        raise ValueError("SEC acceptance clock must match original SEC index")
    session = date(2018, 11, 16)
    cutoff = datetime.combine(session, time(9, 30), tzinfo=NY)
    if (not accepted < cutoff
            or accepted < datetime(2018, 11, 15, 16, 0, tzinfo=NY)):
        raise ValueError("SEC filing cannot leak across the decision cutoff")
    stock, stock_sha = read_daily_prices(stock_csv)
    market, benchmark_sha = read_daily_prices(benchmark_csv)
    stock_idx = next((i for i, bar in enumerate(stock) if bar.day == session), None)
    market_idx = next((i for i, bar in enumerate(market) if bar.day == session), None)
    if stock_idx is None or market_idx is None or stock_idx == 0 or market_idx == 0:
        raise ValueError("Expected actual stock and market session with prior close")
    stock_prev, stock_today = stock[stock_idx - 1], stock[stock_idx]
    market_prev, market_today = market[market_idx - 1], market[market_idx]
    if stock_prev.day != market_prev.day:
        raise ValueError("Incompatible historical trading calendars")
    stock_return = stock_today.adjusted_close / stock_prev.adjusted_close - 1
    market_return = market_today.adjusted_close / market_prev.adjusted_close - 1
    return {
        "schemaVersion": "historical_sec_earnings_price_event_v1",
        "status": "SEC_FILING_PREOPEN_PRICE_OUTCOME_OBSERVED_RETROSPECTIVE",
        "ticker": "NVDA",
        "benchmark": "SPY",
        "eventId": sec["eventId"],
        "researchMode": "RETROSPECTIVE_OUTCOME_ALREADY_KNOWN",
        "corroborationSha256": corroboration_sha,
        "secEvidenceSha256": sec_sha,
        "stockPriceSha256": stock_sha,
        "benchmarkPriceSha256": benchmark_sha,
        "decisionAt": cutoff.astimezone(UTC).isoformat(),
        "features": {
            "sourceType": "SEC_8K_WITH_ISSUER_EXHIBIT",
            "secAcceptedAt": accepted.astimezone(UTC).isoformat(),
            "sourceAvailableByDecisionCutoff": True,
            "secAccession": sec["secAccession"],
            "filingUrl": FILING,
            "exhibitUrl": EXHIBIT,
            "issuerDocumentAvailableAsSecExhibit": True,
            "issuerFacts": corroboration["issuerSource"]["facts"],
            "publisherFirstReleaseTimeVerified": False,
            "gkgCandidateUrlsExcluded": corroboration["gkgCandidateCount"],
            "gkgIssuerLinksVerified": False,
            "previousSession": stock_prev.day.isoformat(),
            "previousStockAdjustedClose": stock_prev.adjusted_close,
            "previousBenchmarkAdjustedClose": market_prev.adjusted_close,
        },
        "outcome": {
            "session": session.isoformat(),
            "returnWindow": "PREVIOUS_CLOSE_TO_EVENT_DAY_CLOSE",
            "stockReturn": stock_return,
            "benchmarkReturn": market_return,
            "excessReturn": stock_return - market_return,
            "directionVersusBenchmark": (
                "UP" if stock_return > market_return else
                "DOWN" if stock_return < market_return else "FLAT"
            ),
            "actualOutcomeUnavailableAtDecision": True,
            "notAnExecutablePreopenEntryReturn": True,
        },
        "secAcceptanceDoesNotProvePublisherFirstRelease": True,
        "secAcceptanceDoesNotProveFirstMarketAvailability": True,
        "publisherArticleUrlsVerified": 0,
        "causalImpactVerified": False,
        "modelPredictionMadeBeforeOutcome": False,
        "trainingReady": False,
        "prospectiveAccuracyMeasured": False,
        "modelPromotionAllowed": False,
        "productionDbModified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One SEC filing's witnessed time to EOD event outcome")
    parser.add_argument("--corroboration", type=Path, required=True)
    parser.add_argument("--sec", type=Path, required=True)
    parser.add_argument("--stock-csv", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        corroboration, corroboration_sha = _read(args.corroboration)
        sec, sec_sha = _read(args.sec)
        result = build_sec_earnings_price_event(
            corroboration=corroboration, corroboration_sha=corroboration_sha,
            sec=sec, sec_sha=sec_sha, stock_csv=args.stock_csv,
            benchmark_csv=args.benchmark_csv,
        )
        _write(args.output, result)
        print(json.dumps({
            "status": result["status"],
            "decisionAt": result["decisionAt"],
            "secAcceptedAt": result["features"]["secAcceptedAt"],
            "eventSession": result["outcome"]["session"],
            "stockReturn": result["outcome"]["stockReturn"],
            "benchmarkReturn": result["outcome"]["benchmarkReturn"],
            "excessReturn": result["outcome"]["excessReturn"],
            "gkgUrlsNotPromoted": result["features"]["gkgCandidateUrlsExcluded"],
            "trainingReady": False,
            "predictionReady": False,
            "outputCreated": True,
        }))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        parser.exit(2, "historical-sec-event: INVALID_SOURCE_OR_PIT_EVIDENCE\n")


if __name__ == "__main__":
    main()
