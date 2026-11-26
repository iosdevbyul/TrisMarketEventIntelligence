"""Compare unchanged sealed replay hypotheses with original-declaration-date price windows.

This is a retrospective sensitivity study, NOT a verified historical forecast.
The original seal used observation-date semantics; the changed anchor is explicit.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.independent_event_replay_cli import (
    VERSION as SEAL_VERSION,
)
from tris_market_event_intelligence.independent_event_replay_cli import (
    evaluate_row,
    write_new,
)
from tris_market_event_intelligence.market_outcome_validation_cli import ZONES
from tris_market_event_intelligence.market_price_recovery_cli import _observations
from tris_market_event_intelligence.unified_event_integration_cli import clock, digest, encode

VERSION = "original_event_date_comparison_v1"


def original_anchor(record):
    field = "secAcceptanceAt" if record["kind"] == "SEC" else "publisherDeclaredAt"
    declared = clock(record.get(field))
    observed = clock(record.get("observedAt"))
    if declared is None or observed is None or declared > observed:
        return None, field
    return declared, field


def coverage(series):
    return {"sessions": len(series), "firstDay": series[0]["date"] if series else None,
            "lastDay": series[-1]["date"] if series else None}


def compare_row(record, prices):
    anchor, field = original_anchor(record)
    mapping = record.get("mapping")
    common = {"anchorField": field, "originalObservedAt": record["observedAt"],
              "historicalSourceVersionVerified": False, "predictionHorizonReanchored": True}
    if not mapping:
        return [{**common, "status": "ABSTAINED_NO_TARGET"}]
    stock = prices.get(mapping["symbol"], [])
    bench = prices.get(mapping["benchmark"], [])
    common.update({"stockCoverage": coverage(stock), "benchmarkCoverage": coverage(bench)})
    if anchor is None:
        return [{**common, "status": "ORIGINAL_DECLARED_CLOCK_INVALID"}]
    day = anchor.astimezone(ZoneInfo(ZONES[mapping["market"]])).date()
    observed = clock(record["observedAt"])
    common.update({"anchorAt": anchor.isoformat(), "anchorLocalDay": day.isoformat(),
                   "collectionLagSeconds": (observed - anchor).total_seconds(),
                   "historicalMappingAvailableAtAnchor": False})
    if not stock or not bench:
        return [{**common, "status": "MISSING_INSTRUMENT_OR_BENCHMARK"}]
    # Do not silently treat a much later first available bar as the next session.
    # A pre-anchor benchmark bar is necessary, not sufficient, to establish coverage.
    if any(date.fromisoformat(series[0]["date"]) > day for series in (stock, bench)):
        return [{**common, "status": "PRICE_HISTORY_STARTS_AFTER_ANCHOR",
                 "horizonSessions": h, "prediction": record["answer"][f"direction_{h}d"]}
                for h in (1, 5)]
    # Reuse the unchanged horizon evaluator; only a temporary copy is reanchored.
    labels = evaluate_row({**record, "observedAt": anchor.isoformat()}, prices)
    return [{**common, **label} for label in labels]


def compare(seal_path, prices_path, manifest_path=None):
    sealed = json.loads(seal_path.read_text(encoding="utf-8"))
    checksum = sealed.pop("sealSha256")
    if (sealed.get("version") != SEAL_VERSION or checksum != digest(encode(sealed))
            or sealed.get("prospectivePrediction") is not False
            or sealed.get("marketReturnsAnalyzed") is not False):
        raise ValueError("Invalid original forecast seal")
    price_sha = digest(prices_path.read_bytes())
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("outputSha256") != price_sha:
            raise ValueError("Price manifest mismatch")
    prices = _observations(prices_path)
    rows = []
    for record in sealed["records"]:
        if record["status"] != "CANDIDATE":
            continue
        for label in compare_row(record, prices):
            rows.append({"eventId": record["eventId"], "kind": record["kind"],
                         "title": record["title"], "target": record["answer"]["target"],
                         "economicEffect": record["answer"]["economic_effect"],
                         "mapping": record["mapping"], **label})
    observed = [r for r in rows if r["status"] == "OBSERVED_RETROSPECTIVE"]
    scored = [r for r in observed if r["verdict"] != "ABSTAINED"]
    return {"version": VERSION, "forecastSealSha256": checksum, "priceSha256": price_sha,
            "priceManifestVerified": manifest_path is not None,
            "modelCalls": 0, "originalForecastsModified": False,
            "comparisonMode": "REANCHORED_HISTORICAL_SENSITIVITY_NOT_ORIGINAL_FORECAST_ACCURACY",
            "rows": rows, "statusCounts": dict(Counter(r["status"] for r in rows)),
            "verdictCounts": dict(Counter(r["verdict"] for r in observed)),
            "relativeVerdictCounts": dict(Counter(r["relativeVerdict"] for r in observed)),
            "observedRows": len(observed), "scoredRows": len(scored),
            "marketReturnsAnalyzed": bool(observed), "prospectivePrediction": False,
            "trainingEligible": False, "firstPublicVerified": False,
            "historicalSourceVersionVerified": False, "calendarCompletenessVerified": False,
            "causalAttributionVerified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seal", "prices", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--price-manifest", type=Path)
    args = parser.parse_args()
    result = compare(args.seal, args.prices, args.price_manifest)
    write_new(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
