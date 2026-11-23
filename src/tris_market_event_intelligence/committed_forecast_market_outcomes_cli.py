"""Read-only genuine daily-price outcomes for locally precommitted Gemma forecasts.

A local SQLite hash-chain is NOT third-party timestamp evidence. This is
descriptive *candidate* forward observation, not authenticated prospective
accuracy. Neither source news, saved commitments nor quotes are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.automatic_forecast_precommitment import (
    CUTOFF,
    ELIGIBLE,
    verify_chain,
)
from tris_market_event_intelligence.market_outcome_validation_cli import (
    NEUTRAL_THRESHOLD,
    ZONES,
    _timestamp,
)
from tris_market_event_intelligence.market_price_recovery_cli import _observations

MAX_ROWS = 10000
MAX_BYTES = 40 * 1024 * 1024
HORIZONS = {1, 5}


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Bounded existing regular price file required")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direction(value: float) -> str:
    if value > NEUTRAL_THRESHOLD:
        return "UP"
    if value < -NEUTRAL_THRESHOLD:
        return "DOWN"
    return "NEUTRAL"


def _clock(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    try:
        return _timestamp(raw).astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def _source_identity(db: sqlite3.Connection, commit: sqlite3.Row) -> str | None:
    """Fail closed if stored model answer or current original body was replaced."""
    matched = db.execute(
        """SELECT n.status, n.processed_at, n.analysis_json,
                  f.body, f.body_sha256, f.status AS body_status,\n                  f.observed_at AS body_observed_at,
                  a.published_at, a.detected_at
           FROM automatic_news_analyses n
           JOIN article_full_text f ON f.article_id=n.article_id
           JOIN articles a ON a.id=n.article_id
           WHERE n.article_id=? AND n.body_sha256=? AND n.model=?""",
        (commit["article_id"], commit["body_sha256"], commit["model"]),
    ).fetchone()
    if matched is None:
        return "ORIGINAL_MODEL_OR_BODY_NOT_FOUND"
    if (matched["status"] != "COMPLETED"
            or matched["body_status"] != "COMPLETE"
            or matched["body_sha256"] != commit["body_sha256"]\n            or matched["body_observed_at"] != commit["original_body_observed_at"]
            or not isinstance(matched["body"], str)
            or hashlib.sha256(matched["body"].encode()).hexdigest() != commit["body_sha256"]
            or matched["processed_at"] != commit["analyzed_at"]
            or matched["published_at"] != commit["publisher_stated_at"]
            or matched["detected_at"] != commit["detected_at"]):
        return "ORIGINAL_MODEL_OR_BODY_REVISED"
    try:
        payload = json.loads(matched["analysis_json"])
        impact = payload["impacts"][commit["impact_index"]]
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        raise ValueError("Corrupt original stored Gemma answer") from exc
    if commit["evidence_quote"] not in matched["body"]:\n        return "ORIGINAL_EVIDENCE_QUOTE_NOT_IN_STORED_BODY"\n    if any(impact[key] != commit[field] for key, field in (
        ("target", "target"), ("direction", "direction"),
        ("reason", "rationale"), ("evidence_quote", "evidence_quote"),
    )):
        return "COMMITTED_FORECAST_DOES_NOT_MATCH_ORIGINAL_MODEL"
    return None


def _bar_prices(
    commit: sqlite3.Row, prices: dict[str, list[dict[str, str]]],
) -> dict:
    symbol, benchmark = commit["symbol"], commit["benchmark"]
    if not symbol or not benchmark or symbol == benchmark:
        return {"status": "NO_PRECOMMITTED_INSTRUMENT"}
    if commit["market"] not in ZONES:
        return {"status": "UNSUPPORTED_ORIGINAL_MARKET"}
    publication = _clock(commit["publisher_stated_at"])
    sealed = _clock(commit["sealed_at"])
    if publication is None or sealed is None:
        return {"status": "SOURCE_CLOCK_NOT_ADMISSIBLE"}
    zone = ZoneInfo(ZONES[commit["market"]])
    publication_day = publication.astimezone(zone).date()
    sealed_day = sealed.astimezone(zone).date()
    if sealed_day > publication_day:
        return {"status": "SEAL_AFTER_SOURCE_DAY"}
    stock = prices.get(symbol)
    bench = prices.get(benchmark)
    if not stock or not bench:
        return {"status": "SYMBOL_OR_BENCHMARK_PRICES_MISSING"}
    # Only a strictly *later local exchange trading date* is eligible.
    # This deliberately discards any contemporaneous news-day gap.
    candidate = [row for row in stock if date.fromisoformat(row["date"]) > publication_day]
    if not candidate:
        return {"status": "NO_POST_PUBLICATION_ENTRY_CLOSE"}
    entry = candidate[0]
    entry_day = date.fromisoformat(entry["date"])
    if entry_day <= sealed_day:
        return {"status": "SEALED_AFTER_OR_DURING_ENTRY_DAY"}
    bench_by_day = {row["date"]: row for row in bench}
    if entry["date"] not in bench_by_day:
        return {"status": "BENCHMARK_ENTRY_CLOSE_MISSING", "entryDay": entry["date"]}
    horizon = commit["horizon_sessions"]
    if horizon >= len(candidate):
        return {"status": "INSUFFICIENT_LATER_SESSIONS", "entryDay": entry["date"]}
    exit_row = candidate[horizon]
    if exit_row["date"] not in bench_by_day:
        return {
            "status": "BENCHMARK_EXIT_CLOSE_MISSING",
            "entryDay": entry["date"], "exitDay": exit_row["date"],
        }
    stock_return = float(exit_row["adjusted_close"]) / float(entry["adjusted_close"]) - 1
    bench_return = (
        float(bench_by_day[exit_row["date"]]["adjusted_close"])
        / float(bench_by_day[entry["date"]]["adjusted_close"]) - 1
    )
    relative_return = stock_return - bench_return
    prediction = commit["direction"]
    if prediction == "UNCERTAIN":
        verdict = relative_verdict = "ABSTAINED"
    else:
        verdict = "CORRECT" if prediction == _direction(stock_return) else "INCORRECT"
        relative_verdict = (
            "CORRECT" if prediction == _direction(relative_return) else "INCORRECT"
        )
    return {
        "status": "OBSERVED_UNAUTHENTICATED_PRECOMMITMENT",
        "entryDay": entry["date"], "exitDay": exit_row["date"],
        "stockReturn": stock_return, "benchmarkReturn": bench_return,
        "excessReturn": relative_return,
        "actualDirection": _direction(stock_return),
        "relativeActualDirection": _direction(relative_return),
        "verdict": verdict, "relativeVerdict": relative_verdict,
    }


def evaluate(db_path: Path, prices_path: Path, price_manifest: Path | None = None) -> dict:
    if not db_path.is_file() or db_path.is_symlink():
        raise ValueError("Existing actual news SQLite required")
    prices_sha = _sha(prices_path)
    if price_manifest is not None:
        _sha(price_manifest)
        manifest = json.loads(price_manifest.read_text(encoding="utf-8"))
        if manifest.get("outputSha256") != prices_sha:
            raise ValueError("Price dataset and manifest digest mismatch")
    price_rows = _observations(prices_path)
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        has_table = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='automatic_forecast_commitments'"
        ).fetchone()
        if has_table is None:
            return {
                "schemaVersion": "committed_forecast_daily_price_evaluation_v1",
                "status": "NOT_OPTED_IN", "rows": [],
                "committedRows": 0, "priceObservations": sum(map(len, price_rows.values())),
                "prospectivePredictionAccuracy": None, "trainingEligible": False,
                "externalTimestampAuthenticated": False,
            }
        chain = verify_chain(db)
        commitments = db.execute(
            "SELECT * FROM automatic_forecast_commitments ORDER BY rowid ASC"
        ).fetchall()
        if len(commitments) > MAX_ROWS:
            raise ValueError("Too many local commitments in bounded read-only report")
        result_rows = []
        status_counts: Counter[str] = Counter()
        verdict_counts: dict[int, Counter[str]] = {
            h: Counter() for h in HORIZONS
        }
        for item in commitments:
            admission = _source_identity(db, item)
            seal = _clock(item["sealed_at"])
            published = _clock(item["publisher_stated_at"])
            detected = _clock(item["detected_at"])
            original = _clock(item["original_body_observed_at"])
            if item["eligibility"] != ELIGIBLE:
                verdict = {"status": item["eligibility"]}
            elif item["market_cutoff_policy"] != CUTOFF:
                verdict = {"status": "COMMITMENT_CUTOFF_POLICY_CHANGED"}
            elif admission is not None:
                verdict = {"status": admission}
            elif (sealed is None or published is None or detected is None\n                  or original is None
                  or published > detected or detected > sealed
                  or (original is not None and original > sealed)):
                verdict = {"status": "COMMITMENT_CLOCK_ORDER_INVALID"}
            elif item["horizon_sessions"] not in HORIZONS:
                verdict = {"status": "UNSUPPORTED_COMMITTED_HORIZON"}
            else:
                verified = _clock(item["mapping_verified_at"])
                if verified is None or verified > sealed:
                    verdict = {"status": "MAPPING_WAS_NOT_VERIFIED_AT_SEAL"}
                else:
                    verdict = _bar_prices(item, price_rows)
            status_counts[verdict["status"]] += 1
            if verdict.get("verdict"):
                verdict_counts[item["horizon_sessions"]][verdict["verdict"]] += 1
            result_rows.append({
                "articleId": item["article_id"],
                "model": item["model"], "bodySha256": item["body_sha256"],
                "impactIndex": item["impact_index"], "horizonSessions": item["horizon_sessions"],
                "commitmentSha256": item["forecast_sha256"],
                "publisherStatedAt": item["publisher_stated_at"],
                "sealedAt": item["sealed_at"],
                "target": item["target"], "direction": item["direction"],
                "market": item["market"], "symbol": item["symbol"],
                "benchmark": item["benchmark"], "precommitmentEligibility": item["eligibility"],
                "sourceIdentity": admission or "MATCHED_ORIGINAL_MODEL_AND_BODY",
                **verdict,
            })
    horizons = {}
    for horizon, counts in sorted(verdict_counts.items()):
        total = counts["CORRECT"] + counts["INCORRECT"]
        horizons[f"{horizon}d"] = {
            "correct": counts["CORRECT"], "incorrect": counts["INCORRECT"],
            "abstained": counts["ABSTAINED"],
            "denominator": total,
            "candidateLocalAgreement": counts["CORRECT"] / total if total else None,
            "independentlyAuthenticatedProspectiveAccuracy": None,
        }
    return {
        "schemaVersion": "committed_forecast_daily_price_evaluation_v1",
        "status": "COMPLETED",
        "committedRows": len(result_rows), "priceSha256": prices_sha,
        "priceManifestSha256": _sha(price_manifest) if price_manifest else None,
        "priceObservations": sum(map(len, price_rows.values())),
        "commitmentChainHeadSha256": chain["chainHeadSha256"],
        "commitmentChainVerifiedLocally": True,
        "statusCounts": dict(sorted(status_counts.items())),
        "horizons": horizons, "rows": result_rows,
        "marketPricesAnalyzed": True,
        "firstPublicAvailabilityIndependentlyVerified": False,
        "externalTimestampAuthenticated": False,
        "historicalTamperProof": False,
        "prospectivePredictionAccuracy": None,
        "trainingEligible": False, "productionDatabaseModified": False,
        "causalityEstablished": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--price-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Result path exists; immutable original results preserved")
        report = evaluate(args.db, args.prices, args.price_manifest)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps({
            "status": report["status"],
            "committedRows": report["committedRows"],
            "statusCounts": report.get("statusCounts", {}),
            "horizons": report.get("horizons", {}),
            "prospectivePredictionAccuracy": None,
            "output": str(args.output),
        }, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        parser.exit(2, f"committed-forecast-real-market: {exc}\n")


if __name__ == "__main__":
    main()
