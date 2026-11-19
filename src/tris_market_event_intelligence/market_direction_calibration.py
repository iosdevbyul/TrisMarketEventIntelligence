"""Frozen price-only direction calibration learned from the training journal.

This is an exploratory price-coincidence calibration, not event causality.
It cannot use any held-out outcomes. Predictions abstain if no well-supported
conditional history exists. The training record is read in SQLite read-only
mode and must precede the first held-out release and its outcomes.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tris_market_event_intelligence.sequential_market_learning_cli import asset_market, timestamp

DIRECTIONS = ("NEGATIVE", "NEUTRAL", "POSITIVE")


def fit_price_calibrator(
    journal: Path, first_holdout_release: datetime, *, minimum_support: int = 8,
) -> dict[str, Any]:
    if journal.is_symlink() or not journal.is_file():
        raise ValueError("Original training journal required")
    if first_holdout_release.utcoffset() is None:
        raise ValueError("Holdout release requires timezone")
    if minimum_support < 2:
        raise ValueError("Minimum support must be at least two")
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    cases = 0
    skipped_unmatured = 0
    skipped_unverified = 0
    with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute(
            "SELECT observed_at, prediction_json, evaluation_json "
            "FROM steps WHERE phase='COMPLETE' ORDER BY observed_at"
        )
        for published, pred_raw, eval_raw in rows:
            if timestamp(published) >= first_holdout_release:
                raise ValueError("Training publication overlaps frozen holdout")
            prediction = json.loads(pred_raw or "{}")
            evaluation = json.loads(eval_raw or "{}")
            targets = {
                (p.get("targetType"), p.get("target")): p
                for p in prediction.get("impacts", [])
            }
            for row in evaluation.get("impacts", []):
                actual = row.get("actualDirection")
                predicted = row.get("direction")
                if (row.get("status") not in {"CORRECT", "WRONG"}
                        or predicted not in DIRECTIONS or actual not in DIRECTIONS):
                    skipped_unverified += 1
                    continue
                end = row.get("outcomeAt")
                if not isinstance(end, str) or timestamp(end) >= first_holdout_release:
                    skipped_unmatured += 1
                    continue
                key = (row.get("targetType"), row.get("target"))
                if key not in targets:
                    skipped_unverified += 1
                    continue
                symbol = row.get("priceProxy")
                if not isinstance(symbol, str) or not symbol:
                    skipped_unverified += 1
                    continue
                market = asset_market(symbol)
                # Use only market and the AI's input direction; retain
                # sector subtype as audit metadata, not a hidden hand-coded
                # issuer-based future-outcome lookup.
                bucket = market + ":" + predicted
                buckets[bucket][actual] += 1
                cases += 1
    rules = {}
    for key, observed in sorted(buckets.items()):
        count = sum(observed.values())
        if count < minimum_support:
            continue
        ranked = sorted(observed.items(), key=lambda item: (-item[1], item[0]))
        winner = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        if winner[1] <= runner_up:
            continue
        rules[key] = {
            "direction": winner[0], "support": count,
            "matchingTrainingCases": winner[1],
            "trainingAgreement": winner[1] / count,
            "distribution": dict(sorted(observed.items())),
        }
    fingerprint = hashlib.sha256(
        json.dumps(rules, sort_keys=True).encode()
    ).hexdigest()
    return {
        "status": "TRAINED" if rules else "NO_SUPPORTED_RULES",
        "schemaVersion": "price_only_frozen_calibrator_v1",
        "minSupport": minimum_support,
        "trainingTargetObservations": cases,
        "skippedUnverified": skipped_unverified,
        "skippedUnmatured": skipped_unmatured,
        "ruleCount": len(rules),
        "rules": rules,
        "fingerprintSha256": fingerprint,
        "frozenBefore": first_holdout_release.astimezone(UTC).isoformat(),
        "causalLearningVerified": False,
        "trainingMode": "SUPERVISED_PRICE_COINCIDENCE_NOT_NEWS_CAUSALITY",
    }


def calibrate_direction(
    fitted: dict[str, Any], symbol: str, news_direction: str,
) -> dict[str, Any]:
    if news_direction not in DIRECTIONS:
        return {"status": "ABSTAIN", "reasonCode": "INVALID_NEWS_DIRECTION"}
    key = asset_market(symbol) + ":" + news_direction
    rule = fitted.get("rules", {}).get(key)
    if rule is None:
        return {"status": "ABSTAIN", "reasonCode": "INSUFFICIENT_PAST_EVIDENCE",
                "bucket": key}
    return {
        "status": "AVAILABLE", "direction": rule["direction"],
        "bucket": key, "support": rule["support"],
        "trainingAgreement": rule["trainingAgreement"],
        "frozenCalibratorHash": fitted["fingerprintSha256"],
        "causalAttributionEstablished": False,
    }
