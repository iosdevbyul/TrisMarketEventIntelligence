"""NEWS-only chronological coverage and immutable source growth audit.

Useful without a model, human labels, stock prices, or a trained candidate.
Coverage is a necessary warning, NEVER independent generalization proof.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import _cohort, _instant
from tris_market_event_intelligence.news_review_sample import _candidate_units

MIN_OBSERVED_DAYS = 7
MIN_CALENDAR_SPAN_DAYS = 14
MAX_SINGLE_DAY_SHARE = 0.75


def _complete_snapshot(path: Path) -> tuple[bytes, dict]:
    raw, doc = _cohort(path)
    articles = doc["items"]
    if (doc.get("hasMore") is not False or doc.get("pageAfterId") != 0
            or doc.get("totalObservedByCutoff") != len(articles)
            or doc.get("nextCursor") != articles[-1]["articleId"]):
        raise ValueError("Complete first-page-through-final NEWS cohort required")
    return raw, doc


def _summary(doc: dict) -> dict[str, object]:
    rows = doc["items"]
    days: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    dated = []
    for article in rows:
        observed = _instant(article["firstObservedAt"])
        dated.append(observed)
        days[observed.date().isoformat()] += 1
        sources[article["source"]] += 1
    ordered = sorted(dated)
    first, last = ordered[0], ordered[-1]
    active_days = len(days)
    # Distinct calendar days spanning first and last observations, inclusive.
    calendar_span = (last.date() - first.date()).days + 1
    largest_day = max(days.values())
    greatest_source = max(sources.values())
    issues = []
    if active_days < MIN_OBSERVED_DAYS:
        issues.append("FEWER_THAN_7_OBSERVED_UTC_DAYS")
    if calendar_span < MIN_CALENDAR_SPAN_DAYS:
        issues.append("LESS_THAN_14_CALENDAR_DAYS_COVERED")
    if largest_day / len(rows) > MAX_SINGLE_DAY_SHARE:
        issues.append("SINGLE_UTC_DAY_OVER_75_PERCENT")
    # Grouping uses original story IDs / exact text / canonical URLs only.
    # A candidate unit count is not independent event evidence.
    candidate_units = _candidate_units(rows)
    return {
        "articleCount": len(rows),
        "firstObservedAt": first.astimezone(UTC).isoformat(),
        "lastObservedAt": last.astimezone(UTC).isoformat(),
        "observedActiveDaysUTC": active_days,
        "calendarSpanDaysUTC": calendar_span,
        "articlesByObservedUTCDate": dict(sorted(days.items())),
        "articlesBySource": dict(sorted(sources.items())),
        "largestSingleUTCDateCount": largest_day,
        "largestSingleUTCDateShare": round(largest_day / len(rows), 6),
        "largestSingleSourceCount": greatest_source,
        "largestSingleSourceShare": round(greatest_source / len(rows), 6),
        "candidateStoryUnits": len(candidate_units),
        "candidateStoryUnitsVerifiedDistinctEvents": False,
        "sourceOnlyChronologyWarnings": issues,
        "meetsPreliminaryCoverageThresholds": not issues,
        "coverageThresholdsProveGeneralization": False,
    }


def audit_news_observation_coverage(*, baseline_path: Path,
                                    newer_path: Path | None = None) -> dict:
    """Audit one full snapshot or compare an immutable older/newer pair.

    Source URLs, capture hashes and raw titles are validated but not printed.
    Any changed, missing or time-reordered source observation fails closed.
    """
    baseline_raw, baseline = _complete_snapshot(baseline_path)
    baseline_summary = _summary(baseline)
    result: dict = {
        "schemaVersion": "news_observation_coverage_v1",
        "status": ("NEEDS_MORE_TEMPORAL_DIVERSITY"
                   if baseline_summary["sourceOnlyChronologyWarnings"]
                   else "PRELIMINARY_COVERAGE_ONLY_NOT_VALIDATED"),
        "baselineSha256": hashlib.sha256(baseline_raw).hexdigest(),
        "baseline": baseline_summary,
        "newer": None,
        "growth": None,
        "modelCalls": 0,
        "networkCalls": 0,
        "humanLabelsGenerated": 0,
        "humanGoldIndependentlyAudited": False,
        "testGoldScored": False,
        "modelImprovementValidated": False,
        "promotionAllowed": False,
        "stockPricesUsed": False,
        "productionDbModified": False,
        "researchArtifactsModified": False,
    }
    if newer_path is None:
        return result
    newer_raw, newer = _complete_snapshot(newer_path)
    if _instant(newer["asOf"]) <= _instant(baseline["asOf"]):
        raise ValueError("Newer snapshot cutoff must be strictly later")
    earlier = {a["articleId"]: a for a in baseline["items"]}
    later = {a["articleId"]: a for a in newer["items"]}
    missing = set(earlier) - set(later)
    if missing:
        raise ValueError("Newer NEWS snapshot removed original article IDs")
    # Mutable pipeline state may change (event count, story indexing),
    # while identity of the first stored article observation must not.
    fixed = (
        "source", "sourceUrl", "externalId", "title", "summary",
        "textSha256", "captureSha256", "firstObservedAt",
    )
    for aid, old in earlier.items():
        current = later[aid]
        if any(old.get(k) != current.get(k) for k in fixed):
            raise ValueError("News article content, identity or receipt changed")
    additions = [a for aid, a in later.items() if aid not in earlier]
    cutoff = _instant(baseline["asOf"])
    if any(_instant(a["firstObservedAt"]) <= cutoff for a in additions):
        raise ValueError("Late-added articles cross the earlier observation cutoff")
    current_summary = _summary(newer)
    additional_days = set(current_summary["articlesByObservedUTCDate"]) - set(
        baseline_summary["articlesByObservedUTCDate"]
    )
    result["newer"] = current_summary
    result["newerSha256"] = hashlib.sha256(newer_raw).hexdigest()
    result["growth"] = {
        "newArticles": len(additions),
        "unchangedPinnedOriginalArticles": len(earlier),
        "newObservedUTCDays": len(additional_days),
        "observedDayCountIncrease": (current_summary["observedActiveDaysUTC"]
                                     - baseline_summary["observedActiveDaysUTC"]),
        "calendarSpanIncreaseDays": (current_summary["calendarSpanDaysUTC"]
                                     - baseline_summary["calendarSpanDaysUTC"]),
        "newArticlesWithHumanVerifiedIndependence": 0,
        "historicalArticlesRevisedOrRemoved": False,
    }
    result["status"] = ("NEWER_SNAPSHOT_NEEDS_MORE_TEMPORAL_DIVERSITY"
                        if current_summary["sourceOnlyChronologyWarnings"]
                        else "NEWER_SNAPSHOT_PRELIMINARY_COVERAGE_ONLY")
    return result
