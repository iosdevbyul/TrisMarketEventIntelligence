"""Deterministic chronological holdout selection across distinct release dates.

Only originally reported publication dates are used. If fewer than the
requested independent dates are present, do not invent articles or claim an
evaluation of 30 days. Selection never consults future returns or any model
prediction outcomes.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.event_evidence_policy import article_market
from tris_market_event_intelligence.sequential_market_learning_cli import publication_time

NEW_YORK = ZoneInfo("America/New_York")
SEOUL = ZoneInfo("Asia/Seoul")


def release_day(article: dict[str, Any]) -> str:
    zone = SEOUL if article_market(article) == "KR" else NEW_YORK
    return publication_time(article).astimezone(zone).date().isoformat()


def spread_date_selection(
    articles: list[dict[str, Any]], count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One unique local market release day per article, maximin time spread.

    Selects the earliest and latest dates first, then the date furthest from
    the previously chosen dates. Ties favor underrepresented sources and then
    stable article IDs. This is a predeclared date-only algorithm, not an
    optimizer based on market outcomes.
    """
    if not 1 <= count <= 1000:
        raise ValueError("Expected 1..1000 dates")
    rows = sorted(articles, key=lambda row: (publication_time(row), row["articleId"]))
    by_day: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_day.setdefault(release_day(row), []).append(row)
    days = sorted(by_day)
    picked_days: list[str] = []
    represented_sources: Counter[str] = Counter()
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count and len(picked_days) < len(days):
        remaining = [day for day in days if day not in picked_days]
        if not picked_days:
            day = remaining[0]
        elif len(picked_days) == 1:
            day = remaining[-1]
        else:
            existing = [datetime.fromisoformat(day).date() for day in picked_days]
            day = max(
                remaining,
                key=lambda candidate: (
                    min(abs((datetime.fromisoformat(candidate).date() - prior).days)
                        for prior in existing),
                    -sum(represented_sources[a["source"]] for a in by_day[candidate]),
                    -days.index(candidate),
                ),
            )
        candidate = min(
            by_day[day],
            key=lambda row: (
                represented_sources[str(row["source"])],
                publication_time(row),
                row["articleId"],
            ),
        )
        chosen.append(candidate)
        picked_days.append(day)
        represented_sources[str(candidate["source"])] += 1
    chosen.sort(key=lambda row: (publication_time(row), row["articleId"]))
    all_span = (
        (datetime.fromisoformat(days[-1]) - datetime.fromisoformat(days[0])).days
        if days else 0
    )
    selected_days = sorted(picked_days)
    chosen_span = (
        (datetime.fromisoformat(selected_days[-1])
         - datetime.fromisoformat(selected_days[0])).days
        if selected_days else 0
    )
    info: dict[str, Any] = {
        "requestedDistinctDates": count,
        "candidateArticles": len(rows),
        "candidateDistinctDates": len(days),
        "availableDateRange": [days[0], days[-1]] if days else [],
        "availableSpanDays": all_span,
        "selected": len(chosen),
        "selectedDistinctDates": len(selected_days),
        "selectedSpanDays": chosen_span,
        "selectedSources": dict(sorted(represented_sources.items())),
        "selectedMonths": dict(sorted(Counter(day[:7] for day in selected_days).items())),
        "sufficientDistinctDates": len(selected_days) == count,
        "items": [
            {
                "articleId": row["articleId"],
                "publishedAtReported": row["publishedAtReported"],
                "releaseDayLocal": release_day(row),
                "source": row["source"],
                "title": row["title"],
                "sourceUrl": row.get("sourceUrl"),
                "contentEvidence": row.get("contentEvidence"),
            }
            for row in chosen
        ],
        "selectionRule": "maximin_published_date_then_underrepresented_source_v1",
        "selectionUsedFuturePrices": False,
        "pointInTimeCertified": False,
    }
    return chosen, info
