"""NEWS chronological coverage is descriptive, source-pinned and label-free."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_news_research_worker import files

from tris_market_event_intelligence.news_observation_coverage import (
    audit_news_observation_coverage,
)


def _full_news(tmp_path: Path) -> Path:
    path = files(tmp_path)[0]
    doc = json.loads(path.read_text())
    doc.update({
        "schemaVersion": "historical_news_cohort_v2",
        "pageAfterId": 0,
        "hasMore": False,
        "nextCursor": 12,
        "totalObservedByCutoff": 12,
    })
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _later(base_path: Path) -> Path:
    doc = json.loads(base_path.read_text())
    new = dict(doc["items"][-1])
    new.update({
        "articleId": 13,
        "title": "New policy proposal creates sector regulatory questions",
        "sourceUrl": "https://example.org/research/13",
        "firstObservedAt": "2026-11-20T15:00:00+00:00",
        "storyGroupId": None,
        "externalId": "13",
    })
    new["textSha256"] = hashlib.sha256(
        (new["title"] + "\n" + (new["summary"] or "")).encode()
    ).hexdigest()
    doc["items"].append(new)
    doc.update({
        "totalObservedByCutoff": 13,
        "nextCursor": 13,
        "asOf": "2026-11-21T00:00:00+00:00",
    })
    path = base_path.parent / "newer.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_three_observed_days_never_marked_as_model_ready(tmp_path: Path) -> None:
    path = _full_news(tmp_path)
    original = path.read_bytes()
    report = audit_news_observation_coverage(baseline_path=path)
    assert report["status"] == "NEEDS_MORE_TEMPORAL_DIVERSITY"
    assert report["baseline"]["articleCount"] == 12
    assert report["baseline"]["observedActiveDaysUTC"] == 3
    assert report["baseline"]["calendarSpanDaysUTC"] == 11
    assert report["baseline"]["candidateStoryUnits"] == 12
    assert "FEWER_THAN_7_OBSERVED_UTC_DAYS" in (
        report["baseline"]["sourceOnlyChronologyWarnings"]
    )
    assert "LESS_THAN_14_CALENDAR_DAYS_COVERED" in (
        report["baseline"]["sourceOnlyChronologyWarnings"]
    )
    assert report["baseline"]["coverageThresholdsProveGeneralization"] is False
    assert report["humanGoldIndependentlyAudited"] is False
    assert report["modelImprovementValidated"] is False
    assert report["stockPricesUsed"] is False
    assert report["networkCalls"] == 0
    assert path.read_bytes() == original
    text = json.dumps(report)
    assert "Major regulatory decision" not in text
    assert "https://example.org/research" not in text
    assert "fixture-human" not in text


def test_comparing_later_cohort_preserves_old_article_identity(tmp_path: Path) -> None:
    baseline = _full_news(tmp_path)
    newer = _later(baseline)
    old = baseline.read_bytes()
    new = newer.read_bytes()
    report = audit_news_observation_coverage(
        baseline_path=baseline, newer_path=newer,
    )
    assert report["growth"]["newArticles"] == 1
    assert report["growth"]["unchangedPinnedOriginalArticles"] == 12
    assert report["growth"]["newObservedUTCDays"] == 1
    assert report["growth"]["historicalArticlesRevisedOrRemoved"] is False
    assert report["growth"]["newArticlesWithHumanVerifiedIndependence"] is None
    assert report["newer"]["articleCount"] == 13
    assert baseline.read_bytes() == old
    assert newer.read_bytes() == new


@pytest.mark.parametrize("alter", ["remove", "title", "timestamp", "backfill", "cutoff"])
def test_growth_fails_closed_on_mutated_history(
    tmp_path: Path, alter: str,
) -> None:
    baseline = _full_news(tmp_path)
    newer = _later(baseline)
    doc = json.loads(newer.read_text())
    if alter == "remove":
        doc["items"].pop(0)
        doc["totalObservedByCutoff"] -= 1
    elif alter == "title":
        doc["items"][0]["title"] = "Forged historical title"
        doc["items"][0]["textSha256"] = hashlib.sha256(
            (doc["items"][0]["title"] + "\nSource-only detail").encode()
        ).hexdigest()
    elif alter == "timestamp":
        doc["items"][0]["firstObservedAt"] = "2026-11-02T12:00:00+00:00"
    elif alter == "backfill":
        doc["items"][-1]["firstObservedAt"] = "2026-11-11T12:00:00+00:00"
    elif alter == "cutoff":
        doc["asOf"] = "2026-11-12T23:00:00+00:00"
    doc["nextCursor"] = doc["items"][-1]["articleId"]
    newer.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        audit_news_observation_coverage(
            baseline_path=baseline, newer_path=newer,
        )


def test_partial_snapshot_is_rejected(tmp_path: Path) -> None:
    path = _full_news(tmp_path)
    doc = json.loads(path.read_text())
    doc["hasMore"] = True
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Complete"):
        audit_news_observation_coverage(baseline_path=path)


def test_empty_or_wrong_version_rejected(tmp_path: Path) -> None:
    path = _full_news(tmp_path)
    doc = json.loads(path.read_text())
    doc["schemaVersion"] = "historical_news_cohort_v1"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="v2"):
        audit_news_observation_coverage(baseline_path=path)
