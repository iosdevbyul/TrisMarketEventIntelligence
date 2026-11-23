"""Documentary evidence and event-family isolation regression tests.

Synthetic source evidence can prove consistency of a contract, never the true
historical instant of public availability or event-news causal attribution.
"""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tris_market_event_intelligence.point_in_time_evidence import (
    evaluate_chronology,
    load_evidence,
)
from tris_market_event_intelligence.point_in_time_outcome import EventClock, SealedForecast

KST = ZoneInfo("Asia/Seoul")
HEADERS = [
    "article_id", "evidence_url", "evidence_type", "evidence_published_at",
    "captured_at", "sha256", "source_url", "first_public_at", "family_id",
    "relationship", "family_evidence_url",
]


def stamp(hour: int, minute: int) -> datetime:
    return datetime(2026, 11, 23, hour, minute, tzinfo=KST)


def prediction(article: int, *, novelty: str = "NEW",
               family: str = "issuer-earnings", direction: str = "DOWN") -> SealedForecast:
    release = stamp(14, 0) if article == 1 else stamp(15, 0)
    event = EventClock(
        article, family, release, release,
        stamp(14, 1) if article == 1 else stamp(15, 1),
        novelty, f"https://publisher.example/{article}", True,
    )
    return SealedForecast(
        event, "STOCK", "INDEX", direction,
        stamp(14, 2) if article == 1 else stamp(15, 2),
        stamp(16, 0), "same-day", "HISTORICAL_REPLAY",
        "model", "rule", f"p{article}",
    )


def evidence_csv(tmp_path: Path, items: list[dict]) -> Path:
    path = tmp_path / "chronology.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(items)
    return path


def evidence(article: int = 1, *, relation: str = "NEW",
             family: str = "issuer-earnings", family_link: str = "") -> dict:
    start = stamp(14, 0) if article == 1 else stamp(15, 0)
    return {
        "article_id": article,
        "evidence_url": f"https://archive.example/{article}",
        "evidence_type": "ARCHIVE_SNAPSHOT",
        "evidence_published_at": start.isoformat(),
        "captured_at": stamp(15, 30).isoformat(),
        "sha256": hashlib.sha256(f"snapshot:{article}".encode()).hexdigest(),
        "source_url": f"https://publisher.example/{article}",
        "first_public_at": start.isoformat(),
        "family_id": family,
        "relationship": relation,
        "family_evidence_url": family_link,
    }


def test_distinct_article_reports_are_single_family_not_independent(tmp_path: Path) -> None:
    files = load_evidence(evidence_csv(tmp_path, [
        evidence(1), evidence(2, relation="REPRINT",
                               family_link="https://publisher.example/1"),
    ]))
    study = evaluate_chronology([
        prediction(1), prediction(1, direction="UP"),
        prediction(2, novelty="REPRINT"),
    ], files)
    assert study["articleCount"] == 2
    assert study["familyCount"] == 1
    assert study["repeatedFamilyCount"] == 1
    assert study["reusedArticleForecastCount"] == 1
    assert study["articles"][1]["isCandidateFamilyRepresentative"] is True
    assert study["articles"][2]["isCandidateFamilyRepresentative"] is False
    assert study["articles"][2]["chronologyStatus"] == "DOCUMENTARY_CONSISTENT_UNATTESTED"
    assert study["independentPublicTimeCertified"] is False


def test_reprint_without_previous_article_link_is_not_proven(tmp_path: Path) -> None:
    study = evaluate_chronology([
        prediction(2, novelty="REPRINT"),
    ], load_evidence(evidence_csv(tmp_path, [
        evidence(2, relation="REPRINT"),
    ])))
    assert study["articles"][2]["chronologyStatus"] == "FAMILY_LINK_EVIDENCE_MISSING"
    assert study["articles"][2]["isCandidateFamilyRepresentative"] is False


def test_conflicting_first_public_time_fails_closed(tmp_path: Path) -> None:
    row = evidence()
    row["first_public_at"] = stamp(13, 0).isoformat()
    study = evaluate_chronology([prediction(1)], load_evidence(evidence_csv(tmp_path, [row])))
    assert study["articles"][1]["chronologyStatus"] == "EVIDENCE_CONFLICT"


def test_missing_evidence_is_not_authentication() -> None:
    study = evaluate_chronology([prediction(1)], {})
    assert study["articles"][1]["chronologyStatus"] == "EVIDENCE_MISSING"
    assert study["articles"][1]["isCandidateFamilyRepresentative"] is False
    assert study["independentEventFamilyCertified"] is False


def test_reject_duplicate_id_and_invalid_family_proof(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate article evidence"):
        load_evidence(evidence_csv(tmp_path, [evidence(1), evidence(1)]))
    bad = evidence(2, relation="REPRINT", family_link="http://example.com")
    with pytest.raises(ValueError, match="HTTPS"):
        load_evidence(evidence_csv(tmp_path, [bad]))


def test_unsupported_earliest_public_time_is_not_verified(tmp_path: Path) -> None:
    row = evidence()
    row["evidence_published_at"] = stamp(14, 1).isoformat()
    study = evaluate_chronology([prediction(1)], load_evidence(evidence_csv(tmp_path, [row])))
    assert study["articles"][1]["chronologyStatus"] == "EARLIEST_PUBLIC_TIME_UNSUPPORTED"
