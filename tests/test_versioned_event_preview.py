"""v3 is a review-only candidate; v2 remains the production extractor."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tris_market_event_intelligence.multi_events import (
    EXTRACTOR_VERSION,
    extract_news_event,
)
from tris_market_event_intelligence.news_classification_evidence import (
    inspect_unclassified_news,
)
from tris_market_event_intelligence.versioned_event_preview import (
    PREVIEW_EXTRACTOR_VERSION,
    preview_news_event_v3,
)

START = datetime(2026, 11, 14, 15, 54, tzinfo=UTC)
END = datetime(2026, 11, 14, 18, tzinfo=UTC)


@pytest.mark.parametrize(("headline", "expected", "tag"), [
    (
        "Microsoft says services fully restored after day-long global outage",
        "SERVICE_INCIDENT", "RECOVERY",
    ),
    (
        "Italy fines Meta €805m over accessing users' photos",
        "REGULATION", "MONETARY_PENALTY",
    ),
    (
        "Regulator fined ExampleCo $2.1 billion after investigation",
        "REGULATION", "MONETARY_PENALTY",
    ),
])
def test_actual_observed_event_wording_is_v3_only(
    headline: str, expected: str, tag: str,
) -> None:
    assert EXTRACTOR_VERSION == "multi-v2"
    assert extract_news_event("bbc_world", headline, None) is None
    preview = preview_news_event_v3("bbc_world", headline, None)
    assert preview is not None
    assert preview.event.classification == expected
    assert preview.tags == (tag,)
    assert preview.event.extractor_version == PREVIEW_EXTRACTOR_VERSION
    assert preview.event.status == "UNVERIFIED"
    assert preview.event.direction is None
    assert preview.event.change_basis_points is None


@pytest.mark.parametrize("headline", [
    "Microsoft services fully restored",
    "A global outage hits services",
    "Microsoft says services were not restored after a global outage",
    "Italy has not fined Meta €805m",
    "Traffic police fines drivers €20",
    "Nigeria cautions against travel via Ethiopia's main airport due to altitude",
    "'Miracle on Mississippi' - All passengers rescued after cruise ship collision",
])
def test_unrelated_and_negated_headlines_do_not_trigger_v3(
    headline: str,
) -> None:
    assert preview_news_event_v3("bbc_world", headline, None) is None


def test_existing_v2_result_remains_unchanged() -> None:
    headline = "Cloud service outage disrupts users"
    baseline = extract_news_event("bbc_technology", headline, None)
    assert baseline is not None
    assert preview_news_event_v3("bbc_technology", headline, None) == baseline


def test_readonly_report_and_deterministic_cross_feed_representative(
    tmp_path: Path,
) -> None:
    dbpath = tmp_path / "news.sqlite3"
    with sqlite3.connect(dbpath) as db:
        db.executescript(
            """CREATE TABLE articles(
                id INTEGER PRIMARY KEY, source TEXT, url TEXT,
                title TEXT, summary TEXT, detected_at TEXT
            );
            CREATE TABLE extraction_runs(
                article_id INTEGER, extractor_version TEXT
            );
            CREATE TABLE economic_events(
                id INTEGER PRIMARY KEY, article_id INTEGER
            );"""
        )
        db.executemany("INSERT INTO articles VALUES (?,?,?,?,?,?)", [
            (
                30, "bbc_world", "https://example.org/meta?utm_source=world",
                "Italy fines Meta €805m over accessing users' photos",
                "The firm is accused of violating a privacy regulation.",
                "2026-11-14T16:00:00+00:00",
            ),
            (
                31, "bbc_technology", "https://example.org/meta?utm_source=tech",
                "Italy fines Meta €805m over accessing users' photos",
                "The firm is accused of violating a privacy regulation.",
                "2026-11-14T16:01:00+00:00",
            ),
            (
                32, "bbc_world", "https://example.org/microsoft",
                "Microsoft says services fully restored after day-long global outage",
                None, "2026-11-14T16:02:00+00:00",
            ),
            (
                33, "bbc_world", "https://example.org/traffic",
                "Traffic police fines drivers €20",
                None, "2026-11-14T16:03:00+00:00",
            ),
        ])
        db.executemany("INSERT INTO extraction_runs VALUES (?,?)", [
            (30, EXTRACTOR_VERSION), (31, EXTRACTOR_VERSION),
            (32, EXTRACTOR_VERSION), (33, EXTRACTOR_VERSION),
        ])
    original = dbpath.read_bytes()
    report = inspect_unclassified_news(dbpath, observed_at=START, as_of=END)
    assert report["v3_preview_matches_by_category"] == {
        "REGULATION": 2, "SERVICE_INCIDENT": 1,
    }
    assert report["v3_preview_article_matches"] == 3
    assert report["v3_preview_distinct_representative_articles"] == 2
    assert report["v3_preview_event_rows_written"] == 0
    assert report["cross_source_canonical_url_groups"] == 1
    samples = {sample["article_id"]: sample for sample in report["article_samples"]}
    assert samples[30]["representative_article_id"] == 30
    assert samples[31]["representative_article_id"] == 30
    assert samples[30]["is_representative"] is True
    assert samples[31]["is_representative"] is False
    assert samples[32]["v3_preview"]["tag"] == "RECOVERY"
    assert samples[33]["v3_preview"] is None
    assert report["database_modified"] is False
    assert dbpath.read_bytes() == original
