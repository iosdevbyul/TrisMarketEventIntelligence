"""Article-level issuer diagnostics preserve news evidence and review safety."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tris_market_event_intelligence.article_issuer_coverage import (
    inspect_article_issuer_coverage,
)
from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_ticker_proposals import ListedIssuer


NOW = datetime(2026, 11, 15, 3, tzinfo=UTC)
OBSERVED = NOW - timedelta(hours=1)


def issuer(
    market: str, symbol: str, name: str, *, alias: str | None = None,
    evidence_at: datetime = OBSERVED,
) -> ListedIssuer:
    return ListedIssuer(
        market, symbol, name, (name, alias) if alias else (name,),
        OBSERVED, None, evidence_at, "operator-reviewed-snapshot",
    )


def store_story(
    store: NewsStore, source: str, guid: str, title: str,
    url: str, detected: datetime,
) -> None:
    payload = (
        f"<rss><channel><item><guid>{guid}</guid><title>{title}</title>"
        f"<link>{url}</link></item></channel></rss>"
    ).encode()
    received = detected.isoformat()
    assert store.save(
        source, "https://example.org/feed", payload,
        parse_rss(source, payload, received), received, None, None,
    ) == 1


def test_real_articles_are_inspected_without_economic_events(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    store_story(
        store, "bbc_technology", "a", "Microsoft launches new software",
        "https://example.org/ms", OBSERVED + timedelta(minutes=5),
    )
    store_story(
        store, "bbc_world", "b", "Microsoft launches new software",
        "https://example.org/ms", OBSERVED + timedelta(minutes=6),
    )
    store_story(
        store, "bbc_business", "c", "Unrelated policy debate",
        "https://example.org/debate", OBSERVED + timedelta(minutes=7),
    )
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='economic_events'"
        ).fetchone()[0] == 0
        before = tuple(conn.iterdump())
    report = inspect_article_issuer_coverage(
        db, (issuer("XNAS", "MSFT", "Microsoft Corporation", alias="Microsoft"),),
        observed_at=OBSERVED, as_of=NOW,
    )
    assert report["post_snapshot_articles_inspected"] == 3
    assert report["candidate_mentions"] == 1
    assert report["status_counts"]["NAME_CANDIDATES"] == 1
    assert report["status_counts"]["CROSS_SOURCE_REPEAT"] == 1
    assert report["status_counts"]["NO_EXACT_ISSUER_NAME"] == 1
    assert report["candidate_samples"][0]["candidates"][0]["symbol"] == "MSFT"
    assert report["candidate_samples"][0]["candidates"][0]["approved"] is False
    assert report["economic_event_required"] is False
    assert report["training_dataset_eligible"] is False
    assert report["database_modified"] is False
    with sqlite3.connect(db) as conn:
        assert tuple(conn.iterdump()) == before


def test_future_registry_evidence_and_ambiguous_issuer_names_are_blocked(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    seen = OBSERVED + timedelta(minutes=10)
    store_story(store, "bbc_business", "one", "Orbit announces deal",
                "https://example.org/one", seen)
    store_story(store, "bbc_world", "two", "Microsoft announces deal",
                "https://example.org/two", seen)
    issuers = (
        issuer("XNAS", "ORBT", "Orbit Global"),
        issuer("XKRX", "000111", "Orbit Group"),
        issuer("XNAS", "MSFT", "Microsoft",
               evidence_at=NOW + timedelta(days=1)),
    )
    # Alias "Orbit" is attributed to two different companies and must not
    # produce links, even if the article title contains the shared word.
    issuers = (
        issuer("XNAS", "ORBT", "Orbit Global", alias="Orbit"),
        issuer("XKRX", "000111", "Orbit Group", alias="Orbit"),
        issuers[2],
    )
    report = inspect_article_issuer_coverage(
        db, issuers, observed_at=OBSERVED, as_of=NOW,
    )
    assert report["candidate_mentions"] == 0
    assert report["status_counts"]["AMBIGUOUS_ISSUER_NAME"] == 1
    assert report["status_counts"]["FUTURE_LISTING_EVIDENCE"] == 1
    assert report["automatic_approvals"] == 0


def test_pre_snapshot_and_future_articles_never_use_current_listing(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    store_story(store, "bbc_world", "old", "Microsoft reports earnings",
                "https://example.org/old", OBSERVED - timedelta(minutes=1))
    store_story(store, "bbc_world", "new", "Microsoft reports earnings",
                "https://example.org/new", NOW + timedelta(hours=1))
    report = inspect_article_issuer_coverage(
        db, (issuer("XNAS", "MSFT", "Microsoft"),),
        observed_at=OBSERVED, as_of=NOW,
    )
    assert report["post_snapshot_articles_inspected"] == 0
    assert report["candidate_mentions"] == 0


def test_bounded_limit_reports_uninspected_articles_instead_of_silent_omission(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    for index in range(3):
        store_story(
            store, "bbc_world", str(index), "Microsoft announces deal",
            f"https://example.org/{index}",
            OBSERVED + timedelta(minutes=index + 1),
        )
    report = inspect_article_issuer_coverage(
        db, (issuer("XNAS", "MSFT", "Microsoft"),),
        observed_at=OBSERVED, as_of=NOW, max_articles=1,
    )
    assert report["post_snapshot_articles_inspected"] == 1
    assert report["has_more_articles"] is True
    assert report["uninspected_articles_due_to_limit"] == 2
    assert report["candidate_mentions"] == 1


def test_missing_database_and_invalid_timestamps_fail_closed(tmp_path: Path) -> None:
    db = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError, match="Existing"):
        inspect_article_issuer_coverage(
            db, (), observed_at=OBSERVED, as_of=NOW,
        )
    assert not db.exists()
    store = NewsStore(db)
    store.initialize()
    with pytest.raises(ValueError, match="Timezone-aware"):
        inspect_article_issuer_coverage(
            db, (), observed_at=datetime(2026, 11, 15), as_of=NOW,
        )
