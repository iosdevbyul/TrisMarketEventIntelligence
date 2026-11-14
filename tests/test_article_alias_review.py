"""Stage 3-A article alias hints must never be treated as approved issuers."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tris_market_event_intelligence.article_issuer_coverage import inspect_article_issuer_coverage
from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_ticker_proposals import ListedIssuer

AT = datetime(2026, 11, 15, 4, tzinfo=UTC)
SNAPSHOT = AT - timedelta(hours=2)


def add(store: NewsStore, guid: str, headline: str, detected: datetime) -> None:
    payload = (
        f"<rss><channel><item><guid>{guid}</guid><title>{headline}</title>"
        f"<link>https://example.org/{guid}</link></item></channel></rss>"
    ).encode()
    stamp = detected.isoformat()
    assert store.save(
        "bbc_business", "https://example.org/rss", payload,
        parse_rss("bbc_business", payload, stamp),
        stamp, None, None,
    ) == 1


def company(
    symbol: str, legal_name: str, *,
    observed: datetime = SNAPSHOT,
    expires: datetime | None = None,
) -> ListedIssuer:
    return ListedIssuer(
        "XNAS", symbol, legal_name, (legal_name,), SNAPSHOT, expires,
        observed, "reviewed-snapshot",
    )


def test_unclassified_article_legal_suffix_hint_is_not_a_ticker_link(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    add(store, "one", "Microsoft launches new cloud products", AT)
    with sqlite3.connect(db) as conn:
        original = tuple(conn.iterdump())
    report = inspect_article_issuer_coverage(
        db, (company("MSFT", "Microsoft Corporation"),),
        observed_at=SNAPSHOT, as_of=AT,
    )
    assert report["status_counts"]["NO_EXACT_ISSUER_NAME"] == 1
    assert report["candidate_mentions"] == 0
    assert report["derived_alias_hints_total"] == 1
    assert report["derived_alias_hint_counts"]["NEEDS_HUMAN_ALIAS_REVIEW"] == 1
    hint = report["derived_alias_hints"][0]
    assert hint["candidate_alias"] == "microsoft"
    assert hint["possible_issuers"][0]["symbol"] == "MSFT"
    assert hint["approved"] is False
    assert hint["human_review_required"] is True
    assert hint["training_dataset_eligible"] is False
    assert report["derived_aliases_approved"] == 0
    assert report["approved_links_created"] == 0
    with sqlite3.connect(db) as conn:
        assert tuple(conn.iterdump()) == original


def test_shared_derived_name_and_future_registry_cannot_promote_issuer(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    add(store, "a", "Contoso announces results", AT)
    add(store, "b", "Alphabet opens new office", AT)
    report = inspect_article_issuer_coverage(
        db, (
            company("CNTS", "Contoso Corporation"),
            company("CNTI", "Contoso Incorporated"),
            company("ABCD", "Alphabet Corporation",
                    observed=AT + timedelta(days=1)),
        ),
        observed_at=SNAPSHOT, as_of=AT,
    )
    assert report["candidate_mentions"] == 0
    assert report["derived_alias_hint_counts"]["AMBIGUOUS_DERIVED_ALIAS"] == 1
    assert report["derived_alias_hint_counts"]["FUTURE_ALIAS_EVIDENCE"] == 1
    assert all(not hint["possible_issuers"] for hint in report["derived_alias_hints"])
    assert report["automatic_approvals"] == 0


def test_inactive_alias_and_duplicate_representative_are_diagnostic_only(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    add(store, "a", "Alphabet announces results", AT)
    report = inspect_article_issuer_coverage(
        db, (company("ABCD", "Alphabet Corporation",
                     expires=SNAPSHOT + timedelta(minutes=30)),),
        observed_at=SNAPSHOT, as_of=AT,
    )
    assert report["derived_alias_hint_counts"]["INACTIVE_ALIAS"] == 1
    assert report["derived_alias_hints"][0]["possible_issuers"] == []
