"""Bounded source-pinned article issuer review exports are not approvals."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tris_market_event_intelligence.article_issuer_coverage import (
    inspect_article_issuer_coverage,
)
from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.event_ticker_proposals import ListedIssuer
from tris_market_event_intelligence.listing_news_validation import (
    validate_existing_listing_news,
)
from tris_market_event_intelligence.listing_research_workflow import (
    acquire_and_reconcile_listings,
)

NOW = datetime(2026, 11, 15, 5, tzinfo=UTC)
SNAPSHOT = NOW - timedelta(hours=3)


def save(store: NewsStore, guid: str, title: str, source: str = "bbc_business") -> None:
    payload = (
        f"<rss><channel><item><guid>{guid}</guid><title>{title}</title>"
        f"<link>https://example.org/article/{guid}</link></item></channel></rss>"
    ).encode()
    receipt = (NOW - timedelta(hours=1)).isoformat()
    assert store.save(
        source, "https://example.org/rss", payload,
        parse_rss(source, payload, receipt),
        receipt, None, None,
    ) == 1


def test_review_bundle_contains_all_real_articles_beyond_sample_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "reviewed.csv"
    source.write_text(
        "market,symbol,company_name,security_type\n"
        "XKRX,005930,삼성전자,COMMON\n", encoding="utf-8",
    )
    bundle = tmp_path / "snapshot"
    acquire_and_reconcile_listings(
        bundle, krx_reviewed=source, now=SNAPSHOT,
    )
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    for i in range(31):
        save(store, str(i), f"삼성전자 announces result {i}")
    EventProcessor(db).initialize()
    with sqlite3.connect(db) as conn:
        before = tuple(conn.iterdump())
    result = validate_existing_listing_news(
        bundle, db, tmp_path / "report", as_of=NOW,
    )
    coverage = result["article_issuer_coverage"]
    assert coverage["post_snapshot_articles_inspected"] == 31
    assert coverage["candidate_mentions"] == 31
    assert len(coverage["candidate_samples"]) == 25
    assert coverage["review_items_total"] == 31
    assert coverage["review_items_truncated"] is False
    assert "review_items" not in coverage
    path = Path(coverage["review_bundle_path"])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == coverage["review_bundle_sha256"]
    exported = json.loads(raw)
    assert exported["schema_version"] == "article_issuer_review_bundle_v1"
    assert exported["status"] == "REVIEW_ONLY_NOT_APPROVED"
    assert exported["listing_manifest_sha256"] == hashlib.sha256(
        (bundle / "manifest.json").read_bytes()
    ).hexdigest()
    assert exported["listing_registry_sha256"] == result["registry_sha256"]
    assert exported["items_total"] == 31
    assert len(exported["items"]) == 31
    assert [item["article_id"] for item in exported["items"]] == list(range(1, 32))
    assert all(item["symbol"] == "005930" for item in exported["items"])
    assert all(item["candidate_type"] == "EXACT_REVIEWED_NAME" for item in exported["items"])
    assert all(item["approved"] is False for item in exported["items"])
    assert exported["point_in_time_training_eligible"] is False
    assert exported["approved_ticker_links_created"] == 0
    with sqlite3.connect(db) as conn:
        assert tuple(conn.iterdump()) == before


def test_alias_review_export_includes_hints_beyond_first_25_samples(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    for i in range(28):
        save(store, str(i), f"Microsoft announces product {i}")
    issuer = ListedIssuer(
        "XNAS", "MSFT", "Microsoft Corporation", ("Microsoft Corporation",),
        SNAPSHOT, None, SNAPSHOT, "observed-us-directory",
    )
    report = inspect_article_issuer_coverage(
        db, (issuer,), observed_at=SNAPSHOT, as_of=NOW,
        include_review_items=True,
    )
    assert report["candidate_mentions"] == 0
    assert report["derived_alias_hints_total"] == 28
    assert len(report["derived_alias_hints"]) == 25
    assert report["review_items_total"] == 28
    assert report["review_items_truncated"] is False
    assert len(report["review_items"]) == 28
    assert all(item["candidate_type"] == "DERIVED_ALIAS_REVIEW_HINT"
               for item in report["review_items"])
    assert all(item["registry_source_id"] == "observed-us-directory"
               for item in report["review_items"])
    assert all(item["approved"] is False for item in report["review_items"])


def test_default_inspection_remains_small_and_does_not_export(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    save(store, "one", "Microsoft announces results")
    issuer = ListedIssuer(
        "XNAS", "MSFT", "Microsoft", ("Microsoft",),
        SNAPSHOT, None, SNAPSHOT, "observed-us-directory",
    )
    report = inspect_article_issuer_coverage(
        db, (issuer,), observed_at=SNAPSHOT, as_of=NOW,
    )
    assert "review_items" not in report
    assert report["review_items_total"] is None
    assert report["approved_links_created"] == 0
