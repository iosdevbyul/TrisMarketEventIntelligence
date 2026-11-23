"""Real-schema fixture tests for the opt-in operational article/price join.

Prices, issuer and timestamp are synthetic. CI never claims real provenance.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.operational_news_price_bridge_cli import (
    evaluate_operational_news_prices,
)
from tris_market_event_intelligence.point_in_time_source_validation_cli import EVENT_COLUMNS

KST = ZoneInfo("Asia/Seoul")


def stamp(hour: int, minute: int) -> str:
    return datetime(2026, 11, 23, hour, minute, tzinfo=KST).isoformat()


def csv_file(path: Path, fields: list[str], items: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


def fixture(tmp: Path) -> tuple[Path, Path, Path, Path]:
    db = tmp / "news.sqlite3"
    feed = b"<rss><item><title>Issuer announcement</title></item></rss>"
    body = "Issuer announcement body and cited original facts."
    raw = b"<html>Issuer announcement body and cited original facts.</html>"
    with sqlite3.connect(db) as connection:
        connection.executescript("""
        CREATE TABLE captures (
            id INTEGER PRIMARY KEY, source TEXT, feed_url TEXT,
            fetched_at TEXT, sha256 TEXT, payload BLOB
        );
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY, source TEXT, external_id TEXT, url TEXT,
            title TEXT, summary TEXT, published_at TEXT, detected_at TEXT,
            capture_id INTEGER
        );
        CREATE TABLE article_full_text (
            article_id INTEGER PRIMARY KEY, status TEXT, observed_at TEXT,
            body TEXT, body_sha256 TEXT, raw_payload BLOB, raw_sha256 TEXT,
            sparse_pages TEXT
        );
        CREATE TABLE story_members (
            article_id INTEGER PRIMARY KEY, story_id INTEGER, match_reason TEXT
        );
        """)
        connection.execute("INSERT INTO captures VALUES (1,?,?,?,?,?)", (
            "fed_all", "https://example.org/rss", stamp(14, 1),
            hashlib.sha256(feed).hexdigest(), feed,
        ))
        connection.execute("INSERT INTO articles VALUES (1,?,?,?,?,?,?,?,?)", (
            "fed_all", "issuer-1", "https://example.org/a", "Announcement",
            "Source summary", stamp(14, 0), stamp(14, 1), 1,
        ))
        connection.execute("INSERT INTO article_full_text VALUES (1,?,?,?,?,?,?,?)", (
            "COMPLETE", stamp(14, 2), body, hashlib.sha256(body.encode()).hexdigest(),
            raw, hashlib.sha256(raw).hexdigest(), "[]",
        ))
        connection.execute("INSERT INTO story_members VALUES (1,10,'new_story')")
    mappings = tmp / "mappings.csv"
    csv_file(mappings, sorted(EVENT_COLUMNS | {"body_sha256"}), [{
        "article_id": 1, "family_id": "event-1",
        "published_at": stamp(14, 0), "first_public_at": stamp(14, 0),
        "detected_at": stamp(14, 1), "novelty": "NEW",
        "source_url": "https://example.org/a", "source_verified": "true",
        "target": "STOCK", "benchmark": "INDEX", "direction": "DOWN",
        "forecast_at": stamp(14, 3), "horizon_end": stamp(15, 30),
        "horizon_name": "session-close", "mode": "LIVE",
        "model_version": "fixture-model", "rule_version": "no-learning",
        "prediction_id": "p1", "market": "XKRX",
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }])
    sessions = tmp / "sessions.csv"
    csv_file(sessions, ["market", "date", "status", "open_at", "close_at"], [{
        "market": "XKRX", "date": "2026-11-23", "status": "OPEN",
        "open_at": stamp(9, 0), "close_at": stamp(15, 30),
    }])
    prices = tmp / "minute.csv"
    records = []
    for sym, values in (("STOCK", [110, 108, 102]), ("INDEX", [100, 100, 99])):
        for (h, m), price in zip(((13, 59), (14, 3), (15, 30)), values):
            records.append({
                "market": "XKRX", "symbol": sym, "close_at": stamp(h, m),
                "adjusted_close": str(price), "adjustment": "adjusted",
                "instrument_kind": "STOCK", "provider": "synthetic-provider",
                "observed_at": stamp(h, m),
            })
    csv_file(prices, [
        "market", "symbol", "close_at", "adjusted_close", "adjustment",
        "instrument_kind", "provider", "observed_at",
    ], records)
    manifest = tmp / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "minute_source_v1",
        "sources": [{
            "filename": "minute.csv", "sha256": hashlib.sha256(prices.read_bytes()).hexdigest(),
            "market": "XKRX", "provider": "synthetic-provider", "source_kind": "STOCK",
        }],
    }), encoding="utf-8")
    return db, mappings, manifest, sessions


def rewrite_mapping(mapping: Path, **updates: object) -> None:
    with mapping.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0].update(updates)
    csv_file(mapping, list(rows[0]), rows)


def test_descriptive_market_return_requires_stored_real_article_identity(
    tmp_path: Path,
) -> None:
    db, mapping, manifest, sessions = fixture(tmp_path)
    before = db.read_bytes()
    result = evaluate_operational_news_prices(db, mapping, manifest, sessions)
    assert db.read_bytes() == before
    assert result["mappingRows"] == 1
    assert result["operationalSourceAdmitted"] == 1
    assert result["operationalSourceBlocked"] == 0
    item = result["items"][0]
    assert item["status"] == "OBSERVED"
    assert item["observation"]["absolute_return"] == 102 / 110 - 1
    assert item["observation"]["forecast_return"] == 102 / 108 - 1
    assert item["observation"]["direction_match"] is True
    assert item["operationalSource"]["admission"] == "ADMITTED_FOR_DESCRIPTIVE_RESEARCH"
    assert item["marketOutcomeCausallyCertified"] is False
    assert item["sealedProspectiveForecastCertified"] is False
    assert result["trainingEligible"] is False
    assert result["productionDatabaseModified"] is False


def test_bad_body_hash_does_not_leak_market_score(tmp_path: Path) -> None:
    db, mapping, manifest, sessions = fixture(tmp_path)
    rewrite_mapping(mapping, body_sha256="0" * 64)
    result = evaluate_operational_news_prices(db, mapping, manifest, sessions)
    assert result["operationalSourceBlocked"] == 1
    item = result["items"][0]
    assert item["status"] == "OPERATIONAL_ARTICLE_NOT_ADMITTED"
    assert "ORIGINAL_BODY_DIGEST_MISMATCH" in item["operationalSource"]["blockers"]
    assert "observation" not in item


def test_post_capture_live_seal_must_not_be_predated(tmp_path: Path) -> None:
    db, mapping, manifest, sessions = fixture(tmp_path)
    rewrite_mapping(mapping, forecast_at=stamp(14, 1))
    result = evaluate_operational_news_prices(db, mapping, manifest, sessions)
    assert result["items"][0]["status"] == "OPERATIONAL_ARTICLE_NOT_ADMITTED"
    assert "ORIGINAL_BODY_CAPTURED_AFTER_LIVE_SEAL" in (
        result["items"][0]["operationalSource"]["blockers"]
    )


def test_mismatched_stored_pub_time_blocks_prediction(tmp_path: Path) -> None:
    db, mapping, manifest, sessions = fixture(tmp_path)
    rewrite_mapping(mapping, published_at=stamp(13, 55))
    result = evaluate_operational_news_prices(db, mapping, manifest, sessions)
    assert result["items"][0]["status"] == "OPERATIONAL_ARTICLE_NOT_ADMITTED"
    assert "NEWS_IDENTITY_OR_CLOCK_MISMATCH" in (
        result["items"][0]["operationalSource"]["blockers"]
    )


def test_nonexistent_article_cannot_be_backfilled_with_fake_price_proof(
    tmp_path: Path,
) -> None:
    db, mapping, manifest, sessions = fixture(tmp_path)
    rewrite_mapping(mapping, article_id="999")
    result = evaluate_operational_news_prices(db, mapping, manifest, sessions)
    assert result["operationalSourceAdmitted"] == 0
    assert result["items"][0]["status"] == "OPERATIONAL_ARTICLE_NOT_ADMITTED"
    assert result["items"][0]["operationalSource"]["blockers"] == [
        "ARTICLE_NOT_IN_OPERATIONAL_DB"
    ]
