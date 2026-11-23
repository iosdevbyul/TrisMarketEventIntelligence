"""Source-time forward-only article capture and real-price observation regressions.

Tests use only synthetic original articles and mocked model responses.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.automatic_news_pipeline import pending, process
from tris_market_event_intelligence.forward_news_market_cycle_cli import cycle


def _clock(minutes_ago: int = 0) -> str:
    return (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()


def _setup_news(path: Path) -> str:
    body = "Original issuer news reports a change in chip demand expectations."
    digest = hashlib.sha256(body.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE articles (
          id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT,
          published_at TEXT, detected_at TEXT
        );
        CREATE TABLE article_full_text (
          article_id INTEGER PRIMARY KEY, status TEXT, body TEXT,
          body_sha256 TEXT, sparse_pages TEXT, observed_at TEXT
        );
        """)
        for article_id, publish, detected, observed in [
            (1, _clock(1440), _clock(1439), _clock(1438)),
            (2, _clock(15), _clock(14), _clock(13)),
            # A historical repost with recent detection/body is not new.
            (3, _clock(400), _clock(11), _clock(10)),
            # Clock-inconsistent source is excluded before Gemma.
            (4, _clock(12), _clock(14), _clock(11)),
        ]:
            db.execute("INSERT INTO articles VALUES (?,?,?,?,?,?)", (
                article_id, "issuer", f"https://example.org/{article_id}",
                f"Original issuer {article_id}", publish, detected,
            ))
            db.execute(
                "INSERT INTO article_full_text VALUES (?,'COMPLETE',?,?,?,?)",
                (article_id, body, digest, "[]", observed),
            )
    return body


def _mapping(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "market", "symbol", "benchmark", "verified_at"])
        writer.writerow([
            "Nvidia", "XNAS", "NVDA", "SPY", _clock(1200),
        ])


def _prices(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "date", "adjusted_close", "source"])
        for symbol, close in (("NVDA", "200"), ("SPY", "400")):
            writer.writerow([symbol, "2026-11-20", close, "synthetic source"])


def _model(body: str) -> dict:
    return {
        "summary": "Increased demand", "limitations": "Uncertain outcome",
        "impacts": [{
            "target": "Nvidia", "direction": "UP",
            "reason": "Potential demand effects",
            "evidence_id": "E1", "evidence_quote": body,
        }],
    }


def test_recent_publication_requires_all_real_source_clocks(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    _setup_news(path)
    with sqlite3.connect(path) as db:
        db.execute("""
            CREATE TABLE automatic_news_analyses (
              article_id INTEGER, body_sha256 TEXT, model TEXT,
              status TEXT, processed_at TEXT, attempts INTEGER,
              PRIMARY KEY(article_id,body_sha256,model)
            )
        """)
        chosen, coverage = pending(
            db, "gemma4:e2b", 10, 30, fresh_only_minutes=120,
        )
    assert [row["id"] for row in chosen] == [2]
    assert coverage["freshAdmissible"] == 1
    assert coverage["freshClockRejected"] == 3
    assert coverage["noHistoricalBackfill"] is True


def test_forward_only_model_does_not_backfill_historical_article(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    body = _setup_news(db)
    mapping = tmp_path / "reviewed.csv"
    _mapping(mapping)
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.ask_gemma",
        return_value=_model(body),
    ) as model:
        report = process(
            db, collect=False, limit=10, seal_forecasts=True,
            mappings_path=mapping, fresh_only_minutes=120,
        )
    model.assert_called_once()
    assert report["completed"] == 1
    assert [row["articleId"] for row in report["results"]] == [2]
    assert report["coverage"]["freshClockRejected"] == 3
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT DISTINCT article_id FROM automatic_forecast_commitments"
        ).fetchall() == [(2,)]
        assert conn.execute(
            "SELECT COUNT(*) FROM automatic_news_analyses"
        ).fetchone()[0] == 1


def test_one_shot_cycle_reuses_actual_price_file_without_quote_provider(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    body = _setup_news(db)
    mapping = tmp_path / "reviewed.csv"
    _mapping(mapping)
    prices = tmp_path / "observed.csv"
    _prices(prices)
    result_file = tmp_path / "first-forward-evaluation.json"
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.ask_gemma",
        return_value=_model(body),
    ) as gemma:
        result = cycle(
            db, mapping, prices, result_file,
            collect=False, fresh_minutes=120,
        )
    gemma.assert_called_once()
    assert result_file.is_file()
    assert result["freshModelPredictionsOnly"] is True
    assert result["newArticleAnalyses"] == 1
    assert result["marketPriceReport"]["recordedCommitmentHorizons"] == 2
    assert result["marketPriceReport"]["horizons"]["1d"]["denominator"] == 0
    assert result["marketPriceReport"]["horizons"]["5d"]["denominator"] == 0
    assert result["newMarketPricesFetched"] is False
    assert result["prospectivePredictionAccuracy"] is None
    assert result["trainingEligible"] is False
    with pytest.raises(ValueError, match="must not exist"):
        cycle(db, mapping, prices, result_file, collect=False)


def test_second_forward_cycle_has_no_reprocessing_and_no_fake_accuracy(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    body = _setup_news(db)
    mapping = tmp_path / "reviewed.csv"
    _mapping(mapping)
    prices = tmp_path / "observed.csv"
    _prices(prices)
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.ask_gemma",
        return_value=_model(body),
    ) as gemma:
        first = cycle(
            db, mapping, prices, tmp_path / "first.json",
            collect=False, fresh_minutes=120,
        )
        second = cycle(
            db, mapping, prices, tmp_path / "second.json",
            collect=False, fresh_minutes=120,
        )
    assert gemma.call_count == 1
    assert first["newArticleAnalyses"] == 1
    assert second["newArticleAnalyses"] == 0
    assert second["marketPriceReport"]["recordedCommitmentHorizons"] == 2
    assert second["marketPriceReport"]["horizons"]["1d"]["denominator"] == 0


def test_fresh_only_rejects_missing_optin_and_invalid_window(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    _setup_news(db)
    with pytest.raises(ValueError, match="requires explicit forecast sealing"):
        process(db, collect=False, fresh_only_minutes=120)
    with pytest.raises(ValueError, match="1..1440"):
        process(db, collect=False, seal_forecasts=True, fresh_only_minutes=9000)


def test_reject_replayed_article_with_recent_detection_and_body(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    _setup_news(db)
    # Article 3 is freshly detected but old publisher-stated material.
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE articles SET detected_at=? WHERE id=3", (_clock(3),),
        )
        conn.execute(
            "UPDATE article_full_text SET observed_at=? WHERE article_id=3",
            (_clock(2),),
        )
        conn.execute("""
        CREATE TABLE automatic_news_analyses (
          article_id INTEGER, body_sha256 TEXT, model TEXT, status TEXT,
          processed_at TEXT, attempts INTEGER,
          PRIMARY KEY(article_id,body_sha256,model)
        )""")
        chosen, _ = pending(
            conn, "gemma4:e2b", 10, 30, fresh_only_minutes=120,
        )
    assert [row["id"] for row in chosen] == [2]
