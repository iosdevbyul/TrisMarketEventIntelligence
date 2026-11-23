"""Real-schema integration tests for persisted Gemma → historical daily prices.

All article and price records in these tests are synthetic fixtures.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from test_market_outcome_validation_cli import _fixture
from test_operational_news_pit_readiness_cli import create_db

from tris_market_event_intelligence.automatic_gemma_market_outcomes_cli import (
    _automatically_analyzed,
    evaluate_automatic_outcomes,
)


def _ledger(tmp_path: Path, *, target: str = "Example Corp",
            corrupt_body: bool = False) -> Path:
    db_path = tmp_path / "real-schema-news.sqlite3"
    create_db(db_path)
    result = {
        "summary": "Economic event", "limitations": "Retrospective only",
        "impacts": [
            {
                "target": target, "direction": "UP",
                "reason": "Original issuer news",
                "evidence_id": "E1",
                "evidence_quote": "Actual issuer filing content",
            },
        ],
    }
    with sqlite3.connect(db_path) as db:
        db.execute("DROP TABLE automatic_news_analyses")
        db.executescript("""
        CREATE TABLE automatic_news_analyses (
          article_id INTEGER NOT NULL,
          body_sha256 TEXT NOT NULL,
          model TEXT NOT NULL,
          status TEXT NOT NULL,
          processed_at TEXT NOT NULL,
          attempts INTEGER NOT NULL,
          error_code TEXT,
          analysis_json TEXT,
          PRIMARY KEY(article_id, body_sha256, model)
        )
        """)
        db.execute("UPDATE articles SET published_at=?, detected_at=? WHERE id=1", (
            "2026-01-05T15:00:00+00:00", "2026-01-05T15:30:00+00:00",
        ))
        db.execute("UPDATE captures SET fetched_at=? WHERE id=1", (
            "2026-01-05T15:30:00+00:00",
        ))
        db.execute("UPDATE article_full_text SET observed_at=? WHERE article_id=1", (
            "2026-01-05T15:45:00+00:00",
        ))
        digest = db.execute(
            "SELECT body_sha256 FROM article_full_text WHERE article_id=1"
        ).fetchone()[0]
        db.execute(
            "INSERT INTO automatic_news_analyses VALUES (?,?,?,?,?,?,?,?)",
            (
                1, digest, "gemma4:e2b", "COMPLETED",
                "2026-01-05T20:00:00+00:00", 1, None,
                json.dumps(result),
            ),
        )
        if corrupt_body:
            db.execute(
                "UPDATE article_full_text SET body_sha256=? WHERE article_id=1",
                ("0" * 64,),
            )
    return db_path


def test_actual_sqlite_completed_gemma_joins_source_and_prices(
    tmp_path: Path,
) -> None:
    _, mappings, prices = _fixture(tmp_path)
    news_db = _ledger(tmp_path)
    before = hashlib.sha256(news_db.read_bytes()).hexdigest()
    report = evaluate_automatic_outcomes(
        news_db, mappings, prices, model="gemma4:e2b",
        price_source="synthetic fixture", limit=20,
    )
    assert hashlib.sha256(news_db.read_bytes()).hexdigest() == before
    assert report["sourceType"] == "REAL_AUTOMATIC_SQLITE_GEMMA_LEDGER"
    assert report["articlesInspected"] == 1
    assert report["impactsInspected"] == 1
    assert report["items"][0]["sourceAuditStatus"] == "STORED_ORIGINAL_SHA_MATCHED"
    assert report["items"][0]["horizons"][0]["stockReturn"] == pytest.approx(0.03)
    assert report["metrics"]["1d"]["correct"] == 1
    assert report["marketPricesAnalyzed"] is True
    assert report["forecastSealedBeforeOutcomes"] is False
    assert report["prospectivePredictionAccuracy"] is None
    assert report["trainingEligible"] is False
    assert report["productionDatabaseModified"] is False


def test_unmapped_stored_gemma_impacts_remain_explicit_unavailable(
    tmp_path: Path,
) -> None:
    _, mappings, prices = _fixture(tmp_path)
    db_path = _ledger(tmp_path, target="Unreviewed target")
    report = evaluate_automatic_outcomes(
        db_path, mappings, prices,
        model="gemma4:e2b", price_source="synthetic fixture",
    )
    assert report["metrics"]["1d"]["denominator"] == 0
    assert report["items"][0]["status"] == "UNMAPPED"
    assert report["metrics"]["1d"]["unavailable"] == 1


def test_corrupted_original_source_never_becomes_market_result(
    tmp_path: Path,
) -> None:
    _, mappings, prices = _fixture(tmp_path)
    db_path = _ledger(tmp_path, corrupt_body=True)
    with pytest.raises(ValueError, match="No source-admissible"):
        evaluate_automatic_outcomes(
            db_path, mappings, prices,
            model="gemma4:e2b", price_source="synthetic fixture",
        )


def test_wrong_model_cannot_reuse_another_models_forecast(
    tmp_path: Path,
) -> None:
    db_path = _ledger(tmp_path)
    result, coverage = _automatically_analyzed(
        db_path, "different-model", limit=10,
    )
    assert result == []
    assert coverage["completedCurrentBodyModelRows"] == 0


def test_latest_original_body_replaces_stale_analysis_for_join(
    tmp_path: Path,
) -> None:
    db_path = _ledger(tmp_path)
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE article_full_text SET body=?, body_sha256=? WHERE article_id=1",
            ("updated original", hashlib.sha256(b"updated original").hexdigest()),
        )
    result, coverage = _automatically_analyzed(
        db_path, "gemma4:e2b", limit=10,
    )
    assert result == []
    assert coverage["completedCurrentBodyModelRows"] == 0
