"""Offline opt-in new-forecast commitment, immutability and integrity regressions."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.automatic_forecast_precommitment import (
    ELIGIBLE,
    INVALID,
    LATE,
    UNMAPPED,
    initialize,
    record_new_analysis,
    verify_chain,
)
from tris_market_event_intelligence.automatic_forecast_precommitment_audit_cli import (
    audit,
)
from tris_market_event_intelligence.automatic_news_pipeline import process


def analysis(target: str = "Nvidia", direction: str = "UP") -> dict:
    return {
        "summary": "Original news", "limitations": "Potential response",
        "impacts": [{
            "target": target, "direction": direction, "reason": "News mechanism",
            "evidence_id": "E1", "evidence_quote": "Original issuer news",
        }],
    }


def article(published: str = "2026-11-23T10:00:00+00:00") -> dict:
    return {
        "articleId": 7, "publishedAt": published,
        "detectedAt": "2026-11-23T10:01:00+00:00",
        "bodyObservedAt": "2026-11-23T10:02:00+00:00",
    }


def reviewed() -> dict:
    return {"Nvidia": {
        "market": "XNAS", "symbol": "NVDA", "benchmark": "SPY",
        "verifiedAt": "2026-11-23T09:00:00+00:00",
    }}


def test_immutable_first_model_completion_with_direction_horizons(
    tmp_path: Path,
) -> None:
    path = tmp_path / "commitments.sqlite3"
    with sqlite3.connect(path) as db:
        initialize(db)
        counts = record_new_analysis(
            db, article=article(), analysis=analysis(), model="gemma4:e2b",
            body_sha256="a" * 64, analyzed_at="2026-11-23T10:03:00+00:00",
            mappings=reviewed(),
        )
        assert counts == {ELIGIBLE: 2}
        assert verify_chain(db)["records"] == 2
        with pytest.raises(sqlite3.IntegrityError, match="IMMUTABLE"):
            db.execute("UPDATE automatic_forecast_commitments SET direction='DOWN'")
        with pytest.raises(sqlite3.IntegrityError, match="IMMUTABLE"):
            db.execute("DELETE FROM automatic_forecast_commitments")
        assert db.execute(
            "SELECT COUNT(*) FROM automatic_forecast_commitments"
        ).fetchone()[0] == 2
    saved = audit(path)
    assert saved["localHashIntegrity"]["integrity"] == "LOCAL_HASH_CHAIN_VALID"
    assert saved["firstPublicAvailabilityVerified"] is False
    assert saved["externallyTimestamped"] is False
    assert saved["trainingEligible"] is False


def test_old_and_unmapped_predictions_are_preserved_but_not_eligible(
    tmp_path: Path,
) -> None:
    path = tmp_path / "commitments.sqlite3"
    with sqlite3.connect(path) as db:
        initialize(db)
        old = record_new_analysis(
            db, article=article(), analysis=analysis(), model="gemma4:e2b",
            body_sha256="b" * 64, analyzed_at="2026-11-24T10:03:00+00:00",
            mappings=reviewed(),
        )
        assert old == {LATE: 2}
        missing = record_new_analysis(
            db, article={**article(), "articleId": 8},
            analysis=analysis(target="Unlisted private bank"), model="gemma4:e2b",
            body_sha256="c" * 64, analyzed_at="2026-11-23T10:03:00+00:00",
            mappings=reviewed(),
        )
        assert missing == {UNMAPPED: 2}
        invalid = record_new_analysis(
            db, article={**article(), "articleId": 9,
                         "publishedAt": "not a timestamp"},
            analysis=analysis(), model="gemma4:e2b",
            body_sha256="d" * 64, analyzed_at="2026-11-23T10:03:00+00:00",
            mappings=reviewed(),
        )
        assert invalid == {INVALID: 2}
        assert verify_chain(db)["records"] == 6


def test_no_automatic_backfill_or_unapproved_mapping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator.sqlite3"
    body = "Original issuer news with a source-backed market mechanism."
    digest = hashlib.sha256(body.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY, source TEXT, title TEXT, url TEXT,
            published_at TEXT, detected_at TEXT
        );
        CREATE TABLE article_full_text (
            article_id INTEGER PRIMARY KEY, status TEXT, body TEXT,
            body_sha256 TEXT, sparse_pages TEXT
        );
        """)
        db.execute(
            "INSERT INTO articles VALUES (1,?,?,?,?,?)",
            ("issuer", "Nvidia release", "https://example.com/issuer",
             "2026-11-23T10:00:00+00:00", "2026-11-23T10:01:00+00:00"),
        )
        db.execute(
            "INSERT INTO article_full_text VALUES (1,'COMPLETE',?,?,?)",
            (body, digest, "[]"),
        )
    mappings = tmp_path / "verified.csv"
    with mappings.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "target", "market", "symbol", "benchmark", "verified_at",
        ])
        writer.writerow([
            "Nvidia", "XNAS", "NVDA", "SPY", "2026-11-23T09:00:00+00:00",
        ])
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.ask_gemma",
        return_value=analysis(),
    ), patch(
        "tris_market_event_intelligence.automatic_news_pipeline.now",
        return_value="2026-11-23T10:03:00+00:00",
    ):
        first = process(
            path, collect=False, limit=5,
            seal_forecasts=True, mappings_path=mappings,
        )
        again = process(
            path, collect=False, limit=5,
            seal_forecasts=True, mappings_path=mappings,
        )
    assert first["completed"] == 1
    assert first["newForecastCommitments"] == {ELIGIBLE: 2}
    assert first["forecastCommitmentAudit"]["records"] == 2
    assert again["completed"] == 0
    assert again["newForecastCommitments"] == {}
    assert again["forecastCommitmentAudit"]["records"] == 2
    assert audit(path)["localHashIntegrity"]["records"] == 2
    with sqlite3.connect(path) as db:
        row = db.execute(
            "SELECT direction, symbol, benchmark, market_cutoff_policy,"
            " external_timestamp_authenticated"
            " FROM automatic_forecast_commitments LIMIT 1"
        ).fetchone()
        assert row[0:3] == ("UP", "NVDA", "SPY")
        assert row[4] == 0
        assert json.loads(db.execute(
            "SELECT analysis_json FROM automatic_news_analyses"
        ).fetchone()[0])["impacts"][0]["direction"] == "UP"


def test_opt_in_refuses_implicit_mapping_and_does_not_edit_original(
    tmp_path: Path,
) -> None:
    path = tmp_path / "not-changed.sqlite3"
    path.write_bytes(b"not a valid SQLite file")
    with pytest.raises(ValueError, match="Explicit reviewed"):
        process(path, collect=False, seal_forecasts=True)
    assert path.read_bytes() == b"not a valid SQLite file"
    assert audit_missing(tmp_path)


def audit_missing(tmp_path: Path) -> bool:
    path = tmp_path / "empty.sqlite3"
    with sqlite3.connect(path):
        pass
    return audit(path)["status"] == "NOT_OPTED_IN"


def test_hash_verifier_detects_tampering_even_without_sql_trigger(
    tmp_path: Path,
) -> None:
    path = tmp_path / "local.sqlite3"
    with sqlite3.connect(path) as db:
        initialize(db)
        record_new_analysis(
            db, article=article(), analysis=analysis(),
            model="gemma4:e2b", body_sha256="e" * 64,
            analyzed_at="2026-11-23T10:03:00+00:00", mappings=reviewed(),
        )
        db.execute("DROP TRIGGER no_forecast_commitment_update")
        db.execute("UPDATE automatic_forecast_commitments SET direction='DOWN' "
                   "WHERE impact_index=0 AND horizon_sessions=1")
        with pytest.raises(ValueError, match="CONTENT_HASH_MISMATCH"):
            verify_chain(db)
