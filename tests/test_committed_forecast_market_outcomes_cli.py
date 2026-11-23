"""Read-only real daily-price scoring of locally committed source forecasts.

All issuer/news/price values here are synthetic and never training labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.automatic_forecast_precommitment import (
    initialize,
    record_new_analysis,
)
from tris_market_event_intelligence.committed_forecast_market_outcomes_cli import (
    evaluate,
)


def _prices(tmp_path: Path) -> Path:
    path = tmp_path / "historical-adjusted.csv"
    values = [
        ("2026-11-20", 95, 390),
        ("2026-11-24", 100, 400),
        ("2026-11-25", 103, 420),
        ("2026-11-27", 104, 421),
        ("2026-11-30", 105, 422),
        ("2026-12-01", 106, 423),
        ("2026-12-02", 110, 424),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "date", "adjusted_close", "source"])
        for day, stock, benchmark in values:
            writer.writerow(["NVDA", day, stock, "synthetic fixture"])
            writer.writerow(["SPY", day, benchmark, "synthetic fixture"])
    return path


def _source(tmp_path: Path, *, direction: str = "UP",
            analyzed_at: str = "2026-11-23T10:03:00+00:00",
            body_observed: str = "2026-11-23T10:02:00+00:00",
            mapped: bool = True) -> Path:
    db_path = tmp_path / "news.sqlite3"
    body = "Original issuer news with exact source quote and financial mechanism."
    digest = hashlib.sha256(body.encode()).hexdigest()
    analysis = {
        "summary": "Reasoning", "limitations": "Outcome unknown",
        "impacts": [{
            "target": "Nvidia", "direction": direction,
            "reason": "Possible demand surprise", "evidence_id": "E1",
            "evidence_quote": "Original issuer news",
        }],
    }
    with sqlite3.connect(db_path) as db:
        db.executescript("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY, source TEXT, title TEXT, url TEXT,
            published_at TEXT, detected_at TEXT
        );
        CREATE TABLE article_full_text (
            article_id INTEGER PRIMARY KEY, status TEXT, body TEXT,
            body_sha256 TEXT, observed_at TEXT
        );
        CREATE TABLE automatic_news_analyses (
            article_id INTEGER NOT NULL, body_sha256 TEXT NOT NULL, model TEXT NOT NULL,
            status TEXT NOT NULL, processed_at TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            PRIMARY KEY(article_id,body_sha256,model)
        );
        """)
        db.execute(
            "INSERT INTO articles VALUES (1,?,?,?,?,?)",
            ("issuer", "Original issuer announcement", "https://example.org/issuer",
             "2026-11-23T10:00:00+00:00", "2026-11-23T10:01:00+00:00"),
        )
        db.execute(
            "INSERT INTO article_full_text VALUES (1,'COMPLETE',?,?,?)",
            (body, digest, body_observed),
        )
        db.execute(
            "INSERT INTO automatic_news_analyses VALUES (?,?,?,?,?,?)",
            (1, digest, "gemma4:e2b", "COMPLETED", analyzed_at,
             json.dumps(analysis)),
        )
        initialize(db)
        record_new_analysis(
            db,
            article={
                "articleId": 1, "publishedAt": "2026-11-23T10:00:00+00:00",
                "detectedAt": "2026-11-23T10:01:00+00:00",
                "bodyObservedAt": body_observed,
            },
            analysis=analysis, model="gemma4:e2b", body_sha256=digest,
            analyzed_at=analyzed_at,
            mappings=({
                "Nvidia": {
                    "market": "XNAS", "symbol": "NVDA", "benchmark": "SPY",
                    "verifiedAt": "2026-11-23T09:00:00+00:00",
                }
            } if mapped else {}),
        )
    return db_path


def test_committed_1d_and_5d_original_source_actual_market_labels(
    tmp_path: Path,
) -> None:
    db = _source(tmp_path)
    prices = _prices(tmp_path)
    old_news = hashlib.sha256(db.read_bytes()).hexdigest()
    old_prices = hashlib.sha256(prices.read_bytes()).hexdigest()
    report = evaluate(db, prices)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == old_news
    assert hashlib.sha256(prices.read_bytes()).hexdigest() == old_prices
    assert report["committedRows"] == 2
    assert report["commitmentChainVerifiedLocally"] is True
    assert report["statusCounts"] == {"OBSERVED_UNAUTHENTICATED_PRECOMMITMENT": 2}
    for row in report["rows"]:
        assert row["entryDay"] == "2026-11-24"
        assert row["symbol"] == "NVDA"
        assert row["sourceIdentity"] == "MATCHED_ORIGINAL_MODEL_AND_BODY"
        assert row["verdict"] == "CORRECT"
    assert report["horizons"]["1d"]["correct"] == 1
    assert report["horizons"]["1d"]["candidateLocalAgreement"] == 1.0
    assert report["horizons"]["1d"]["independentlyAuthenticatedProspectiveAccuracy"] is None
    assert report["horizons"]["5d"]["correct"] == 1
    assert report["rows"][0]["relativeVerdict"] == "INCORRECT"
    assert report["prospectivePredictionAccuracy"] is None
    assert report["trainingEligible"] is False
    assert report["externalTimestampAuthenticated"] is False
    assert report["causalityEstablished"] is False


def test_unmapped_and_late_forecasts_never_get_price_verdict(tmp_path: Path) -> None:
    prices = _prices(tmp_path)
    for name, options in (
        ("unmapped", {"mapped": False}),
        ("late", {"analyzed_at": "2026-11-24T10:03:00+00:00"}),
    ):
        folder = tmp_path / name
        folder.mkdir()
        db = _source(folder, **options)
        report = evaluate(db, prices)
        assert all("verdict" not in row for row in report["rows"])
        assert report["horizons"]["1d"]["denominator"] == 0
        assert report["prospectivePredictionAccuracy"] is None


def test_no_post_publication_or_short_horizon_return_is_unavailable(
    tmp_path: Path,
) -> None:
    db = _source(tmp_path)
    prices = _prices(tmp_path)
    with prices.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "date", "adjusted_close", "source"])
        for symbol, value in (("NVDA", 100), ("SPY", 400)):
            writer.writerow([symbol, "2026-11-20", value, "synthetic"])
    report = evaluate(db, prices)
    assert report["statusCounts"] == {"NO_POST_PUBLICATION_ENTRY_CLOSE": 2}
    assert report["horizons"]["1d"]["candidateLocalAgreement"] is None
    assert report["horizons"]["5d"]["denominator"] == 0


def test_replaced_model_output_and_original_source_clock_fail_closed(
    tmp_path: Path,
) -> None:
    db = _source(tmp_path)
    prices = _prices(tmp_path)
    with sqlite3.connect(db) as conn:
        changed = json.dumps({
            "impacts": [{
                "target": "Nvidia", "direction": "DOWN",
                "reason": "Possible demand surprise",
                "evidence_quote": "Original issuer news",
            }],
        })
        conn.execute(
            "UPDATE automatic_news_analyses SET analysis_json=?",
            (changed,),
        )
    report = evaluate(db, prices)
    assert report["statusCounts"] == {
        "COMMITTED_FORECAST_DOES_NOT_MATCH_ORIGINAL_MODEL": 2,
    }
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE article_full_text SET observed_at=?",
            ("2026-11-24T12:00:00+00:00",),
        )
    report = evaluate(db, prices)
    assert report["statusCounts"] == {"ORIGINAL_MODEL_OR_BODY_REVISED": 2}


def test_abstained_retained_but_never_claimed_correct(tmp_path: Path) -> None:
    db = _source(tmp_path, direction="UNCERTAIN")
    report = evaluate(db, _prices(tmp_path))
    assert report["horizons"]["1d"]["abstained"] == 1
    assert report["horizons"]["1d"]["denominator"] == 0
    assert report["rows"][0]["verdict"] == "ABSTAINED"


def test_manifest_mismatch_and_local_chain_tamper_refused(tmp_path: Path) -> None:
    db = _source(tmp_path)
    prices = _prices(tmp_path)
    manifest = tmp_path / "quotes.manifest.json"
    manifest.write_text(json.dumps({"outputSha256": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        evaluate(db, prices, manifest)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TRIGGER no_forecast_commitment_update")
        conn.execute("UPDATE automatic_forecast_commitments SET direction='DOWN' "
                     "WHERE horizon_sessions=1")
    with pytest.raises(ValueError, match="CONTENT_HASH_MISMATCH"):
        evaluate(db, prices)


def test_no_opt_in_never_generates_old_prediction_or_scores(tmp_path: Path) -> None:
    db = tmp_path / "empty.sqlite3"
    with sqlite3.connect(db):
        pass
    result = evaluate(db, _prices(tmp_path))
    assert result["status"] == "NOT_OPTED_IN"
    assert result["committedRows"] == 0
    assert result["prospectivePredictionAccuracy"] is None
