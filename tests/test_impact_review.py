"""Regression: no DB writes, stale-sheet protection, and honest human-only scoring."""
import csv
import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.impact_review import (
    DIMENSIONS, export_impact_review, score_impact_review,
)


def database(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, status TEXT, source TEXT,
                source_url TEXT, capture_sha256 TEXT, evidence TEXT
            );
            CREATE TABLE event_metadata (event_id INTEGER, claim_kind TEXT);
            CREATE TABLE market_impacts (
                id INTEGER PRIMARY KEY, event_id INTEGER, model_id TEXT,
                prompt_version TEXT, rationale TEXT, evidence_excerpt TEXT,
                market TEXT, sector TEXT, direction TEXT, horizon TEXT,
                uncertainty TEXT
            );
        """)
        db.execute("INSERT INTO economic_events VALUES (6,'UNVERIFIED',?,?,?,?)", (
            "news", "https://example.org/article", "abc123",
            "반도체 경기 호조의 영향으로 수출과 투자의 높은 증가세가 지속되고",
        ))
        db.execute("INSERT INTO event_metadata VALUES (6,'REPORTED')")
        db.execute("INSERT INTO market_impacts VALUES (1,6,?,?,?,?,?,?,?,?,?)", (
            "gemma2:latest", "market-impact-v6",
            "반도체 경기 호조의 영향으로 수출과 투자의 높은 증가세가 지속되고",
            "반도체 경기 호조의 영향으로 수출과 투자의 높은 증가세",
            "GLOBAL", "SEMICONDUCTORS", "POSITIVE", "MEDIUM_TERM", "HIGH",
        ))


def rewrite(sheet: Path, *, value: str | None = None) -> None:
    with sheet.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if value is not None:
        for field in DIMENSIONS:
            rows[0][field] = value if value in DIMENSIONS[field] else DIMENSIONS[field][0]
    with sheet.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_real_v6_case_export_and_manual_score(tmp_path: Path) -> None:
    db, sheet = tmp_path / "news.sqlite3", tmp_path / "review.csv"
    database(db)
    before = db.read_bytes()
    report = export_impact_review(db, sheet, count=30)
    assert report["exported_impacts"] == 1
    assert report["distinct_events"] == 1
    assert not report["quality_verified"]
    with sheet.open(encoding="utf-8-sig", newline="") as stream:
        row = list(csv.DictReader(stream))[0]
    assert row["automated_quality_flag"] == "RATIONALE_INCOMPLETE"
    assert row["source_support"] == ""
    assert score_impact_review(db, sheet)["reviewed_impacts"] == 0
    rewrite(sheet, value="UNSUPPORTED")
    scores = score_impact_review(db, sheet)
    assert scores["reviewed_impacts"] == 1
    assert scores["dimension_counts"]["source_support"]["UNSUPPORTED"] == 1
    assert scores["quality_verified"] is False
    assert db.read_bytes() == before


def test_stale_record_and_tampered_sheet_rejected(tmp_path: Path) -> None:
    db, sheet = tmp_path / "news.sqlite3", tmp_path / "review.csv"
    database(db)
    export_impact_review(db, sheet)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE market_impacts SET rationale='changed' WHERE id=1")
    with pytest.raises(ValueError, match="no longer matches"):
        score_impact_review(db, sheet)


def test_no_overwrite_and_no_missing_db_creation(tmp_path: Path) -> None:
    db, sheet = tmp_path / "news.sqlite3", tmp_path / "review.csv"
    with pytest.raises(FileNotFoundError):
        export_impact_review(db, sheet)
    assert not db.exists()
    database(db)
    export_impact_review(db, sheet)
    with pytest.raises(FileExistsError):
        export_impact_review(db, sheet)


def test_incomplete_or_bad_labels_rejected(tmp_path: Path) -> None:
    db, sheet = tmp_path / "news.sqlite3", tmp_path / "review.csv"
    database(db)
    export_impact_review(db, sheet)
    with sheet.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        row = next(reader)
    row["source_support"] = "SUPPORTED"
    with sheet.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="Partially labelled"):
        score_impact_review(db, sheet)
    row.update({dimension: "FAKE" for dimension in DIMENSIONS})
    with sheet.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="Unknown human review label"):
        score_impact_review(db, sheet)
