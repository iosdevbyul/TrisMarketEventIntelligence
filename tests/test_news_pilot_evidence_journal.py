"""Isolated pilot evidence archiving, with full source provenance checks."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.news_pilot_evidence_journal import (
    NewsPilotEvidenceJournal,
    validate_pilot_report,
)

EVIDENCE = (
    "A Microsoft outage is affecting services including Teams, "
    "Outlook and AI software Copilot."
)
MECHANISM = (
    "The reported outage could disrupt productivity and costs, but "
    "any financial reaction or customer churn remains unverified."
)


def seed(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript(
            """CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, classification TEXT, status TEXT,
                source_url TEXT, evidence TEXT
            );
            CREATE TABLE event_metadata (
                event_id INTEGER, claim_kind TEXT
            );"""
        )
        db.execute(
            "INSERT INTO economic_events VALUES (?,?,?,?,?)",
            (69, "SERVICE_INCIDENT", "UNVERIFIED",
             "https://example.org/service/outage", EVIDENCE),
        )
        db.execute(
            "INSERT INTO event_metadata VALUES (?,?)", (69, "REPORTED")
        )


def sample() -> dict:
    return {
        "schema_version": "news_two_stage_pilot_v1",
        "selected_events": 2,
        "direction_model_calls": 2,
        "strength_model_calls": 1,
        "model_id": "gemma2:latest",
        "stock_returns_predicted": False,
        "production_database_modified": False,
        "operational_quality_verified": False,
        "items": [
            {
                "event_id": 69, "classification": "SERVICE_INCIDENT",
                "event_status": "UNVERIFIED",
                "source_url": "https://example.org/service/outage",
                "status": "TWO_STAGE_OUTPUT_VALIDATED",
                "signal": {
                    "event_id": 69, "market": "GLOBAL", "sector": "TECH",
                    "direction": "NEGATIVE", "intensity": "MODERATE",
                    "intensity_score": 55.0, "horizon": "SHORT_TERM",
                    "uncertainty": "HIGH",
                    "rationale": MECHANISM + " News pressure: Outage affects services.",
                    "evidence_excerpt": EVIDENCE,
                    "source_url": "https://example.org/service/outage",
                    "model_id": "gemma2:latest",
                    "prompt_version": "news-pressure-intensity-v1",
                    "generated_at": "2026-11-15T06:10:43+09:00",
                    "evidence_assessment": {
                        "reported_fact": EVIDENCE,
                        "conditional_mechanism": MECHANISM,
                        "verification_needed": [
                            "CONFIRM_OUTAGE_DURATION", "CONFIRM_AFFECTED_USER_SCOPE",
                            "CONFIRM_SERVICE_RECOVERY_STATUS",
                            "CONFIRM_MATERIAL_BUSINESS_IMPACT",
                        ],
                        "fact_status": "UNVERIFIED",
                        "strength_basis": "HEURISTIC_UNCALIBRATED",
                    },
                },
            },
            {
                "event_id": 70, "classification": "ECONOMIC_INDICATOR",
                "event_status": "UNVERIFIED",
                "source_url": "https://example.org/indicators",
                "status": "NO_SUPPORTED_DIRECTION",
            },
        ],
    }


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.sqlite3"
    output = tmp_path / "separate-review.sqlite3"
    report = tmp_path / "pilot.json"
    seed(source)
    with sqlite3.connect(source) as db:
        db.execute(
            "INSERT INTO economic_events VALUES (?,?,?,?,?)",
            (70, "ECONOMIC_INDICATOR", "UNVERIFIED",
             "https://example.org/indicators",
             "The indicator report contained no assessed market exposure."),
        )
    report.write_text(json.dumps(sample()), encoding="utf-8")
    return source, output, report


def test_preview_then_commit_isolated_review_journal_idempotent(tmp_path: Path) -> None:
    source, journal_path, report_path = fixture(tmp_path)
    with sqlite3.connect(source) as db:
        before = tuple(db.iterdump())
    journal = NewsPilotEvidenceJournal(source_db=source, journal_db=journal_path)
    preview = journal.import_report(report_path)
    assert preview["acceptedCount"] == 1
    assert preview["rejectedOrUnassessedCount"] == 1
    assert preview["status"] == "VALIDATED_PREVIEW_ONLY"
    assert not journal_path.exists()
    first = journal.import_report(report_path, commit=True)
    assert first["inserted"] == 1
    assert first["reviewStatus"] == "REQUIRES_HUMAN_REVIEW"
    assert journal_path.exists()
    retry = journal.import_report(report_path, commit=True)
    assert retry["inserted"] == 0
    with sqlite3.connect(journal_path) as db:
        rows = db.execute(
            "SELECT event_id, signal_json, review_status FROM news_pilot_evidence"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 69
    assert json.loads(rows[0][1])["evidence_assessment"]["fact_status"] == "UNVERIFIED"
    assert rows[0][2] == "REQUIRES_HUMAN_REVIEW"
    with sqlite3.connect(source) as db:
        assert tuple(db.iterdump()) == before


@pytest.mark.parametrize("mutation", [
    "wrong_quote", "wrong_source_url", "fake_status", "fake_mechanism",
    "high_unverified", "bad_model", "forged_event", "fake_price",
    "missing_assessment", "wrong_model_calls", "unmatched_event",
])
def test_forged_or_unsupported_reports_fail_without_writes(
    tmp_path: Path, mutation: str,
) -> None:
    source, journal_path, report_path = fixture(tmp_path)
    data = sample()
    item = data["items"][0]
    signal = item["signal"]
    if mutation == "wrong_quote":
        signal["evidence_excerpt"] = "Invented outage"
        signal["evidence_assessment"]["reported_fact"] = "Invented outage"
    elif mutation == "wrong_source_url":
        signal["source_url"] = "https://other.example.org/outage"
    elif mutation == "fake_status":
        signal["evidence_assessment"]["fact_status"] = "VERIFIED"
    elif mutation == "fake_mechanism":
        signal["evidence_assessment"]["conditional_mechanism"] = "False cause"
    elif mutation == "high_unverified":
        signal["intensity"] = "HIGH"
        signal["intensity_score"] = 80
    elif mutation == "bad_model":
        signal["model_id"] = "remote-price-forecast"
    elif mutation == "forged_event":
        signal["event_id"] = 999
    elif mutation == "fake_price":
        signal["predicts_stock_return"] = True
    elif mutation == "missing_assessment":
        signal.pop("evidence_assessment")
    elif mutation == "wrong_model_calls":
        data["strength_model_calls"] = 0
    else:
        item["event_id"] = 999
        signal["event_id"] = 999
    report_path.write_text(json.dumps(data), encoding="utf-8")
    journal = NewsPilotEvidenceJournal(source_db=source, journal_db=journal_path)
    with pytest.raises(ValueError):
        journal.import_report(report_path, commit=True)
    assert not journal_path.exists()


def test_unrelated_or_same_db_cannot_be_used_as_journal(tmp_path: Path) -> None:
    source, journal_path, report_path = fixture(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        NewsPilotEvidenceJournal(source_db=source, journal_db=source)
    with sqlite3.connect(journal_path) as db:
        db.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    with pytest.raises(ValueError, match="unrelated"):
        NewsPilotEvidenceJournal(
            source_db=source, journal_db=journal_path,
        ).import_report(report_path, commit=True)
    with sqlite3.connect(journal_path) as db:
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE name='unrelated'"
        ).fetchone() is not None


def test_report_sha_and_missing_source_checks(tmp_path: Path) -> None:
    source, journal_path, report_path = fixture(tmp_path)
    checked = validate_pilot_report(source_db=source, report_path=report_path)
    assert checked["acceptedCount"] == 1
    assert len(checked["reportSha256"]) == 64
    missing = tmp_path / "not-present.sqlite3"
    with pytest.raises(ValueError, match="source DB"):
        NewsPilotEvidenceJournal(
            source_db=missing, journal_db=journal_path,
        ).import_report(report_path, commit=True)
    assert not missing.exists()
    assert not journal_path.exists()
