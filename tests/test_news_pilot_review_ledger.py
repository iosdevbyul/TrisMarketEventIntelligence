"""Read-only queue and immutable decisions never change production data."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.news_pilot_review_ledger import (
    NewsPilotReviewLedger,
)


def seeded(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.sqlite3"
    journal = tmp_path / "research.sqlite3"
    evidence = "A Microsoft outage is affecting Teams, Outlook and Copilot."
    with sqlite3.connect(source) as db:
        db.executescript(
            """CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, evidence TEXT,
                source_url TEXT, status TEXT
            );"""
        )
        db.execute(
            "INSERT INTO economic_events VALUES (?,?,?,?)",
            (69, evidence, "https://example.org/69", "UNVERIFIED"),
        )
    signal = {
        "event_id": 69, "market": "GLOBAL", "sector": "TECH",
        "direction": "NEGATIVE", "intensity": "MODERATE",
        "intensity_score": 55, "horizon": "SHORT_TERM",
        "uncertainty": "HIGH",
        "rationale": (
            "Service disruption could have costs if prolonged, "
            "but user exposure and market reactions are unverified."
        ),
        "evidence_excerpt": evidence,
        "source_url": "https://example.org/69",
        "model_id": "gemma2:latest",
        "prompt_version": "news-pressure-intensity-v1",
        "generated_at": "2026-11-15T06:10:43+09:00",
        "evidence_assessment": {
            "reported_fact": evidence,
            "conditional_mechanism": (
                "Service disruption could have costs if prolonged, "
                "but user exposure and market reactions are unverified."
            ),
            "verification_needed": ["CONFIRM_OUTAGE_DURATION"],
            "fact_status": "UNVERIFIED",
            "strength_basis": "HEURISTIC_UNCALIBRATED",
        },
    }
    raw = json.dumps(signal)
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    with sqlite3.connect(journal) as db:
        db.executescript(
            """CREATE TABLE news_pilot_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_sha256 TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                source_evidence_sha256 TEXT NOT NULL,
                signal_json TEXT NOT NULL,
                signal_sha256 TEXT NOT NULL,
                review_status TEXT NOT NULL
                    CHECK(review_status='REQUIRES_HUMAN_REVIEW'),
                UNIQUE(report_sha256, event_id)
            );"""
        )
        db.execute(
            "INSERT INTO news_pilot_evidence("
            "report_sha256,event_id,source_evidence_sha256,"
            "signal_json,signal_sha256,review_status) VALUES(?,?,?,?,?,?)",
            (digest("report"), 69, digest(evidence), raw,
             digest(raw), "REQUIRES_HUMAN_REVIEW"),
        )
    return source, journal


def test_list_preview_commit_and_idempotent_review_without_publish(
    tmp_path: Path,
) -> None:
    source, journal = seeded(tmp_path)
    ledger = NewsPilotReviewLedger(source_db=source, journal_db=journal)
    with sqlite3.connect(source) as db:
        baseline = tuple(db.iterdump())
    queue = ledger.list_entries()
    assert len(queue["items"]) == 1
    assert queue["items"][0]["review_status"] == "REQUIRES_HUMAN_REVIEW"
    assert queue["items"][0]["published"] is False
    with sqlite3.connect(journal) as db:
        before = tuple(db.iterdump())
    kwargs = {
        "evidence_id": 1, "decision": "APPROVED_FOR_RESEARCH",
        "reviewer": "local-reviewer",
        "reason": "Original source excerpt verified as quoted; hypothesis is conditional.",
    }
    preview = ledger.decide(**kwargs)
    assert preview["status"] == "REVIEW_PREVIEW_ONLY"
    assert preview["journalModified"] is False
    with sqlite3.connect(journal) as db:
        assert tuple(db.iterdump()) == before
    saved = ledger.decide(**kwargs, commit=True)
    assert saved["status"] == "REVIEW_RECORDED"
    assert saved["published"] is False
    assert ledger.decide(**kwargs, commit=True)["status"] == "ALREADY_REVIEWED"
    assert ledger.list_entries()["items"][0]["review_status"] == "APPROVED_FOR_RESEARCH"
    with sqlite3.connect(journal) as db:
        assert db.execute(
            "SELECT review_status FROM news_pilot_evidence WHERE id=1"
        ).fetchone()[0] == "REQUIRES_HUMAN_REVIEW"
        assert db.execute(
            "SELECT COUNT(*) FROM news_pilot_review_decisions"
        ).fetchone()[0] == 1
    with sqlite3.connect(source) as db:
        assert tuple(db.iterdump()) == baseline


def test_reject_and_conflicting_decision_fail_closed(tmp_path: Path) -> None:
    source, journal = seeded(tmp_path)
    ledger = NewsPilotReviewLedger(source_db=source, journal_db=journal)
    kwargs = {
        "evidence_id": 1, "decision": "REJECTED",
        "reviewer": "local-reviewer",
        "reason": "The source did not establish a measurable financial consequence.",
    }
    assert ledger.decide(**kwargs, commit=True)["status"] == "REVIEW_RECORDED"
    assert ledger.list_entries()["items"][0]["review_status"] == "REJECTED"
    with pytest.raises(ValueError, match="Immutable"):
        ledger.decide(**{**kwargs, "decision": "APPROVED_FOR_RESEARCH"}, commit=True)
    with pytest.raises(ValueError, match="Immutable"):
        ledger.decide(**{**kwargs, "reason": "Different explanation recorded later"}, commit=True)
    with sqlite3.connect(journal) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM news_pilot_review_decisions"
        ).fetchone()[0] == 1


@pytest.mark.parametrize("change", ["tampered_signal", "tampered_source",
                                    "tampered_source_url", "tampered_fact_status"])
def test_tampered_provenance_blocks_review_before_write(
    tmp_path: Path, change: str,
) -> None:
    source, journal = seeded(tmp_path)
    if change == "tampered_signal":
        with sqlite3.connect(journal) as db:
            db.execute(
                "UPDATE news_pilot_evidence SET signal_json='{}' WHERE id=1"
            )
    elif change == "tampered_source":
        with sqlite3.connect(source) as db:
            db.execute("UPDATE economic_events SET evidence='changed' WHERE id=69")
    elif change == "tampered_source_url":
        with sqlite3.connect(source) as db:
            db.execute(
                "UPDATE economic_events SET source_url='https://wrong' WHERE id=69"
            )
    else:
        with sqlite3.connect(source) as db:
            db.execute(
                "UPDATE economic_events SET status='VERIFIED' WHERE id=69"
            )
    ledger = NewsPilotReviewLedger(source_db=source, journal_db=journal)
    with pytest.raises(ValueError):
        ledger.decide(
            evidence_id=1, decision="APPROVED_FOR_RESEARCH",
            reviewer="reviewer", reason="The review cannot trust mutated evidence.",
            commit=True,
        )
    with sqlite3.connect(journal) as db:
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE name='news_pilot_review_decisions'"
        ).fetchone() is None


@pytest.mark.parametrize("invalid", [
    {"evidence_id": 0}, {"evidence_id": 2},
    {"reviewer": " "}, {"reason": "short"},
    {"decision": "PUBLISHED"},
])
def test_bad_decisions_fail_closed(tmp_path: Path, invalid: dict) -> None:
    source, journal = seeded(tmp_path)
    ledger = NewsPilotReviewLedger(source_db=source, journal_db=journal)
    kwargs = {
        "evidence_id": 1, "decision": "REJECTED",
        "reviewer": "reviewer",
        "reason": "Insufficient independently confirmed impact evidence.",
    }
    with pytest.raises(ValueError):
        ledger.decide(**{**kwargs, **invalid}, commit=True)
    with sqlite3.connect(journal) as db:
        assert db.execute(
            "SELECT name FROM sqlite_master WHERE name='news_pilot_review_decisions'"
        ).fetchone() is None


def test_live_source_path_as_journal_prohibited(tmp_path: Path) -> None:
    source, _ = seeded(tmp_path)
    with pytest.raises(ValueError, match="production"):
        NewsPilotReviewLedger(source_db=source, journal_db=source)
