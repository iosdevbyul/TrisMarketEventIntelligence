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
                source_url TEXT, status TEXT, classification TEXT
            );"""
        )
        db.execute(
            "INSERT INTO economic_events VALUES (?,?,?,?,?)",
            (69, evidence, "https://example.org/69", "UNVERIFIED",
             "SERVICE_INCIDENT"),
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


def test_review_audit_reports_pending_and_approved_without_publishing(
    tmp_path: Path,
) -> None:
    from tris_market_event_intelligence.news_pilot_review_audit import (
        NewsPilotReviewAuditor,
    )

    source, journal = seeded(tmp_path)
    auditor = NewsPilotReviewAuditor(source_db=source, journal_db=journal)
    before = auditor.audit()
    assert before["status"] == "LOCAL_CONSISTENCY_VERIFIED"
    assert before["counts"]["REQUIRES_HUMAN_REVIEW"] == 1
    assert before["counts"]["APPROVED_FOR_RESEARCH"] == 0
    assert before["items"][0]["published"] is False
    assert before["reportIsCryptographicallyAuthentic"] is False
    assert before["independentSourceVerification"] is False
    assert before["journalModified"] is False
    assert before["productionDbModified"] is False
    assert auditor.audit(status="REJECTED")["items"] == []

    ledger = NewsPilotReviewLedger(source_db=source, journal_db=journal)
    ledger.decide(
        evidence_id=1, decision="APPROVED_FOR_RESEARCH",
        reviewer="local-reviewer",
        reason="The reported outage quote is intact; economic effects remain speculative.",
        commit=True,
    )
    after = auditor.audit(status="APPROVED_FOR_RESEARCH")
    assert after["counts"]["REQUIRES_HUMAN_REVIEW"] == 0
    assert after["counts"]["APPROVED_FOR_RESEARCH"] == 1
    assert after["reviewDecisions"] == 1
    assert len(after["items"]) == 1
    assert after["items"][0]["review_status"] == "APPROVED_FOR_RESEARCH"
    assert after["items"][0]["reviewer"] == "local-reviewer"
    assert after["items"][0]["published"] is False
    assert auditor.audit(status="REQUIRES_HUMAN_REVIEW")["items"] == []
    assert auditor.audit(after_id=1)["items"] == []
    assert auditor.audit(after_id=1)["nextCursor"] == 1


def test_review_audit_keeps_rejected_signals_separate(
    tmp_path: Path,
) -> None:
    from tris_market_event_intelligence.news_pilot_review_audit import (
        NewsPilotReviewAuditor,
    )

    source, journal = seeded(tmp_path)
    NewsPilotReviewLedger(source_db=source, journal_db=journal).decide(
        evidence_id=1, decision="REJECTED",
        reviewer="human",
        reason="No independent materiality or magnitude evidence was supplied.",
        commit=True,
    )
    auditor = NewsPilotReviewAuditor(source_db=source, journal_db=journal)
    report = auditor.audit(status="REJECTED")
    assert report["counts"]["REJECTED"] == 1
    assert len(report["items"]) == 1
    assert report["items"][0]["review_status"] == "REJECTED"
    assert auditor.audit(status="APPROVED_FOR_RESEARCH")["items"] == []


@pytest.mark.parametrize("change", [
    "decision_source_sha", "decision_signal_sha",
    "orphan_decision", "invalid_review_reason", "invalid_reviewed_at",
    "mutated_source", "mutated_signal",
])
def test_review_audit_fails_closed_on_inconsistent_history(
    tmp_path: Path, change: str,
) -> None:
    from tris_market_event_intelligence.news_pilot_review_audit import (
        NewsPilotReviewAuditor,
    )

    source, journal = seeded(tmp_path)
    ledger = NewsPilotReviewLedger(source_db=source, journal_db=journal)
    ledger.decide(
        evidence_id=1, decision="APPROVED_FOR_RESEARCH",
        reviewer="human",
        reason="The facts are quoted but financial impact remains speculative.",
        commit=True,
    )
    if change == "mutated_source":
        with sqlite3.connect(source) as db:
            db.execute("UPDATE economic_events SET evidence='incorrect' WHERE id=69")
    elif change == "mutated_signal":
        with sqlite3.connect(journal) as db:
            db.execute(
                "UPDATE news_pilot_evidence SET signal_json='{}' WHERE id=1"
            )
    else:
        column, value = {
            "decision_source_sha": ("source_evidence_sha256", "a" * 64),
            "decision_signal_sha": ("signal_sha256", "b" * 64),
            "orphan_decision": ("evidence_id", 999),
            "invalid_review_reason": ("reason", "short"),
            "invalid_reviewed_at": ("reviewed_at", "invalid"),
        }[change]
        with sqlite3.connect(journal) as db:
            db.execute("PRAGMA foreign_keys=OFF")
            db.execute(
                f"UPDATE news_pilot_review_decisions SET {column}=? WHERE id=1",
                (value,),
            )
    with pytest.raises(ValueError):
        NewsPilotReviewAuditor(source_db=source, journal_db=journal).audit()


def test_audit_cursor_validation_and_source_db_isolation(tmp_path: Path) -> None:
    from tris_market_event_intelligence.news_pilot_review_audit import (
        NewsPilotReviewAuditor,
    )

    source, journal = seeded(tmp_path)
    auditor = NewsPilotReviewAuditor(source_db=source, journal_db=journal)
    with sqlite3.connect(source) as db:
        source_snapshot = tuple(db.iterdump())
    with sqlite3.connect(journal) as db:
        journal_snapshot = tuple(db.iterdump())
    report = auditor.audit(after_id=0, limit=1)
    assert report["nextCursor"] == 1
    assert report["hasMore"] is False
    for invalid in [
        {"status": "PUBLISHED"},
        {"limit": 0},
        {"limit": 501},
        {"after_id": -1},
        {"after_id": True},
    ]:
        with pytest.raises(ValueError):
            auditor.audit(**invalid)
    with sqlite3.connect(source) as db:
        assert tuple(db.iterdump()) == source_snapshot
    with sqlite3.connect(journal) as db:
        assert tuple(db.iterdump()) == journal_snapshot


def test_full_review_packet_exposes_context_hypothesis_and_gaps_without_writes(
    tmp_path: Path,
) -> None:
    from tris_market_event_intelligence.news_pilot_review_packet import (
        NewsPilotReviewPacket,
    )

    source, journal = seeded(tmp_path)
    with sqlite3.connect(source) as db:
        source_before = tuple(db.iterdump())
    with sqlite3.connect(journal) as db:
        journal_before = tuple(db.iterdump())
    packet = NewsPilotReviewPacket(source_db=source, journal_db=journal)
    result = packet.build(evidence_id=1)
    assert result["schemaVersion"] == "news_pilot_review_packet_v1"
    assert result["eventId"] == 69
    assert result["reviewStatus"] == "REQUIRES_HUMAN_REVIEW"
    assert result["source"]["literalEvidenceQuote"] in (
        result["source"]["reportedEvidenceContext"]
    )
    assert result["source"]["factStatus"] == "UNVERIFIED"
    assert result["hypothesis"]["direction"] == "NEGATIVE"
    assert result["hypothesis"]["intensityScore"] == 55
    assert result["hypothesis"]["scoreIsCalibratedToReturns"] is False
    assert "financial" in result["hypothesis"]["conditionalMechanism"]
    assert result["independentVerificationChecklist"] == [
        {"check": "CONFIRM_OUTAGE_DURATION", "verified": False}
    ]
    assert result["reviewDecision"] is None
    assert result["published"] is False
    assert result["journalModified"] is False
    assert result["productionDbModified"] is False
    with sqlite3.connect(source) as db:
        assert tuple(db.iterdump()) == source_before
    with sqlite3.connect(journal) as db:
        assert tuple(db.iterdump()) == journal_before


def test_review_packet_retains_rejection_without_publishing(
    tmp_path: Path,
) -> None:
    from tris_market_event_intelligence.news_pilot_review_packet import (
        NewsPilotReviewPacket,
    )

    source, journal = seeded(tmp_path)
    NewsPilotReviewLedger(source_db=source, journal_db=journal).decide(
        evidence_id=1, decision="REJECTED", reviewer="editor",
        reason="No independent evidence of duration and business impact.",
        commit=True,
    )
    result = NewsPilotReviewPacket(source_db=source, journal_db=journal).build(
        evidence_id=1
    )
    assert result["reviewStatus"] == "REJECTED"
    assert result["reviewDecision"]["reviewer"] == "editor"
    assert result["reviewDecision"]["decision"] == "REJECTED"
    assert result["published"] is False


@pytest.mark.parametrize("alteration", [
    "tamper_any_source", "tamper_any_signal", "tamper_decision_hash",
])
def test_packet_refuses_tampered_data(
    tmp_path: Path, alteration: str,
) -> None:
    from tris_market_event_intelligence.news_pilot_review_packet import (
        NewsPilotReviewPacket,
    )

    source, journal = seeded(tmp_path)
    if alteration == "tamper_any_source":
        with sqlite3.connect(source) as db:
            db.execute("UPDATE economic_events SET evidence='forged' WHERE id=69")
    elif alteration == "tamper_any_signal":
        with sqlite3.connect(journal) as db:
            db.execute("UPDATE news_pilot_evidence SET signal_json='{}' WHERE id=1")
    else:
        NewsPilotReviewLedger(source_db=source, journal_db=journal).decide(
            evidence_id=1, decision="REJECTED", reviewer="editor",
            reason="News has insufficient confirmed impact evidence.",
            commit=True,
        )
        with sqlite3.connect(journal) as db:
            db.execute(
                "UPDATE news_pilot_review_decisions SET signal_sha256=? WHERE id=1",
                ("e" * 64,),
            )
    with pytest.raises(ValueError):
        NewsPilotReviewPacket(source_db=source, journal_db=journal).build(
            evidence_id=1
        )


def test_packet_does_not_guess_journal_id_or_allow_invalid_id(tmp_path: Path) -> None:
    from tris_market_event_intelligence.news_pilot_review_packet import (
        NewsPilotReviewPacket,
    )

    source, journal = seeded(tmp_path)
    packet = NewsPilotReviewPacket(source_db=source, journal_db=journal)
    with pytest.raises(ValueError, match="not found"):
        packet.build(evidence_id=69)
    for invalid in [0, -1, True]:
        with pytest.raises(ValueError, match="Positive"):
            packet.build(evidence_id=invalid)
