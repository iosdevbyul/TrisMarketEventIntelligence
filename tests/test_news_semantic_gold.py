"""Research-only human gold scoring; source and journal are never rewritten."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tris_market_event_intelligence.news_semantic_gold import (
    evaluate_gold, load_gold,
)

from test_news_pilot_review_ledger import seeded


def adjudication(journal: Path, **overrides: object) -> dict[str, object]:
    with sqlite3.connect(journal) as db:
        sha, source_sha = db.execute(
            "SELECT signal_sha256, source_evidence_sha256 "
            "FROM news_pilot_evidence WHERE id=1"
        ).fetchone()
    record = {
        "journalEvidenceId": 1,
        "signalSha256": sha,
        "sourceEvidenceSha256": source_sha,
        "reviewer": "human-analyst",
        "reviewedAt": datetime.now(UTC).isoformat(),
        "decision": "SUPPORTED",
        "errorTypes": [],
        "reason": "The cautious mechanism only describes potential outcomes.",
    }
    record.update(overrides)
    return record


def write_gold(path: Path, *records: dict[str, object]) -> None:
    path.write_text(json.dumps({
        "schemaVersion": "news_semantic_gold_v1", "judgments": list(records),
    }), encoding="utf-8")


def test_gold_unflagged_supported_no_false_accuracy(tmp_path: Path) -> None:
    source, journal = seeded(tmp_path)
    gold = tmp_path / "gold.json"
    write_gold(gold, adjudication(journal))
    with sqlite3.connect(source) as db:
        before_source = tuple(db.iterdump())
    with sqlite3.connect(journal) as db:
        before_journal = tuple(db.iterdump())
    report = evaluate_gold(source_db=source, journal_db=journal, gold_path=gold)
    assert report["status"] == "INSUFFICIENT_GOLD"
    assert report["totalHumanJudgments"] == 1
    assert report["confusion"]["trueNegative"] == 1
    assert report["triagePrecision"] is None
    assert report["triageRecall"] is None
    assert report["modelAccuracyVerified"] is False
    assert report["humanReviewerAuthenticated"] is False
    assert report["sourceDbModified"] is False
    assert report["journalModified"] is False
    with sqlite3.connect(source) as db:
        assert tuple(db.iterdump()) == before_source
    with sqlite3.connect(journal) as db:
        assert tuple(db.iterdump()) == before_journal


def test_gold_detects_false_negative_without_auto_approval(tmp_path: Path) -> None:
    source, journal = seeded(tmp_path)
    gold = tmp_path / "gold.json"
    write_gold(gold, adjudication(
        journal, decision="UNRELATED", errorTypes=["UNRELATED_EVIDENCE"],
        reason="The explanation discusses financial exposure absent in the cited text.",
    ))
    result = evaluate_gold(source_db=source, journal_db=journal, gold_path=gold)
    assert result["confusion"]["falseNegative"] == 1
    assert result["items"][0]["humanVerdict"] == "UNRELATED"
    assert result["items"][0]["triageFlagged"] is False
    assert result["semanticallyVerified"] is False


def test_gold_flags_false_positive_and_unscored(tmp_path: Path) -> None:
    source, journal = seeded(tmp_path)
    gold = tmp_path / "gold.json"
    with sqlite3.connect(source) as db:
        original = db.execute(
            "SELECT evidence FROM economic_events WHERE id=69"
        ).fetchone()[0]
        qualified = "The reporter did not confirm this incident. " + original
        db.execute(
            "UPDATE economic_events SET evidence=? WHERE id=69", (qualified,),
        )
    with sqlite3.connect(journal) as db:
        db.execute(
            "UPDATE news_pilot_evidence SET source_evidence_sha256=? WHERE id=1",
            (hashlib.sha256(qualified.encode()).hexdigest(),),
        )
    write_gold(gold, adjudication(journal))
    result = evaluate_gold(source_db=source, journal_db=journal, gold_path=gold)
    assert result["confusion"]["falsePositive"] == 1
    assert "NEGATION_OUTSIDE_EXCERPT_REVIEW" in (
        result["items"][0]["triageReviewCodes"]
    )
    write_gold(gold, adjudication(
        journal, decision="INSUFFICIENT_CONTEXT",
        reason="Insufficient independent evidence to adjudicate the explanation.",
    ))
    result = evaluate_gold(source_db=source, journal_db=journal, gold_path=gold)
    assert result["confusion"]["unscored"] == 1
    assert result["scoredJudgments"] == 0


@pytest.mark.parametrize("change", [
    {"journalEvidenceId": True},
    {"journalEvidenceId": 99},
    {"signalSha256": "0" * 64},
    {"sourceEvidenceSha256": "1" * 64},
    {"reviewer": ""},
    {"reviewedAt": "2026-11-15T00:00:00"},
    {"reason": "short"},
    {"decision": "VERIFIED"},
    {"decision": "CONTRADICTED", "errorTypes": []},
    {"decision": "SUPPORTED", "errorTypes": ["TIME_MISMATCH"]},
    {"decision": "UNRELATED", "errorTypes": ["INVENTED_ERROR"]},
])
def test_invalid_or_unverifiable_gold_rejected(
    tmp_path: Path, change: dict[str, object],
) -> None:
    source, journal = seeded(tmp_path)
    gold = tmp_path / "gold.json"
    write_gold(gold, adjudication(journal, **change))
    with pytest.raises(ValueError):
        evaluate_gold(source_db=source, journal_db=journal, gold_path=gold)


def test_duplicate_or_symlink_gold_rejected(tmp_path: Path) -> None:
    source, journal = seeded(tmp_path)
    gold = tmp_path / "gold.json"
    label = adjudication(journal)
    write_gold(gold, label, label)
    with pytest.raises(ValueError, match="duplicate"):
        load_gold(gold)
    link = tmp_path / "link.json"
    link.symlink_to(gold)
    with pytest.raises(ValueError, match="non-symlink"):
        load_gold(link)


def test_mutated_source_refuses_gold_after_hash_check(tmp_path: Path) -> None:
    source, journal = seeded(tmp_path)
    gold = tmp_path / "gold.json"
    write_gold(gold, adjudication(journal))
    with sqlite3.connect(source) as db:
        db.execute(
            "UPDATE economic_events SET evidence='Changed report' WHERE id=69"
        )
    with pytest.raises(ValueError):
        evaluate_gold(source_db=source, journal_db=journal, gold_path=gold)
