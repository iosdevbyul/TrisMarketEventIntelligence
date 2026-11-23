"""Offline regression coverage for collection and repeated news interpretation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_gemma_original_article_batch_cli import database

from tris_market_event_intelligence.gemma_original_article_batch_cli import (
    evidence_passages,
)
from tris_market_event_intelligence.news_collection_semantic_audit_cli import (
    collection_audit,
    repeatability_audit,
)


def _run(path: Path, digest: str, *, direction: str = "UP") -> Path:
    path.mkdir()
    report = {
        "schemaVersion": "gemma_original_article_review_v2",
        "model": "gemma4:e2b",
        "article": {
            "articleId": 1,
            "bodySha256": digest,
        },
        "analysis": {
            "summary": "Policy decision",
            "limitations": "Source alone cannot prove a market outcome.",
            "impacts": [{
                "target": "Financial Markets",
                "direction": direction,
                "reason": "Market policy may affect risk appetite",
                "evidence_quote": "This report describes the regional market",
            }],
        },
    }
    (path / "article-1.json").write_text(json.dumps(report))
    return path


def test_sentence_and_newline_evidence_are_not_arbitrary_chunks() -> None:
    body = (
        "The bank held rates steady after its scheduled meeting. "
        "Officials discussed consumer spending and near-term inflation.\n"
        "Manufacturing demand remained steady in the quarter."
    )
    fragments = evidence_passages(body)
    assert [item["text"] for item in fragments] == [
        "The bank held rates steady after its scheduled meeting.",
        "Officials discussed consumer spending and near-term inflation.",
        "Manufacturing demand remained steady in the quarter.",
    ]
    assert all(item["text"] in body for item in fragments)


def test_collection_audit_is_read_only_and_reports_source_integrity(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    report = collection_audit(db)
    after = hashlib.sha256(db.read_bytes()).hexdigest()
    assert before == after
    assert report["totalArticles"] == 3
    assert report["bySource"] == {"bbc_business": 3}
    assert report["fullTextStatuses"] == {
        "COMPLETE": 1,
        "NON_ARTICLE": 1,
        "REVIEW_REQUIRED": 1,
    }
    assert report["completeBodyIntegrity"]["validCompleteRows"] == 1
    assert report["duplicateUrls"] == 0
    assert report["duplicateExtraArticleRows"] == 0
    assert report["duplicateUrlExamples"] == []
    assert report["articlesWithoutFullTextStatus"] == 0
    assert report["missingStatusArticles"] == []
    assert len(report["reviewRequiredExamples"]) == 1
    assert report["fullTextStatusBySource"]["bbc_business"] == {
        "COMPLETE": 1, "NON_ARTICLE": 1, "REVIEW_REQUIRED": 1,
    }


def test_repeatability_identifies_divergent_direction(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    digest = hashlib.sha256(
        ("This report describes the regional market and company activity. " * 8).encode()
    ).hexdigest()
    first = _run(tmp_path / "run-1", digest)
    second = _run(tmp_path / "run-2", digest, direction="DOWN")
    result = repeatability_audit(db, [first, second])
    assert result["comparableArticles"] == 1
    assert result["stableArticles"] == 0
    assert result["stableEmptyArticles"] == 0
    assert result["stableNonEmptyArticles"] == 0
    assert result["allRunsNonEmptyArticles"] == 1
    assert result["mixedEmptyNonEmptyArticles"] == 0
    assert result["comparableEventKindArticles"] == 0
    assert result["agreement"] == 0
    assert result["cases"][0]["status"] == "DIVERGENT_TARGET_DIRECTIONS"
    assert result["sameImpactCountArticles"] == 1
    assert result["sameTargetSetArticles"] == 1
    assert result["sameSharedTargetDirectionArticles"] == 0
    assert result["noSharedTargetArticles"] == 0
    assert result["changedDirectionArticles"] == 1
    assert result["sharedTargetPairs"] == 1
    assert result["changedDirectionPairs"] == 1
    assert result["meanTargetJaccard"] == 1.0
    assert result["cases"][0]["sharedDirectionDisagreements"] == [
        {"target": "financial markets", "directionsByRun": ["UP", "DOWN"]},
    ]
    assert result["cases"][0]["impactCountsStable"] is True
    assert result["cases"][0]["targetSetsIdentical"] is True
    assert result["cases"][0]["sharedTargetDirectionsIdentical"] is False
    assert result["cases"][0]["sharedTargets"] == ["financial markets"]
    assert result["cases"][0]["analysisByRun"][0]["impacts"][0]["evidenceQuote"]


def test_missing_run_does_not_count_as_agreement(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    digest = hashlib.sha256(
        ("This report describes the regional market and company activity. " * 8).encode()
    ).hexdigest()
    first = _run(tmp_path / "run-1", digest)
    second = tmp_path / "run-2"
    second.mkdir()
    result = repeatability_audit(db, [first, second])
    assert result["agreement"] is None
    assert result["cases"][0]["status"] == "MISSING_RUN"


def test_repeated_review_different_source_is_not_comparable(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    digest = hashlib.sha256(
        ("This report describes the regional market and company activity. " * 8).encode()
    ).hexdigest()
    first = _run(tmp_path / "run-1", digest)
    second = _run(tmp_path / "run-2", "0" * 64)
    result = repeatability_audit(db, [first, second])
    assert result["comparableArticles"] == 0
    assert result["cases"][0]["status"] == "INCOMPARABLE_SOURCE_OR_MODEL"


def test_repeatability_requires_distinct_folders(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    folder = tmp_path / "single"
    folder.mkdir()
    with pytest.raises(ValueError, match="distinct"):
        repeatability_audit(db, [folder, folder])


def test_target_change_separated_from_direction_change(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    digest = hashlib.sha256(
        ("This report describes the regional market and company activity. " * 8).encode()
    ).hexdigest()
    first = _run(tmp_path / "run-a", digest)
    second = _run(tmp_path / "run-b", digest)
    saved = second / "article-1.json"
    report = json.loads(saved.read_text())
    report["analysis"]["impacts"][0]["target"] = "Technology Sector"
    saved.write_text(json.dumps(report))
    result = repeatability_audit(db, [first, second])
    case = result["cases"][0]
    assert result["comparableArticles"] == 1
    assert result["stableArticles"] == 0
    assert result["sameImpactCountArticles"] == 1
    assert result["sameTargetSetArticles"] == 0
    assert case["sharedTargets"] == []
    assert case["sharedTargetDirectionsIdentical"] is True
    assert case["targetJaccard"] == 0.0
    assert case["sharedDirectionDisagreements"] == []
    assert result["noSharedTargetArticles"] == 1
    assert result["changedDirectionArticles"] == 0
    assert result["sharedTargetPairs"] == 0
    assert result["meanTargetJaccard"] == 0.0


def test_narrow_alias_agreement_is_reported_without_upgrading_raw_agreement(
    tmp_path: Path,
) -> None:
    db = database(tmp_path / "news.sqlite3")
    digest = hashlib.sha256(
        ("This report describes the regional market and company activity. " * 8).encode()
    ).hexdigest()
    first = _run(tmp_path / "first", digest)
    second = _run(tmp_path / "second", digest)
    report_path = second / "article-1.json"
    report = json.loads(report_path.read_text())
    report["analysis"]["impacts"][0]["target"] = "General market"
    report_path.write_text(json.dumps(report))
    results = repeatability_audit(db, [first, second])
    assert results["stableArticles"] == 0
    assert results["limitedAliasTargetDirectionArticles"] == 1
    assert results["cases"][0]["limitedAliasTargetDirectionsIdentical"] is True
    assert results["cases"][0]["status"] == "DIVERGENT_TARGET_DIRECTIONS"


def test_double_empty_is_not_substantive_agreement(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    digest = hashlib.sha256(
        ("This report describes the regional market and company activity. " * 8).encode()
    ).hexdigest()
    first = _run(tmp_path / "first-empty", digest)
    second = _run(tmp_path / "second-empty", digest)
    for directory in (first, second):
        path = directory / "article-1.json"
        data = json.loads(path.read_text())
        data["analysis"]["impacts"] = []
        data["event"] = {
            "eventKind": "POLICY_DECISION", "eventFact": "Policy was discussed"
        }
        path.write_text(json.dumps(data))
    result = repeatability_audit(db, [first, second])
    assert result["stableArticles"] == 1
    assert result["stableEmptyArticles"] == 1
    assert result["stableNonEmptyArticles"] == 0
    assert result["allRunsNonEmptyArticles"] == 0
    assert result["comparableEventKindArticles"] == 1
    assert result["matchingEventKindArticles"] == 1
    assert result["cases"][0]["allRunsEmpty"] is True


def test_empty_and_nonempty_predictions_are_not_merged(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    digest = hashlib.sha256(
        ("This report describes the regional market and company activity. " * 8).encode()
    ).hexdigest()
    first = _run(tmp_path / "first-mixed", digest)
    second = _run(tmp_path / "second-mixed", digest)
    path = second / "article-1.json"
    data = json.loads(path.read_text())
    data["analysis"]["impacts"] = []
    path.write_text(json.dumps(data))
    result = repeatability_audit(db, [first, second])
    assert result["stableArticles"] == 0
    assert result["mixedEmptyNonEmptyArticles"] == 1
    assert result["cases"][0]["mixedEmptyAndNonEmpty"] is True


def test_different_event_kinds_do_not_hide_behind_matching_empty_impacts(
    tmp_path: Path,
) -> None:
    db = database(tmp_path / "news.sqlite3")
    digest = hashlib.sha256(
        ("This report describes the regional market and company activity. " * 8).encode()
    ).hexdigest()
    first = _run(tmp_path / "first-kind", digest)
    second = _run(tmp_path / "second-kind", digest)
    for directory, kind in ((first, "SCHEDULE_ONLY"), (second, "POLICY_DECISION")):
        path = directory / "article-1.json"
        data = json.loads(path.read_text())
        data["analysis"]["impacts"] = []
        data["event"] = {"eventKind": kind}
        data["eventAssessment"] = {
            "status": "REVIEW_REQUIRED" if kind == "POLICY_DECISION" else "UNVERIFIED"
        }
        path.write_text(json.dumps(data))
    result = repeatability_audit(db, [first, second])
    assert result["stableEmptyArticles"] == 1
    assert result["comparableEventKindArticles"] == 1
    assert result["matchingEventKindArticles"] == 0
    assert result["eventReviewRequiredArticles"] == 1
    assert result["cases"][0]["eventKindsIdentical"] is False
