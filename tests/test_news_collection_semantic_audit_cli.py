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
    assert result["agreement"] == 0
    assert result["cases"][0]["status"] == "DIVERGENT_TARGET_DIRECTIONS"


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
