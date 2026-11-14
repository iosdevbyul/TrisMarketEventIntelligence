"""Cross-page news audit must flag candidate leakage and incomplete pagination."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_news_holdout_plan import cohort

from tris_market_event_intelligence.news_cross_page_holdout_audit import (
    audit_cross_page_news,
)
from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout

TRAIN = datetime(2026, 11, 2, tzinfo=UTC)
VALIDATION = datetime(2026, 11, 6, tzinfo=UTC)


def pages(tmp_path: Path) -> list[tuple[Path, Path]]:
    original = json.loads(cohort(tmp_path).read_text())
    titles = [
        "Regional banking authority announces revised consumer loan policy next week",
        "Completely unrelated manufacturing survey short note",
        "Regional banking authority announces revised consumer loan policy next month",
        "Unrelated weather and transport update with no banking reference",
    ]
    for row, title in zip(original["items"], titles, strict=True):
        row["title"] = title
        row["textSha256"] = hashlib.sha256(
            (title + "\n" + (row["summary"] or "")).encode()
        ).hexdigest()
    result = []
    for index in range(2):
        page = dict(original)
        page["items"] = original["items"][index * 2:index * 2 + 2]
        page["pageAfterId"] = index * 2
        page["nextCursor"] = (index + 1) * 2
        page["hasMore"] = index == 0
        page["pageLimit"] = 2
        cohort_path = tmp_path / f"cohort-{index}.json"
        cohort_path.write_text(json.dumps(page), encoding="utf-8")
        holdout = plan_news_holdout(
            cohort_path, train_end=TRAIN, validation_end=VALIDATION,
        )
        plan_path = tmp_path / f"holdout-{index}.json"
        plan_path.write_text(json.dumps(holdout), encoding="utf-8")
        result.append((cohort_path, plan_path))
    return result


def test_two_pages_detect_lexical_cross_split_candidates(tmp_path: Path) -> None:
    inputs = pages(tmp_path)
    originals = [(a.read_bytes(), b.read_bytes()) for a, b in inputs]
    report = audit_cross_page_news(pages=inputs)
    assert report["crossPageCandidates"] == 1
    assert report["pageCount"] == 2
    assert report["articlesChecked"] == 4
    assert report["continuousPaginationChecked"] is True
    candidate = report["items"][0]
    assert (candidate["earlierArticleId"], candidate["laterArticleId"]) == (1, 3)
    assert candidate["relation"] == "LEXICAL_CANDIDATE"
    assert candidate["earlierPartition"] == "TRAIN"
    assert candidate["laterPartition"] == "TEST"
    assert candidate["confirmedSameStory"] is None
    assert report["holdoutCanBeCertified"] is False
    assert report["humanGoldExposed"] is False
    assert report["stockPricesUsed"] is False
    assert [(a.read_bytes(), b.read_bytes()) for a, b in inputs] == originals


def test_exact_source_url_cross_page_flagged_despite_different_titles(
    tmp_path: Path,
) -> None:
    inputs = pages(tmp_path)
    doc = json.loads(inputs[1][0].read_text())
    doc["items"][0]["sourceUrl"] = (
        json.loads(inputs[0][0].read_text())["items"][0]["sourceUrl"]
    )
    doc["items"][0]["title"] = "Breaking different content with unrelated text"
    doc["items"][0]["textSha256"] = hashlib.sha256(
        (doc["items"][0]["title"] + "\n" + doc["items"][0]["summary"]).encode()
    ).hexdigest()
    inputs[1][0].write_text(json.dumps(doc))
    plan = plan_news_holdout(inputs[1][0], train_end=TRAIN, validation_end=VALIDATION)
    inputs[1][1].write_text(json.dumps(plan))
    result = audit_cross_page_news(pages=inputs)
    assert result["crossPageCandidates"] == 1
    assert result["items"][0]["relation"] == "EXACT_KEY"


def test_bad_cursor_or_missing_last_page_fails_closed(tmp_path: Path) -> None:
    inputs = pages(tmp_path)
    doc = json.loads(inputs[1][0].read_text())
    doc["pageAfterId"] = 1
    inputs[1][0].write_text(json.dumps(doc))
    plan = plan_news_holdout(inputs[1][0], train_end=TRAIN, validation_end=VALIDATION)
    inputs[1][1].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="pagination"):
        audit_cross_page_news(pages=inputs)


def test_hole_in_pages_and_no_terminal_page_fails(tmp_path: Path) -> None:
    inputs = pages(tmp_path)
    doc = json.loads(inputs[1][0].read_text())
    doc["hasMore"] = True
    inputs[1][0].write_text(json.dumps(doc))
    plan = plan_news_holdout(inputs[1][0], train_end=TRAIN, validation_end=VALIDATION)
    inputs[1][1].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="pagination"):
        audit_cross_page_news(pages=inputs)


def test_tampered_plan_cannot_be_used(tmp_path: Path) -> None:
    inputs = pages(tmp_path)
    doc = json.loads(inputs[1][1].read_text())
    doc["items"][0]["textSha256"] = "a" * 64
    inputs[1][1].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="provenance"):
        audit_cross_page_news(pages=inputs)


def test_mixed_holdout_cutoffs_are_rejected(tmp_path: Path) -> None:
    inputs = pages(tmp_path)
    doc = json.loads(inputs[1][1].read_text())
    doc["trainEnd"] = "2026-10-01T00:00:00+00:00"
    inputs[1][1].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="boundaries"):
        audit_cross_page_news(pages=inputs)


@pytest.mark.parametrize("settings", [
    {"threshold": 0.69}, {"threshold": 0.96}, {"threshold": True},
    {"max_days": 0}, {"max_days": 31}, {"max_days": True},
])
def test_invalid_limits_rejected(tmp_path: Path, settings: dict) -> None:
    inputs = pages(tmp_path)
    with pytest.raises(ValueError):
        audit_cross_page_news(pages=inputs, **settings)


def test_single_page_not_a_full_cross_page_audit(tmp_path: Path) -> None:
    inputs = pages(tmp_path)
    with pytest.raises(ValueError, match="2..16"):
        audit_cross_page_news(pages=inputs[:1])
