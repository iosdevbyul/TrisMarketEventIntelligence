"""Only manual, source-pinned sample judgments may enter full NEWS reviews."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_news_research_worker import files

from tris_market_event_intelligence.news_importance_review import prepare_review
from tris_market_event_intelligence.news_materiality_review import (
    prepare_materiality_review,
)
from tris_market_event_intelligence.news_review_sample import prepare_news_review_sample
from tris_market_event_intelligence.news_review_sample_bridge import (
    bridge_review_sample,
)


def fixture(tmp_path: Path):
    cohort = files(tmp_path)[0]
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(
        prepare_news_review_sample(cohort_path=cohort, max_selected=5)
    ))
    imp = tmp_path / "importance.json"
    imp.write_text(json.dumps(prepare_review(cohort)))
    mag = tmp_path / "materiality.json"
    mag.write_text(json.dumps(prepare_materiality_review(cohort)))
    return cohort, sample_path, imp, mag


def bridge(paths):
    return bridge_review_sample(
        cohort_path=paths[0], sample_path=paths[1],
        importance_template_path=paths[2], materiality_template_path=paths[3],
    )


def test_empty_sample_never_creates_human_gold(tmp_path: Path):
    paths = fixture(tmp_path)
    originals = [x.read_bytes() for x in paths]
    report = bridge(paths)
    assert report["status"] == "NO_HUMAN_IMPORTANCE_OR_MAGNITUDE_JUDGMENTS"
    assert report["importanceHumanJudgments"] == 0
    assert report["materialityHumanJudgments"] == 0
    assert report["modelCalls"] == 0
    assert report["promotionAllowed"] is False
    assert report["stockPricesUsed"] is False
    assert [x.read_bytes() for x in paths] == originals


def test_manually_reviewed_article_maps_only_that_article(tmp_path: Path):
    paths = fixture(tmp_path)
    sample = json.loads(paths[1].read_text())
    reviewed_id = sample["items"][0]["articleId"]
    sample["items"][0].update({
        "importance": "IMPORTANT", "sourceAlignment": "SUPPORTED",
        "potentialMagnitude": "CRITICAL",
        "sameStoryConfirmedByHuman": False,
        "reviewer": "actual-human-test",
        "reviewedAt": "2026-11-16T00:00:00+00:00",
        "reason": "An independently written human hypothesis on consequences.",
    })
    paths[1].write_text(json.dumps(sample))
    report = bridge(paths)
    assert report["importanceHumanJudgments"] == 1
    assert report["materialityHumanJudgments"] == 1
    assert report["humanStoryRelationsNotPropagated"] == 1
    assert sum(x["importance"] is not None
               for x in report["importanceReview"]["judgments"]) == 1
    assert sum(x["potentialMagnitude"] is not None
               for x in report["materialityReview"]["judgments"]) == 1
    assert next(x for x in report["importanceReview"]["judgments"]
                if x["articleId"] == reviewed_id)["importance"] == "IMPORTANT"
    assert len(report["importanceReview"]["judgments"]) == 12
    assert len(report["materialityReview"]["judgments"]) == 12
    assert report["humanGoldIndependentlyAudited"] is False


def test_importance_and_magnitude_are_independent(tmp_path: Path):
    paths = fixture(tmp_path)
    sample = json.loads(paths[1].read_text())
    sample["items"][0].update({
        "potentialMagnitude": "UNASSESSABLE",
        "reviewer": "reviewer",
        "reviewedAt": "2026-11-16T00:00:00+00:00",
        "reason": "The independent news has too little detail to assess impact.",
    })
    paths[1].write_text(json.dumps(sample))
    report = bridge(paths)
    assert report["materialityHumanJudgments"] == 1
    assert report["importanceHumanJudgments"] == 0


@pytest.mark.parametrize("edit", [
    {"importance": "IMPORTANT"},
    {"importance": "IMPORTANT", "sourceAlignment": "SUPPORTED"},
    {"potentialMagnitude": "EXTREME"},
    {"sameStoryConfirmedByHuman": "yes"},
    {"reviewer": "fiction"},
])
def test_invalid_or_partial_reviews_cannot_generate_gold(tmp_path: Path, edit: dict):
    paths = fixture(tmp_path)
    sample = json.loads(paths[1].read_text())
    sample["items"][0].update(edit)
    paths[1].write_text(json.dumps(sample))
    with pytest.raises((ValueError, TypeError)):
        bridge(paths)


def test_provenance_changes_rejected(tmp_path: Path):
    paths = fixture(tmp_path)
    sample = json.loads(paths[1].read_text())
    sample["items"][0]["title"] = "Forged press release"
    paths[1].write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="provenance"):
        bridge(paths)


def test_edited_original_full_template_is_rejected(tmp_path: Path):
    paths = fixture(tmp_path)
    importance = json.loads(paths[2].read_text())
    importance["judgments"][0]["importance"] = "IMPORTANT"
    paths[2].write_text(json.dumps(importance))
    with pytest.raises(ValueError, match="untouched"):
        bridge(paths)


def test_unsupported_review_timestamp_fails(tmp_path: Path):
    paths = fixture(tmp_path)
    sample = json.loads(paths[1].read_text())
    sample["items"][0].update({
        "potentialMagnitude": "LOW",
        "reviewer": "editor",
        "reviewedAt": "2020-01-01T00:00:00+00:00",
        "reason": "A human reasoning assertion predating the news observation.",
    })
    paths[1].write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="review requires"):
        bridge(paths)
