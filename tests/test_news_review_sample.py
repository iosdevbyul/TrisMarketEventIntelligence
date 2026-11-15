"""Human review sample never invents news gold or market features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_news_research_worker import files

from tris_market_event_intelligence.news_review_sample import (
    prepare_news_review_sample,
)


def _files(tmp_path: Path):
    return files(tmp_path)[0]


def test_original_news_sample_deterministic_and_no_leaked_gold(tmp_path: Path):
    path = _files(tmp_path)
    a = prepare_news_review_sample(cohort_path=path, max_selected=5)
    b = prepare_news_review_sample(cohort_path=path, max_selected=5)
    assert a == b
    assert a["totalArticles"] == 12
    assert a["candidateUnitCount"] == 12
    assert a["selectedArticles"] == 5
    assert a["humanGoldCreated"] is False
    assert a["stockPricesUsed"] is False
    assert a["testGoldScored"] is False
    assert all(item["potentialMagnitude"] is None and item["importance"] is None
               for item in a["items"])
    assert len({item["articleId"] for item in a["items"]}) == 5


def test_same_story_and_exact_duplicates_are_only_candidate_units(tmp_path: Path):
    path = _files(tmp_path)
    doc = json.loads(path.read_text())
    first, second = doc["items"][0], doc["items"][1]
    second["storyGroupId"] = 900
    first["storyGroupId"] = 900
    # Link by source-declared group even with different headline.
    path.write_text(json.dumps(doc))
    result = prepare_news_review_sample(cohort_path=path)
    assert result["candidateUnitCount"] == 11
    assert result["multiArticleCandidateUnits"] == 1
    assert result["articlesInMultiArticleUnits"] == 2
    assert result["exactOrDeclaredLinksNotHumanVerified"] is True
    grouped = [row for row in result["items"]
               if len(row["candidateUnitArticleIds"]) == 2]
    assert len(grouped) == 1
    assert grouped[0]["sameStoryConfirmedByHuman"] is None


def test_lexical_candidates_never_silently_merged(tmp_path: Path):
    path = _files(tmp_path)
    doc = json.loads(path.read_text())
    a = doc["items"][0]
    b = doc["items"][1]
    b["title"] = a["title"] + " Update"
    b["textSha256"] = hashlib.sha256(
        (b["title"] + "\n" + (b["summary"] or "")).encode()
    ).hexdigest()
    path.write_text(json.dumps(doc))
    result = prepare_news_review_sample(cohort_path=path)
    assert result["candidateUnitCount"] == 12
    assert result["crossUnitLexicalCandidatePairs"] > 0
    assert result["lexicalWarningsNotConfirmedDuplicates"] is True


def test_invalid_cohort_hash_fails_closed(tmp_path: Path):
    path = _files(tmp_path)
    doc = json.loads(path.read_text())
    doc["items"][0]["title"] = "Tampered news"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_news_review_sample(cohort_path=path)


@pytest.mark.parametrize("n", [0, -1, 101, True, 2.0])
def test_sample_limit_is_bounded(tmp_path: Path, n):
    path = _files(tmp_path)
    with pytest.raises(ValueError, match="1..100"):
        prepare_news_review_sample(cohort_path=path, max_selected=n)
