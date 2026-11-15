"""Real Mac NEWS inventory reveals readiness without exposing private article data."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from test_news_research_worker import files

from tris_market_event_intelligence.news_mac_learning_inventory import (
    inspect_mac_news_learning,
)


def test_missing_private_directory_is_safe(tmp_path: Path) -> None:
    result = inspect_mac_news_learning(data_dir=tmp_path / "not-created")
    assert result["status"] == "PRIVATE_DATA_DIRECTORY_MISSING"
    assert result["modelCalls"] == 0
    assert result["nextAction"] == "VERIFY_PRIVATE_NEWS_COLLECTOR_STORAGE"
    assert result["stockPricesUsed"] is False


def test_private_sqlite_only_requires_news_cohort(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO articles VALUES (1)")
    report = inspect_mac_news_learning(data_dir=tmp_path)
    assert report["database"]["articleCount"] == 1
    assert report["database"]["status"] == "READ_ONLY_SQLITE_SNAPSHOT"
    assert report["nextAction"] == "EXPORT_READ_ONLY_NEWS_COHORT"
    assert report["artifactCounts"]["cohort"] == 0
    assert report["researchArtifactsModified"] is False


def test_one_pinned_private_triple_preflights_without_outputting_private_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    paths = files(source)
    private = tmp_path / "private"
    private.mkdir()
    for path in paths[:3]:
        shutil.copyfile(path, private / path.name)
    report = inspect_mac_news_learning(data_dir=private)
    assert report["matchedResearchTriples"] == 1
    assert report["artifactCounts"] == {
        "cohort": 1, "review": 1, "holdout": 1,
    }
    assert report["readiness"]["trainingArticlesWithTimeGatedHumanLabels"] == 6
    assert report["readiness"]["trainFoldResearchFeasible"] is True
    assert report["readiness"]["humanGoldIndependentlyAdjudicated"] is False
    assert report["testGoldScored"] is False
    assert report["modelImprovementValidated"] is False
    payload = json.dumps(report)
    assert "Major regulatory decision" not in payload
    assert "fixture-human" not in payload
    assert str(private) not in payload
    assert "sourceUrl" not in payload


def test_missing_human_review_explicitly_reported(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    paths = files(source)
    shutil.copyfile(paths[0], tmp_path / "news-cohort.json")
    report = inspect_mac_news_learning(data_dir=tmp_path)
    assert report["nextAction"] == "PREPARE_AND_COMPLETE_HUMAN_NEWS_REVIEW"
    assert report["readiness"] is None


def test_mismatched_holdout_is_not_pretended_ready(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    paths = files(source)
    for path in paths[:3]:
        shutil.copyfile(path, tmp_path / path.name)
    holdout = tmp_path / "plan.json"
    contents = json.loads(holdout.read_text())
    contents["humanReviewSha256"] = "0" * 64
    holdout.write_text(json.dumps(contents))
    report = inspect_mac_news_learning(data_dir=tmp_path)
    assert report["matchedResearchTriples"] == 0
    assert report["nextAction"] == "RECONCILE_COHORT_REVIEW_HOLDOUT_SHA"


def test_ambiguous_pinned_sets_not_auto_selected(tmp_path: Path) -> None:
    source = tmp_path / "sources"
    source.mkdir()
    paths = files(source)
    for path in paths[:3]:
        shutil.copyfile(path, tmp_path / path.name)
    shutil.copyfile(paths[1], tmp_path / "another-holdout.json")
    report = inspect_mac_news_learning(data_dir=tmp_path)
    assert report["matchedResearchTriples"] == 2
    assert report["nextAction"] == "SELECT_EXPLICIT_PINNED_RESEARCH_TRIPLE"
    assert report["readiness"] is None


def test_invalid_json_and_symlink_skipped_without_following(tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "alias.json").symlink_to(tmp_path / "bad.json")
    output = inspect_mac_news_learning(data_dir=tmp_path)
    assert output["boundedInvalidOrOversizedJsonFiles"] == 1
    assert output["artifactCounts"]["cohort"] == 0
    assert output["networkCalls"] == 0
