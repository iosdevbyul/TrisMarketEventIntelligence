"""Persisted NEWS model research candidates must replay exactly, fail closed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_news_research_worker import TRAIN_END, VALIDATION_END, files, step

from tris_market_event_intelligence.news_candidate_artifact import (
    build_candidate_artifact,
    compare_candidate_replay,
    read_candidate_artifact,
    write_candidate_artifact,
)
from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout
from tris_market_event_intelligence.news_naive_bayes_model import NewsImportanceModel


def setup(tmp_path: Path) -> tuple[tuple[Path, Path, Path, Path], Path, str]:
    paths = files(tmp_path)
    result = step(paths)
    assert result["status"] == "DESCRIPTIVE_RESEARCH_ONLY"
    return paths, tmp_path / "candidate.private.json", result["experimentSha256"]


def build(paths: tuple[Path, Path, Path, Path], sha: str) -> dict:
    return build_candidate_artifact(
        cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
        experiment_sha256=sha,
    )


def replay(paths: tuple[Path, Path, Path, Path], artifact: Path) -> dict:
    return compare_candidate_replay(
        candidate_path=artifact, cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
    )


def test_create_reload_and_replay_exact_same_model(tmp_path: Path) -> None:
    paths, path, sha = setup(tmp_path)
    originals = [p.read_bytes() for p in paths[:3]]
    bundle = build(paths, sha)
    write_candidate_artifact(path, bundle)
    model, payload = read_candidate_artifact(path)
    assert payload["experimentSha256"] == sha
    assert payload["productionEligible"] is False
    assert model.to_state() == bundle["payload"]["modelState"]
    report = replay(paths, path)
    assert report["status"] == "REPRODUCIBLE_RESEARCH_ONLY"
    assert report["validationPredictionsCompared"] == 4
    assert report["testGoldAccessed"] is False
    assert report["modelQualityImprovementValidated"] is False
    assert all(p.read_bytes() == old for p, old in zip(paths[:3], originals, strict=True))
    assert not (tmp_path / "news.sqlite3").exists()


def test_artifact_create_only_and_no_overwriting(tmp_path: Path) -> None:
    paths, artifact, sha = setup(tmp_path)
    data = build(paths, sha)
    write_candidate_artifact(artifact, data)
    original = artifact.read_bytes()
    with pytest.raises(ValueError, match="new"):
        write_candidate_artifact(artifact, data)
    assert artifact.read_bytes() == original


def test_source_mutation_and_nonexistent_experiment_fails(tmp_path: Path) -> None:
    paths, artifact, sha = setup(tmp_path)
    with pytest.raises(ValueError, match="research record"):
        build(paths, "0" * 64)
    doc = json.loads(paths[1].read_text())
    doc["humanReviewSha256"] = "0" * 64
    paths[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="time-gated"):
        build(paths, sha)
    assert not artifact.exists()


@pytest.mark.parametrize("change", [
    lambda data: data["payload"]["modelState"]["counts"]["IMPORTANT"].update({"bad": -1}),
    lambda data: data["payload"]["modelState"]["documents"].update({"IMPORTANT": 999}),
    lambda data: data["payload"].update({"productionEligible": True}),
    lambda data: data["payload"].update({"stockPricesUsed": True}),
    lambda data: data["payload"].update({"modelState": {}}),
    lambda data: data.update({"payloadSha256": "0" * 64}),
])
def test_tampering_invalidates_private_model(
    tmp_path: Path, change,
) -> None:
    paths, artifact, sha = setup(tmp_path)
    data = build(paths, sha)
    change(data)
    artifact.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        read_candidate_artifact(artifact)


def test_artifact_valid_but_different_research_inputs_fail_replay(tmp_path: Path) -> None:
    paths, artifact, sha = setup(tmp_path)
    write_candidate_artifact(artifact, build(paths, sha))
    doc = json.loads(paths[2].read_text())
    doc["judgments"][10]["importance"] = "UNIMPORTANT"
    paths[2].write_text(json.dumps(doc))
    plan = plan_news_holdout(
        paths[0], train_end=TRAIN_END, validation_end=VALIDATION_END,
        review_path=paths[2],
    )
    paths[1].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="research record"):
        replay(paths, artifact)


def test_invalid_manual_model_state_rejected() -> None:
    with pytest.raises(ValueError):
        NewsImportanceModel.from_state({
            "alpha": 1.0,
            "documents": {"IMPORTANT": 0, "UNIMPORTANT": 1},
            "counts": {"IMPORTANT": {"token": 2}, "UNIMPORTANT": {"text": 3}},
        })
