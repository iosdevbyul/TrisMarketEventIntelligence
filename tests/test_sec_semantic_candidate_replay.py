"""Real SEC replay stays matched to verified archived originals and baselines."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
from test_sec_primary_semantic_review import _packet, _Provider

from tris_market_event_intelligence import sec_semantic_candidate_replay_cli as replay_cli
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    prepare_primary_cases,
    run_semantic_research,
)
from tris_market_event_intelligence.sec_semantic_candidate_replay_cli import (
    prepare_candidate_replay,
    replay_candidate,
)


def _inputs(tmp_path: Path, count: int = 3) -> tuple[dict, list[tuple[dict, str]]]:
    primary = _packet(tmp_path, count=count)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    baseline = run_semantic_research(
        prepared, _Provider(prepared["cases"]), start_index=0, limit=count,
    )
    return primary, [(baseline, "b" * 64)]


def test_replay_only_exact_existing_baseline_cases(tmp_path: Path) -> None:
    primary, baseline = _inputs(tmp_path)
    prepared, plan = prepare_candidate_replay(
        primary, "a" * 64, tmp_path, baseline, start_index=1, limit=2,
    )
    assert plan["caseIndexes"] == [1, 2]
    assert plan["baselineCasesValidated"] == 3
    assert plan["qualityImprovementVerified"] is False
    provider = _Provider(prepared["cases"])
    candidate = replay_candidate(prepared, plan, provider)
    assert [c["caseIndex"] for c in candidate["cases"]] == [1, 2]
    assert len(provider.calls) == 0
    assert candidate["modelCalls"] == 0
    assert candidate["casesAbstained"] == 2
    assert candidate["trainingReady"] is False
    assert candidate["promotionAllowed"] is False


@pytest.mark.parametrize(("start", "limit"), [(-1, 1), (3, 1), (2, 2), (0, 4)])
def test_invalid_replay_windows_fail_before_provider(
    tmp_path: Path, start: int, limit: int,
) -> None:
    primary, baseline = _inputs(tmp_path)
    with pytest.raises(ValueError, match="SEC_CANDIDATE_RANGE_INVALID"):
        prepare_candidate_replay(
            primary, "a" * 64, tmp_path, baseline,
            start_index=start, limit=limit,
        )


def test_replay_requires_verified_baseline_for_every_case(tmp_path: Path) -> None:
    primary, baseline = _inputs(tmp_path, count=3)
    shorter = copy.deepcopy(baseline[0][0])
    shorter["cases"] = shorter["cases"][:1]
    shorter["casesAnalyzed"] = 1
    shorter["modelCalls"] = 1
    with pytest.raises(ValueError, match="SEC_CANDIDATE_BASELINE_CASES_MISSING"):
        prepare_candidate_replay(
            primary, "a" * 64, tmp_path, [(shorter, "b" * 64)],
            start_index=1, limit=2,
        )


def test_replay_rejects_original_source_mutation(tmp_path: Path) -> None:
    primary, baseline = _inputs(tmp_path)
    original = tmp_path / f"sec-{primary['cases'][0]['accession']}-primary.source"
    original.write_bytes(original.read_bytes().replace(b"CORPORATION", b"MODIFIED"))
    with pytest.raises(ValueError, match="SEC_SEMANTIC_SOURCE_SHA_MISMATCH"):
        prepare_candidate_replay(
            primary, "a" * 64, tmp_path, baseline, start_index=0, limit=1,
        )


def test_prepare_mode_does_not_call_model_or_write_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    primary, baseline = _inputs(tmp_path)
    original = tmp_path / "primary.json"
    report = tmp_path / "baseline.json"
    original.write_text(json.dumps(primary), encoding="utf-8")
    report.write_text(json.dumps(baseline[0][0]), encoding="utf-8")
    def read_pinned(path: Path, limit: int) -> tuple[dict, str]:
        if path == original:
            return primary, "a" * 64
        if path == report:
            return baseline[0][0], "b" * 64
        raise AssertionError("Unexpected file")
    monkeypatch.setattr(replay_cli, "_read_json", read_pinned)
    monkeypatch.setattr(
        replay_cli, "build_runtime",
        lambda *args, **kwargs: pytest.fail("prepare called a model"),
    )
    target = tmp_path / "candidate.json"
    monkeypatch.setattr(sys, "argv", [
        "sec-candidate-replay", "--primary", str(original),
        "--source-dir", str(tmp_path), "--baseline-report", str(report),
        "--start-index", "0", "--limit", "1", "--prepare",
        "--output", str(target),
    ])
    replay_cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "VERIFIED_BASELINE_REPLAY_READY"
    assert not target.exists()


def test_run_existing_target_fails_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "candidate.json"
    target.write_text("immutable existing data", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "sec-candidate-replay", "--primary", str(tmp_path / "primary.json"),
        "--source-dir", str(tmp_path), "--baseline-report", str(tmp_path / "base.json"),
        "--start-index", "0", "--limit", "1", "--run", "--output", str(target),
    ])
    with pytest.raises(SystemExit) as exc:
        replay_cli.main()
    assert exc.value.code == 2
    assert "ERROR_CODE=SEC_CANDIDATE_OUTPUT_NOT_CREATE_ONLY" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "immutable existing data"
