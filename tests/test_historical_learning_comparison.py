"""Baseline and paired comparisons never silently promote retrospective research."""
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.historical_learning_comparison_cli import (
    build_comparison,
    main,
)
from tris_market_event_intelligence.historical_research_quality_cli import evaluate_research


def _research(*, coached: bool = False) -> dict:
    return {
        "schemaVersion": "historical_multi_hypothesis_research_v1",
        "status": "RETROSPECTIVE_HYPOTHESES_UNVERIFIED",
        "modelId": "gemma4:e2b",
        "promptVersion": "historical-multi-evidence-v2",
        "reviewPacketSha256": "a" * 64,
        "humanReviewedLessonsSha256": None,
        "usedHumanReviewedLessons": False,
        "promotionAllowed": False,
        "weightTrainingPerformed": False,
        "allCandidateReferenceCount": 44,
        "hypotheses": [
            {
                "category": "COMPANY_EARNINGS_OR_GUIDANCE",
                "evidence_strength": "PRIMARY_DOCUMENT",
                "reference_ids": ["issuer-primary", "ref-027"],
                "mechanism": (
                    "Expected company revenue growth could revise investors' future "
                    "cash flow projections."
                ) if coached else "Firm-Specific Performance and Guidance",
                "alternative_explanations": "Other sector news may confound this explanation.",
                "source_limitations": "No verified first tradable release clock.",
            },
            {
                "category": "COMPETITOR_OR_SECTOR",
                "evidence_strength": "URL_HINT_ONLY",
                "reference_ids": ["ref-001", "ref-003"],
                "mechanism": "Industry supply changes could influence sentiment.",
                "alternative_explanations": "Issuer guidance remains an alternative.",
                "source_limitations": "Archive URL hints were not verified full articles.",
            },
            {
                "category": "UNASSESSED",
                "evidence_strength": "INSUFFICIENT",
                "reference_ids": [],
                "mechanism": "Missing macroeconomic indicators prevent any ranking.",
                "alternative_explanations": "Policy news may be a confounding factor.",
                "source_limitations": "No independent timestamped macro data provided.",
            },
        ],
        "researchSummary": "Several tentative explanations require verification.",
        "missingEvidence": "Missing release timestamps and article bodies.",
    }


def _lessons(tmp_path: Path) -> Path:
    path = tmp_path / "approved.json"
    path.write_text(json.dumps({
        "schemaVersion": "human_reviewed_research_lessons_v1",
        "reviewedByHuman": True,
        "reviewer": "human-analyst",
        "auditSha256": "b" * 64,
        "researchSha256": "c" * 64,
        "lessons": [{
            "lesson": "Do not overstate first-party evidence when timestamps are unknown.",
            "evidence": "The analyst independently inspected the release and news archive.",
        }],
    }))
    return path


def test_baseline_recorded_without_human_approval() -> None:
    baseline = _research()
    report = build_comparison(
        baseline_research=baseline, baseline_audit=evaluate_research(baseline),
    )
    assert report["status"] == "BASELINE_RECORDED"
    assert report["baseline"]["totalFindings"] == 1
    assert report["candidate"] is None
    assert report["independentCaseEvaluationCompleted"] is False
    assert report["promotionAllowed"] is False


def test_paired_comparison_tracks_heuristics_not_generalization(tmp_path: Path) -> None:
    from tris_market_event_intelligence.historical_multi_hypothesis import read_reviewed_lessons

    baseline = _research()
    candidate = _research(coached=True)
    approved = _lessons(tmp_path)
    _, digest = read_reviewed_lessons(approved)
    candidate["usedHumanReviewedLessons"] = True
    candidate["humanReviewedLessonsSha256"] = digest
    report = build_comparison(
        baseline_research=baseline, baseline_audit=evaluate_research(baseline),
        candidate_research=candidate, candidate_audit=evaluate_research(candidate),
        approved_lessons=approved,
    )
    assert report["status"] == "PAIRED_DIAGNOSTIC_RECORDED"
    assert report["findingCountDelta"] == -1
    assert report["findingDeltasByCode"] == {"MECHANISM_TITLE_ONLY": -1}
    assert report["generalizationImprovementEstablished"] is False


def test_rejects_tampered_audit_and_unapproved_candidate(tmp_path: Path) -> None:
    baseline = _research()
    audit = evaluate_research(baseline)
    audit["findings"] = []
    with pytest.raises(ValueError, match="match exactly"):
        build_comparison(baseline_research=baseline, baseline_audit=audit)
    candidate = _research(coached=True)
    with pytest.raises(ValueError, match="Candidate provenance"):
        build_comparison(
            baseline_research=baseline, baseline_audit=evaluate_research(baseline),
            candidate_research=candidate, candidate_audit=evaluate_research(candidate),
            approved_lessons=_lessons(tmp_path),
        )


def test_rejects_nonmatching_study_packet(tmp_path: Path) -> None:
    from tris_market_event_intelligence.historical_multi_hypothesis import read_reviewed_lessons

    baseline = _research()
    candidate = _research(coached=True)
    lessons = _lessons(tmp_path)
    _, digest = read_reviewed_lessons(lessons)
    candidate["usedHumanReviewedLessons"] = True
    candidate["humanReviewedLessonsSha256"] = digest
    candidate["reviewPacketSha256"] = "0" * 64
    with pytest.raises(ValueError, match="Candidate provenance"):
        build_comparison(
            baseline_research=baseline, baseline_audit=evaluate_research(baseline),
            candidate_research=candidate, candidate_audit=evaluate_research(candidate),
            approved_lessons=lessons,
        )


def test_cli_baseline_creates_private_output_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    research = _research()
    research_file = tmp_path / "baseline.json"
    audit_file = tmp_path / "audit.json"
    output = tmp_path / "comparison.json"
    research_file.write_text(json.dumps(research))
    audit_file.write_text(json.dumps(evaluate_research(research)))
    monkeypatch.setattr(sys, "argv", [
        "historical-learning-compare", "--baseline-research", str(research_file),
        "--baseline-audit", str(audit_file), "--output", str(output),
    ])
    main()
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
