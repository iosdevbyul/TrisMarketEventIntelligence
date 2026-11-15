"""Compare historical research quality without turning audit heuristics into Gold.

Baseline capture works before human approval. A paired comparison requires
a separately human-reviewed lesson file and the exact lesson hash on the new
research result. Same-case changes are diagnostic, never generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.historical_multi_hypothesis import read_reviewed_lessons
from tris_market_event_intelligence.historical_research_quality_cli import evaluate_research

MAX_BYTES = 120_000


def _read_private(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Invalid private input")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Expected JSON object")
    return data


def _sha(data: dict) -> str:
    return hashlib.sha256(json.dumps(
        data, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()


def _verified_pair(research: dict, audit: dict) -> None:
    recomputed = evaluate_research(research)
    if audit != recomputed or audit.get("sourceArtifactSha256") != _sha(research):
        raise ValueError("Research and quality audit must match exactly")


def _snapshot(research: dict, audit: dict) -> dict:
    findings = Counter(item["code"] for item in audit["findings"])
    return {
        "researchSha256": _sha(research),
        "reviewPacketSha256": research.get("reviewPacketSha256"),
        "qualityAuditSha256": _sha(audit),
        "modelId": research.get("modelId"),
        "promptVersion": research.get("promptVersion"),
        "humanReviewedLessonsSha256": research.get("humanReviewedLessonsSha256"),
        "reviewedLessonsUsed": research.get("usedHumanReviewedLessons"),
        "findingsByCode": dict(sorted(findings.items())),
        "totalFindings": sum(findings.values()),
        "candidateReferenceCount": audit["candidateReferenceCount"],
        "explicitlyCitedArchiveReferenceCount": audit[
            "explicitlyCitedArchiveReferenceCount"
        ],
        "coverageIsNotEvidenceOfReading": True,
    }


def build_comparison(
    *, baseline_research: dict, baseline_audit: dict,
    candidate_research: dict | None = None, candidate_audit: dict | None = None,
    approved_lessons: Path | None = None,
) -> dict:
    _verified_pair(baseline_research, baseline_audit)
    if (baseline_research.get("usedHumanReviewedLessons") is not False
            or baseline_research.get("humanReviewedLessonsSha256") is not None):
        raise ValueError("Baseline must not include reviewed lessons")
    baseline = _snapshot(baseline_research, baseline_audit)
    result = {
        "schemaVersion": "historical_research_learning_comparison_v1",
        "status": "BASELINE_RECORDED",
        "baseline": baseline,
        "candidate": None,
        "findingCountDelta": None,
        "findingDeltasByCode": None,
        "sameCaseDiagnosticOnly": True,
        "independentCaseEvaluationCompleted": False,
        "generalizationImprovementEstablished": False,
        "causalityOrForecastSkillEstablished": False,
        "humanReviewStillRequired": True,
        "independentGoldCreated": False,
        "promotionAllowed": False,
    }
    if (candidate_research is None and candidate_audit is None
            and approved_lessons is None):
        return result
    if (candidate_research is None or candidate_audit is None
            or approved_lessons is None):
        raise ValueError("Paired comparison requires candidate, audit, approved lessons")
    _verified_pair(candidate_research, candidate_audit)
    _, lesson_sha = read_reviewed_lessons(approved_lessons)
    if (candidate_research.get("usedHumanReviewedLessons") is not True
            or candidate_research.get("humanReviewedLessonsSha256") != lesson_sha
            or not isinstance(baseline["reviewPacketSha256"], str)
            or candidate_research.get("reviewPacketSha256") != baseline["reviewPacketSha256"]
            or candidate_research.get("modelId") != baseline["modelId"]
            or candidate_research.get("promptVersion") != baseline["promptVersion"]
            or candidate_research.get("allCandidateReferenceCount") != baseline[
                "candidateReferenceCount"
            ]
            or candidate_research.get("weightTrainingPerformed") is not False
            or candidate_research.get("promotionAllowed") is not False
            or baseline["researchSha256"] == _sha(candidate_research)):
        raise ValueError("Candidate provenance, model, or reviewed lessons mismatch")
    candidate = _snapshot(candidate_research, candidate_audit)
    categories = sorted(set(baseline["findingsByCode"]) | set(candidate["findingsByCode"]))
    result.update({
        "status": "PAIRED_DIAGNOSTIC_RECORDED",
        "candidate": candidate,
        "findingCountDelta": candidate["totalFindings"] - baseline["totalFindings"],
        "findingDeltasByCode": {
            key: candidate["findingsByCode"].get(key, 0) - baseline[
                "findingsByCode"
            ].get(key, 0)
            for key in categories
        },
    })
    return result


def _write_private(path: Path, data: dict) -> None:
    if path.is_symlink() or path.exists() or not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("Unsafe output file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare unverified research quality")
    parser.add_argument("--baseline-research", required=True, type=Path)
    parser.add_argument("--baseline-audit", required=True, type=Path)
    parser.add_argument("--candidate-research", type=Path)
    parser.add_argument("--candidate-audit", type=Path)
    parser.add_argument("--reviewed-lessons", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = build_comparison(
            baseline_research=_read_private(args.baseline_research),
            baseline_audit=_read_private(args.baseline_audit),
            candidate_research=(
                _read_private(args.candidate_research)
                if args.candidate_research else None
            ),
            candidate_audit=(
                _read_private(args.candidate_audit) if args.candidate_audit else None
            ),
            approved_lessons=args.reviewed_lessons,
        )
        _write_private(args.output, result)
        print(json.dumps({
            "status": result["status"],
            "baselineFindings": result["baseline"]["totalFindings"],
            "candidateFindings": (
                result["candidate"]["totalFindings"] if result["candidate"] else None
            ),
            "findingCountDelta": result["findingCountDelta"],
            "generalizationImprovementEstablished": False,
            "outputCreated": True,
        }))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        parser.exit(2, "historical-learning-compare: INVALID_OR_MISMATCHED_INPUT\n")


if __name__ == "__main__":
    main()
