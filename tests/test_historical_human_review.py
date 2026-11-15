"""Human review has no implicit approval, and cannot bypass evidence checks."""
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.historical_human_review_cli import (
    finalize_review,
    main,
    prepare_review,
)
from tris_market_event_intelligence.historical_multi_hypothesis import (
    read_reviewed_lessons,
)


def _audit() -> dict:
    return {
        "schemaVersion": "historical_research_quality_review_v1",
        "status": "HUMAN_REVIEW_REQUIRED",
        "lessonsApproved": False,
        "independentGoldCreated": False,
        "promotionAllowed": False,
        "sourceArtifactSha256": "a" * 64,
        "findings": [
            {"code": "CATEGORY_STRENGTH_CONFLICT",
             "location": "hypotheses[0]", "detail": "Conflicting evidence"}
        ],
        "suggestedMethodLessons": [
            {"lesson": "Do not overstate evidence strength in retrospective studies.",
             "evidence": "Unverified first-party document timestamp."},
        ],
    }


def _edited_review(audit: dict) -> dict:
    review = prepare_review(audit)
    review["reviewer"] = "human-analyst"
    review["sourceEvidenceReviewed"] = True
    review["approved"] = True
    review["findings"][0]["decision"] = "CONFIRMED"
    review["findings"][0]["reviewNote"] = "Checked the unverified classification conflict."
    review["lessons"][0]["selected"] = True
    review["lessons"][0]["independentEvidence"] = (
        "Independent inspection of primary issuer evidence reveals "
        "that market causality is not established."
    )
    return review


def test_prepare_is_not_approval() -> None:
    review = prepare_review(_audit())
    assert review["status"] == "AWAITING_HUMAN_REVIEW"
    assert review["findings"][0]["decision"] == "PENDING"
    assert review["sourceEvidenceReviewed"] is False
    assert review["approved"] is False
    with pytest.raises(ValueError):
        finalize_review(audit=_audit(), review=review, reviewer="human-analyst")


def test_finalize_rejects_missing_decisions_and_tampering() -> None:
    audit = _audit()
    review = _edited_review(audit)
    review["findings"][0]["decision"] = "PENDING"
    with pytest.raises(ValueError):
        finalize_review(audit=audit, review=review, reviewer="human-analyst")
    review = _edited_review(audit)
    audit["findings"].append({
        "code": "ADDITIONAL", "location": "researchSummary", "detail": "Changed"
    })
    with pytest.raises(ValueError):
        finalize_review(audit=audit, review=review, reviewer="human-analyst")


def test_finalized_lessons_are_compatible_with_existing_input(tmp_path: Path) -> None:
    audit = _audit()
    data = finalize_review(
        audit=audit, review=_edited_review(audit), reviewer="human-analyst",
    )
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(data))
    lessons, sha = read_reviewed_lessons(path)
    assert len(lessons) == 1
    assert sha is not None
    assert data["reviewedByHuman"] is True
    assert data["auditSha256"] == prepare_review(audit)["auditSha256"]


def test_cli_prepare_and_explicit_finalize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    audit_file = tmp_path / "audit.json"
    review_file = tmp_path / "template.json"
    lessons_file = tmp_path / "lessons.json"
    audit_file.write_text(json.dumps(_audit()))
    monkeypatch.setattr(sys, "argv", [
        "review", "--prepare", "--audit", str(audit_file),
        "--output", str(review_file),
    ])
    main()
    assert review_file.stat().st_mode & 0o777 == 0o600
    review = _edited_review(_audit())
    review_file.write_text(json.dumps(review))
    monkeypatch.setattr(sys, "argv", [
        "review", "--finalize", "--audit", str(audit_file),
        "--review", str(review_file), "--reviewer", "human-analyst",
        "--output", str(lessons_file),
    ])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not lessons_file.exists()
    monkeypatch.setattr(sys, "argv", [
        "review", "--finalize", "--audit", str(audit_file),
        "--review", str(review_file), "--reviewer", "human-analyst",
        "--confirm-human-review", "--output", str(lessons_file),
    ])
    main()
    assert lessons_file.stat().st_mode & 0o777 == 0o600
    assert json.loads(lessons_file.read_text())["reviewedByHuman"] is True


def test_selected_lesson_requires_independent_evidence() -> None:
    audit = _audit()
    review = _edited_review(audit)
    review["lessons"][0]["independentEvidence"] = "too short"
    with pytest.raises(ValueError, match="independently"):
        finalize_review(audit=audit, review=review, reviewer="human-analyst")
