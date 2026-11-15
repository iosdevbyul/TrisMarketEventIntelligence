"""Private manual-review workflow for retrospective Gemma research quality reports.

Preparing a review is never an approval. Only an explicit human attestation
with independently inspected evidence may export procedural lessons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tris_market_event_intelligence.historical_multi_hypothesis import read_reviewed_lessons

MAX_BYTES = 100_000
DECISIONS = {"CONFIRMED", "REJECTED", "UNCERTAIN"}


def _sha(document: dict) -> str:
    return hashlib.sha256(json.dumps(
        document, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Invalid review input")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Expected JSON object")
    return document


def _check_audit(audit: dict) -> None:
    if (audit.get("schemaVersion") != "historical_research_quality_review_v1"
            or audit.get("status") != "HUMAN_REVIEW_REQUIRED"
            or audit.get("lessonsApproved") is not False
            or audit.get("independentGoldCreated") is not False
            or audit.get("promotionAllowed") is not False
            or not isinstance(audit.get("sourceArtifactSha256"), str)
            or len(audit["sourceArtifactSha256"]) != 64
            or not isinstance(audit.get("findings"), list)
            or not isinstance(audit.get("suggestedMethodLessons"), list)
            or len(audit["suggestedMethodLessons"]) > 6):
        raise ValueError("Unsupported audit artifact")
    for finding in audit["findings"]:
        if (not isinstance(finding, dict)
                or not all(isinstance(finding.get(k), str)
                           for k in ("code", "location", "detail"))):
            raise ValueError("Invalid audit finding")
    for candidate in audit["suggestedMethodLessons"]:
        if (not isinstance(candidate, dict)
                or not all(isinstance(candidate.get(k), str)
                           for k in ("lesson", "evidence"))):
            raise ValueError("Invalid suggested lesson")


def prepare_review(audit: dict) -> dict:
    _check_audit(audit)
    return {
        "schemaVersion": "historical_research_human_review_v1",
        "status": "AWAITING_HUMAN_REVIEW",
        "auditSha256": _sha(audit),
        "researchSha256": audit["sourceArtifactSha256"],
        "reviewer": "",
        "sourceEvidenceReviewed": False,
        "findings": [
            {"code": f["code"], "location": f["location"],
             "decision": "PENDING", "reviewNote": ""}
            for f in audit["findings"]
        ],
        "lessons": [
            {"lesson": item["lesson"], "selected": False,
             "independentEvidence": ""}
            for item in audit["suggestedMethodLessons"]
        ],
        "approved": False,
        "independentGoldCreated": False,
        "promotionAllowed": False,
    }


def finalize_review(*, audit: dict, review: dict, reviewer: str) -> dict:
    """Require affirmative human testimony; no automatic approval path."""
    _check_audit(audit)
    expected = prepare_review(audit)
    if (review.get("schemaVersion") != expected["schemaVersion"]
            or review.get("auditSha256") != expected["auditSha256"]
            or review.get("researchSha256") != expected["researchSha256"]
            or review.get("status") != "AWAITING_HUMAN_REVIEW"
            or review.get("sourceEvidenceReviewed") is not True
            or review.get("approved") is not True
            or review.get("independentGoldCreated") is not False
            or review.get("promotionAllowed") is not False
            or not isinstance(reviewer, str) or not 2 <= len(reviewer.strip()) <= 80
            or review.get("reviewer") != reviewer
            or not isinstance(review.get("findings"), list)
            or not isinstance(review.get("lessons"), list)
            or len(review["findings"]) != len(expected["findings"])
            or len(review["lessons"]) != len(expected["lessons"])):
        raise ValueError("Explicit human review attestation missing or mismatched")

    for actual, template in zip(review["findings"], expected["findings"], strict=True):
        if (not isinstance(actual, dict)
                or actual.get("code") != template["code"]
                or actual.get("location") != template["location"]
                or actual.get("decision") not in DECISIONS
                or not isinstance(actual.get("reviewNote"), str)
                or not 15 <= len(actual["reviewNote"].strip()) <= 1000):
            raise ValueError("Every finding needs a manual decision and note")

    accepted = []
    for actual, template in zip(review["lessons"], expected["lessons"], strict=True):
        if not isinstance(actual, dict) or actual.get("lesson") != template["lesson"]:
            raise ValueError("Lesson identity mismatch")
        if type(actual.get("selected")) is not bool:
            raise ValueError("Lesson choice must be explicit")
        evidence = actual.get("independentEvidence")
        if not isinstance(evidence, str):
            raise TypeError("Invalid independent evidence")
        if actual["selected"]:
            if not 30 <= len(evidence.strip()) <= 300:
                raise ValueError("Accepted lesson needs independently inspected evidence")
            accepted.append({"lesson": actual["lesson"], "evidence": evidence.strip()})
        elif evidence.strip():
            raise ValueError("Unselected lessons must not claim evidence")
    if not 1 <= len(accepted) <= 6:
        raise ValueError("Select one to six manually reviewed lessons")

    result = {
        "schemaVersion": "human_reviewed_research_lessons_v1",
        "reviewedByHuman": True,
        "reviewer": reviewer,
        "auditSha256": expected["auditSha256"],
        "researchSha256": expected["researchSha256"],
        "lessons": accepted,
    }
    # Check the real existing Gemma input contract before publishing.
    # No temporary local file is needed to validate the equivalent shape.
    for item in accepted:
        if any(not 10 <= len(item[key]) <= 300 for key in ("lesson", "evidence")):
            raise ValueError("Lesson exceeds Gemma memory limit")
    return result


def _write_private(path: Path, document: dict) -> None:
    if (path.is_symlink() or path.exists() or not path.parent.is_dir()
            or path.parent.is_symlink()):
        raise ValueError("Unsafe output path")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicit manual historical research review")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--confirm-human-review", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit = _read(args.audit)
        if args.prepare:
            if args.review or args.reviewer or args.confirm_human_review:
                raise ValueError("Prepare never approves lessons")
            result = prepare_review(audit)
        else:
            if not (args.review and args.reviewer and args.confirm_human_review):
                raise ValueError("Explicit human confirmation required")
            result = finalize_review(
                audit=audit, review=_read(args.review), reviewer=args.reviewer,
            )
        _write_private(args.output, result)
        # Never print private lesson contents or evidence.
        print(json.dumps({
            "status": "HUMAN_REVIEW_COMPLETE" if args.finalize else "AWAITING_HUMAN_REVIEW",
            "findings": len(audit["findings"]),
            "approvedLessons": len(result["lessons"]) if args.finalize else 0,
            "outputCreated": True,
        }))
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        parser.exit(2, "historical-human-review: INVALID_OR_UNAPPROVED_INPUT\n")


if __name__ == "__main__":
    main()
