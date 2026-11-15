"""Deterministic retrospective research audit; never a source of Gold labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from tris_market_event_intelligence.historical_multi_hypothesis import PRIMARY_ID
from tris_market_event_intelligence.historical_story_review import CATEGORIES

CAUSAL_ASSERTIONS = re.compile(
    r"\b(?:primary (?:driver|catalyst)|"
    r"(?:was|were|is) (?:primarily |mainly )?driven by|"
    r"served as (?:the |a )?(?:primary |main )?catalyst|"
    r"(?:definitely |directly )?caused (?:the |a )?(?:rally|decline|move|return))\b",
    re.IGNORECASE,
)
EXPLANATION_FIELDS = ("mechanism", "alternative_explanations", "source_limitations")
TITLE_ONLY = re.compile(
    r"^(?:firm.specific performance and guidance|industry/competitive dynamics|"
    r"macro/market sentiment)$",
    re.IGNORECASE,
)


def evaluate_research(research: dict) -> dict:
    """Flag checkable contradictions, without promoting model output to truth."""
    if (research.get("schemaVersion") != "historical_multi_hypothesis_research_v1"
            or research.get("status") != "RETROSPECTIVE_HYPOTHESES_UNVERIFIED"
            or research.get("promotionAllowed") is not False
            or research.get("weightTrainingPerformed") is not False
            or not isinstance(research.get("hypotheses"), list)
            or len(research["hypotheses"]) != 3):
        raise ValueError("Unsupported or unsafe research artifact")

    cited: set[str] = set()
    findings: list[dict] = []
    def flag(code: str, location: str, detail: str) -> None:
        findings.append({"code": code, "location": location, "detail": detail})

    for index, row in enumerate(research["hypotheses"]):
        if not isinstance(row, dict):
            raise TypeError("Invalid hypothesis entry")
        location = f"hypotheses[{index}]"
        refs = row.get("reference_ids")
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise ValueError("Invalid reference IDs")
        cited.update(ref for ref in refs if ref != PRIMARY_ID)
        strength = row.get("evidence_strength")
        category = row.get("category")
        if category not in (*CATEGORIES, "UNASSESSED"):
            raise ValueError("Invalid hypothesis category")
        if strength not in ("PRIMARY_DOCUMENT", "URL_HINT_ONLY", "INSUFFICIENT"):
            raise ValueError("Invalid evidence strength")
        if strength == "PRIMARY_DOCUMENT" and PRIMARY_ID not in refs:
            flag("PRIMARY_REFERENCE_MISSING", location, "Primary-document claim has no issuer citation")
        if strength != "PRIMARY_DOCUMENT" and PRIMARY_ID in refs:
            flag("PRIMARY_REFERENCE_CLASS_CONFLICT", location,
                 "Issuer citation and evidence strength should be reviewed together")
        if category == "UNASSESSED" and strength == "PRIMARY_DOCUMENT":
            flag("CATEGORY_STRENGTH_CONFLICT", location,
                 "Unassessed classification conflicts with strong primary-document evidence")
        if strength == "INSUFFICIENT" and refs:
            flag("INSUFFICIENT_WITH_REFERENCES_REVIEW", location,
                 "References are candidates only; confirm why evidence is insufficient")
        mechanism = row.get("mechanism")
        if not isinstance(mechanism, str):
            raise TypeError("Invalid mechanism")
        if TITLE_ONLY.fullmatch(mechanism.strip()):
            flag("MECHANISM_TITLE_ONLY", location, "Mechanism is a topic label, not an economic chain")
        for key in EXPLANATION_FIELDS:
            value = row.get(key)
            if not isinstance(value, str):
                raise TypeError("Invalid explanation text")
            if CAUSAL_ASSERTIONS.search(value):
                flag("UNVERIFIED_CAUSAL_LANGUAGE", f"{location}.{key}",
                     "Causal language exceeds unverified retrospective evidence")
    for field in ("researchSummary", "missingEvidence"):
        value = research.get(field)
        if not isinstance(value, str):
            raise TypeError("Invalid summary field")
        if CAUSAL_ASSERTIONS.search(value):
            flag("UNVERIFIED_CAUSAL_LANGUAGE", field,
                 "Causal language exceeds unverified retrospective evidence")

    count = research.get("allCandidateReferenceCount")
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("Invalid reference cohort count")
    # This measures explicit citation coverage, NOT whether the model read each input.
    coverage = len(cited)
    if coverage > count:
        flag("REFERENCE_COUNT_CONFLICT", "hypotheses",
             "Unique cited archive IDs exceed provided cohort size")
    return {
        "schemaVersion": "historical_research_quality_review_v1",
        "status": "HUMAN_REVIEW_REQUIRED",
        "sourceArtifactSha256": hashlib.sha256(
            json.dumps(research, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "modelId": research.get("modelId"),
        "candidateReferenceCount": count,
        "explicitlyCitedArchiveReferenceCount": coverage,
        "uncitedArchiveReferenceCount": max(0, count - coverage),
        "coverageIsNotEvidenceOfReading": True,
        "findings": findings,
        "suggestedMethodLessons": [
            {
                "lesson": "Use an assessed category only when it matches the cited evidence strength.",
                "evidence": "Review category and evidence-strength conflicts against the actual sources.",
            },
            {
                "lesson": "Describe a plausible economic mechanism without declaring market causality.",
                "evidence": "Retrospective URL hints and issuer documents do not establish a causal return.",
            },
            {
                "lesson": "Explain source limits and distinguish URL hints from verified article text.",
                "evidence": "Archive reference coverage alone cannot verify publisher text or timestamps.",
            },
        ],
        "lessonsApproved": False,
        "independentGoldCreated": False,
        "promotionAllowed": False,
        "weightTrainingPerformed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit unverified historical Gemma research")
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        source, output = args.research, args.output
        if (source.is_symlink() or not source.is_file()
                or source.stat().st_size > 100_000
                or output.is_symlink() or output.exists()
                or not output.parent.is_dir() or output.parent.is_symlink()):
            raise ValueError("Unsafe input or existing output")
        report = evaluate_research(json.loads(source.read_text(encoding="utf-8")))
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(output, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        print(json.dumps({
            "status": report["status"],
            "findings": len(report["findings"]),
            "citedArchiveReferences": report["explicitlyCitedArchiveReferenceCount"],
            "candidateReferences": report["candidateReferenceCount"],
            "lessonsApproved": False,
            "outputCreated": True,
        }))
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
        parser.exit(2, "historical-quality-review: INVALID_OR_UNSAFE_ARTIFACT\n")


if __name__ == "__main__":
    main()
