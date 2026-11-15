"""Retrospective audit tests never grant approval or model Gold."""
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.historical_research_quality_cli import (
    evaluate_research,
    main,
)


def _research() -> dict:
    return {
        "schemaVersion": "historical_multi_hypothesis_research_v1",
        "status": "RETROSPECTIVE_HYPOTHESES_UNVERIFIED",
        "promotionAllowed": False,
        "weightTrainingPerformed": False,
        "modelId": "gemma4:e2b",
        "allCandidateReferenceCount": 44,
        "hypotheses": [
            {
                "category": "UNASSESSED",
                "reference_ids": ["ref-027", "ref-040", "issuer-primary"],
                "evidence_strength": "PRIMARY_DOCUMENT",
                "mechanism": "Firm-Specific Performance and Guidance",
                "alternative_explanations": "Financial results may influence demand expectations.",
                "source_limitations": "The release served as the primary catalyst.",
            },
            {
                "category": "OTHER_OR_UNCLEAR",
                "reference_ids": ["ref-001", "ref-003", "ref-013"],
                "evidence_strength": "URL_HINT_ONLY",
                "mechanism": "Industry/Competitive Dynamics",
                "alternative_explanations": "Sector commentary might provide a confounder.",
                "source_limitations": "Only URL hints were available.",
            },
            {
                "category": "UNASSESSED",
                "reference_ids": ["ref-035", "ref-014", "ref-019",
                                  "ref-036", "ref-038", "ref-043"],
                "evidence_strength": "INSUFFICIENT",
                "mechanism": "Macro/Market Sentiment",
                "alternative_explanations": "Market sentiment could be relevant.",
                "source_limitations": "Publisher clock is unverified.",
            },
        ],
        "researchSummary": "Strong guidance was the primary driver of the rally.",
        "missingEvidence": "No independent attribution is available.",
    }


def test_actual_nvidia_style_findings_and_no_automatic_learning() -> None:
    report = evaluate_research(_research())
    codes = [entry["code"] for entry in report["findings"]]
    assert "CATEGORY_STRENGTH_CONFLICT" in codes
    assert "MECHANISM_TITLE_ONLY" in codes
    assert "UNVERIFIED_CAUSAL_LANGUAGE" in codes
    assert report["candidateReferenceCount"] == 44
    assert report["explicitlyCitedArchiveReferenceCount"] == 11
    assert report["uncitedArchiveReferenceCount"] == 33
    assert report["lessonsApproved"] is False
    assert report["independentGoldCreated"] is False
    assert report["promotionAllowed"] is False


def test_unsupported_status_or_promotion_is_rejected() -> None:
    research = _research()
    research["promotionAllowed"] = True
    with pytest.raises(ValueError, match="Unsupported"):
        evaluate_research(research)
    research["promotionAllowed"] = False
    research["status"] = "PROMOTED"
    with pytest.raises(ValueError, match="Unsupported"):
        evaluate_research(research)


def test_cli_writes_private_new_report_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import sys

    source = tmp_path / "research.json"
    output = tmp_path / "review.json"
    source.write_text(json.dumps(_research()))
    monkeypatch.setattr(sys, "argv", [
        "historical-quality", "--research", str(source), "--output", str(output),
    ])
    main()
    assert output.exists()
    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(output.read_text())["status"] == "HUMAN_REVIEW_REQUIRED"
    assert '"outputCreated": true' in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_primary_document_citation_without_primary_id_is_flagged() -> None:
    research = _research()
    research["hypotheses"][0]["reference_ids"].remove("issuer-primary")
    report = evaluate_research(research)
    assert any(x["code"] == "PRIMARY_REFERENCE_MISSING" for x in report["findings"])
