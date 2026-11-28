"""Deterministic checks for retrospective reflection diagnostics."""
from tris_market_event_intelligence.prediction_failure_analysis_cli import _summary
from tris_market_event_intelligence.reflection_quality_analysis_cli import (
    citation_flags,
    paired_effect,
)


def test_citation_provenance_is_not_semantic_verification():
    packet = {"sourcePassages": [{"id": "E1", "text": "Original source"}]}
    reflection = {
        "category": "MARKET_RESPONSE",
        "evidence_id": "E1",
        "evidenceQuote": "Original source",
        "hypothesis": "A hypothesis",
        "missing_evidence": "Other information",
        "falsification": "A test",
        "semanticEntailmentVerified": False,
    }
    assert citation_flags(reflection, packet) == ["SEMANTIC_SUPPORT_UNVERIFIED"]
    reflection["evidenceQuote"] = "Different text"
    assert "CITATION_MISMATCH" in citation_flags(reflection, packet)


def test_paired_effect_excludes_no_memory_and_unpaired_predictions():
    common = {
        "articleId": 1, "market": "XNYS", "symbol": "ABC", "horizon": 1,
        "pairedAB": True, "retrievedExperience": True,
    }
    rows = [
        {**common, "arm": "A", "prediction": "UP",
         "verdict": "CORRECT", "relativeVerdict": "CORRECT"},
        {**common, "arm": "B", "prediction": "NEUTRAL",
         "verdict": "INCORRECT", "relativeVerdict": "INCORRECT"},
        {**common, "articleId": 2, "arm": "A", "prediction": "UP",
         "retrievedExperience": False, "verdict": "CORRECT"},
        {**common, "articleId": 3, "arm": "B", "prediction": "UP",
         "pairedAB": False, "verdict": "CORRECT"},
    ]
    result = paired_effect(rows)["1d:verdict"]
    assert result["pairs"] == 1
    assert result["netCorrectChange"] == -1
    assert result["directionTransitions"] == {"UP_TO_NEUTRAL": 1}


def test_false_retrieval_is_not_unknown():
    rows = [{"retrievedExperience": False, "prediction": "UP", "actual": "UP"}]
    assert _summary(rows, "retrievedExperience", actual_key="actual") == {
        "False": {"count": 1, "correct": 1, "accuracy": 1.0}
    }
