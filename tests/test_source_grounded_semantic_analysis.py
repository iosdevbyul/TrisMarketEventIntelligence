"""Exact-source semantic evidence and source-only replay invariants."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.source_grounded_semantic_analysis import (
    issuer_provenance,
    semantic_summary,
    validate_semantic_interpretation,
)
from tris_market_event_intelligence.source_grounded_semantic_replay_cli import (
    semantic_replay,
)


def article(source: str = "sec_8k_2488") -> dict:
    return {
        "articleId": 20873, "source": source,
        "title": "ADVANCED MICRO DEVICES INC SEC 8-K Item 2.02 EARNINGS_8K",
        "summary": (
            "The company reported revenue of $8.2 billion for the quarter. "
            "The filing states that company expenses also increased in the "
            "period compared with the prior year, with several risks noted. "
            "The announced earnings information was included as Item 2.02."
        ),
        "sourceUrl": (
            "https://www.sec.gov/Archives/edgar/data/2488/"
            "000000248826000072/amd-20260505.htm"
        ),
        "publishedAtReported": "2026-05-05T20:16:06+00:00",
    }


def valid() -> dict:
    quote = "The company reported revenue of $8.2 billion for the quarter."
    return {
        "eventCategory": "EARNINGS_OR_GUIDANCE",
        "eventDescription": "Company released quarterly results",
        "sourceStatements": [
            {"quote": quote, "interpretation": "Revenue amount is disclosed",
             "factualStatus": "OFFICIAL"},
        ],
        "economicMechanisms": [
            {"affectedScope": "ISSUER", "economicDirection": "MIXED",
             "reasoning": "Hypothesis: operating performance may affect earnings",
             "groundingQuote": quote,
             "uncertainty": "Consensus and prior sales are unavailable"},
        ],
        "unverifiedAssumptions": ["Share-price reaction"],
        "missingEvidence": ["Historical consensus estimates"],
        "overallEconomicDirection": "MIXED",
    }


def test_semantic_claims_require_true_verbatim_source_quote():
    output = validate_semantic_interpretation(article(), valid())
    assert output["status"] == "SOURCE_GROUNDED_NOT_FACTUALLY_ADJUDICATED"
    assert output["issuerProvenance"]["symbol"] == "AMD"
    assert output["economicTargetLinks"][0]["target"] == "AMD"
    assert output["economicTargetLinks"][0]["priceDirectionUnverified"] is True
    assert output["priceDirectionPredicted"] is False
    fact = output["modelAnalysis"]["sourceStatements"][0]
    body = article()["summary"]
    assert body[fact["sourceStart"]:fact["sourceEnd"]] == fact["quote"]
    assert len(output["sourceBodySha256"]) == 64


def test_invented_numeric_claim_and_mechanism_are_rejected():
    result = valid()
    result["sourceStatements"][0]["quote"] = (
        "The company beat consensus EPS expectations by 45 percent."
    )
    with pytest.raises(ValueError, match="SEMANTIC_FACT_QUOTE_NOT_IN_SOURCE"):
        validate_semantic_interpretation(article(), result)
    result = valid()
    result["economicMechanisms"][0]["groundingQuote"] = (
        "Stock rallied immediately after the release."
    )
    with pytest.raises(ValueError, match="SEMANTIC_MECHANISM_QUOTE_NOT_IN_SOURCE"):
        validate_semantic_interpretation(article(), result)


def test_without_grounded_mechanism_model_cannot_claim_economic_direction():
    result = valid()
    result["economicMechanisms"] = []
    result["overallEconomicDirection"] = "POSITIVE"
    with pytest.raises(ValueError, match="SEMANTIC_UNSUPPORTED_DIRECTION"):
        validate_semantic_interpretation(article(), result)


def test_official_sec_issuer_not_inferred_from_model_prose():
    missing = article("bbc_business")
    assert issuer_provenance(missing)["status"] == "NO_OFFICIAL_ISSUER_BINDING"
    tampered = article()
    tampered["sourceUrl"] = "https://example.org/Archives/edgar/data/2488/filing"
    assert issuer_provenance(tampered)["status"] == "NO_OFFICIAL_ISSUER_BINDING"


def test_summary_counts_model_failures_without_fabricating_success():
    rows = [
        {"status": "COMPLETE", "analysis": {
            "modelAnalysis": {"eventCategory": "EARNINGS_OR_GUIDANCE",
                              "overallEconomicDirection": "MIXED"},
        }},
        {"status": "INVALID_MODEL_ANALYSIS", "analysis": None},
    ]
    output = semantic_summary(rows)
    assert output["articles"] == 2
    assert output["statusCounts"]["INVALID_MODEL_ANALYSIS"] == 1
    assert output["economicDirectionIsNotPriceForecast"]


def test_replay_preserves_original_article_set_and_does_not_pass_price_labels(
    monkeypatch, tmp_path: Path,
):
    from tris_market_event_intelligence import (
        source_grounded_semantic_replay_cli as replay,
    )

    row = article()
    replay_cohort = {"inventoryAsOf": "2026-11-20T03:55:45+00:00"}
    monkeypatch.setattr(replay, "source_items", lambda *_: (
        [row], "abc123", replay_cohort,
    ))
    calls = []

    class SemanticModel:
        def generate_json(self, *, system, user, schema):
            payload = json.loads(user)
            calls.append(payload)
            assert payload["sourceExcerpts"][0]["evidenceId"] == "E001"
            assert payload["sourceExcerpts"][0]["text"] in row["summary"]
            assert "sourceBody" not in payload
            assert "market" not in payload
            assert "actualDirection" not in payload
            assert "evaluation" not in payload
            response = valid()
            response["sourceStatements"] = [
                {"evidenceId": "E001", "interpretation": "Reported result",
                 "factualStatus": "OFFICIAL"},
            ]
            response["economicMechanisms"] = [
                {"evidenceId": "E001", "affectedScope": "ISSUER",
                 "economicDirection": "MIXED",
                 "reasoning": "Hypothesis: results may affect earnings",
                 "uncertainty": "Consensus unavailable"},
            ]
            return json.dumps(response)

    target = tmp_path / "semantic-review.json"
    output = semantic_replay(
        holdout_result=tmp_path / "already-scored.json",
        model=SemanticModel(), output=target,
    )
    assert output["articleCount"] == 1
    assert output["summary"]["statusCounts"]["COMPLETE"] == 1
    assert len(calls) == 1
    assert target.exists()
    report = json.loads(target.read_text())
    assert report["priceDataAccessed"] is False
    assert report["articles"][0]["semanticAnalysis"]["issuerProvenance"]["symbol"] == "AMD"
    assert report["articles"][0]["sourceBodySha256"] == hashlib.sha256(
        row["summary"].encode(),
    ).hexdigest()
    with pytest.raises(ValueError, match="SEMANTIC_REPORT_EXISTS"):
        semantic_replay(
            holdout_result=tmp_path / "already-scored.json",
            model=SemanticModel(), output=target,
        )
