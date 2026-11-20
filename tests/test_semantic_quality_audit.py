"""Adversarial quality audit of real source-grounded economic reports."""
from __future__ import annotations

import copy

import pytest

from tris_market_event_intelligence.semantic_quality_audit_cli import (
    audit_article,
    audit_report,
)


def record(
    *, article_id=20873, title="AMD SEC 8-K Item 2.02 EARNINGS_8K",
    source="sec_8k_2488", category="CORPORATE_ACTION_OR_CONTRACT",
    original="EARNINGS_OR_GUIDANCE", direction="POSITIVE", status="NEEDS_CATEGORY_REVIEW",
) -> dict:
    return {
        "articleId": article_id, "title": title, "source": source,
        "status": status,
        "semanticAnalysis": {
            "classification": {"eventCategory": original},
            "modelAnalysis": {
                "eventCategory": category, "overallEconomicDirection": direction,
                "sourceStatements": [{
                    "quote": "The company reported revenue of $11 billion.",
                    "interpretation": "Revenue reported as 11 billion.",
                    "factualStatus": "REPORTED",
                }],
                "economicMechanisms": [{
                    "affectedScope": "INDUSTRY",
                    "economicDirection": direction,
                    "reasoning": "The company's own revenue increased.",
                    "uncertainty": "Unknown consensus expectation",
                }],
                "missingEvidence": [],
            },
        },
    }


def codes(result: dict) -> set[str]:
    return {row["code"] for row in result["findings"]}


def test_real_earnings_filing_is_not_normalized_to_wrong_model_category():
    article = record()
    before = copy.deepcopy(article)
    audit = audit_article(article)
    assert "SEC_EARNINGS_ITEM_CATEGORY_CONFLICT" in codes(audit)
    assert "SEC_ISSUER_TRANSMISSION_NOT_IDENTIFIED" in codes(audit)
    assert audit["qualityStatus"] == "REQUIRES_FACTUAL_REVIEW"
    assert audit["originalModelCategory"] == "CORPORATE_ACTION_OR_CONTRACT"
    assert article == before


def test_issuer_and_market_scope_anomalies_are_kept_separate():
    macro = record(
        article_id=208, title="2026년 1/4분기중 가계신용(잠정)",
        source="bok_statistics", category="ECONOMIC_INDICATOR_RELEASE",
        original="ECONOMIC_INDICATOR_RELEASE", direction="UNKNOWN",
        status="COMPLETE",
    )
    macro["semanticAnalysis"]["modelAnalysis"]["economicMechanisms"][0][
        "affectedScope"
    ] = "ISSUER"
    assert "MACRO_REPORT_UNGROUNDED_SINGLE_ISSUER" in codes(audit_article(macro))


def test_unsupported_official_allegation_and_statistical_market_mechanism():
    article = record(
        article_id=187, title="2010-2015-2020년 접속산업연관표 작성결과",
        source="bok_statistics", category="ECONOMIC_INDICATOR_RELEASE",
        original="ECONOMIC_INDICATOR_RELEASE", direction="MIXED",
        status="COMPLETE",
    )
    model = article["semanticAnalysis"]["modelAnalysis"]
    model["sourceStatements"][0]["factualStatus"] = "ALLEGED"
    model["economicMechanisms"][0]["uncertainty"] = "UNKNOWN"
    finding_codes = codes(audit_article(article))
    assert "OFFICIAL_SOURCE_ALL_STATEMENTS_ALLEGED" in finding_codes
    assert "METHODOLOGY_OR_REVIEW_MARKET_EFFECT_UNPROVEN" in finding_codes
    assert "UNCERTAINTY_NOT_EXPLAINED" in finding_codes


def test_xbrl_identifiers_are_not_reliable_new_issuance_evidence():
    article = record(
        title="Alphabet SEC 8-K Item 1.01 MATERIAL_CONTRACT_8K",
        source="sec_8k_1652044", original="CORPORATE_ACTION_OR_CONTRACT",
        status="COMPLETE",
    )
    article["semanticAnalysis"]["modelAnalysis"]["sourceStatements"][0]["quote"] = (
        "2026-06-04 us-gaap:SeniorNotesMember goog:CapitalClassCMember"
    )
    assert "SEC_XBRL_IDENTIFIER_MISTAKEN_FOR_EVENT_EVIDENCE" in codes(
        audit_article(article),
    )


def test_korean_large_unit_conversion_is_flagged_not_autocorrected():
    article = record(
        article_id=205, title="2026년 1/4분기 차주별 가계부채(잠정)",
        source="bok_statistics", category="ECONOMIC_INDICATOR_RELEASE",
        original="ECONOMIC_INDICATOR_RELEASE", status="COMPLETE",
    )
    article["semanticAnalysis"]["modelAnalysis"]["sourceStatements"][0] = {
        "quote": "차주당 주택담보대출 신규취급액은 2억 2,939만원",
        "interpretation": "2.2939 million KRW were borrowed.",
        "factualStatus": "REPORTED",
    }
    result = audit_article(article)
    assert "KOREAN_LARGE_NUMBER_UNIT_CONVERSION_REVIEW" in codes(result)
    assert result["qualityStatus"] == "REQUIRES_FACTUAL_REVIEW"


def test_qoq_yoy_claim_mismatch_is_flagged_without_rewriting():
    article = record(
        article_id=198, title="2026년 1/4분기 국민소득(잠정)",
        source="bok_statistics", category="ECONOMIC_INDICATOR_RELEASE",
        original="ECONOMIC_INDICATOR_RELEASE", status="COMPLETE",
    )
    article["semanticAnalysis"]["modelAnalysis"]["sourceStatements"][0] = {
        "quote": "명목 국내총생산은 전기대비 10.5% 성장",
        "interpretation": "Nominal GDP rose 10.5% year-over-year.",
        "factualStatus": "REPORTED",
    }
    assert "PERIOD_COMPARISON_QOQ_VS_YOY_REVIEW" in codes(audit_article(article))


def test_positive_north_korea_gdp_is_not_automatically_korean_equity_positive():
    article = record(
        article_id=175, title="2025년 북한 경제성장률 추정 결과",
        source="bok_statistics", category="ECONOMIC_INDICATOR_RELEASE",
        original="ECONOMIC_INDICATOR_RELEASE", status="COMPLETE",
    )
    assert "CROSS_BORDER_MARKET_MAPPING_UNPROVEN" in codes(audit_article(article))


def test_clean_case_is_only_structurally_clear_not_human_verified():
    article = record(
        article_id=77, title="2026년 5월 경기조사",
        source="bok_statistics", category="ECONOMIC_INDICATOR_RELEASE",
        original="ECONOMIC_INDICATOR_RELEASE", status="COMPLETE",
    )
    article["semanticAnalysis"]["modelAnalysis"]["economicMechanisms"][0][
        "affectedScope"
    ] = "BROAD_MARKET"
    result = audit_article(article)
    assert result["findings"] == []
    assert result["qualityStatus"] == "STRUCTURE_ONLY_NO_GOLD_VALIDATION"
    assert result["semanticClaimsIndependentlyCertified"] is False


def test_read_only_audit_preserves_every_frozen_article_and_original_summary():
    article = record()
    source = {
        "schemaVersion": "frozen_thirty_source_grounded_semantic_v1",
        "originalHoldoutSha256": "unchanged-study",
        "articleCount": 1,
        "articles": [article],
    }
    old = copy.deepcopy(source)
    report = audit_report(source)
    assert source == old
    assert report["reviewedArticles"] == 1
    assert report["modelWasRerun"] is False
    assert report["priceDataAccessed"] is False
    assert report["independentlyGoldLabeled"] is False
    assert report["findingCodeCounts"]["SEC_EARNINGS_ITEM_CATEGORY_CONFLICT"] == 1
    with pytest.raises(ValueError, match="SEMANTIC_COHORT_COUNT_MISMATCH"):
        audit_report({**source, "articleCount": 30})
