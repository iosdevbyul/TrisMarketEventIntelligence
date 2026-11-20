"""Crosscheck the 30-case frozen model without labeling heuristics as accuracy."""
import pytest

from tris_market_event_intelligence.bok_model_numeric_crosscheck_cli import (
    audit,
    compare_claim,
    numeric_mentions,
)


def fixtures():
    semantic = {
        "schemaVersion": "frozen_thirty_source_grounded_semantic_v1",
        "originalHoldoutSha256": "frozen", "articleCount": 1,
        "articles": [{
            "articleId": 210, "status": "COMPLETE",
            "semanticAnalysis": {"modelAnalysis": {
                "overallEconomicDirection": "UNKNOWN",
                "sourceStatements": [{
                    "quote": "국내총생산은 전기 대비 2.5% 증가했다.",
                    "interpretation": "국내총생산은 전기 대비 2.5% 증가했다.",
                }],
                "economicMechanisms": [{
                    "groundingQuote": "전기 대비 2.5% 증가했다.",
                    "reasoning": "전년 대비 3.5% 상승 효과가 있을 가능성",
                }],
            }},
        }],
    }
    inventory = {
        "schemaVersion": "bok_source_numeric_inventory_v1",
        "originalHoldoutSha256": "frozen", "articleCount": 1,
        "articles": [{
            "articleId": 210,
            "attachmentEvidence": [{
                "numericCandidates": [
                    {"numberText": "2.5", "unit": "%"},
                    {"numberText": "2026", "unit": "NOT_EXPLICIT"},
                ],
            }],
        }],
    }
    return semantic, inventory


def test_exact_quote_and_comparison_mismatch_are_distinct():
    quoted = compare_claim("국내총생산은 전기 대비 2.5% 증가", "전기 대비 2.5% 증가", set())
    assert quoted["reviewStatus"] == "SOURCE_NUMBER_PRESENCE_ONLY"
    assert quoted["numericClaims"][0]["evidenceStatus"] == "MATCHED_IN_MODEL_QUOTE"
    mismatch = compare_claim("전년 대비 3.5% 상승", "전기 대비 2.5% 증가", {("2.5", "PERCENT")})
    assert mismatch["comparisonPeriodConflict"]
    assert mismatch["reviewStatus"] == "PERIOD_CONFLICT_REVIEW"
    assert mismatch["factualCorrectnessIndependentlyVerified"] is False


def test_date_and_unqualified_digits_are_not_numeric_claims():
    assert numeric_mentions("2026년 1월의 수치와 행 번호 100") == []
    rows = numeric_mentions("1.5%와 1.5억원")
    assert [r["unit"] for r in rows] == ["PERCENT", "KRW_HUNDRED_MILLION"]


def test_frozen_model_and_source_inventory_are_joined_by_article_id():
    semantic, inventory = fixtures()
    result = audit(semantic, inventory)
    assert result["articleCount"] == 1
    assert result["reviewStatusCounts"] == {
        "PERIOD_CONFLICT_REVIEW": 1, "SOURCE_NUMBER_PRESENCE_ONLY": 1,
    }
    assert result["articles"][0]["originalEconomicDirection"] == "UNKNOWN"
    assert result["originalPredictionsModified"] is False
    assert result["modelWasRerun"] is False
    assert result["independentlyGoldLabeled"] is False


def test_wrong_cohort_and_unknown_article_fail_closed():
    semantic, inventory = fixtures()
    inventory["originalHoldoutSha256"] = "other"
    with pytest.raises(ValueError, match="FROZEN_COHORT_HASH_MISMATCH"):
        audit(semantic, inventory)
    inventory["originalHoldoutSha256"] = "frozen"
    inventory["articles"][0]["articleId"] = 999
    with pytest.raises(ValueError, match="ARTICLE_ID_NOT_IN_FROZEN_COHORT"):
        audit(semantic, inventory)


def test_no_model_analysis_is_visible_and_not_silently_scored():
    semantic, inventory = fixtures()
    semantic["articles"][0]["semanticAnalysis"] = None
    result = audit(semantic, inventory)
    assert result["articles"][0]["numericClaimReviews"] == []
    assert result["articles"][0]["originalEconomicDirection"] is None
    assert result["reviewStatusCounts"] == {}


def test_percentage_point_and_basis_point_units_are_not_percentages():
    rows = numeric_mentions("기준금리 0.25%p 인상, 국채금리 25bps 상승, 물가 2.5% 증가")
    assert [row["unit"] for row in rows] == [
        "PERCENT_POINT", "BASIS_POINT", "PERCENT",
    ]
