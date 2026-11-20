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


def test_quote_number_omission_is_review_not_factual_failure():
    result = compare_claim(
        "전기 대비 증가했다는 설명입니다.",
        "전기 대비 2.5% 증가했습니다.",
        {("2.5", "PERCENT")},
    )
    assert result["reviewStatus"] == "NO_EXPLICIT_NUMERIC_CLAIM"
    assert result["sourceQuotedNumberOmissionReview"] is True
    assert result["sourceQuotedNumbersNotRepeated"][0]["verbatim"].strip() == "2.5%"
    assert result["factualCorrectnessIndependentlyVerified"] is False


def test_quote_number_presence_does_not_create_omission_review():
    result = compare_claim(
        "전기 대비 2.5% 증가했습니다.",
        "전기 대비 2.5% 증가했습니다.",
        set(),
    )
    assert result["sourceQuotedNumberOmissionReview"] is False


def test_report_exposes_omission_count_without_changing_original_direction():
    semantic, inventory = fixtures()
    semantic["articles"][0]["semanticAnalysis"]["modelAnalysis"]["sourceStatements"][0][
        "interpretation"
    ] = "전기 대비 증가했음을 설명합니다."
    result = audit(semantic, inventory)
    assert result["sourceQuoteNumberOmissionReviewCount"] >= 1
    assert result["articles"][0]["originalEconomicDirection"] == "UNKNOWN"
    assert result["originalPredictionsModified"] is False


def test_real_bok_money_is_not_misread_as_only_manwon():
    rows = numeric_mentions("차주당 신규취급액은 2억 2,939만원, 증가폭은 1,653만원")
    assert rows[0]["verbatim"] == "2억 2,939만원"
    assert rows[0]["value"] == "229390000"
    assert rows[0]["unit"] == "KRW_COMPOUND_WON"
    assert rows[1]["value"] == "1653"
    assert rows[1]["unit"] == "KRW_TEN_THOUSAND"
    assert len(rows) == 2


def test_real_bok_m2_two_comparison_bases_are_not_omission_errors():
    quote = (
        "M2는 전월대비 0.4% 증가 [원계열, 전년동월대비 +5.6%]"
    )
    result = compare_claim("전월대비 0.4% 증가", quote, set())
    assert result["sourceQuotedNumberOmissionReview"] is False
    assert result["sourceQuoteSelectiveSummary"] is True
    assert result["comparisonPeriodConflict"] is False
    assert result["comparisonPeriodInQuotedEvidence"] == ["MOM", "YOY"]


def test_negative_percentage_does_not_match_positive_rate():
    result = compare_claim("문화 서비스는 0.3% 증가", "문화 서비스는 -0.3% 감소", set())
    assert result["reviewStatus"] == "NUMERIC_SOURCE_REVIEW"
    assert result["numericClaims"][0]["evidenceStatus"] == "NO_EXACT_UNIT_VALUE_MATCH"


def test_mechanism_explanation_may_summarize_single_number_without_omission_error():
    result = compare_claim(
        "금융환경 변화 가능성은 불확실합니다.",
        "전월대비 0.4% 증가",
        set(), fact_interpretation=False,
    )
    assert result["sourceQuotedNumberOmissionReview"] is False
    assert result["sourceQuotedNumbersNotRepeated"][0]["value"] == "0.4"
