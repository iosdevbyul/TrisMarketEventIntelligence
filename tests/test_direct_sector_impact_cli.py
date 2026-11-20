"""Core product behavior: news article directly produces sector and direction."""
import json

import pytest

from tris_market_event_intelligence.direct_sector_impact_cli import (
    analyze,
    run,
    validate,
)

NEWS = (
    "미국 정부는 첨단 인공지능 반도체의 중국 수출 규제를 강화한다고 발표했다. "
    "새로운 수출 허가 기준에 따라 일부 기업의 중국향 AI 반도체 판매가 제한될 수 있으며, "
    "당국은 다음 달부터 강화된 허가 절차를 시행할 계획이라고 밝혔다."
)
QUOTE = "일부 기업의 중국향 AI 반도체 판매가 제한될 수 있으며"


class FakeModel:
    model_id = "fake-sector-model"

    def __init__(self, prediction):
        self.prediction = prediction
        self.calls = 0

    def generate_json(self, *, system, user, schema):
        self.calls += 1
        assert "sourceBody" in json.loads(user)
        assert "SECTORS" in system
        assert schema["properties"]["sectors"]["maxItems"] == 6
        return json.dumps(self.prediction, ensure_ascii=False)


def prediction(direction="NEGATIVE"):
    return {
        "event": "미국의 AI 반도체 대중국 수출 규제 강화",
        "summary": "반도체 기업의 일부 시장 매출 기회를 제한할 수 있다.",
        "overallMarketDirection": "UNKNOWN",
        "sectors": [{
            "sector": "SEMICONDUCTORS",
            "direction": direction,
            "sourceQuote": QUOTE,
            "reason": "규제 대상 반도체 제조사의 중국향 판매 기회가 줄어들 수 있다.",
            "uncertainty": "세부 규제 대상 제품과 기존 매출 노출도는 확인되지 않았다.",
        }],
    }


def test_one_article_gives_clear_negative_sector_hypothesis():
    model = FakeModel(prediction())
    result = analyze({"articleId": 1, "title": "AI 수출 제한", "summary": NEWS}, model)
    assert model.calls == 1
    assert result["status"] == "ANALYZED"
    assert result["sectors"][0]["sector"] == "SEMICONDUCTORS"
    assert result["sectors"][0]["direction"] == "NEGATIVE"
    assert result["overallMarketDirection"] == "UNKNOWN"
    assert NEWS[result["sectors"][0]["sourceStart"]:result["sectors"][0]["sourceEnd"]] == QUOTE
    assert not result["priceDirectionVerified"]


def test_fake_source_quote_and_duplicate_sector_rejected():
    fake = prediction()
    fake["sectors"][0]["sourceQuote"] = "원문에 없는 문장으로 섹터 방향을 주장한다."
    with pytest.raises(ValueError, match="SECTOR_EVIDENCE_QUOTE_NOT_IN_ARTICLE"):
        validate(fake, NEWS)
    duplicate = prediction()
    duplicate["sectors"].append(dict(duplicate["sectors"][0]))
    with pytest.raises(ValueError, match="SECTOR_MAPPING_OR_DIRECTION_INVALID"):
        validate(duplicate, NEWS)


def test_direction_change_is_preserved_and_not_forced_negative():
    positive = prediction("POSITIVE")
    model = FakeModel(positive)
    result = analyze({"articleId": 2, "summary": NEWS}, model)
    assert result["status"] == "ANALYZED"
    assert result["sectors"][0]["direction"] == "POSITIVE"


def test_no_sector_requires_unknown_market_direction():
    empty = {
        "event": "검토할 사건", "summary": "영향이 확인되지 않는다.",
        "sectors": [], "overallMarketDirection": "UNKNOWN",
    }
    assert validate(empty, NEWS)["sectors"] == []
    empty["overallMarketDirection"] = "POSITIVE"
    with pytest.raises(ValueError, match="BROAD_MARKET_WITHOUT_ANY_EVIDENCE"):
        validate(empty, NEWS)


def test_inadequate_article_does_not_call_model_or_guess_direction():
    model = FakeModel(prediction())
    result = analyze({"articleId": 3, "summary": "짧은 소식"}, model)
    assert result["status"] == "INSUFFICIENT_ARTICLE_TEXT"
    assert result["sectors"] == []
    assert model.calls == 0


def test_batch_shows_each_article_and_keeps_predictions_separate():
    model = FakeModel(prediction())
    result = run([
        {"articleId": 1, "title": "규제 발표", "summary": NEWS},
        {"articleId": 2, "title": "확인 불가", "summary": ""},
    ], model, frozen_hash="sealed")
    assert result["articleCount"] == 2
    assert result["statusCounts"] == {
        "ANALYZED": 1, "INSUFFICIENT_ARTICLE_TEXT": 1,
    }
    assert result["sectorDirectionCounts"] == [
        {"sector": "SEMICONDUCTORS", "direction": "NEGATIVE", "count": 1},
    ]
    assert result["originalHoldoutSha256"] == "sealed"
    assert result["originalPredictionsModified"] is False
    assert result["priceDataAccessed"] is False


def test_alphabet_filing_cannot_create_energy_or_real_estate_impact():
    body = (
        "On April 29, 2026, Alphabet Inc. is issuing a press release and "
        "holding a conference call regarding its financial results for "
        "the quarter ended March 31, 2026."
    )
    invalid = {
        "event": "Alphabet results press release notice",
        "summary": "Results are being released without figures in this text.",
        "overallMarketDirection": "UNKNOWN",
        "sectors": [
            {"sector": sector, "direction": "UNKNOWN", "sourceQuote": body[:110],
             "reason": "General economic influence might extend to this industry.",
             "uncertainty": "There is no actual earnings figure in the source."}
            for sector in ("ENERGY", "CONSUMER", "REAL_ESTATE")
        ],
    }
    article = {
        "articleId": 20884, "source": "sec_8k_1652044",
        "title": "Alphabet Inc. SEC 8-K Item 2.02 EARNINGS_8K",
        "summary": body,
    }
    result = analyze(article, FakeModel(invalid))
    assert result["status"] == "ANALYZED"
    assert result["sectors"][0]["sector"] == "TECH"
    assert result["sectors"][0]["direction"] == "UNKNOWN"
    assert {x["sector"] for x in result["excludedSectorSuggestions"]} == {
        "ENERGY", "CONSUMER", "REAL_ESTATE",
    }
    assert result["overallMarketDirection"] == "UNKNOWN"


def test_amd_filing_primary_sector_is_semiconductors():
    body = (
        "Advanced Micro Devices Inc. is furnishing the accompanying "
        "press release announcing financial results for the quarter ended "
        "March 31, 2026 and providing information about the conference call."
    )
    article = {
        "articleId": 20873, "source": "sec_8k_2488",
        "title": "AMD SEC 8-K Item 2.02 EARNINGS_8K",
        "summary": body,
    }
    result = analyze(article, FakeModel({
        "event": "AMD earnings filing",
        "summary": "The filing announces its earnings.",
        "overallMarketDirection": "UNKNOWN",
        "sectors": [],
    }))
    assert result["status"] == "ANALYZED"
    assert result["sectors"][0]["sector"] == "SEMICONDUCTORS"
    assert result["sectors"][0]["direction"] == "UNKNOWN"


def test_m2_liquidity_alone_does_not_create_consumer_upside():
    body = (
        "M2(광의통화, 계절조정계열, 평잔)는 전월대비 0.4% 증가하며 "
        "원계열은 전년동월대비 5.6% 증가한 것으로 발표되었다."
    )
    original = {
        "event": "Broad money supply increase",
        "summary": "Broad money increased.",
        "overallMarketDirection": "UNKNOWN",
        "sectors": [{
            "sector": "CONSUMER", "direction": "POSITIVE",
            "sourceQuote": "M2(광의통화, 계절조정계열, 평잔)는 전월대비 0.4% 증가",
            "reason": "Money supply growth could theoretically support consumption.",
            "uncertainty": "There is no evidence of consumer spending growth.",
        }],
    }
    result = analyze({
        "articleId": 210, "source": "bok_monetary",
        "title": "2026년 3월 통화 및 유동성", "summary": body,
    }, FakeModel(original))
    assert result["status"] == "ANALYZED"
    assert result["sectors"] == []
    assert result["excludedSectorSuggestions"][0]["reasonCode"] == (
        "MONEY_SUPPLY_NOT_PROOF_OF_CONSUMER_DEMAND"
    )


def test_invalid_model_quote_is_retried_once_without_faking_grounding():
    class RecoverModel(FakeModel):
        def generate_json(self, *, system, user, schema):
            self.calls += 1
            parsed = json.loads(user)
            answer = prediction()
            if self.calls == 1:
                answer["sectors"][0]["sourceQuote"] = "This is not the source quote"
            else:
                assert "correction" in parsed
            return json.dumps(answer, ensure_ascii=False)

    model = RecoverModel(prediction())
    result = analyze({"articleId": 1, "summary": NEWS}, model)
    assert model.calls == 2
    assert result["status"] == "ANALYZED"
    assert result["modelAttempts"] == 2


def test_table_numeric_sequence_does_not_count_as_sector_evidence():
    fragment = "91.3 91.0 89.6 92.5 92.3 93.2"
    sample = dict(prediction())
    sample["sectors"] = [{**prediction()["sectors"][0], "sourceQuote": fragment}]
    with pytest.raises(ValueError, match="SECTOR_EVIDENCE_TABLE_ONLY"):
        validate(sample, NEWS + " " + fragment)


def test_real_article_not_rejected_as_too_long_when_model_sees_compact_input():
    from tris_market_event_intelligence.direct_sector_excerpt_selection import compact_source
    table = " ".join(["91.3 91.0 89.6 92.5 92.3"] * 600)
    body = NEWS + " " + table + " □ 반도체 신규투자가 전분기 대비 15% 증가했다."
    selected = compact_source(body, "반도체 투자", max_chars=2500)
    assert len(selected) <= 2500
    assert "반도체 신규투자가 전분기 대비 15% 증가했다." in selected
    assert selected.startswith(NEWS)


def test_directional_article_count_not_equal_analyzed_count():
    none = {"event": "알림", "summary": "추가 정보 부족", "sectors": [],
            "overallMarketDirection": "UNKNOWN"}
    model = FakeModel(none)
    result = run([{"articleId": 1, "summary": NEWS}], model)
    assert result["statusCounts"] == {"ANALYZED": 1}
    assert result["directionalArticleCount"] == 0
