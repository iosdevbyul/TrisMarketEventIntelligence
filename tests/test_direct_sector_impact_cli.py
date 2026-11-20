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
