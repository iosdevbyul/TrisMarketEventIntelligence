"""Core contract: full article to model, free-form themes, unchanged judgments."""
import json

from tris_market_event_intelligence.direct_sector_impact_cli import analyze, run, validate


class FakeModel:
    model_id = "test-model"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_json(self, *, system, user, schema):
        self.calls.append({"system": system, "user": json.loads(user), "schema": schema})
        return json.dumps(self.response, ensure_ascii=False)


def test_freeform_themes_are_not_forced_into_fixed_sector_taxonomy():
    raw = {"sectors": [
        {"sector": "의료로봇", "direction": "POSITIVE", "reason": "사업화 자금 확보"},
        {"sector": "AI 의료기기", "direction": "POSITIVE", "reason": "시장 진입"},
        {"sector": "일본 내시경 제조", "direction": "NEGATIVE", "reason": "경쟁 심화 가능"},
        {"sector": "피지컬 AI", "direction": "NEUTRAL", "reason": "시장 전체 수요는 미확인"},
    ]}
    model = FakeModel(raw)
    body = ("메디인테크가 338억원 시리즈C 투자를 유치했다. "
            "국내 판매망과 해외 초기 판매를 확보했다. ") * 80
    result = analyze({"articleId": 1, "title": "의료로봇 투자", "body": body,
                      "summary": "요약으로 대체해서는 안 된다"}, model)
    assert result["status"] == "ANALYZED"
    assert result["sectors"] == raw["sectors"]
    assert len(result["sectors"]) == 4
    assert model.calls[0]["user"]["article"] == body
    assert model.calls[0]["schema"]["properties"]["sectors"]["items"]["properties"]["sector"] == {"type": "string"}
    assert result["modelInputChars"] == len(body)
    assert result["modelAttempts"] == 1


def test_model_direction_and_reason_never_rewritten():
    raw = {"sectors": [{"sector": "전기차", "direction": "NEGATIVE", "reason": "원가 부담"}]}
    assert validate(raw, "any text")["sectors"] == raw["sectors"]
    assert analyze({"body": "news"}, FakeModel(raw))["sectors"] == raw["sectors"]


def test_invalid_shape_rejected_without_reinterpretation():
    raw = {"sectors": [{"sector": "전기차", "direction": "UNKNOWN", "reason": "불명"}]}
    result = analyze({"body": "news"}, FakeModel(raw))
    assert result["status"] == "MODEL_RESULT_REJECTED"
    assert result["sectors"] == []


def test_no_body_uses_summary_and_empty_source_skips_model():
    model = FakeModel({"sectors": []})
    assert analyze({"summary": "실제 기사 요약"}, model)["status"] == "ANALYZED"
    assert model.calls[0]["user"]["article"] == "실제 기사 요약"
    assert analyze({"body": "", "summary": ""}, model)["status"] == "INSUFFICIENT_ARTICLE_TEXT"
    assert len(model.calls) == 1


def test_report_preserves_per_article_results():
    model = FakeModel({"sectors": [
        {"sector": "자유로운 세부 테마", "direction": "POSITIVE", "reason": "경제적 수혜"}
    ]})
    report = run([{"articleId": 1, "body": "내용"}, {"articleId": 2, "body": ""}], model)
    assert report["schemaVersion"] == "direct_news_sector_impact_v2"
    assert report["articleCount"] == 2
    assert report["directionalArticleCount"] == 1
    assert report["sectorDirectionCounts"][0]["sector"] == "자유로운 세부 테마"
    assert report["originalPredictionsModified"] is False
