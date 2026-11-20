"""Direct analysis contract: grounded quotes and unmodified AI decisions."""
import json

from tris_market_event_intelligence.direct_sector_impact_cli import (
    SCHEMA,
    SYSTEM,
    analyze,
    check_source_quotes,
    render_review,
    run,
    validate,
)


class FakeModel:
    model_id = "fake-test-model"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate_json(self, *, system, user, schema):
        self.calls.append({"system": system, "user": json.loads(user), "schema": schema})
        return json.dumps(self.reply, ensure_ascii=False)


def item(sector="의료로봇", direction="POSITIVE", quote="메디인테크가 투자를 유치했다.",
         reason="제품 사업화 자금을 확보했다."):
    return {"sector": sector, "direction": direction, "reason": reason,
            "sourceQuote": quote}


def test_full_article_and_freeform_theme_schema():
    body = ("메디인테크가 투자를 유치했다. 의료기기 사업화가 시작됐다. " * 110).strip()
    answer = {"sectors": [
        item(),
        item("AI 의료기기", "NEUTRAL", "의료기기 사업화가 시작됐다.", "산업 전체 영향 미확인"),
        item("일본 내시경 업체", "NEGATIVE", "메디인테크가 투자를 유치했다.",
             "장래 경쟁 강도 증가 가능"),
    ]}
    model = FakeModel(answer)
    result = analyze({"articleId": 1, "title": "의료로봇 투자", "body": body,
                      "summary": "요약만 사용하면 안 된다",
                      "publishedAtReported": "2026-11-21T07:00:00+09:00"}, model)
    assert result["status"] == "ANALYZED"
    assert result["sectors"] == answer["sectors"]
    assert result["modelInputChars"] == len(body)
    assert result["evidenceQuoteFoundCount"] == 3
    assert result["evidenceQuoteFlaggedCount"] == 0
    assert model.calls[0]["user"]["article"] == body
    assert model.calls[0]["user"]["publishedAtReported"] == "2026-11-21T07:00:00+09:00"
    assert SCHEMA["properties"]["sectors"]["items"]["properties"]["sector"] == {"type": "string"}
    assert "maxItems" not in SCHEMA["properties"]["sectors"]
    assert len(model.calls) == 1


def test_invalid_model_shape_is_rejected_not_reinterpreted():
    invalid = {"sectors": [{"sector": "의료로봇", "direction": "UNKNOWN",
                            "reason": "근거 부족", "sourceQuote": "기사 발췌"}]}
    result = analyze({"body": "기사 발췌입니다."}, FakeModel(invalid))
    assert result["status"] == "MODEL_RESULT_REJECTED"
    assert result["sectors"] == []


def test_source_quote_checks_preserve_predictions_when_quote_is_wrong():
    body = "회사는 의료기기 생산 투자를 확대했다."
    answer = {"sectors": [
        item("의료기기", "POSITIVE", "회사는 의료기기 생산 투자를 확대했다."),
        item("반도체", "NEGATIVE", "기사에 없는 주가 폭락 사실입니다."),
        item("자동화", "NEUTRAL", ""),
        item("정밀 기계", "POSITIVE", "투자"),
    ]}
    result = analyze({"articleId": 7, "body": body}, FakeModel(answer))
    assert result["status"] == "ANALYZED"
    assert result["sectors"] == answer["sectors"]
    assert [c["status"] for c in result["evidenceChecks"]] == [
        "QUOTE_FOUND", "QUOTE_NOT_IN_SOURCE", "MISSING_QUOTE", "QUOTE_TOO_SHORT",
    ]
    assert result["evidenceQuoteFoundCount"] == 1
    assert result["evidenceQuoteFlaggedCount"] == 3
    assert result["sectors"][1]["direction"] == "NEGATIVE"


def test_quote_whitespace_normalization_only():
    source = "제조업 생산이\n 전월 대비 3.2% 증가했다."
    checks = check_source_quotes(
        [item(quote="제조업 생산이 전월 대비 3.2% 증가했다.")], source
    )
    assert checks[0]["status"] == "QUOTE_FOUND"
    assert check_source_quotes([item(quote="제조업 생산이 32% 증가했다.")], source)[0]["status"] == (
        "QUOTE_NOT_IN_SOURCE"
    )


def test_analysis_never_changes_model_reason_direction_or_theme():
    answer = {"sectors": [item("아주 자유로운 투자 테마", "NEGATIVE",
                               "회사 신제품 출시가 발표됐다.", "원가 부담을 추정한다.")]}
    assert validate(answer, "임의의 본문")["sectors"] == answer["sectors"]
    assert analyze({"body": "관련 기사 본문."}, FakeModel(answer))["sectors"] == answer["sectors"]


def test_missing_body_uses_summary_and_does_not_call_when_empty():
    model = FakeModel({"sectors": []})
    assert analyze({"summary": "기사는 짧다."}, model)["status"] == "ANALYZED"
    assert model.calls[0]["user"]["article"] == "기사는 짧다."
    assert analyze({"body": "", "summary": ""}, model)["status"] == "INSUFFICIENT_ARTICLE_TEXT"
    assert len(model.calls) == 1


def test_multi_article_review_preserves_positive_negative_neutral():
    source = "국제 유가가 상승했고 항공사의 연료비 부담이 커졌다."
    class Responses(FakeModel):
        def __init__(self):
            super().__init__({})
            self.replies = [
                {"sectors": [item("항공운송", "NEGATIVE", source, "연료비 부담")]},
                {"sectors": [item("정유", "POSITIVE", source, "수익성 개선 가능성")]},
                {"sectors": [item("소매유통", "NEUTRAL", source, "효과 불확실")]},
            ]

        def generate_json(self, *, system, user, schema):
            self.calls.append(json.loads(user))
            return json.dumps(self.replies.pop(0), ensure_ascii=False)

    model = Responses()
    report = run([{"articleId": i, "title": f"기사 {i}", "body": source}
                  for i in range(1, 4)], model)
    markdown = render_review(report)
    assert report["schemaVersion"] == "direct_news_sector_impact_v3"
    assert report["articleCount"] == 3
    assert report["directionalArticleCount"] == 2
    assert report["evidenceQuoteFoundCount"] == 3
    assert "항공운송** — NEGATIVE" in markdown
    assert "정유** — POSITIVE" in markdown
    assert "소매유통** — NEUTRAL" in markdown
    assert "원문 근거:" in markdown and "근거 대조: QUOTE_FOUND" in markdown
    assert len(model.calls) == 3
    assert report["originalPredictionsModified"] is False


def test_review_identifies_unverified_quote_without_scoring_accuracy():
    model = FakeModel({"sectors": [item(quote="실제로 등장하지 않는 문장입니다.")]})
    report = run([{"articleId": 1, "title": "테스트", "body": "공식 자료가 공개됐다."}],
                 model)
    review = render_review(report)
    assert "QUOTE_NOT_IN_SOURCE" in review
    assert "정확도·실제 주가 반응은 검증하지 않았습니다" in review
    assert report["articles"][0]["sectors"] == model.reply["sectors"]


def test_prompt_requests_verbatim_quote_and_no_future_facts():
    assert "직접 복사한 짧은 문장" in SYSTEM
    assert "발표 시점까지 확인된 사실" in SYSTEM
    assert "사건·활동·결과를 테마 이름으로 쓰지 마세요" in SYSTEM
    assert "미리 정해진 섹터 목록과 개수 제한은 없습니다" in SYSTEM
    assert "이름과 이유는 한국어로 쓰고" in SYSTEM
