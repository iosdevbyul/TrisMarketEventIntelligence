"""Regression checks for country-specific price scoring and evidence abstention."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tris_market_event_intelligence import event_evidence_policy as evidence
from tris_market_event_intelligence import sequential_market_learning_cli as seq


def article(summary: str, *, source: str = "bok_corrections") -> dict:
    return {
        "articleId": 101,
        "source": source,
        "title": "한국은행의 금융안정 조치에 대한 보도해명자료",
        "summary": summary,
        "publishedAtReported": "2019-06-01T00:00:00+00:00",
        "firstObservedAt": "2019-06-01T00:00:00+00:00",
    }


def test_attachment_pointer_is_not_analyzable_body():
    row = article("<p>&nbsp;</p><p>자세한 내용은 첨부파일을 참고하시기 바랍니다.</p>")
    quality = evidence.content_quality(row)
    assert quality["status"] == "INSUFFICIENT"
    assert quality["reasonCode"] == "ATTACHMENT_CONTENT_MISSING"
    assert quality["requiresSourceEnrichment"]


def test_title_only_and_html_markup_cannot_pass_content_check():
    row = article("<p>한국은행의 금융안정 조치에 대한 보도해명자료</p>")
    assert evidence.content_quality(row)["reasonCode"] == "TITLE_ONLY"
    assert evidence.plain_text("&lt;p&gt;실질적인 본문&lt;/p&gt;") == "실질적인 본문"


def test_korean_market_and_sector_are_not_us_etfs():
    raw = seq.canonical_prediction({
        "impacts": [
            {"targetType": "MARKET", "target": "BROAD_MARKET",
             "direction": "NEGATIVE", "reason": "Local policy shock"},
            {"targetType": "SECTOR", "target": "FINANCIALS",
             "direction": "NEGATIVE", "reason": "Local bank exposure"},
            {"targetType": "SECTOR", "target": "TECH",
             "direction": "NEGATIVE", "reason": "Unsupported Korean technology proxy"},
        ],
    })
    prediction = evidence.localize_prediction(
        raw, article("한국은행이 금융기관에 대한 정책 조치를 발표하였고 실제 세부 정책과 적용 대상이 명시되어 있습니다."),
    )
    assert prediction["eventMarket"] == "KR"
    assert [p["priceProxy"] for p in prediction["impacts"]] == [
        "069500.KS", "091170.KS",
    ]
    assert prediction["targetMappingRejections"][0]["reasonCode"] == (
        "NO_VERIFIED_KR_SECTOR_PROXY"
    )


def test_nvidia_us_article_keeps_original_us_proxy():
    row = article(
        "NVIDIA announced a revised earnings forecast affecting the semiconductor "
        "industry and its supply chain.", source="industry_rss",
    )
    row["title"] = "NVIDIA financial forecast"
    prediction = seq.canonical_prediction({
        "impacts": [
            {"targetType": "MARKET", "target": "BROAD_MARKET",
             "direction": "NEGATIVE", "reason": "US stocks"},
        ],
    })
    assert evidence.localize_prediction(prediction, row)["impacts"][0]["priceProxy"] == "SPY"


def test_korean_sectors_use_korean_benchmark(monkeypatch):
    called = []

    def price(client, symbol, observed, horizon, interval):
        called.append(symbol)
        return {
            "return": {"091170.KS": -0.015, "069500.KS": -0.007}[symbol],
            "baselineAt": "2019-05-31T06:30:00+00:00",
            "outcomeAt": "2019-06-03T06:30:00+00:00",
        }

    monkeypatch.setattr(seq, "price_return", price)
    prediction = {"impacts": [
        {"targetType": "SECTOR", "target": "FINANCIALS",
         "priceProxy": "091170.KS", "direction": "NEGATIVE", "reason": "Korean rates"}
    ]}
    result = seq.evaluate(
        prediction, datetime(2019, 6, 1, tzinfo=UTC),
        timedelta(hours=24), "1d", object(),
    )
    assert set(called) == {"069500.KS", "091170.KS"}
    assert result["impacts"][0]["excessReturn"] == -0.008


def test_market_index_moves_do_not_create_causal_learning_rules():
    row = article(
        "한국은행이 경제상황과 시장에 관한 공식 해명 발표를 하였으나, 구체적인 조치와 영향은 확인되지 않았습니다.",
    )
    class RejectCritic:
        def generate_json(self, **kwargs):
            raise AssertionError("An unrelated market movement must not teach causality")

    evaluation = {"impacts": [{
        "targetType": "MARKET", "status": "WRONG",
        "excessReturn": 0, "absoluteReturn": -0.02,
    }]}
    feedback = seq.next_lesson(RejectCritic(), row, {}, evaluation)
    assert feedback["lesson"] == ""
    assert feedback["diagnosis"].startswith("NO_CAUSAL_EVIDENCE")


def test_eligibility_filter_skips_missing_official_attachment(monkeypatch, tmp_path):
    now = datetime(2019, 6, 1, tzinfo=UTC)
    missing = article("자세한 내용은 첨부파일을 참고하시기 바랍니다.")
    valid = article("금융시장 안정화 조치가 2019년 6월 1일 공표되었으며 금융기관 적용 범위와 시행 내용이 함께 설명됩니다.")
    valid["articleId"] = 102
    valid["publishedAtReported"] = (now + timedelta(days=1)).isoformat()
    valid["firstObservedAt"] = valid["publishedAtReported"]
    monkeypatch.setattr(seq, "historical_news_cohort", lambda **kw: {
        "items": [missing, valid, {**valid, "articleId": 103}],
    })
    items, heldout = seq.load_articles(
        tmp_path / "news.sqlite3", 5, now + timedelta(days=4),
    )
    # The 80% chronology split retains only the first eligible event.
    assert [row["articleId"] for row in items] == [102]
    assert heldout == 1
