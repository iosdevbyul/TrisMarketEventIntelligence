"""Regression checks for country-specific price scoring and evidence abstention."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

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


def test_enriched_cache_requires_matching_text_and_original_capture(
    monkeypatch, tmp_path,
):
    import hashlib
    import json

    valid = article(
        "자세한 내용은 첨부파일을 참고하시기 바랍니다.",
    )
    valid["captureSha256"] = "original-archive-capture"
    valid["sourceUrl"] = "https://www.bok.or.kr/portal/example"
    text_body = (
        "한국은행은 금융시장 불안을 완화하기 위한 구체적인 조치와 "
        "금융기관 적용 방안을 포함하는 공식 자료를 발표했습니다."
    )
    cache = {
        "schemaVersion": "retrospective_official_attachment_cache_v1",
        "sourceDb": str((tmp_path / "news.sqlite3").resolve()),
        "originalArchiveModified": False,
        "articles": {
            "101": {
                "status": "AVAILABLE",
                "sourceUrl": valid["sourceUrl"],
                "captureSha256": valid["captureSha256"],
                "publishedAtReported": valid["publishedAtReported"],
                "contentText": text_body,
                "contentTextSha256": hashlib.sha256(text_body.encode()).hexdigest(),
                "attachmentUrl": "https://www.bok.or.kr/portal/example.pdf",
                "contentSha256": "pdf-checksum",
                "acquiredAt": "2026-11-19T00:00:00+00:00",
                "pointInTimeCertified": False,
            }
        },
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(cache))
    monkeypatch.setattr(seq, "historical_news_cohort", lambda **kw: {
        "items": [valid, {**valid, "articleId": 102, "summary": "A fully described release, covering new detailed policy measures and institutional effects."}],
    })
    selected, _ = seq.load_articles(
        tmp_path / "news.sqlite3", 10,
        datetime(2019, 6, 9, tzinfo=UTC), content_cache=path,
    )
    assert selected[0]["articleId"] == 101
    assert selected[0]["contentEvidence"]["pointInTimeCertified"] is False
    cache["articles"]["101"]["contentText"] = "tampered"
    path.write_text(json.dumps(cache))
    selected, _ = seq.load_articles(
        tmp_path / "news.sqlite3", 10,
        datetime(2019, 6, 9, tzinfo=UTC), content_cache=path,
    )
    assert selected == []


def test_official_attachment_source_rejects_external_hosts():
    from tris_market_event_intelligence import source_enrichment_cli as source

    assert source.authorized("https://www.bok.or.kr/portal/a.pdf")
    assert not source.authorized("https://evil.example/files/a.pdf")
    assert not source.authorized("http://www.bok.or.kr/portal/a.pdf")
    assert not source.authorized("https://www.bok.or.kr.evil.example/a.pdf")


def test_missing_official_pdf_is_explicit_abstention():
    import httpx

    from tris_market_event_intelligence import source_enrichment_cli as source

    class Client:
        def get(self, url, **kwargs):
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                headers={"content-type": "text/html"},
                text='<html><body><a href="unsupported.hwp">HWP</a></body></html>',
            )

    row = {
        "sourceUrl": "https://www.bok.or.kr/portal/test",
        "articleId": 7,
    }
    output = source.source_document(Client(), row)
    assert output == {
        "status": "UNAVAILABLE",
        "reasonCode": "PDF_ATTACHMENT_NOT_LINKED_OR_UNSUPPORTED",
    }


def test_korean_headline_cannot_price_unmentioned_american_stock():
    row = article(
        "한국은행은 금융안정을 위한 조치를 발표하였으며 구체적인 적용 대상과 정책 기준을 함께 공지했습니다."
    )
    prediction = seq.canonical_prediction({
        "impacts": [
            {"targetType": "STOCK", "target": "NVDA",
             "direction": "POSITIVE", "reason": "Unsupported global spillover"},
            {"targetType": "MARKET", "target": "BROAD_MARKET",
             "direction": "NEUTRAL", "reason": "Domestic equity benchmark"},
        ],
    })
    scoped = evidence.localize_prediction(prediction, row)
    assert [item["priceProxy"] for item in scoped["impacts"]] == ["069500.KS"]
    assert scoped["targetMappingRejections"] == [{
        "targetType": "STOCK", "target": "NVDA",
        "reasonCode": "NO_VERIFIED_ARTICLE_ISSUER_LINK",
    }]


def test_bok_cdn_pdf_href_and_safe_redirect_are_supported(monkeypatch):
    import httpx

    from tris_market_event_intelligence import source_enrichment_cli as source

    page_url = "https://www.bok.or.kr/portal/bbs/P0000560/view.do?nttId=123"
    first = "https://www.bok.or.kr/common/download.do?file=123"
    cdn = "https://file-cdn.bok.or.kr/portal/123/2/FILE_123.pdf?token=signed"
    assert source.authorized(cdn)
    responses = {
        page_url: httpx.Response(
            200, request=httpx.Request("GET", page_url),
            headers={"content-type": "text/html"},
            text=f'<a href="{first}">보도자료.pdf 다운로드</a>',
        ),
        first: httpx.Response(
            302, request=httpx.Request("GET", first),
            headers={"location": cdn},
        ),
        cdn: httpx.Response(
            200, request=httpx.Request("GET", cdn),
            headers={"content-type": "application/pdf"}, content=b"%PDF-test",
        ),
    }

    class Client:
        def get(self, url, **kwargs):
            return responses[url]

    monkeypatch.setattr(
        source, "extract_pdf", lambda _: "한국은행은 금융안정 조치와 시장 대응을 구체적으로 발표했습니다. "
        "금융시장 대응 주체와 세부 조건에 관한 내용을 포함합니다.",
    )
    row = article("첨부파일을 참조하시기 바랍니다.")
    row.update(sourceUrl=page_url, captureSha256="sha")
    result = source.source_document(Client(), row)
    assert result["status"] == "AVAILABLE"
    assert result["attachmentUrl"] == first
    assert result["contentTextSha256"]


def test_bok_external_pdf_redirect_is_rejected():
    import httpx

    from tris_market_event_intelligence import source_enrichment_cli as source

    url = "https://www.bok.or.kr/common/download.do"
    class Client:
        def get(self, url, **kwargs):
            return httpx.Response(
                302, request=httpx.Request("GET", url),
                headers={"location": "https://evil.example/private.pdf"},
            )

    with pytest.raises(ValueError, match="OUTSIDE_OFFICIAL_HOSTS"):
        source.fetch(Client(), url)


def test_duplicate_gemma_targets_preserve_single_agreeing_direction():
    class Model:
        def generate_json(self, **kwargs):
            return json.dumps({
                "impacts": [
                    {"targetType": "MARKET", "target": "SPY",
                     "direction": "NEGATIVE", "reason": "Domestic risk"},
                    {"targetType": "MARKET", "target": "BROAD_MARKET",
                     "direction": "NEGATIVE", "reason": "Same market exposure"},
                ]
            })

    result = seq.predict_with_repair(Model(), {})
    assert len(result["impacts"]) == 1
    assert result["impacts"][0]["direction"] == "NEGATIVE"
    assert result["targetMappingRejections"][0]["reasonCode"] == (
        "DUPLICATE_TARGET_COLLAPSED"
    )


def test_conflicting_gemma_duplicate_abstains_without_invented_verdict():
    class Model:
        def generate_json(self, **kwargs):
            return json.dumps({
                "impacts": [
                    {"targetType": "STOCK", "target": "NVDA",
                     "direction": "POSITIVE", "reason": "Opportunity"},
                    {"targetType": "STOCK", "target": "NVDA",
                     "direction": "NEGATIVE", "reason": "Risk"},
                    {"targetType": "SECTOR", "target": "FINANCIALS",
                     "direction": "NEUTRAL", "reason": "Uncertain market"},
                ]
            })

    result = seq.predict_with_repair(Model(), {})
    assert [r["target"] for r in result["impacts"]] == ["FINANCIALS"]
    assert any(x["reasonCode"] == "CONFLICTING_DUPLICATE_TARGET"
               for x in result["targetMappingRejections"])
