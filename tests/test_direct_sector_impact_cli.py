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


def test_hallucinated_no_article_message_is_not_scored_as_analyzed():
    answer = {
        "event": "No article or data was provided for analysis.",
        "summary": "There is no information.",
        "overallMarketDirection": "UNKNOWN",
        "sectors": [],
    }
    class AbsentSourceModel(FakeModel):
        def generate_json(self, *, system, user, schema):
            self.calls += 1
            return json.dumps(answer)

    model = AbsentSourceModel(answer)
    result = analyze({"articleId": 192, "summary": NEWS + NEWS}, model)
    assert result["status"] == "MODEL_RESULT_REJECTED"
    assert result["errorCode"] == "MODEL_FAILED_TO_READ_SOURCE"
    assert model.calls == 3


def test_official_bok_tables_do_not_replace_relevant_announcement_prose():
    from tris_market_event_intelligence.direct_sector_excerpt_selection import compact_source

    title = "2026년 5월 기업경기조사 결과 및 경제심리지수"
    prefix = "한국은행은 기업경기조사 결과를 발표했다."
    rows = " ".join(["91.3 91.0 89.6 92.5 92.3 93.2"] * 900)
    core = " □ 2026년 5월 제조업 기업심리지수는 전월 대비 하락했으며 비제조업도 감소했다."
    source = prefix + " OFFICIAL BOK ATTACHMENT TEXT: " + rows + core
    concise = compact_source(source, title, max_chars=2000)
    assert concise.startswith(prefix)
    assert "제조업 기업심리지수는 전월 대비 하락" in concise
    assert len(concise) <= 2000


def test_financing_a_specific_company_not_entire_semiconductor_sector():
    body = (
        "Advanced Micro Devices, Inc. entered into a Credit Agreement with "
        "the lenders named therein, JPMorgan Chase Bank N.A. as agent. "
        "The Credit Agreement establishes a revolving credit facility."
    )
    raw = {
        "event": "AMD financing",
        "summary": "Company financing agreement",
        "overallMarketDirection": "UNKNOWN",
        "sectors": [{
            "sector": "SEMICONDUCTORS", "direction": "POSITIVE",
            "sourceQuote": "Advanced Micro Devices, Inc. entered into a Credit Agreement",
            "reason": "AMD has access to liquidity, but sector-wide profitability is unknown.",
            "uncertainty": "No sector-wide earnings effect is documented.",
        }],
    }
    article = {
        "articleId": 20874, "title": "AMD SEC 8-K Item 1.01 MATERIAL_CONTRACT_8K",
        "source": "sec_8k_2488", "summary": body,
    }
    result = analyze(article, FakeModel(raw))
    assert result["status"] == "ANALYZED"
    assert result["sectors"] == []
    assert result["excludedSectorSuggestions"][0]["reasonCode"] == (
        "ISSUER_FINANCING_NOT_SECTOR_DIRECTION"
    )


def test_monetary_balance_increase_does_not_prove_financials_upside():
    body = (
        "2026년 3월 통화 및 유동성 발표에 따르면 Lf(금융기관유동성)는 "
        "전월 대비 0.5% 증가했고 M2(광의통화)도 증가했다."
    )
    raw = {
        "event": "M2 growth", "summary": "Liquidity increase",
        "overallMarketDirection": "UNKNOWN",
        "sectors": [{
            "sector": "FINANCIALS", "direction": "POSITIVE",
            "sourceQuote": "Lf(금융기관유동성)는 전월 대비 0.5% 증가",
            "reason": "More liquid balances are not evidence of bank profitability.",
            "uncertainty": "Funding costs and asset quality are not known.",
        }],
    }
    result = analyze({
        "articleId": 210, "source": "bok_monetary",
        "title": "2026년 3월 통화 및 유동성", "summary": body,
    }, FakeModel(raw))
    assert result["status"] == "ANALYZED"
    assert result["sectors"] == []
    assert result["excludedSectorSuggestions"][0]["reasonCode"] == (
        "MACRO_BALANCE_NOT_SECTOR_EARNINGS"
    )


def test_pdf_whitespace_variance_is_recovered_only_with_real_substring():
    from tris_market_event_intelligence.direct_sector_impact_cli import (
        locate_verbatim_quote,
    )
    original = "실제 보고서는 금융기관 유동성이 전월대비 0.5% 증가했다고 발표했다."
    quote, start = locate_verbatim_quote(original, "금융기관유동성이 전월대비 0.5% 증가")
    assert original[start:start + len(quote)] == quote
    with pytest.raises(ValueError, match="SECTOR_EVIDENCE_QUOTE_NOT_IN_ARTICLE"):
        locate_verbatim_quote(original, "금융기관 유동성이 20% 증가")


def test_truncation_retries_with_shorter_source_and_keeps_failure_visible():
    from tris_market_event_intelligence.llm_runtime import LLMExecutionError

    class TruncatedModel(FakeModel):
        def generate_json(self, *, system, user, schema):
            self.calls += 1
            if self.calls < 3:
                raise LLMExecutionError("MODEL_GENERATION_TRUNCATED")
            return json.dumps(prediction(), ensure_ascii=False)

    long_text = NEWS + " " + ("반도체 시장 전망과 산업 정책입니다. " * 150)
    model = TruncatedModel(prediction())
    result = analyze({"articleId": 1, "summary": long_text}, model)
    assert model.calls == 3
    assert result["status"] == "ANALYZED"
    assert result["sectors"][0]["direction"] == "NEGATIVE"


def test_actual_ppi_example_does_not_report_unsupported_positive_industrials():
    quote = (
        "공 산 품 :화학제품(1.8%), 1차금속제품(1.4%), "
        "컴퓨터･ 전자및광학기기 (1.6%)등이올라전월대비 0.7%상승"
    )
    source = "한국은행은 2026년 5월 생산자물가지수 결과를 발표했다. " + quote
    raw = {
        "event": "Producer prices rose",
        "summary": "Several manufactured goods producer price indices rose.",
        "overallMarketDirection": "UNKNOWN",
        "sectors": [{
            "sector": "INDUSTRIALS", "direction": "POSITIVE",
            "sourceQuote": quote,
            "reason": "Rising manufactured goods costs are good for the industry.",
            "uncertainty": "Margin effects unknown.",
        }],
    }
    result = analyze({
        "articleId": 194, "source": "bok_release",
        "title": "2026년 5월 생산자물가지수(잠정)",
        "summary": source,
    }, FakeModel(raw))
    assert result["status"] == "ANALYZED"
    assert result["sectors"][0]["direction"] == "UNKNOWN"
    assert result["sectors"][0]["modelOriginalDirection"] == "POSITIVE"
    assert result["sectors"][0]["modelOriginalReason"] == raw["sectors"][0]["reason"]
    assert "costs are good" not in result["sectors"][0]["reason"]
    assert result["sectors"][0]["economicReasoningReview"]["reviewReasons"] == [
        "PPI_LEVEL_NOT_SECTOR_MARGIN",
    ]


def test_bok_business_sentiment_cannot_assign_aggregate_to_materials_sector():
    quote = "5월 중 제조업 기업심리지수는 100.8로 전월에 비해 1.7p 상승하였으며"
    text = "한국은행 기업경기조사와 경제심리지수 발표: " + quote
    raw = {
        "event": "Manufacturing business sentiment rose",
        "summary": "Manufacturers reported an improvement in sentiment.",
        "overallMarketDirection": "UNKNOWN",
        "sectors": [{
            "sector": "MATERIALS", "direction": "POSITIVE",
            "sourceQuote": quote,
            "reason": "This supposedly establishes improving materials-company earnings.",
            "uncertainty": "No materials-only series is given.",
        }],
    }
    result = analyze({
        "articleId": 204, "source": "bok_release",
        "title": "2026년 5월 기업경기조사 결과 및 경제심리지수(ESI)",
        "summary": text,
    }, FakeModel(raw))
    assert result["status"] == "ANALYZED"
    assert result["sectors"] == []
    assert result["excludedSectorSuggestions"][0]["reasonCode"] == (
        "SENTIMENT_AGGREGATE_WRONG_SECTOR"
    )


def test_real_semiconductor_boom_retains_direct_positive_signal():
    quote = "반도체 경기 호황과 파급효과 확산"
    content = ("한국은행은 2026년 8월 경제전망에서 " + quote + "을 설명하였으며, "
               "관련 산업의 사업 전망과 수출 수요에도 영향을 미칠 수 있다고 발표하였다.")
    raw = {
        "event": "Semiconductor boom",
        "summary": "Semiconductor industry conditions strengthened.",
        "overallMarketDirection": "UNKNOWN",
        "sectors": [{
            "sector": "SEMICONDUCTORS", "direction": "POSITIVE",
            "sourceQuote": quote,
            "reason": "The report describes favorable conditions in the semiconductor sector.",
            "uncertainty": "This is not a measured abnormal stock return.",
        }],
    }
    result = analyze({
        "articleId": 117, "source": "bok_release",
        "title": "경제전망(2026년 8월)",
        "summary": content,
    }, FakeModel(raw))
    assert result["status"] == "ANALYZED"
    assert result["sectors"][0]["direction"] == "POSITIVE"
    assert result["sectors"][0]["economicReasoningReview"]["reviewReasons"] == []
