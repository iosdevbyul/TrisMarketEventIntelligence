"""Actual BOK 2026 excerpts: measured fact vs listed-sector impact."""
from tris_market_event_intelligence.sector_economic_reasoning import (
    economic_category,
    review_sector_hypothesis,
)


def review(title, sector, direction, quote, reason="Measured index increased"):
    return review_sector_hypothesis(
        title=title, sector=sector, direction=direction,
        quote=quote, reason=reason,
    )


def test_ppi_manu_price_increase_not_industrials_profit():
    quote = (
        "공 산 품 :화학제품(1.8%), 1차금속제품(1.4%), "
        "컴퓨터･ 전자및광학기기 (1.6%)등이올라전월대비 0.7%상승"
    )
    result = review("2026년 5월 생산자물가지수(잠정)",
                    "INDUSTRIALS", "POSITIVE", quote)
    assert result["originalDirection"] == "POSITIVE"
    assert result["reviewedDirection"] == "UNKNOWN"
    assert result["reviewReasons"] == ["PPI_LEVEL_NOT_SECTOR_MARGIN"]


def test_ppi_energy_cost_increase_not_automatic_energy_stock_upside():
    quote = "전력 ·가스 ·수도및폐기물 : 산업용도시가스(10.3%) 등이 올라 전월대비 0.5% 상승"
    result = review("2026년 5월 생산자물가지수(잠정)",
                    "ENERGY", "POSITIVE", quote)
    assert result["reviewedDirection"] == "UNKNOWN"


def test_manu_sentiment_does_not_attribute_index_to_materials():
    quote = "5월 중 제조업 기업심리지수는 100.8로 전월에 비해 1.7p 상승하였으며"
    result = review("2026년 5월 기업경기조사 결과 및 경제심리지수(ESI)",
                    "MATERIALS", "POSITIVE", quote)
    assert result["disposition"] == "EXCLUDE_SECTOR"
    assert result["reviewReasons"] == ["SENTIMENT_AGGREGATE_WRONG_SECTOR"]


def test_nonmanufacturing_sentiment_is_not_consumer_stocks():
    quote = "5월 중 비제조업 기업심리지수는 97.5로 전월에 비해 5.4p 상승하였고"
    result = review("2026년 5월 기업경기조사 결과 및 경제심리지수(ESI)",
                    "CONSUMER", "POSITIVE", quote)
    assert result["disposition"] == "EXCLUDE_SECTOR"


def test_gdp_growth_not_automatic_industrials_or_consumer_stock_upside():
    quote = "실질 국내총생산(GDP)은 전기대비 1.8% 성장, 전년동기대비 3.8% 성장"
    for sector in ("CONSUMER", "INDUSTRIALS", "BROAD_MARKET"):
        result = review("2026년 1/4분기 국민소득(잠정)",
                        sector, "POSITIVE", quote)
        assert result["reviewedDirection"] == "UNKNOWN"


def test_bok_117_direct_semiconductor_boom_remains_positive_hypothesis():
    result = review("경제전망(2026년 8월)",
                    "SEMICONDUCTORS", "POSITIVE",
                    "반도체 경기 호황과 파급효과 확산")
    assert economic_category("경제전망(2026년 8월)") == "OTHER"
    assert result["reviewedDirection"] == "POSITIVE"
    assert result["reviewReasons"] == []


def test_explicit_earnings_effect_does_not_get_blanket_ppi_veto():
    result = review("2026년 5월 생산자물가지수",
                    "MATERIALS", "POSITIVE",
                    "화학 업체의 영업이익이 크게 증가했다.")
    assert result["reviewedDirection"] == "POSITIVE"


def test_quantitative_statement_not_verified_as_price_accuracy():
    result = review("2026년 5월 생산자물가지수",
                    "INDUSTRIALS", "NEGATIVE", "원재료 가격이 상승했다.")
    assert result["economicOutcomeIndependentlyVerified"] is False
    assert result["reviewedDirection"] == "UNKNOWN"
