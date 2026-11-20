"""Economic interpretation checks for direct news -> sector impact hypotheses.

A factual rise in an index is NOT proof of a company's share-price upside.
This module distinguishes the measured object from exposed listed sectors.
Never rewrites the event/source quote, never sets a positive price prediction.
"""
from __future__ import annotations

from typing import Any

DIRECTIONS = frozenset({"POSITIVE", "NEGATIVE"})
SECTOR_SCOPE = frozenset({
    "SEMICONDUCTORS", "TECH", "CLOUD", "FINANCIALS", "ENERGY",
    "DEFENSE", "INDUSTRIALS", "CONSUMER", "TRANSPORT", "HEALTHCARE",
    "REAL_ESTATE", "MATERIALS", "UTILITIES", "BROAD_MARKET",
})


def economic_category(title: str) -> str:
    lower = title.lower()
    if "생산자물가지수" in lower or "producer price" in lower:
        return "PRODUCER_PRICES"
    if ("기업경기조사" in lower or "기업심리지수" in lower
            or "business sentiment" in lower):
        return "BUSINESS_SENTIMENT"
    if ("국민소득" in lower or "국내총생산" in lower
            or "gross domestic product" in lower):
        return "GDP_AGGREGATE"
    if ("통화 및 유동성" in lower or "가계신용" in lower
            or "가계부채" in lower):
        return "MONETARY_OR_CREDIT_AGGREGATE"
    return "OTHER"


def review_sector_hypothesis(
    *, title: str, sector: str, direction: str,
    quote: str, reason: str,
) -> dict[str, Any]:
    """Conservatively qualify model sector-level claims.

    No keyword match constitutes a gold label. Only narrow, observable
    mismatches between statistic type and claimed economic exposure are gated.
    """
    category = economic_category(title)
    notes: list[str] = []
    revised = direction
    disposition = "UNCHANGED"
    evidence = quote.lower()
    rationale = reason.lower()

    if category == "PRODUCER_PRICES" and direction in DIRECTIONS:
        # Higher output prices for a seller can be beneficial; higher input costs
        # to a buyer can be harmful. A PPI change alone gives neither margin.
        exposure_known = any(word in evidence for word in (
            "영업이익", "매출총이익", "이익률", "원가율",
            "gross margin", "operating margin",
        ))
        if not exposure_known:
            notes.append("PPI_LEVEL_NOT_SECTOR_MARGIN")
            revised = "UNKNOWN"
    elif category == "BUSINESS_SENTIMENT" and direction in DIRECTIONS:
        if ("제조업" in evidence and sector in {"MATERIALS", "CONSUMER"}
                or "비제조업" in evidence and sector in {"CONSUMER", "MATERIALS"}):
            # Broad manufacturing/non-manufacturing series cannot attribute
            # the value to these specific tradable sector classifications.
            notes.append("SENTIMENT_AGGREGATE_WRONG_SECTOR")
            disposition = "EXCLUDE_SECTOR"
        else:
            # Direction of an industry sentiment survey is informative but not
            # an identified earnings or equity-price shock.
            if not any(x in evidence for x in (
                "영업이익", "매출액", "수주", "출하", "operating profit",
            )):
                notes.append("SURVEY_SENTIMENT_NOT_EARNINGS")
                revised = "UNKNOWN"
    elif category == "GDP_AGGREGATE" and direction in DIRECTIONS:
        if ("국내총생산" in evidence or "gdp" in evidence
                or "국민소득" in evidence or "gni" in evidence):
            if sector not in {"BROAD_MARKET"}:
                notes.append("GDP_GROWTH_NOT_SPECIFIC_SECTOR_PROFIT")
                revised = "UNKNOWN"
            else:
                notes.append("GDP_GROWTH_NOT_MARKET_SURPRISE")
                revised = "UNKNOWN"
    elif category == "MONETARY_OR_CREDIT_AGGREGATE" and direction in DIRECTIONS:
        if not any(term in evidence for term in (
            "영업이익", "소비지출", "소매판매", "매출", "부실률",
            "net interest margin",
        )):
            notes.append("MONEY_STOCK_NOT_SECTOR_EARNINGS")
            revised = "UNKNOWN"

    # This check is a semantic guardrail, not a proof of correctness.
    return {
        "sector": sector, "originalDirection": direction,
        "reviewedDirection": revised, "disposition": disposition,
        "economicCategory": category, "reviewReasons": notes,
        "economicOutcomeIndependentlyVerified": False,
    }
