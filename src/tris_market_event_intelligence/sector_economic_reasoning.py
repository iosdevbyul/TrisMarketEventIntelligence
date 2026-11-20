"""Economic interpretation checks for direct news -> sector impact hypotheses.

A factual rise in an index is NOT proof of a company's share-price upside.
This module distinguishes the measured object from exposed listed sectors.
Never rewrites the event/source quote, never sets a positive price prediction.
"""
from __future__ import annotations

from typing import Any

DIRECTIONS = frozenset({"POSITIVE", "NEGATIVE"})

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
    elif category == "BUSINESS_SENTIMENT":
        if (("비제조업" in evidence and sector in {"INDUSTRIALS", "MATERIALS", "CONSUMER"})
                or ("제조업" in evidence and sector in {"MATERIALS", "CONSUMER"})):
            notes.append("SENTIMENT_AGGREGATE_WRONG_SECTOR")
            disposition = "EXCLUDE_SECTOR"
        elif direction in DIRECTIONS and not any(
            x in evidence for x in ("영업이익", "매출액", "수주", "출하", "operating profit")
        ):
            notes.append("SURVEY_SENTIMENT_NOT_EARNINGS")
            revised = "UNKNOWN"
    elif category == "GDP_AGGREGATE":
        if any(x in evidence for x in ("국내총생산", "gdp", "국민소득", "gni")):
            if sector != "BROAD_MARKET":
                notes.append("GDP_AGGREGATE_NO_SECTOR_IDENTITY")
                disposition = "EXCLUDE_SECTOR"
            elif direction in DIRECTIONS:
                notes.append("GDP_GROWTH_NOT_MARKET_SURPRISE")
                revised = "UNKNOWN"
    elif (category == "MONETARY_OR_CREDIT_AGGREGATE" and direction in DIRECTIONS
          and not any(term in evidence for term in (
              "영업이익", "소비지출", "소매판매", "매출", "부실률",
              "net interest margin",
          ))):
            notes.append("MONEY_STOCK_NOT_SECTOR_EARNINGS")
            revised = "UNKNOWN"

    # Preserve model-authored reasoning separately when its conclusion was gated.
    # Explanations must not claim a positive/negative result for an UNKNOWN row.
    explanation = {
        "PPI_LEVEL_NOT_SECTOR_MARGIN":
            "The release shows producer-price changes, not whether this "
            "sector's revenue gains exceed its higher input costs.",
        "SURVEY_SENTIMENT_NOT_EARNINGS":
            "A sentiment survey changed; this alone does not establish a "
            "listed sector's earnings or price direction.",
        "GDP_GROWTH_NOT_SPECIFIC_SECTOR_PROFIT":
            "GDP growth is an economy-wide statistic and does not demonstrate "
            "this particular listed sector's profitability.",
        "GDP_GROWTH_NOT_MARKET_SURPRISE":
            "GDP growth was reported, but a positive stock-market surprise "
            "cannot be inferred without the market's prior expectations.",
        "MONEY_STOCK_NOT_SECTOR_EARNINGS":
            "Changes in money or credit aggregates do not directly demonstrate "
            "this sector's earnings, spending or share-price direction.",
    }
    return {
        "sector": sector, "originalDirection": direction,
        "reviewedDirection": revised, "disposition": disposition,
        "economicCategory": category, "reviewReasons": notes,
        "reviewedReason": explanation.get(notes[0]) if notes else None,
        "economicOutcomeIndependentlyVerified": False,
    }
