"""Narrow detection of market-impact summaries without any market transmission link.

This checks for an explicit market-facing channel, NOT whether causal reasoning
is economically correct. False positives are avoided by scoping to BROAD_MARKET
or the observed generic macro-strength boilerplate.
"""
from __future__ import annotations

import re

# A hypothesis must speak to transmission into a tradable market, valuation,
# earnings expectation, cash flow or investor behavior, not just macro strength.
_MARKET_LINK = re.compile(
    r"\b(?:equity|equities|stock|stocks|share|shares|bond|bonds|yield|yields|"
    r"market|markets|valuation|valuations|investor|investors|earnings|"
    r"risk.premium|discount.rate|cost.of.capital|capital.flow|"
    r"cash.flow|asset.price|asset.prices|credit.spread|"
    r"exchange.rate|currency|currencies|demand.for|sector)\b"
    r"|주식|주가|증시|시장|채권|금리|환율|투자자|밸류에이션|가치평가|"
    r"할인율|자본비용|이익전망|실적전망|현금흐름|위험프리미엄|수급|자산가격|업종",
    re.IGNORECASE,
)


def mechanism_quality_code(sector: str, rationale: str) -> str | None:
    """Reject broad-market prose with no explicit market transmission mechanism.

    Never infer factuality, plausibility, or direction from keyword presence.
    Other sectors receive prompt guidance, not this coarse automatic veto.
    """
    if sector == "BROAD_MARKET" and not _MARKET_LINK.search(rationale):
        return "MARKET_MECHANISM_MISSING"
    return None
