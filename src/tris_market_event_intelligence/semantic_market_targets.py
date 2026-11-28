"""Conservative retrospective target taxonomy; never guess regional exposure."""
from __future__ import annotations

from datetime import UTC, datetime

# Broad target labels are not evidence of an individual issuer.
AMBIGUOUS_NAMES = frozenset({
    "financial institutions", "nasdaq", "dana", "citizens", "grand",
    "financials", "banks", "ai", "stocks", "market", "securities",
})
INDEXES = {
    "kospi": ("XKRX", "^KS11", "^KQ11", "https://global.krx.co.kr/"),
    "코스피": ("XKRX", "^KS11", "^KQ11", "https://global.krx.co.kr/"),
    "korean stock market (kospi)": ("XKRX", "^KS11", "^KQ11", "https://global.krx.co.kr/"),
    "kosdaq": ("XKRX", "^KQ11", "^KS11", "https://global.krx.co.kr/"),
    "코스닥": ("XKRX", "^KQ11", "^KS11", "https://global.krx.co.kr/"),
    "코스닥 지수": ("XKRX", "^KQ11", "^KS11", "https://global.krx.co.kr/"),
    "s&p 500": ("XNYS", "^GSPC", "^DJI", "https://www.spglobal.com/spdji/"),
    "s&p500지수": ("XNYS", "^GSPC", "^DJI", "https://www.spglobal.com/spdji/"),
    "nasdaq composite": ("XNAS", "^IXIC", "^GSPC", "https://indexes.nasdaq.com/"),
    "nasdaq composite index": ("XNAS", "^IXIC", "^GSPC", "https://indexes.nasdaq.com/"),
    "나스닥지수": ("XNAS", "^IXIC", "^GSPC", "https://indexes.nasdaq.com/"),
    "필라델피아 반도체 지수": ("XNAS", "^SOX", "^GSPC", "https://indexes.nasdaq.com/"),
}
SECTORS = {
    "banking sector (u.s.)": ("XLF", "Broad US financial equities, not banks alone."),
    "us airlines": ("JETS", "Airline ETF includes international exposure."),
    "us defense sector": ("ITA", "US aerospace and defense equities."),
}


def target_key(target: str) -> str:
    return " ".join(target.casefold().split())


def block_equity(target: str) -> bool:
    key = target_key(target)
    return (
        key in AMBIGUOUS_NAMES or key in INDEXES or key in SECTORS
        or key.endswith((" sector", " sectors", " industry", " industries", " market"))
    )


def typed_mapping(target: str) -> dict[str, str] | None:
    key = target_key(target)
    if key in INDEXES:
        market, symbol, benchmark, source = INDEXES[key]
        kind = "MARKET_INDEX"
        limitation = "Relative return versus a distinct comparison index, not causal impact."
    elif key in SECTORS:
        symbol, limitation = SECTORS[key]
        market, benchmark = "XNYS", "SPY"
        source = {
            "XLF": "https://www.ssga.com/",
            "JETS": "https://www.usglobaletfs.com/",
            "ITA": "https://www.ishares.com/",
        }[symbol]
        kind = "SECTOR_ETF_PROXY"
    else:
        return None
    return {
        "target": target, "market": market, "symbol": symbol, "benchmark": benchmark,
        "verified_at": datetime.now(UTC).isoformat(), "mapping_kind": kind,
        "proxy_limitation": limitation, "evidence_url": source,
    }
