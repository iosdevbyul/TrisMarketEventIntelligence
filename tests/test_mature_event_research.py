from datetime import datetime

from tris_market_event_intelligence.mature_event_research_cli import balanced, mature


def test_maturity_uses_only_aligned_dates():
    mapping = {"symbol": "NVDA", "benchmark": "SPY", "market": "XNAS"}
    anchor = datetime.fromisoformat("2026-05-20T20:21:19+00:00")
    days = ("2026-05-20", "2026-05-21", "2026-05-22", "2026-05-26",
            "2026-05-27", "2026-05-28", "2026-05-29")
    assert mature(mapping, anchor, {"NVDA": days, "SPY": days})
    assert not mature(mapping, anchor, {"NVDA": days[:-1], "SPY": days})
    assert not mature(mapping, anchor, {"NVDA": days[1:], "SPY": days})


def test_round_robin_does_not_exhaust_one_issuer_first():
    rows = [{"eventId": str(i), "kind": "SEC", "symbol": symbol,
             "anchorAt": f"2026-05-{i + 1:02}"} for i, symbol in
            enumerate(["NVDA", "NVDA", "NVDA", "AAPL", "MSFT"])]
    assert len({r["symbol"] for r in balanced(rows, 3)}) == 3
