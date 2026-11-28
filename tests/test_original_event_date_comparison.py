from tris_market_event_intelligence.original_event_date_comparison_cli import compare_row


def fixture():
    record = {"kind": "SEC", "secAcceptanceAt": "2026-05-20T20:21:19+00:00",
              "observedAt": "2026-11-24T11:40:44+00:00",
              "mapping": {"market": "XNAS", "symbol": "NVDA", "benchmark": "SPY"},
              "answer": {"direction_1d": "UP", "direction_5d": "DOWN"}}
    prices = {
        symbol: [{"date": day, "adjusted_close": value, "source": "fixture"}
                 for day, value in zip(
                     ["2026-05-20", "2026-05-21", "2026-05-22"], values, strict=True)]
        for symbol, values in {"NVDA": ["100", "101", "105"], "SPY": ["100", "101", "102"]}.items()
    }
    return record, prices


def test_original_date_comparison_preserves_sealed_record():
    record, prices = fixture()
    rows = compare_row(record, prices)
    assert rows[0]["anchorLocalDay"] == "2026-05-20"
    assert rows[0]["entryDay"] == "2026-05-21"
    assert rows[0]["verdict"] == "CORRECT"
    assert rows[1]["status"] == "INSUFFICIENT_FUTURE_SESSIONS"
    assert record["observedAt"] == "2026-11-24T11:40:44+00:00"


def test_price_history_starting_late_is_not_next_session():
    record, prices = fixture()
    prices["SPY"].pop(0)
    assert compare_row(record, prices)[0]["status"] == "PRICE_HISTORY_STARTS_AFTER_ANCHOR"


def test_invalid_declared_clock_does_not_fallback():
    record, prices = fixture()
    record["secAcceptanceAt"] = None
    assert compare_row(record, prices)[0]["status"] == "ORIGINAL_DECLARED_CLOCK_INVALID"
