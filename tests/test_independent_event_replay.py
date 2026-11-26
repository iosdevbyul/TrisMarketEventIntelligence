from tris_market_event_intelligence.independent_event_replay_cli import evaluate_row


def test_retrospective_price_labels_use_later_aligned_sessions():
    record = {"observedAt": "2026-11-02T20:00:00+00:00",
              "mapping": {"symbol": "NVDA", "benchmark": "SPY", "market": "XNAS"},
              "answer": {"direction_1d": "UP", "direction_5d": "DOWN"}}
    prices = {
        "NVDA": [{"date": "2026-11-03", "adjusted_close": "100", "source": "fixture"},
                 {"date": "2026-11-04", "adjusted_close": "110", "source": "fixture"}],
        "SPY": [{"date": "2026-11-03", "adjusted_close": "100", "source": "fixture"},
                {"date": "2026-11-04", "adjusted_close": "102", "source": "fixture"}],
    }
    rows = evaluate_row(record, prices)
    assert rows[0]["entryDay"] == "2026-11-03"
    assert rows[0]["verdict"] == "CORRECT"
    assert rows[1]["status"] == "INSUFFICIENT_FUTURE_SESSIONS"
    prices["NVDA"].pop()
    assert evaluate_row(record, prices)[0]["status"] == "MISSING_ALIGNED_STOCK_SESSION"
