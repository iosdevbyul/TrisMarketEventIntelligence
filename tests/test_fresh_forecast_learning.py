import pytest

from tris_market_event_intelligence.fresh_forecast_learning_cli import metrics, publication_rows


def test_publication_window_captures_early_reaction(tmp_path):
    prices = tmp_path / "prices.csv"
    prices.write_text(
        "symbol,date,adjusted_close,source\n"
        "AAA,2026-01-05,100,test\nAAA,2026-01-06,110,test\n"
        "AAA,2026-01-07,108,test\nAAA,2026-01-08,107,test\n"
        "SPY,2026-01-05,100,test\nSPY,2026-01-07,101,test\n"
        "SPY,2026-01-08,101,test\n",
        encoding="utf-8",
    )
    rows = publication_rows([{
        "id": "forecast", "recordSha256": "hash", "generatedAt": "2026-11-27T00:00:00+00:00",
        "identity": {"articleId": 1, "bodySha256": "body", "packet": {
            "publishedAt": "2026-01-06T14:00:00+00:00",
        }},
        "forecasts": [{"instrument": {"name": "AAA", "symbol": "AAA", "market": "XNAS",
                                      "benchmark": "SPY"},
                       "direction": "UP", "mechanism": "new event",
                       "counterargument": "uncertain", "evidenceQuote": "source"}],
    }], prices)
    assert rows[0]["entryDay"] == "2026-01-05"
    assert rows[0]["exitDay"] == "2026-01-07"
    assert rows[0]["stockReturn"] == pytest.approx(0.08)
    assert rows[0]["verdict"] == "CORRECT"
    assert rows[1]["verdict"] == "UNAVAILABLE"
    result = metrics(rows)
    assert result["1d:verdict"]["constantBaselines"]["UP"] == 1
    assert result["5d:verdict"]["accuracy"] is None
