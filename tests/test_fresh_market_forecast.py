import pytest

from tris_market_event_intelligence.fresh_market_forecast_cli import universe, validate


def test_macro_forecast_is_not_bound_to_old_target():
    options = universe({"title": "Federal Reserve raises rates", "body": "Policy tightening"}, {})
    tech = next(x for x in options if x["symbol"] == "XLK")
    passages = [{"id": "E1", "text": "The central bank raised its policy rate by 25 basis points."}]
    row = {"instrument_id": tech["id"], "direction": "DOWN", "evidence_id": "E1",
           "mechanism": "Higher discount rates may reduce valuations",
           "counterargument": "The hike may already be priced in"}
    result = validate({"forecasts": [row]}, options, passages)
    assert result[0]["instrument"]["symbol"] == "XLK"
    assert result[0]["direction"] == "DOWN"
    row["instrument_id"] = "M999"
    with pytest.raises(ValueError, match="Invalid or duplicate"):
        validate({"forecasts": [row]}, options, passages)


def test_no_forced_forecast_or_page_marker():
    options = universe({"title": "Meeting announcement", "body": "Schedule only"}, {})
    assert validate({"forecasts": []}, options, []) == []
    with pytest.raises(ValueError, match="Invalid source evidence"):
        validate({"forecasts": [{
            "instrument_id": options[0]["id"], "direction": "UP", "evidence_id": "E1",
            "mechanism": "Unfounded reason", "counterargument": "Unknown",
        }]}, options, [{"id": "E1", "text": "[PDF PAGE 1]"}])


def test_generic_issuer_name_not_admitted():
    mapping = {"Financial Institutions": {"mappingKind": "DIRECT_EQUITY",
               "market": "XNAS", "symbol": "FISI", "benchmark": "SPY"}}
    options = universe({"title": "Financial Institutions report", "body": "Sector data"}, mapping)
    assert all(x["symbol"] != "FISI" for x in options)
