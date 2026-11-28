"""Regression tests for abstentions and fixed-population evaluation."""
from pathlib import Path

import pytest

from tris_market_event_intelligence import independent_market_validation_cli as validation


def test_uncertain_does_not_improve_population_correct_rate(monkeypatch):
    instruments = [
        {"id": "M1", "symbol": "ABC", "market": "XNYS"},
        {"id": "M2", "symbol": "DEF", "market": "XNYS"},
    ]
    artifact = {
        "identity": {"articleId": 1, "packet": {"instruments": instruments}},
        "recordSha256": "sealed", "generatedAt": "2026-01-01T00:00:00+00:00",
        "arms": {
            "A": [{"instrument": instruments[0], "direction": "UP"},
                  {"instrument": instruments[1], "direction": "DOWN"}],
            "D": [{"instrument": instruments[0], "direction": "UP"},
                  {"instrument": instruments[1], "direction": "UNCERTAIN"}],
        },
    }

    def outcomes(records, prices):
        assert len(records) == 2
        return [
            {"articleId": 1, "symbol": symbol, "horizonSessions": horizon,
             "actualDirection": "UP", "relativeActualDirection": "UP",
             "direction": "NEUTRAL", "verdict": "INCORRECT"}
            for symbol in ("ABC", "DEF") for horizon in (1, 5)
        ]

    monkeypatch.setattr(validation, "publication_rows", outcomes)
    result = validation.score([artifact], Path("unused"))
    metrics = result["metrics"]
    a = metrics["1d:actualDirection:universe:A"]
    d = metrics["1d:actualDirection:universe:D"]
    assert a["accuracy"] == 0.5
    assert d["accuracy"] == 1.0
    assert d["coverage"] == 0.5
    assert a["allPopulationCorrectRate"] == d["allPopulationCorrectRate"] == 0.5
    assert d["decisionCounts"]["UNCERTAIN"] == 1
    assert all("verdict" not in row for row in result["outcomes"])
    assert len(result["decisions"]) == 8


def test_invalid_validation_limit_rejected(tmp_path):
    import argparse

    args = argparse.Namespace(limit=31, model="gemma4:e2b", output_dir=tmp_path)
    with pytest.raises(ValueError, match="1..30"):
        validation.run(args, None)
