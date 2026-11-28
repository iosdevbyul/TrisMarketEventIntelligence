from datetime import date

import pytest

from tris_market_event_intelligence.autonomous_market_experience_cli import (
    covered_instruments,
    dated_articles,
    eligible_memory,
    read_sealed,
)
from tris_market_event_intelligence.historical_experience_abc_v2_cli import _sealed


def test_only_matured_prior_experiences():
    packet = {"publishedAt": "2025-01-10T10:00:00+00:00"}
    past = {"outcome": {"publishedAt": "2025-01-01T10:00:00+00:00",
                        "exitDay": "2025-01-09"}}
    same_day = {"outcome": {"publishedAt": "2025-01-01T10:00:00+00:00",
                            "exitDay": "2025-01-10"}}
    future = {"outcome": {"publishedAt": "2025-01-11T10:00:00+00:00",
                          "exitDay": "2025-01-12"}}
    assert eligible_memory([past, same_day, future], packet) == [past]


def test_checkpoint_rejects_mutation(tmp_path):
    path = tmp_path / "checkpoint.json"
    record = _sealed(path, {"value": 1})
    assert read_sealed(path) == record
    path.write_text('{"value":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid research checkpoint"):
        read_sealed(path)


def test_dated_articles_reject_invalid_timestamps_before_sorting():
    valid = {"articleId": 1, "publishedAt": "2025-01-01T10:00:00+00:00",
             "detectedAt": "2025-01-01T11:00:00+00:00"}
    later = {**valid, "articleId": 2, "publishedAt": "2025-01-01T10:30:00+00:00"}
    invalid = [
        {**valid, "publishedAt": None},
        {**valid, "publishedAt": 1735725600},
        {**valid, "publishedAt": "invalid"},
        {**valid, "publishedAt": "2025-01-01"},
        {**valid, "detectedAt": None},
        {**valid, "detectedAt": "2024-12-31T11:00:00+00:00"},
    ]
    assert dated_articles([later, *invalid, valid]) == [valid, later]


def test_coverage_requires_both_symbol_and_benchmark_full_windows():
    instrument = {"market": "XNYS", "symbol": "ABC", "benchmark": "SPY"}
    packet = {"publishedAt": "2025-01-06T15:00:00+00:00",
              "instruments": [instrument]}
    days = [date(2025, 1, d) for d in (3, 7, 8, 9, 10, 13)]
    assert covered_instruments(packet, {"ABC": days, "SPY": days}) == [instrument]
    assert covered_instruments(packet, {"ABC": days[:-1], "SPY": days}) == []
    assert covered_instruments(packet, {"ABC": days, "SPY": days[:-1]}) == []
    assert covered_instruments(packet, {"ABC": days[1:], "SPY": days}) == []
    assert covered_instruments(packet, {"ABC": days}) == []
