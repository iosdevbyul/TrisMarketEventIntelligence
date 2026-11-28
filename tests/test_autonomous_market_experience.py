import pytest

from tris_market_event_intelligence.autonomous_market_experience_cli import (
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
