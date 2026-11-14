"""Synthetic boundary tests for leak-resistant KR/US research samples."""
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tris_market_event_intelligence.event_ticker_screen import EventTickerLink
from tris_market_event_intelligence.intraday_event_study import (
    IntradayLabel,
    IntradayNews,
    IntradayResult,
)
from tris_market_event_intelligence.walkforward_dataset import build_walkforward_dataset

NY = ZoneInfo("America/New_York")
CUTOFF = datetime(2026, 11, 17, 12, tzinfo=NY)
HORIZONS = (1, 5, 15, 30, 60)


def item(event_id: int, published: datetime, *, url: str | None = None) -> tuple:
    available = published + timedelta(seconds=30)
    entry = available.replace(second=0) + timedelta(minutes=1)
    event = IntradayNews(
        event_id, "XNAS", "NVDA", "SPY", published, published, available,
    )
    link = EventTickerLink(
        event_id, "XNAS", "NVDA", "SPY",
        url or f"https://example.org/story/{event_id}",
        published, "DIRECT",
    )
    labels = tuple(
        IntradayLabel(
            h, (entry + timedelta(minutes=h)).astimezone(UTC).isoformat(),
            0.02, 0.005, 0.015, "COMPLETED",
        ) for h in HORIZONS
    )
    result = IntradayResult(
        event_id, "XNAS", "NVDA", "SPY",
        entry.astimezone(UTC).isoformat(), "COMPLETED", 30.0, labels,
    )
    return event, link, result


def split(items: list[tuple], *, embargo: int = 60) -> dict[str, object]:
    return build_walkforward_dataset(
        [x[0] for x in items], [x[1] for x in items],
        [x[2] for x in items], cutoff=CUTOFF, embargo_minutes=embargo,
    )


def test_temporal_samples_use_only_preavailable_features_and_expost_target() -> None:
    before = item(1, datetime(2026, 11, 16, 10, tzinfo=NY))
    after = item(2, datetime(2026, 11, 18, 10, tzinfo=NY))
    dataset = split([before, after])
    assert dataset["counts"] == {"HOLDOUT": 5, "TRAIN": 5}
    assert dataset["prediction_model_trained"] is False
    assert dataset["training_dataset_eligible"] is False
    assert dataset["point_in_time_verified"] is False
    assert {sample["split"] for sample in dataset["samples"]} == {"TRAIN", "HOLDOUT"}
    assert len(dataset["samples"]) == 10
    for sample in dataset["samples"]:
        assert "target_excess_return" not in sample["features"]
        assert "event_id" not in sample["features"]
        assert "label_exit_at" not in sample["features"]
        assert set(sample["features"]) == {
            "market", "symbol", "relationship", "hour_of_day", "weekday",
            "source_latency_seconds", "horizon_minutes",
        }
        assert sample["target_excess_return"] == pytest.approx(0.015)


def test_embargo_and_unmatured_train_labels_are_purged() -> None:
    # At 10:55, +1 min has matured; +60 min has not matured before
    # the noon cutoff minus a full hour embargo.
    before = item(1, datetime(2026, 11, 17, 10, 55, tzinfo=NY))
    after = item(2, datetime(2026, 11, 17, 12, 30, tzinfo=NY))
    dataset = split([before, after])
    assert dataset["counts"].get("PURGED_TIME_BOUNDARY") == 9
    assert dataset["counts"].get("TRAIN") == 1
    assert len(dataset["samples"]) == 1


def test_no_random_split_and_late_holdout_starts_after_embargo() -> None:
    entries = [
        item(7, datetime(2026, 11, 17, 11, 59, tzinfo=NY)),
        item(8, datetime(2026, 11, 17, 13, 0, tzinfo=NY)),
    ]
    dataset = split(entries)
    assert dataset["counts"]["HOLDOUT"] == 5
    assert dataset["counts"]["PURGED_TIME_BOUNDARY"] == 5
    assert all(row["event_id"] == 8 for row in dataset["samples"])


def test_censored_returns_are_not_resurrected_as_targets() -> None:
    e, link, r = item(3, datetime(2026, 11, 16, 10, tzinfo=NY))
    replacement = IntradayLabel(
        1, r.labels[0].exit_at, None, None, None, "OVERLAPPING_NEWS",
    )
    revised = IntradayResult(
        r.event_id, r.market, r.symbol, r.benchmark_symbol, r.entry_at,
        "PARTIAL", r.latency_seconds, (replacement, *r.labels[1:]),
    )
    result = build_walkforward_dataset(
        [e], [link], [revised], cutoff=CUTOFF,
    )
    assert result["counts"]["OVERLAPPING_NEWS"] == 1
    assert len(result["samples"]) == 4


def test_future_evidence_and_unmatched_links_fail_before_training() -> None:
    e, link, r = item(1, datetime(2026, 11, 16, 10, tzinfo=NY))
    late = EventTickerLink(
        1, "XNAS", "NVDA", "SPY", link.source_url,
        e.available_at + timedelta(seconds=1), "DIRECT",
    )
    with pytest.raises(ValueError, match="Future-discovered"):
        build_walkforward_dataset([e], [late], [r], cutoff=CUTOFF)
    with pytest.raises(ValueError, match="exactly one"):
        build_walkforward_dataset([e], [], [r], cutoff=CUTOFF)


def test_invalid_cutoff_and_insufficient_embargo_fail_closed() -> None:
    with pytest.raises(ValueError, match="timezone"):
        build_walkforward_dataset([], [], [], cutoff=datetime(2026, 11, 17))
    with pytest.raises(ValueError, match="maximum"):
        split([], embargo=59)
    with pytest.raises(ValueError, match="maximum"):
        build_walkforward_dataset([], [], [], cutoff=CUTOFF, embargo_minutes=True)


def test_duplicate_source_does_not_create_second_research_row() -> None:
    stamp = datetime(2026, 11, 16, 10, tzinfo=NY)
    first = item(1, stamp, url="https://example.org/a?utm_source=rss")
    second = item(2, stamp + timedelta(minutes=2), url="https://example.org/a")
    data = split([first, second])
    assert data["counts"]["DUPLICATE_SOURCE"] == 5
    assert {s["event_id"] for s in data["samples"]} == {1}


def test_invalid_completed_exit_does_not_create_fake_target() -> None:
    event, link, r = item(1, datetime(2026, 11, 16, 10, tzinfo=NY))
    bad = IntradayLabel(1, event.available_at.isoformat(), 0.1, 0, 0.1, "COMPLETED")
    result = IntradayResult(
        r.event_id, r.market, r.symbol, r.benchmark_symbol,
        r.entry_at, "PARTIAL", r.latency_seconds,
        (bad, *r.labels[1:]),
    )
    with pytest.raises(ValueError, match="historical target timestamp"):
        build_walkforward_dataset([event], [link], [result], cutoff=CUTOFF)


def test_report_input_feature_times_use_exchange_local_zone() -> None:
    # A UTC timestamp must still produce New York local hour and weekday.
    event, link, result = item(1, datetime(2026, 11, 16, 10, tzinfo=NY))
    shifted = IntradayNews(
        event.event_id, event.market, event.symbol, event.benchmark_symbol,
        event.published_at.astimezone(UTC),
        event.received_at.astimezone(UTC), event.available_at.astimezone(UTC),
    )
    dataset = build_walkforward_dataset(
        [shifted], [link], [result], cutoff=CUTOFF,
    )
    assert dataset["samples"][0]["features"]["hour_of_day"] == 10
