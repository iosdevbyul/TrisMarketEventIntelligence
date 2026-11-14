"""Point-in-time links and overlapping-news screens never create extra labels."""
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tris_market_event_intelligence.event_ticker_screen import (
    EventTickerLink,
    screen_intraday_links,
    source_identity,
)
from tris_market_event_intelligence.intraday_event_study import (
    IntradayNews,
    MinuteClose,
    study_intraday,
)
from tris_market_event_intelligence.intraday_event_study_cli import intraday_csv

ZONE = ZoneInfo("America/New_York")


def at(minute: int, *, seconds: int = 0) -> datetime:
    return datetime(2026, 11, 16, 10, minute, seconds, tzinfo=ZONE)


def news(event_id: int, minute: int) -> IntradayNews:
    stamp = at(minute)
    return IntradayNews(
        event_id, "XNAS", "NVDA", "SPY",
        stamp, stamp, stamp,
    )


def prices() -> list[MinuteClose]:
    return [
        MinuteClose("XNAS", sym, at(15) + timedelta(minutes=index), 100 + index)
        for sym in ("NVDA", "SPY") for index in range(70)
    ]


def link(event_id: int, minute: int, url: str) -> EventTickerLink:
    return EventTickerLink(
        event_id, "XNAS", "NVDA", "SPY", url, at(minute), "DIRECT",
    )


def run(
    news_rows: list[IntradayNews], link_rows: list[EventTickerLink],
) -> tuple[list, list]:
    outcome = study_intraday(news_rows, prices())
    return screen_intraday_links(news_rows, link_rows, outcome, horizons=(1, 5, 15, 30, 60))


def test_known_direct_symbol_is_accepted_before_news_is_available() -> None:
    e = news(7, 15)
    approved, audit = run([e], [link(7, 14, "https://example.org/story/7")])
    assert approved[0].status == "COMPLETED"
    assert all(label.status == "COMPLETED" for label in approved[0].labels)
    assert audit[0]["status"] == "SCREENED"


def test_future_discovered_ticker_link_is_not_usable_at_news_arrival() -> None:
    result, audit = run([
        news(7, 15),
    ], [link(7, 16, "https://example.org/story/7")])
    assert result[0].status == "LINK_KNOWN_AFTER_AVAILABILITY"
    assert all(x.excess_return is None for x in result[0].labels)
    assert audit[0]["status"] == "LINK_KNOWN_AFTER_AVAILABILITY"


def test_duplicate_article_from_another_feed_is_only_counted_once() -> None:
    rows = [news(7, 15), news(8, 16)]
    links = [
        link(7, 14, "https://example.org/news/123?utm_source=rss&nttId=48"),
        link(8, 15, "https://example.org/news/123?nttId=48&utm_medium=feed"),
    ]
    approved, audit = run(rows, links)
    assert approved[0].status == "COMPLETED"
    assert approved[1].status == "DUPLICATE_SOURCE_EVENT"
    assert audit[1]["status"] == "DUPLICATE_SOURCE_EVENT"
    assert all(label.excess_return is None for label in approved[1].labels)
    # The excluded repeat must not contaminate the first source's labels.
    assert approved[0].labels[0].status == "COMPLETED"


def test_independent_second_news_censors_only_affected_early_horizons() -> None:
    rows = [news(7, 15), news(8, 20)]
    links = [
        link(7, 14, "https://example.org/news/7"),
        link(8, 19, "https://example.org/news/8"),
    ]
    screened, audit = run(rows, links)
    first = screened[0]
    assert first.labels[0].status == "COMPLETED"
    assert first.labels[1].status == "OVERLAPPING_NEWS"
    assert first.labels[2].status == "OVERLAPPING_NEWS"
    assert first.labels[-1].excess_return is None
    assert audit[0]["censored_horizons"] == [5, 15, 30, 60]
    assert screened[1].labels[0].status == "COMPLETED"


def test_simultaneous_independent_events_do_not_fake_isolation() -> None:
    rows = [news(7, 15), news(8, 15)]
    links = [
        link(7, 15, "https://example.org/7"),
        link(8, 15, "https://example.org/8"),
    ]
    results, _ = run(rows, links)
    assert all(all(
        x.status == "OVERLAPPING_NEWS" for x in r.labels
    ) for r in results)


def test_missing_link_duplicate_links_and_wrong_benchmark_fail_closed() -> None:
    rows = [news(7, 15)]
    result = study_intraday(rows, prices())
    with pytest.raises(ValueError, match="Exactly one"):
        screen_intraday_links(rows, [], result, horizons=(1,))
    a = link(7, 14, "https://example.org/7")
    with pytest.raises(ValueError, match="Duplicate event-to-ticker"):
        screen_intraday_links(rows, [a, a], result, horizons=(1,))
    wrong = EventTickerLink(
        7, "XNAS", "NVDA", "QQQ", "https://example.org/7", at(14), "DIRECT",
    )
    screened, _ = screen_intraday_links(rows, [wrong], result, horizons=(1,))
    assert screened[0].status == "BENCHMARK_LINK_MISMATCH"


def test_source_url_preserves_identifiers_and_drops_known_tracking() -> None:
    first = source_identity("https://Example.org/a/?nttId=4&utm_source=rss")
    assert first == "https://example.org/a?nttId=4"
    assert source_identity("https://example.org/a?nttId=5") != first
    assert source_identity("javascript:alert(1)") is None


def test_link_validation_rejects_malformed_and_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        EventTickerLink(1, "XNYS", "A", "SPY", "https://example.org", at(12), "NOPE")
    with pytest.raises(ValueError, match="timezone"):
        EventTickerLink(
            1, "XNYS", "A", "SPY", "https://example.org",
            datetime(2026, 11, 16, 10, 15), "DIRECT",
        )


def test_cli_with_explicit_link_csv_is_read_only_and_reports_provenance(
    tmp_path: Path,
) -> None:
    events, closes, bindings = [
        tmp_path / file for file in ("events.csv", "closes.csv", "links.csv")
    ]
    events.write_text(
        "event_id,market,symbol,benchmark_symbol,published_at,received_at,available_at\n"
        "7,XNAS,NVDA,SPY,2026-11-16T10:15:00-05:00,"
        "2026-11-16T10:15:30-05:00,2026-11-16T10:15:45-05:00\n",
        encoding="utf-8",
    )
    closes.write_text(
        "market,symbol,close_at,adjusted_close,adjustment\n"
        "XNAS,NVDA,2026-11-16T10:16:00-05:00,100,adjusted\n"
        "XNAS,NVDA,2026-11-16T10:17:00-05:00,101,adjusted\n"
        "XNAS,SPY,2026-11-16T10:16:00-05:00,100,adjusted\n"
        "XNAS,SPY,2026-11-16T10:17:00-05:00,100.5,adjusted\n",
        encoding="utf-8",
    )
    bindings.write_text(
        "event_id,market,symbol,benchmark_symbol,source_url,evidence_seen_at,relationship\n"
        "7,XNAS,NVDA,SPY,https://example.org/story,2026-11-16T10:15:30-05:00,DIRECT\n",
        encoding="utf-8",
    )
    old = (events.read_bytes(), closes.read_bytes(), bindings.read_bytes())
    report = intraday_csv(events, closes, links_path=bindings)
    assert report["point_in_time_links_supplied"] is True
    assert report["source_links_independently_verified"] is False
    assert report["training_dataset_eligible"] is False
    assert report["completed_label_count"] == 1
    assert report["link_screening"][0]["status"] == "SCREENED"
    assert old == (events.read_bytes(), closes.read_bytes(), bindings.read_bytes())
