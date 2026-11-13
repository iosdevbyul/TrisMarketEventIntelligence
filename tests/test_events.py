from pathlib import Path

import pytest

from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.events import clean_text, extract_policy_event


@pytest.mark.parametrize(
    ("title", "summary", "status", "direction", "delta"),
    [
        ("Federal funds target rate", "The Committee raised the interest rate by 25 basis points.", "VERIFIED", "UP", 25),
        ("Federal funds rate", "The Committee lowered the policy rate by 0.50 percentage points.", "VERIFIED", "DOWN", -50),
        ("기준금리 결정", "한국은행이 기준금리를 25bp 인상했다고 발표했다.", "VERIFIED", "UP", 25),
        ("기준금리 결정", "한국은행이 기준금리를 동결했다고 발표했다.", "VERIFIED", "UNCHANGED", 0),
        ("Federal funds rate", "The Committee is expected to raise rates by 25 basis points.", "CANDIDATE", None, None),
        ("Federal funds rate", "The Committee may cut the interest rate by 25 basis points.", "CANDIDATE", None, None),
        ("Federal funds rate", "The Committee raised the interest rate.", "CANDIDATE", "UP", None),
        ("Federal funds rate", "The Committee raised the interest rate by 25 basis points and 50 basis points.", "CANDIDATE", "UP", None),
        ("Federal funds rate", "The Committee raised and lowered interest rates by 25 basis points.", "CANDIDATE", None, None),
        ("Federal funds rate", "The Committee discussed interest rates.", "CANDIDATE", None, None),
    ],
)
def test_policy_decision_classification(
    title: str, summary: str, status: str, direction: str | None, delta: int | None
) -> None:
    event = extract_policy_event("fed_monetary", title, summary)
    assert event is not None
    assert (event.status, event.direction, event.change_basis_points) == (
        status, direction, delta
    )


def test_untrusted_source_never_verified() -> None:
    event = extract_policy_event(
        "random_blog", "Federal funds rate",
        "The Committee raised the interest rate by 25 basis points."
    )
    assert event is not None
    assert event.status == "UNVERIFIED"


def test_unrelated_news_does_not_generate_event() -> None:
    assert extract_policy_event("fed_monetary", "Holiday announcement", "Office closed") is None


def test_summary_html_is_stripped() -> None:
    assert clean_text("<b>Policy</b> &amp; <i>rates</i>") == "Policy & rates"


def test_process_persists_provenance_and_resume_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    payload = b"""<rss><channel><item><title>Federal funds rate</title>
    <link>https://www.federalreserve.gov/a</link><guid>a1</guid>
    <description>The Committee raised the interest rate by 25 basis points.</description>
    </item><item><title>Office schedule</title>
    <link>https://www.federalreserve.gov/b</link><guid>b1</guid>
    <description>Office closed.</description></item></channel></rss>"""
    articles = parse_rss("fed_monetary", payload, "2026-11-14T00:00:00+00:00")
    assert store.save(
        "fed_monetary", "https://www.federalreserve.gov/feeds/press_monetary.xml",
        payload, articles, "2026-11-14T00:00:00+00:00", None, None
    ) == 2
    processor = EventProcessor(db)
    first = processor.process_pending()
    assert (first.processed, first.events, first.verified) == (2, 1, 1)
    assert processor.process_pending().processed == 0
    events = processor.list_events()
    assert len(events) == 1
    assert events[0]["source"] == "fed_monetary"
    assert events[0]["capture_sha256"]
    assert events[0]["source_url"] == "https://www.federalreserve.gov/a"
    assert events[0]["detected_at"] == "2026-11-14T00:00:00+00:00"


def test_batch_resume(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    payload = b"""<rss><channel><item><title>Policy rates</title>
    <link>https://example.org/a</link><guid>a</guid>
    <description>The Committee raised rates by 25 basis points.</description>
    </item><item><title>Policy rates</title><link>https://example.org/b</link>
    <guid>b</guid><description>The Committee may raise rates.</description>
    </item></channel></rss>"""
    items = parse_rss("fed_monetary", payload, "2026-11-14T00:00:00+00:00")
    store.save("fed_monetary", "https://example.org/feed", payload, items, "now", None, None)
    processor = EventProcessor(db)
    assert processor.process_pending(batch_size=1).processed == 1
    assert processor.process_pending(batch_size=1).processed == 1
    assert processor.process_pending(batch_size=1).processed == 0
    assert len(processor.list_events()) == 2


def test_invalid_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        EventProcessor(tmp_path / "db.sqlite3").process_pending(0)
