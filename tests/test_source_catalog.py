import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.source_catalog import DEFAULT_FEEDS, load_feeds


def test_defaults_cover_economy_regulation_world_and_technology() -> None:
    assert {"fed_monetary", "bok_monetary", "bls_cpi", "bls_ppi", "bls_employment",
            "sec_press", "bbc_world", "bbc_business", "bbc_technology"} <= DEFAULT_FEEDS.keys()
    assert len(load_feeds()) == len(DEFAULT_FEEDS)


def test_local_extension_preserves_default_feed_ids(tmp_path: Path) -> None:
    path = tmp_path / "feeds.json"
    path.write_text(json.dumps({"feeds": {
        "kr_industry": "https://news.example.kr/rss/industry.xml"
    }}), encoding="utf-8")
    feeds = load_feeds(path, only={"kr_industry", "fed_monetary"})
    assert list(feeds) == ["fed_monetary", "kr_industry"]
    assert feeds["kr_industry"] == "https://news.example.kr/rss/industry.xml"


@pytest.mark.parametrize("feed", [
    {"bad id": "https://example.org/feed"},
    {"kr_source": "http://example.org/feed"},
    {"kr_source": "https://user:pass@example.org/feed"},
    {"kr_source": "https://127.0.0.1/feed"},
    {"kr_source": "file:///etc/passwd"},
])
def test_invalid_external_feed_rejected(tmp_path: Path, feed: dict[str, str]) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"feeds": feed}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_feeds(path)


def test_cannot_repoint_existing_historical_source(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"feeds": {"fed_monetary": "https://example.org/news"}}))
    with pytest.raises(ValueError, match="silently replaced"):
        load_feeds(path)


def test_unknown_selection_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown feed"):
        load_feeds(only={"missing_source"})
