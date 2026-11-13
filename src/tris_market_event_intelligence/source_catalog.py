"""Financial news feed registry with optional, validated local extensions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

# Existing IDs remain unchanged; historical SQLite checkpoints depend on them.
DEFAULT_FEEDS: dict[str, str] = {
    "fed_monetary": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "bok_monetary": "https://www.bok.or.kr/portal/bbs/P0000559/news.rss?menuNo=200690",
    # Published directly by the Bank of Korea: separate economics and corrections sources.
    "bok_all": "https://www.bok.or.kr/portal/bbs/B0000552/news.rss?menuNo=200690",
    "bok_statistics": "https://www.bok.or.kr/portal/bbs/B0000501/news.rss?menuNo=201264",
    "bok_corrections": "https://www.bok.or.kr/portal/bbs/P0000560/news.rss?menuNo=200040",
    "fed_all": "https://www.federalreserve.gov/feeds/press_all.xml",
    "bls_cpi": "https://www.bls.gov/feed/cpi.rss",
    "bls_ppi": "https://www.bls.gov/feed/ppi.rss",
    # Separate official indicator feed; it does not replace or disguise a blocked PPI feed.
    "bls_latest": "https://www.bls.gov/feed/bls_latest.rss",
    "bls_employment": "https://www.bls.gov/feed/empsit.rss",
    "sec_press": "https://www.sec.gov/news/pressreleases.rss",
    "bbc_business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "bbc_technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
}
SOURCE_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
MAX_SOURCES = 50
MAX_CONFIG_BYTES = 65536


def validate_feeds(feeds: dict[str, str]) -> dict[str, str]:
    if not feeds or len(feeds) > MAX_SOURCES:
        raise ValueError("Feed count must be between 1 and 50")
    for name, url in feeds.items():
        if not isinstance(name, str) or SOURCE_ID.fullmatch(name) is None:
            raise ValueError(f"Invalid feed source ID: {name!r}")
        if not isinstance(url, str) or len(url) > 2048:
            raise ValueError(f"Invalid feed URL for {name}")
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username
                or parsed.password or parsed.fragment):
            raise ValueError(f"Feed {name} must be a public HTTPS URL without credentials")
        # Custom sources are external trusted configurations, not user-supplied requests.
        # Restrict URL-form localhost to avoid common accidental local API access.
        if parsed.hostname.lower() in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Loopback feed URLs are not permitted")
    return feeds.copy()


def load_feeds(config_path: Path | None = None, *, only: set[str] | None = None) -> dict[str, str]:
    feeds = DEFAULT_FEEDS.copy()
    if config_path is not None:
        if config_path.stat().st_size > MAX_CONFIG_BYTES:
            raise ValueError("Feed configuration exceeds 64 KiB")
        obj = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict) or set(obj) != {"feeds"} or not isinstance(obj["feeds"], dict):
            raise ValueError("Config must contain a feeds object")
        for key, value in obj["feeds"].items():
            if key in feeds and value != feeds[key]:
                raise ValueError(f"Built-in feed {key} cannot be silently replaced")
            feeds[key] = value
    feeds = validate_feeds(feeds)
    if only is not None:
        unknown = only.difference(feeds)
        if unknown:
            raise ValueError(f"Unknown feed IDs: {', '.join(sorted(unknown))}")
        feeds = {key: value for key, value in feeds.items() if key in only}
        if not feeds:
            raise ValueError("At least one feed must be selected")
    return feeds
