"""Registered publisher text feeds; expansion is opt-in until real acceptance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from tris_market_event_intelligence.news_content_quality import media_page


@dataclass(frozen=True)
class Publisher:
    source: str
    feed: str
    hosts: tuple[str, ...]


BBC = (
    Publisher("bbc_business", "https://feeds.bbci.co.uk/news/business/rss.xml",
              ("www.bbc.com", "bbc.com", "www.bbc.co.uk", "bbc.co.uk")),
    Publisher("bbc_technology", "https://feeds.bbci.co.uk/news/technology/rss.xml",
              ("www.bbc.com", "bbc.com", "www.bbc.co.uk", "bbc.co.uk")),
    Publisher("bbc_world", "https://feeds.bbci.co.uk/news/world/rss.xml",
              ("www.bbc.com", "bbc.com", "www.bbc.co.uk", "bbc.co.uk")),
)
CANDIDATES = (
    Publisher("hankyung_all", "https://www.hankyung.com/feed/all-news",
              ("www.hankyung.com", "hankyung.com")),
    Publisher("mk_all", "https://www.mk.co.kr/rss/40300001/",
              ("www.mk.co.kr", "mk.co.kr", "m.mk.co.kr")),
    Publisher("yna_economy", "https://www.yna.co.kr/rss/economy.xml",
              ("www.yna.co.kr", "yna.co.kr", "m.yna.co.kr")),
    Publisher("cnbc_top", "https://www.cnbc.com/id/100003114/device/rss/rss.html",
              ("www.cnbc.com", "cnbc.com")),
)
PUBLISHERS = {p.source: p for p in (*BBC, *CANDIDATES)}
MEDIA_SQL = "a.source IN (" + ",".join("'" + key + "'" for key in PUBLISHERS) + ")"


class NewsSourceProvider(Protocol):
    """Transport seam: return original discovery bytes, not generated summaries."""

    def fetch(self, client: httpx.Client, url: str,
              headers: dict[str, str]) -> httpx.Response: ...


class RSSNewsProvider:
    def fetch(self, client: httpx.Client, url: str,
              headers: dict[str, str]) -> httpx.Response:
        return client.get(url, headers=headers)


def accepts(source: str, url: str) -> bool:
    publisher = PUBLISHERS.get(source)
    if publisher is None:
        return True
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == "https" and parsed.hostname in publisher.hosts
                and parsed.port in (None, 443) and not parsed.username
                and not parsed.password and not media_page(url))
    except ValueError:
        return False


def canonical_url(url: str) -> str:
    """Only remove known tracking fields; retain content-affecting queries."""
    p = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in
             {"at_medium", "at_campaign", "at_source", "fbclid", "gclid"}]
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path,
                       urlencode(query), ""))
