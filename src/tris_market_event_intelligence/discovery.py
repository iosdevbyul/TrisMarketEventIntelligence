"""Optional Korean/international news discovery using public search APIs.

Discovery returns articles or leads, not confirmed economic events.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import urlsplit

import httpx

from tris_market_event_intelligence.collector import Article, NewsStore, utc_now
from tris_market_event_intelligence.events import clean_text

NAVER_URL = "https://openapi.naver.com/v1/search/news.json"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RESPONSE_BYTES = 2_000_000

# Broad categories are searches, not a guarantee of news coverage.
NAVER_TOPICS = (
    "경제 금융 증시",
    "반도체 AI 기업",
    "금리 환율 물가 고용",
    "기업 실적 CEO 인수합병",
    "관세 무역 규제 에너지",
)
GDELT_TOPICS = (
    '(inflation OR "interest rates" OR unemployment OR tariffs)',
    '(semiconductor OR "artificial intelligence" OR earnings OR "chief executive")',
    '(sanctions OR "military conflict" OR "energy supply" OR "service outage")',
)


@dataclass(frozen=True)
class SourceResult:
    status: str
    inserted: int
    returned: int


def _json_payload(data: bytes) -> dict[str, object]:
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("Discovery response exceeds 2MB")
    obj = json.loads(data)
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


def _published(value: str) -> str | None:
    if not value:
        return None
    try:
        stamp = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        try:
            stamp = datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            return None
    if stamp.tzinfo is None:
        return None
    return stamp.astimezone(UTC).isoformat()


def _public_url(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 2048:
        return None
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return None
    if parts.username or parts.password:
        return None
    return value


def parse_naver(data: bytes, detected_at: str) -> list[Article]:
    items = _json_payload(data).get("items", [])
    if not isinstance(items, list):
        raise ValueError("Invalid Naver items")
    output = []
    for item in items[:100]:
        if not isinstance(item, dict):
            continue
        url = _public_url(item.get("originallink")) or _public_url(item.get("link"))
        title = clean_text(str(item.get("title") or ""))
        if not url or not title:
            continue
        output.append(Article(
            "naver_news", url, url, title,
            clean_text(str(item.get("description") or "")) or None,
            _published(str(item.get("pubDate") or "")), detected_at,
        ))
    return output


def parse_gdelt(data: bytes, detected_at: str) -> list[Article]:
    items = _json_payload(data).get("articles", [])
    if not isinstance(items, list):
        raise ValueError("Invalid GDELT articles")
    output = []
    for item in items[:250]:
        if not isinstance(item, dict):
            continue
        url = _public_url(item.get("url"))
        title = clean_text(str(item.get("title") or ""))
        if not url or not title:
            continue
        output.append(Article(
            "gdelt_doc", url, url, title, None,
            _published(str(item.get("seendate") or "")), detected_at,
        ))
    return output


class SearchDiscovery:
    def __init__(
        self, store: NewsStore, client: httpx.Client | None = None,
        naver_client_id: str | None = None,
        naver_client_secret: str | None = None,
        naver_topics: tuple[str, ...] = NAVER_TOPICS,
        gdelt_topics: tuple[str, ...] = GDELT_TOPICS,
    ) -> None:
        self.store = store
        self.client = client
        self.naver_client_id = naver_client_id
        self.naver_client_secret = naver_client_secret
        self.naver_topics = naver_topics
        self.gdelt_topics = gdelt_topics

    def collect_once(self, *, naver: bool = True, gdelt: bool = True) -> dict[str, SourceResult]:
        self.store.initialize()
        owned = self.client is None
        client = self.client or httpx.Client(
            timeout=20, follow_redirects=True, trust_env=False,
            headers={"User-Agent": "TrisMarketEventIntelligence/0.1 (+local research)"},
        )
        results: dict[str, SourceResult] = {}
        try:
            if naver:
                if not self.naver_client_id or not self.naver_client_secret:
                    results["naver_news"] = SourceResult("not_configured", 0, 0)
                else:
                    results["naver_news"] = self._collect(
                        client, source="naver_news", endpoint=NAVER_URL,
                        topics=self.naver_topics,
                        headers={
                            "X-Naver-Client-Id": self.naver_client_id,
                            "X-Naver-Client-Secret": self.naver_client_secret,
                        },
                        parse=parse_naver,
                    )
            if gdelt:
                results["gdelt_doc"] = self._collect(
                    client, source="gdelt_doc", endpoint=GDELT_URL,
                    topics=self.gdelt_topics, headers={}, parse=parse_gdelt,
                )
        finally:
            if owned:
                client.close()
        return results

    def _collect(
        self, client: httpx.Client, *, source: str, endpoint: str,
        topics: tuple[str, ...], headers: dict[str, str], parse: object,
    ) -> SourceResult:
        inserted = returned = 0
        errors = 0
        for query in topics:
            now = utc_now()
            params: dict[str, str | int]
            if source == "naver_news":
                params = {"query": query, "display": 100, "start": 1, "sort": "date"}
            else:
                params = {
                    "query": query, "mode": "artlist", "format": "json",
                    "maxrecords": 75, "timespan": "1d", "sort": "datedesc",
                }
            try:
                response = client.get(endpoint, params=params, headers=headers)
                response.raise_for_status()
                articles = parse(response.content, now)
                count = self.store.save(
                    source, str(response.url), response.content, articles, now, None, None
                )
                inserted += count
                returned += len(articles)
            except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                # Never record exception text: some HTTP exceptions include URLs/query values.
                self.store.record_error(source, type(exc).__name__, now)
                errors += 1
        status = "ok" if not errors else "partial_error" if errors < len(topics) else "error"
        return SourceResult(status, inserted, returned)
