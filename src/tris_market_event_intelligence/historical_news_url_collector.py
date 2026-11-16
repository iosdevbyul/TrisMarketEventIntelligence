"""Bounded HTTPS news article collection for a blind historical test.

Fail closed on inaccessible original article, unknown publication timestamp,
revisions after cutoff, missing substantive article body, or disallowed host.
These are *currently served* copies, not independently archived PIT snapshots.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx

ALLOWED = frozenset({
    "techcrunch.com", "www.techcrunch.com", "axios.com", "www.axios.com",
    "www.theguardian.com", "theguardian.com", "www.reuters.com", "reuters.com",
    "www.washingtonpost.com", "washingtonpost.com", "apnews.com", "www.apnews.com",
})
MAX_HTML = 1_800_000
MAX_ARTICLE = 2800
PUBLISHED_META = frozenset({
    "article:published_time", "article:published", "datepublished",
    "date", "pubdate", "publishdate", "citation_publication_date",
})
MODIFIED_META = frozenset({
    "article:modified_time", "datemodified", "lastmod",
})
CANONICAL_DATE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d")


def _date(raw: str) -> datetime:
    try:
        result = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("NEWS_URL_PUBLICATION_TIME_INVALID") from exc
    if result.utcoffset() is None or not CANONICAL_DATE.search(raw):
        raise ValueError("NEWS_URL_PUBLICATION_TIME_AMBIGUOUS")
    return result.astimezone(UTC)


def _url(value: str) -> str:
    p = urlsplit(value)
    if (p.scheme != "https" or p.hostname not in ALLOWED
            or p.username or p.password or p.port not in (None, 443)
            or not p.path or p.query or p.fragment):
        raise ValueError("NEWS_URL_NOT_ALLOWED")
    return value


def _public_dns(url: str) -> None:
    """Never permit resolver results that reach loopback, LAN or reserved IPs."""
    host = urlsplit(url).hostname
    if not host:
        raise ValueError("NEWS_URL_NOT_ALLOWED")
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("NEWS_URL_DNS_FAILED") from exc
    if not addresses or any(
        not ipaddress.ip_address(record[4][0]).is_global for record in addresses
    ):
        raise ValueError("NEWS_URL_PRIVATE_NETWORK_BLOCKED")


class ArticleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.title: list[str] = []
        self.in_title = False
        self.article_depth = 0
        self.depth = 0
        self.p_depth = 0
        self.pieces: list[str] = []
        self.paragraphs: list[str] = []
        self.ld_mode = False
        self.ld_scripts: list[str] = []
        self.ld_current: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "meta":
            name = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            value = a.get("content")
            if name and value:
                self.meta.setdefault(name, value)
        if tag == "title":
            self.in_title = True
        if tag == "script" and a.get("type") == "application/ld+json":
            self.ld_mode = True
            self.ld_current = []
        if tag == "article" and not self.article_depth:
            self.article_depth = self.depth + 1
        if tag == "p" and self.article_depth and not self.p_depth:
            self.p_depth = self.depth + 1
            self.pieces = []
        if tag not in {"meta", "link", "img", "br", "hr", "input", "source", "wbr"}:
            self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag == "script" and self.ld_mode:
            self.ld_scripts.append("".join(self.ld_current))
            self.ld_mode = False
        if tag == "p" and self.p_depth:
            value = " ".join("".join(self.pieces).split())
            if len(value) >= 35:
                self.paragraphs.append(value)
            self.p_depth = 0
            self.pieces = []
        if tag == "article" and self.article_depth:
            self.article_depth = 0
        if tag not in {"meta", "link", "img", "br", "hr", "input", "source", "wbr"}:
            self.depth = max(0, self.depth - 1)

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title.append(data)
        if self.ld_mode:
            self.ld_current.append(data)
        if self.p_depth:
            self.pieces.append(data)


def _jsonld_dates(doc: ArticleHTML) -> tuple[list[str], list[str]]:
    published: list[str] = []
    modified: list[str] = []
    def walk(obj: object) -> None:
        if isinstance(obj, list):
            for x in obj:
                walk(x)
        elif isinstance(obj, dict):
            category = obj.get("@type", "")
            is_article = any(
                "Article" in str(x) or "News" in str(x)
                for x in (category if isinstance(category, list) else [category])
            )
            if is_article:
                for field, dest in (
                    ("datePublished", published), ("dateModified", modified),
                ):
                    value = obj.get(field)
                    if isinstance(value, str):
                        dest.append(value)
            for value in obj.values():
                if isinstance(value, (list, dict)):
                    walk(value)
    for script in doc.ld_scripts[:20]:
        try:
            walk(json.loads(script))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return published, modified


def parse_article(raw: bytes, url: str, as_of: str, article_id: str) -> dict:
    try:
        doc = ArticleHTML()
        doc.feed(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("NEWS_URL_HTML_INVALID") from exc
    ld_published, ld_modified = _jsonld_dates(doc)
    published = [
        value for key, value in doc.meta.items() if key in PUBLISHED_META
    ] + ld_published
    modified = [
        value for key, value in doc.meta.items() if key in MODIFIED_META
    ] + ld_modified
    if not published:
        raise ValueError("NEWS_URL_PUBLICATION_TIME_MISSING")
    dates = [_date(x) for x in published]
    # Conflicting article clocks cannot be silently resolved by guesswork.
    if max(dates) - min(dates) > __import__("datetime").timedelta(hours=2):
        raise ValueError("NEWS_URL_PUBLICATION_TIME_CONFLICT")
    effective = min(dates)
    cutoff = _date(as_of)
    if effective > cutoff:
        raise ValueError("NEWS_URL_ARTICLE_AFTER_CUTOFF")
    if any(_date(x) > cutoff for x in modified):
        raise ValueError("NEWS_URL_REVISED_AFTER_CUTOFF")
    title = doc.meta.get("og:title") or " ".join("".join(doc.title).split())
    body = "\n".join(doc.paragraphs)
    if not 8 <= len(title) <= 240 or len(body) < 250 or len(doc.paragraphs) < 3:
        raise ValueError("NEWS_URL_ORIGINAL_BODY_NOT_AVAILABLE")
    if len(body) > MAX_ARTICLE:
        # Bounded beginning of *extracted primary text*, no invented content.
        body = body[:MAX_ARTICLE].rsplit(" ", 1)[0]
    return {
        "articleId": article_id,
        "publishedAt": effective.isoformat(),
        "source": urlsplit(url).hostname or "unknown",
        "title": title,
        "body": body,
    }


def collect_article(url: str, as_of: str, article_id: str, *, client: httpx.Client) -> dict:
    target = _url(url)
    for _ in range(3):
        _public_dns(target)
        try:
            response = client.get(target, follow_redirects=False,
                                  headers={"User-Agent": "TrisMarketEventIntelligence/1.0 research"})
        except httpx.HTTPError as exc:
            raise ValueError("NEWS_URL_FETCH_FAILED") from exc
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("location")
            if not location:
                raise ValueError("NEWS_URL_REDIRECT_INVALID")
            target = _url(urljoin(target, location))
            continue
        if response.status_code != 200:
            raise ValueError("NEWS_URL_FETCH_HTTP_FAILURE")
        if "text/html" not in response.headers.get("content-type", ""):
            raise ValueError("NEWS_URL_HTML_REQUIRED")
        if len(response.content) > MAX_HTML:
            raise ValueError("NEWS_URL_HTML_OVERSIZED")
        return parse_article(response.content, target, as_of, article_id)
    raise ValueError("NEWS_URL_TOO_MANY_REDIRECTS")


def collect_urls(
    urls: list[str], *, as_of: str, case_id: str, horizon: int,
    client: httpx.Client | None = None,
) -> dict:
    if not 2 <= len(urls) <= 6 or len(set(urls)) != len(urls):
        raise ValueError("NEWS_URL_COUNT_OR_DUPLICATES")
    _date(as_of)
    owned = client is None
    session = client or httpx.Client(
        timeout=httpx.Timeout(15), trust_env=False, follow_redirects=False,
    )
    try:
        articles = [
            collect_article(url, as_of, f"a{i+1}", client=session)
            for i, url in enumerate(urls)
        ]
    finally:
        if owned:
            session.close()
    return {
        "schemaVersion": "historical_news_impact_test_v1",
        "caseId": case_id,
        "asOf": as_of,
        "horizonTradingDays": horizon,
        "articles": articles,
        "knownTickers": [],
    }
